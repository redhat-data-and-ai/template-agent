"""Request-scoped MCP Apps host proxy (resources + app tools/call).

These helpers open a short-lived MCP session per HTTP request. Auth tokens
come from Redis (oauth/dcr) or the caller's SSO bearer — nothing is stored
in process memory for multi-pod safety.
"""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast

from fastapi import HTTPException
from langchain_mcp_adapters.client import MultiServerMCPClient

from deep_agent.aegra.mcp import (
    _build_server_config,
    _get_server_configs,
    _resolve_connection_token,
)
from deep_agent.aegra.mcp_apps import (
    ensure_mcp_apps_capability_advertised,
    get_tool_visibility,
    is_app_callable,
)
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

# Cap tools/list pagination when resolving a tool by name (host tools/call).
_MAX_TOOLS_LIST_PAGES = settings.MCP_TOOLS_LIST_MAX_PAGES


def _tool_from_mcp_meta(name: str, meta: dict[str, Any] | None) -> SimpleNamespace:
    """Adapt an MCP tool meta dict into the shape used by mcp_apps helpers."""
    return SimpleNamespace(name=name, metadata={"_meta": meta or {}})


def _authorization_required(mcp_name: str) -> HTTPException:
    from deep_agent.aegra.mcp_auth import get_mcp_credential_resolver

    return HTTPException(
        status_code=401,
        detail={
            "error": "authorization_required",
            "mcp_name": mcp_name,
            "connect_url": get_mcp_credential_resolver().connect_url(mcp_name),
        },
    )


def _get_enabled_server(mcp_name: str) -> dict[str, Any]:
    entry = _get_server_configs().get(mcp_name)
    if not entry or not entry.get("enabled", False):
        raise HTTPException(
            status_code=404,
            detail=f"Unknown or disabled MCP server: {mcp_name}",
        )
    return entry


async def _resolve_bearer(
    mcp_name: str,
    entry: dict[str, Any],
    *,
    user_id: str,
    sso_token: str | None,
) -> str | None:
    """Resolve a bearer token for this request or raise HTTP 401 when required."""
    from deep_agent.aegra.mcp_auth import NeedsAuthorization

    auth_mode = entry.get("auth_mode", "sso")
    auth_required = entry.get("auth", True)

    try:
        bearer = await _resolve_connection_token(mcp_name, entry, sso_token, user_id)
    except NeedsAuthorization:
        logger.info(
            "[%s] NeedsAuthorization during host bearer resolution",
            mcp_name,
        )
        raise _authorization_required(mcp_name) from None

    if not auth_required:
        return bearer

    if auth_mode in ("oauth", "dcr") and not bearer:
        logger.info(
            "[%s] no %s bearer token in host proxy — authorization required",
            mcp_name,
            auth_mode,
        )
        raise _authorization_required(mcp_name)

    if auth_mode == "sso" and not bearer:
        logger.info(
            "[%s] missing SSO bearer token in host proxy",
            mcp_name,
        )
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization bearer token for SSO MCP access",
        )

    if auth_mode == "api_key" and not bearer:
        logger.info(
            "[%s] api_key not configured on agent — host proxy blocked",
            mcp_name,
        )
        raise HTTPException(
            status_code=500,
            detail=f"MCP '{mcp_name}' api_key is not configured on the agent",
        )

    return bearer


# Sessions open for the current turn, keyed by (server, user).
#
# Every host-proxy operation used to build its own client and pay a
# full MCP `initialize` handshake. Two resource listings in one turn
# meant two handshakes and twelve seconds against a server that
# answers each call in well under one — and they ran back to back,
# which is what "things happen sequentially in the subagent" looks
# like from outside.
#
# Scoped to the turn and keyed on the user, because a session is
# established with that user's bearer and must never be handed to
# anyone else.
_open_sessions: contextvars.ContextVar[dict[tuple[str, str], Any] | None] = (
    contextvars.ContextVar("_mcp_host_sessions", default=None)
)


async def enter_mcp_session_scope() -> None:
    """Begin reusing MCP sessions for this turn.

    There is no matching exit. The graph is built in the request
    context and runs in a worker task, so a context manager spanning
    the turn has nowhere to close — the same boundary that made a
    contextvar unreadable at serialisation time.

    Sessions are instead bounded by the client's own timeout and by
    the context dying with the task. That is a deliberate trade:
    reusing a handshake across a turn is worth more than closing it
    a few seconds earlier, and the alternative measured at twelve
    seconds for two listings.
    """
    if _open_sessions.get() is None:
        cache: dict[tuple[str, str], Any] = {}
        _open_sessions.set(cache)
        await _evict_old_session_stacks()
        _session_stacks[id(cache)] = (cache, AsyncExitStack())


@asynccontextmanager
async def mcp_session_scope() -> AsyncIterator[None]:
    """Reuse MCP sessions for the duration of one turn.

    Outside this scope every call opens and closes its own session,
    which is the safe default: a session left open past the request
    that authorised it is a connection nobody owns.
    """
    stack = AsyncExitStack()
    token = _open_sessions.set({})
    try:
        async with stack:
            scoped = _open_sessions.get()
            if scoped is not None:
                _session_stacks[id(scoped)] = (scoped, stack)
            yield
    finally:
        _session_stacks.pop(id(_open_sessions.get() or {}), None)
        _open_sessions.reset(token)


# The per-turn session cache alongside its stack, so eviction can
# empty the cache as well as drop the stack. Dropping the stack alone
# left the cache handing out sessions whose transport was gone.
_session_stacks: dict[int, tuple[dict[tuple[str, str], Any], AsyncExitStack]] = {}

# How many turn-scoped stacks may be held at once.
#
# `enter_mcp_session_scope` has no matching exit on purpose — the
# graph is built in the request context and runs in a worker task —
# so nothing ever removed its entry and the dict, with the MCP
# sessions each stack holds open, grew for the life of the process.
# The reuse-within-a-turn trade stands; the unbounded growth was not
# part of it. Oldest entries are closed first, and a stack still in
# use by a live turn is within this many turns of being created.
#
# Above `MAX_CONCURRENT_RUNS`, deliberately. At 32 against a limit of
# 40 concurrent runs the cap was reachable while every stack in it
# belonged to a turn still running, so eviction closed a session
# underneath a live answer. The cap is here to stop unbounded growth,
# not to ration sessions between concurrent turns.
_MAX_SESSION_STACKS = max(64, settings.MAX_CONCURRENT_RUNS * 2)


async def _evict_old_session_stacks() -> None:
    """Close the oldest stacks once there are more than the cap."""
    while len(_session_stacks) >= _MAX_SESSION_STACKS:
        oldest = next(iter(_session_stacks))
        cache, stack = _session_stacks.pop(oldest)
        # Empty the cache first. `aclose()` here runs in whichever
        # task tripped the cap, not the one that entered the stack,
        # so anyio raises and the close is only best-effort — but a
        # cache still holding those sessions would hand a dead one to
        # the next call in that turn, which is the worse failure.
        cache.clear()
        try:
            await stack.aclose()
        except Exception:
            # A stack whose task is already gone raises on close.
            # Dropping the reference is the point; the error is not.
            logger.debug("Could not close an expired MCP session stack", exc_info=True)


@asynccontextmanager
async def mcp_session(
    mcp_name: str,
    *,
    user_id: str,
    sso_token: str | None,
) -> AsyncIterator[Any]:
    """Open an MCP client session, reusing one inside a session scope."""
    ensure_mcp_apps_capability_advertised()

    cache = _open_sessions.get()
    key = (mcp_name, user_id or "")
    if cache is not None and key in cache:
        yield cache[key]
        return

    entry = _get_enabled_server(mcp_name)
    bearer = await _resolve_bearer(
        mcp_name, entry, user_id=user_id, sso_token=sso_token
    )
    config = _build_server_config(entry, bearer)
    client = MultiServerMCPClient({mcp_name: config})
    timeout = float(entry.get("timeout", 30))

    if cache is not None:
        # Inside a scope: hand the session's lifetime to the scope's
        # exit stack so later calls in this turn reuse it and it is
        # still closed exactly once, when the turn ends.
        entry_stack = _session_stacks.get(id(cache))
        stack = entry_stack[1] if entry_stack is not None else None
        if stack is not None:
            async with asyncio.timeout(timeout):
                session = await stack.enter_async_context(client.session(mcp_name))
            cache[key] = session
            logger.debug("[%s] MCP session opened for this turn", mcp_name)
            yield session
            return

    try:
        async with asyncio.timeout(timeout):
            async with client.session(mcp_name) as session:
                yield session
    except TimeoutError:
        logger.error(
            "[%s] host proxy MCP session timed out after %.0fs",
            mcp_name,
            timeout,
        )
        raise
    except Exception:
        logger.error(
            "[%s] host proxy MCP session failed",
            mcp_name,
            exc_info=True,
        )
        raise


async def _find_mcp_tool(session: Any, tool_name: str) -> Any | None:
    """Find a tool by name using paginated tools/list (live server, not cache)."""
    cursor: str | None = None
    for _ in range(_MAX_TOOLS_LIST_PAGES):
        page = await session.list_tools(cursor=cursor)
        for tool in page.tools or []:
            if tool.name == tool_name:
                return tool
        cursor = page.nextCursor
        if not cursor:
            return None
    return None


async def list_tools(
    mcp_name: str,
    *,
    cursor: str | None = None,
    user_id: str,
    sso_token: str | None,
) -> dict[str, Any]:
    """Proxy ``tools/list`` on *mcp_name* (host metadata; live server)."""
    async with mcp_session(mcp_name, user_id=user_id, sso_token=sso_token) as session:
        result = await session.list_tools(cursor=cursor)

    payload = cast(
        dict[str, Any],
        result.model_dump(by_alias=True, mode="json", exclude_none=True),
    )
    logger.info(
        "MCP Apps tools/list ok server=%s count=%d",
        mcp_name,
        len(payload.get("tools") or []),
    )
    return payload


async def count_capabilities(
    mcp_name: str, *, user_id: str, sso_token: str | None
) -> dict[str, int | None]:
    """How many tools, resources and prompts *mcp_name* exposes.

    One session for all three, because the connection handshake is the
    expensive part and the Settings list asks this for every connected
    server at once.

    A count is `None`, not `0`, when the server does not implement that
    capability — "no prompts" and "does not do prompts" are different
    facts, and a column of zeroes hides which is which. Counts are of
    the first page only; servers that paginate will under-report, and
    that is better than walking every cursor to put a number in a list.
    """
    counts: dict[str, int | None] = {"tools": None, "resources": None, "prompts": None}

    async with mcp_session(mcp_name, user_id=user_id, sso_token=sso_token) as session:
        for key, method in (
            ("tools", "list_tools"),
            ("resources", "list_resources"),
            ("prompts", "list_prompts"),
        ):
            call = getattr(session, method, None)
            if call is None:
                continue
            try:
                result = await call()
            except Exception as exc:
                # One unsupported capability must not cost the other
                # two: servers routinely reject `prompts/list` with a
                # method-not-found rather than an empty list.
                logger.debug("MCP %s %s failed on %s: %s", mcp_name, method, key, exc)
                continue
            items = getattr(result, key, None)
            if items is not None:
                counts[key] = len(items)

    logger.info("MCP capabilities server=%s %s", mcp_name, counts)
    return counts


async def inventory(
    mcp_name: str, *, user_id: str, sso_token: str | None
) -> dict[str, list[dict[str, str]]]:
    """Name and description for every tool, resource and prompt.

    One session for all three: the handshake is the expensive part,
    and a reader opening a server wants the whole picture, not three
    spinners.

    Descriptions are what make this worth showing — a list of forty
    tool names tells you nothing about whether the server is useful.
    They are truncated here because a tool description can be a page
    of prompt engineering, and this is a list, not a manual.
    """
    out: dict[str, list[dict[str, str]]] = {
        "tools": [],
        "resources": [],
        "prompts": [],
    }

    async with mcp_session(mcp_name, user_id=user_id, sso_token=sso_token) as session:
        for key, method in (
            ("tools", "list_tools"),
            ("resources", "list_resources"),
            ("prompts", "list_prompts"),
        ):
            call = getattr(session, method, None)
            if call is None:
                continue
            try:
                result = await call()
            except Exception as exc:
                # A server that does not implement prompts answers
                # with method-not-found; that must not cost the
                # tools we already have.
                logger.debug("MCP %s %s failed: %s", mcp_name, method, exc)
                continue
            for item in getattr(result, key, None) or []:
                uri = str(getattr(item, "uri", "") or "")
                # `title` is the spec's display name and wins when the
                # server bothers to set one.
                label = str(
                    getattr(item, "title", "") or getattr(item, "name", "") or uri
                )
                if not label:
                    continue
                description = str(getattr(item, "description", "") or "")
                entry: dict[str, Any] = {
                    "name": label,
                    "description": description[:300],
                }
                if uri and uri != label:
                    entry["uri"] = uri

                # The server's own statement about what the tool does,
                # which beats guessing from the name — there is no
                # `write_*` convention across providers, and the names
                # are theirs: `merge_merge_request`, `create_issue`,
                # `canvas_update_from_svg`.
                #
                # 261 of 323 tools across the eight configured servers
                # populate `readOnlyHint` (gworkspace 101/101, gitlab
                # 63/116, another 0/6). Absent stays absent rather
                # than defaulting to either value: "unknown" and
                # "writes" must be distinguishable, because the caller
                # treats unknown as needing approval.
                annotations = getattr(item, "annotations", None)
                if annotations is not None:
                    read_only = getattr(annotations, "readOnlyHint", None)
                    destructive = getattr(annotations, "destructiveHint", None)
                    if read_only is not None:
                        entry["read_only"] = bool(read_only)
                    if destructive is not None:
                        entry["destructive"] = bool(destructive)

                out[key].append(entry)

        # A label repeated across the list identifies nothing. Atlan
        # returns six resources all called `_resource` — FastMCP uses
        # the decorated function's name, and theirs is a private
        # helper — while every `uri` is distinct and meaningful
        # (`ui://atlan/asset-detail`, `ui://atlan/query-results`).
        #
        # Keyed on repetition rather than on the string `_resource`,
        # because the next server will produce its own flavour of the
        # same uselessness and a name that is unique is worth keeping
        # even when it is ugly.
        for key, items in out.items():
            seen: dict[str, int] = {}
            for entry in items:
                seen[entry["name"]] = seen.get(entry["name"], 0) + 1
            for entry in items:
                if seen[entry["name"]] > 1 and entry.get("uri"):
                    entry["name"] = entry["uri"]
                    entry.pop("uri", None)

    logger.info(
        "MCP inventory server=%s tools=%d resources=%d prompts=%d",
        mcp_name,
        len(out["tools"]),
        len(out["resources"]),
        len(out["prompts"]),
    )
    return out


async def list_resources(
    mcp_name: str,
    *,
    cursor: str | None = None,
    user_id: str,
    sso_token: str | None,
) -> dict[str, Any]:
    """Proxy ``resources/list`` on *mcp_name* (View → Host → Server)."""
    async with mcp_session(mcp_name, user_id=user_id, sso_token=sso_token) as session:
        result = await session.list_resources(cursor=cursor)

    payload = cast(
        dict[str, Any],
        result.model_dump(by_alias=True, mode="json", exclude_none=True),
    )
    logger.info(
        "MCP Apps resources/list ok server=%s count=%d",
        mcp_name,
        len(payload.get("resources") or []),
    )
    return payload


async def list_resource_templates(
    mcp_name: str,
    *,
    cursor: str | None = None,
    user_id: str,
    sso_token: str | None,
) -> dict[str, Any]:
    """Proxy ``resources/templates/list`` on *mcp_name* (View → Host → Server)."""
    async with mcp_session(mcp_name, user_id=user_id, sso_token=sso_token) as session:
        result = await session.list_resource_templates(cursor=cursor)

    payload = cast(
        dict[str, Any],
        result.model_dump(by_alias=True, mode="json", exclude_none=True),
    )
    logger.info(
        "MCP Apps resources/templates/list ok server=%s count=%d",
        mcp_name,
        len(
            payload.get("resourceTemplates") or payload.get("resource_templates") or []
        ),
    )
    return payload


async def read_resource(
    mcp_name: str,
    uri: str,
    *,
    user_id: str,
    sso_token: str | None,
) -> dict[str, Any]:
    """Proxy ``resources/read`` for any resource URI on *mcp_name*.

    Used both for host HTML fetch (``ui://``) and View ``readServerResource``
    (any server-registered URI, e.g. ``showcase://sample.json``).
    """
    if not isinstance(uri, str) or not uri.strip():
        raise HTTPException(status_code=400, detail="uri is required")

    async with mcp_session(mcp_name, user_id=user_id, sso_token=sso_token) as session:
        result = await session.read_resource(uri)

    payload = cast(
        dict[str, Any],
        result.model_dump(by_alias=True, mode="json", exclude_none=True),
    )
    logger.info(
        "MCP Apps resources/read ok server=%s uri=%s contents=%d",
        mcp_name,
        uri,
        len(payload.get("contents") or []),
    )
    return payload


# Back-compat alias for callers/tests that used the old name.
read_ui_resource = read_resource


async def call_app_tool(
    mcp_name: str,
    tool_name: str,
    arguments: dict[str, Any] | None,
    *,
    user_id: str,
    sso_token: str | None,
) -> dict[str, Any]:
    r"""Proxy ``tools/call`` for an app-visible tool on *mcp_name*.

    Enforces ``_meta.ui.visibility`` includes ``\"app\"`` using a live
    ``tools/list`` (not the per-pod LLM tool cache).
    """
    if not tool_name or not isinstance(tool_name, str):
        raise HTTPException(status_code=400, detail="tool name is required")

    # Defer HTTPException until after mcp_session exits. Raising inside the
    # streamable-HTTP TaskGroup wraps FastAPI errors as ExceptionGroup → 500.
    not_found_detail: str | None = None
    deny_detail: dict[str, Any] | None = None
    result: Any | None = None

    async with mcp_session(mcp_name, user_id=user_id, sso_token=sso_token) as session:
        tool = await _find_mcp_tool(session, tool_name)
        if tool is None:
            not_found_detail = f"Tool not found on MCP server '{mcp_name}': {tool_name}"
        else:
            adapted = _tool_from_mcp_meta(tool.name, getattr(tool, "meta", None))
            if not is_app_callable(adapted):
                deny_detail = {
                    "error": "tool_not_app_callable",
                    "mcp_name": mcp_name,
                    "tool": tool_name,
                    "visibility": get_tool_visibility(adapted),
                }
            else:
                result = await session.call_tool(tool_name, arguments or {})

    if not_found_detail is not None:
        raise HTTPException(status_code=404, detail=not_found_detail)
    if deny_detail is not None:
        raise HTTPException(status_code=403, detail=deny_detail)
    if result is None:
        raise HTTPException(status_code=500, detail="tools/call produced no result")

    payload = cast(
        dict[str, Any],
        result.model_dump(by_alias=True, mode="json", exclude_none=True),
    )
    logger.info(
        "MCP Apps tools/call ok server=%s tool=%s isError=%s",
        mcp_name,
        tool_name,
        payload.get("isError", False),
    )
    return payload
