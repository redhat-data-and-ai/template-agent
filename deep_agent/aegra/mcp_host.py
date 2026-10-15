"""Request-scoped MCP Apps host proxy (resources + app tools/call).

These helpers open a short-lived MCP session per HTTP request. Auth tokens
come from Redis (oauth/dcr) or the caller's SSO bearer — nothing is stored
in process memory for multi-pod safety.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
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
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

# Cap tools/list pagination when resolving a tool by name (host tools/call).
_MAX_TOOLS_LIST_PAGES = 20


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


@asynccontextmanager
async def mcp_session(
    mcp_name: str,
    *,
    user_id: str,
    sso_token: str | None,
) -> AsyncIterator[Any]:
    """Open a short-lived MCP client session for host proxy operations."""
    ensure_mcp_apps_capability_advertised()
    entry = _get_enabled_server(mcp_name)
    bearer = await _resolve_bearer(
        mcp_name, entry, user_id=user_id, sso_token=sso_token
    )
    config = _build_server_config(entry, bearer)
    client = MultiServerMCPClient({mcp_name: config})
    timeout = float(entry.get("timeout", 30))
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
                # 63/116, dataverse 0/6). Absent stays absent rather
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
