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
        raise _authorization_required(mcp_name) from None

    if not auth_required:
        return bearer

    if auth_mode in ("oauth", "dcr") and not bearer:
        raise _authorization_required(mcp_name)

    if auth_mode == "sso" and not bearer:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization bearer token for SSO MCP access",
        )

    if auth_mode == "api_key" and not bearer:
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
    async with asyncio.timeout(timeout):
        async with client.session(mcp_name) as session:
            yield session


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
