"""HTTP routes for per-MCP OAuth/DCR connect, callback, status, and Apps host proxy."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from deep_agent.src.agent.config import agent_config

router = APIRouter(tags=["mcp"])


async def _authenticated_user_id(request: Request) -> str:
    """Return the SSO ``sub`` from the incoming Bearer token."""
    from deep_agent.aegra.auth import (
        DEV_USER_ID,
        ENABLE_AUTH,
        ENVIRONMENT,
        _decode_token,
    )
    from deep_agent.utils.pylogger import get_python_logger

    logger = get_python_logger()

    # Block auth bypass in production
    if ENVIRONMENT == "production" and not ENABLE_AUTH:
        raise HTTPException(
            status_code=500, detail="Authentication bypass disabled in production"
        )

    if not ENABLE_AUTH:
        logger.warning("Auth bypass active for MCP routes (development mode)")
        return DEV_USER_ID

    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Missing or invalid Authorization header"
        )

    payload = _decode_token(auth_header[7:])
    return str(payload["sub"])


def _sso_bearer_from_request(request: Request) -> str | None:
    """Return the raw Bearer token (SSO access token) when present."""
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip() or None
    return None


def _http_exception_response(exc: HTTPException) -> JSONResponse | None:
    """Return JSONResponse for dict ``detail`` so aegra does not stringify-fail.

    aegra's ``AgentProtocolError.message`` is a string; dict details (e.g.
    ``authorization_required``, ``tool_not_app_callable``) must bypass that
    handler or the client sees 500 instead of 401/403.
    """
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return None


@router.post("/mcp/{mcp_name}/connect")
async def mcp_connect(mcp_name: str, request: Request) -> JSONResponse:
    """Start OAuth/DCR authorization for an MCP server."""
    from deep_agent.aegra.mcp_oauth_handlers import handle_mcp_connect
    from deep_agent.src.agent.config import agent_config
    from deep_agent.src.settings import settings

    servers = agent_config.get_mcp_servers()
    cfg = servers.get(mcp_name, {})
    if cfg.get("auth_mode") == "dcr" and not settings.MCP_DCR_ENABLED:
        return JSONResponse(
            status_code=403,
            content={"detail": "DCR is disabled"},
        )

    user_id = await _authenticated_user_id(request)
    caller_origin = request.headers.get("origin")
    result = await handle_mcp_connect(user_id, mcp_name, caller_origin=caller_origin)
    return JSONResponse(content=result)


@router.get("/mcp/oauth/callback")
async def mcp_oauth_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
) -> HTMLResponse:
    """Handle the OAuth redirect — exchange code and notify the UI opener."""
    from deep_agent.aegra.mcp_oauth_handlers import handle_mcp_oauth_callback

    return await handle_mcp_oauth_callback(code, state, request)


@router.get("/mcp/{mcp_name}/status")
async def mcp_status(mcp_name: str, request: Request) -> JSONResponse:
    """Return whether the current user has a valid token for the MCP."""
    from deep_agent.aegra.mcp_oauth_handlers import handle_mcp_status

    user_id = await _authenticated_user_id(request)
    result = await handle_mcp_status(user_id, mcp_name)
    return JSONResponse(content=result)


@router.get("/info")
async def get_agent_info() -> dict[str, Any]:
    """Return agent identity metadata from config."""
    servers = agent_config.get_mcp_servers()
    from deep_agent.src.settings import settings

    allowed_auth_modes = {"oauth", "dcr"} if settings.MCP_DCR_ENABLED else {"oauth"}
    oauth_mcps = sorted(
        name
        for name, cfg in servers.items()
        if cfg.get("enabled")
        and cfg.get("auth_mode") in allowed_auth_modes
        and (cfg.get("oauth") or {}).get("grant_type") != "client_credentials"
    )
    return {"name": agent_config.get_name(), "oauth_mcps": oauth_mcps}


@router.post("/mcp/{mcp_name}/tools/list")
async def mcp_tools_list(mcp_name: str, request: Request) -> JSONResponse:
    """Proxy MCP ``tools/list`` for Apps host metadata (stateless)."""
    from deep_agent.aegra.mcp_host import list_tools

    user_id = await _authenticated_user_id(request)
    cursor: str | None = None
    try:
        body = await request.json()
    except Exception:
        body = {}
    if body is not None and not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    if isinstance(body, dict):
        raw_cursor = body.get("cursor")
        if raw_cursor is not None and not isinstance(raw_cursor, str):
            raise HTTPException(status_code=400, detail="cursor must be a string")
        cursor = raw_cursor

    try:
        result = await list_tools(
            mcp_name,
            cursor=cursor,
            user_id=user_id,
            sso_token=_sso_bearer_from_request(request),
        )
    except HTTPException as exc:
        as_json = _http_exception_response(exc)
        if as_json is not None:
            return as_json
        raise
    return JSONResponse(content=result)


@router.post("/mcp/{mcp_name}/resources/list")
async def mcp_resources_list(mcp_name: str, request: Request) -> JSONResponse:
    """Proxy MCP ``resources/list`` for Apps Views (stateless)."""
    from deep_agent.aegra.mcp_host import list_resources

    user_id = await _authenticated_user_id(request)
    cursor: str | None = None
    try:
        body = await request.json()
    except Exception:
        body = {}
    if body is not None and not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    if isinstance(body, dict):
        raw_cursor = body.get("cursor")
        if raw_cursor is not None and not isinstance(raw_cursor, str):
            raise HTTPException(status_code=400, detail="cursor must be a string")
        cursor = raw_cursor

    try:
        result = await list_resources(
            mcp_name,
            cursor=cursor,
            user_id=user_id,
            sso_token=_sso_bearer_from_request(request),
        )
    except HTTPException as exc:
        as_json = _http_exception_response(exc)
        if as_json is not None:
            return as_json
        raise
    return JSONResponse(content=result)


@router.post("/mcp/{mcp_name}/resources/templates/list")
async def mcp_resource_templates_list(mcp_name: str, request: Request) -> JSONResponse:
    """Proxy MCP ``resources/templates/list`` for Apps Views (stateless)."""
    from deep_agent.aegra.mcp_host import list_resource_templates

    user_id = await _authenticated_user_id(request)
    cursor: str | None = None
    try:
        body = await request.json()
    except Exception:
        body = {}
    if body is not None and not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    if isinstance(body, dict):
        raw_cursor = body.get("cursor")
        if raw_cursor is not None and not isinstance(raw_cursor, str):
            raise HTTPException(status_code=400, detail="cursor must be a string")
        cursor = raw_cursor

    try:
        result = await list_resource_templates(
            mcp_name,
            cursor=cursor,
            user_id=user_id,
            sso_token=_sso_bearer_from_request(request),
        )
    except HTTPException as exc:
        as_json = _http_exception_response(exc)
        if as_json is not None:
            return as_json
        raise
    return JSONResponse(content=result)


@router.post("/mcp/{mcp_name}/resources/read")
async def mcp_resources_read(mcp_name: str, request: Request) -> JSONResponse:
    """Proxy MCP ``resources/read`` for Apps (any resource URI; stateless)."""
    from deep_agent.aegra.mcp_host import read_resource

    user_id = await _authenticated_user_id(request)
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")

    uri = body.get("uri")
    try:
        result = await read_resource(
            mcp_name,
            uri if isinstance(uri, str) else "",
            user_id=user_id,
            sso_token=_sso_bearer_from_request(request),
        )
    except HTTPException as exc:
        as_json = _http_exception_response(exc)
        if as_json is not None:
            return as_json
        raise
    return JSONResponse(content=result)


@router.post("/mcp/{mcp_name}/tools/call")
async def mcp_tools_call(mcp_name: str, request: Request) -> JSONResponse:
    """Proxy MCP ``tools/call`` for app-initiated tools (stateless).

    Enforces tool ``visibility`` includes ``app``. Does not use the per-pod
    LLM tool cache — each call lists/calls against the live MCP server.
    """
    from deep_agent.aegra.mcp_host import call_app_tool

    user_id = await _authenticated_user_id(request)
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")

    tool_name = body.get("name")
    arguments = body.get("arguments")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise HTTPException(
            status_code=400, detail="arguments must be a JSON object when provided"
        )

    try:
        result = await call_app_tool(
            mcp_name,
            tool_name if isinstance(tool_name, str) else "",
            arguments,
            user_id=user_id,
            sso_token=_sso_bearer_from_request(request),
        )
    except HTTPException as exc:
        as_json = _http_exception_response(exc)
        if as_json is not None:
            return as_json
        raise
    return JSONResponse(content=result)
