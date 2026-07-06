"""HTTP routes for per-MCP OAuth/DCR connect, callback, and status."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from deep_agent.src.agent.config import agent_config

router = APIRouter(tags=["mcp-oauth"])


async def _authenticated_user_id(request: Request) -> str:
    """Return the SSO ``sub`` from the incoming Bearer token."""
    from deep_agent.aegra.auth import DEV_USER_ID, ENABLE_AUTH, _decode_token

    if not ENABLE_AUTH:
        return DEV_USER_ID

    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Missing or invalid Authorization header"
        )

    payload = _decode_token(auth_header[7:])
    return str(payload["sub"])


@router.post("/mcp/{mcp_name}/connect")
async def mcp_connect(mcp_name: str, request: Request) -> JSONResponse:
    """Start OAuth/DCR authorization for an MCP server."""
    from deep_agent.aegra.mcp_oauth_handlers import handle_mcp_connect

    user_id = await _authenticated_user_id(request)
    result = await handle_mcp_connect(user_id, mcp_name)
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
    oauth_mcps = sorted(
        name
        for name, cfg in servers.items()
        if cfg.get("enabled") and cfg.get("auth_mode") in ("oauth", "dcr")
    )
    return {"name": agent_config.get_name(), "oauth_mcps": oauth_mcps}
