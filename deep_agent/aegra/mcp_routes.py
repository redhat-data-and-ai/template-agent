"""HTTP routes for per-MCP OAuth/DCR connect, callback, status, disconnect, and Apps host proxy."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse

from deep_agent.src.agent.config import agent_config
from deep_agent.src.settings import settings

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


@router.get("/mcp/oauth/connections")
async def mcp_connections(request: Request) -> JSONResponse:
    """Return OAuth/DCR MCP connection status for the current user."""
    from deep_agent.aegra.mcp_oauth_handlers import handle_mcp_connections

    user_id = await _authenticated_user_id(request)
    result = await handle_mcp_connections(user_id)
    return JSONResponse(content=result)


@router.get("/.well-known/oauth-client")
async def oauth_client_metadata() -> JSONResponse:
    """This agent's OAuth client, as a document (CIMD).

    An authorization server that supports Client ID Metadata
    Documents accepts a **URL** as the `client_id` and fetches that
    URL to learn who the client is. No registration step, no issued
    secret, nothing to go stale.

    This is how a client gets into a provider that publishes no
    `registration_endpoint` — Sumologic and ServiceNow both advertise
    `client_id_metadata_document_supported`, which is why an MCP
    client can reach them from a bare URL while dynamic registration
    is impossible.

    Deliberately unauthenticated: the authorization server fetches it
    server-to-server, with no credentials, before any user is
    involved. It contains nothing secret — it is the public
    description of this client.
    """
    from deep_agent.src.settings import settings as _settings

    base = _settings.agent_public_base_url
    document = {
        "client_id": f"{base}/.well-known/oauth-client",
        "client_name": _settings.agent_deployment_id or "Agent",
        "client_uri": base,
        "redirect_uris": [_settings.oauth_callback_url],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        # No secret exists to present, which is the point.
        "token_endpoint_auth_method": "none",
    }
    return JSONResponse(content=document)


# Fetched icons, kept for the life of the process. A favicon is a
# few kilobytes and does not change; re-fetching one per page load
# would put thirteen third-party requests behind every render.
_ICON_CACHE: dict[str, tuple[bytes, str]] = {}

# Enough for any favicon. A URL that returns something larger is not
# serving an icon.
_MAX_ICON_BYTES = settings.MCP_MAX_ICON_BYTES

# Hops followed by hand, each one re-validated.
_MAX_ICON_REDIRECTS = 3


@router.get("/mcp/{mcp_name}/icon")
async def mcp_icon(mcp_name: str, request: Request) -> Response:
    """Serve a server's icon from this origin.

    Proxied rather than linked. The UI's CSP allows images from
    `self` only, and widening it to `https:` to show a favicon would
    trade a real control for decoration. It also keeps the browser
    from announcing to thirteen vendors every time somebody opens
    Settings, and it is the only way the internal hosts work at all:
    their certificates chain to the corporate CA, which the pod
    trusts and a laptop browser may not.
    """
    await _authenticated_user_id(request)

    from deep_agent.src.agent.config import agent_config

    cached = _ICON_CACHE.get(mcp_name)
    if cached is None:
        cfg = (agent_config.get_mcp_servers() or {}).get(mcp_name) or {}
        url = str(cfg.get("icon") or "")
        if not url.startswith("https://"):
            raise HTTPException(status_code=404, detail="No icon for this server")

        import httpx

        from deep_agent.src.knowledge.fetch import UnsafeURLError, assert_safe_url

        try:
            # Redirects are followed by hand, one hop at a time, each
            # hop checked. `follow_redirects=True` let an icon URL
            # from config bounce the pod onto an internal HTTP
            # address, which is the pod's own network reachable from
            # a config file.
            body = b""
            media = "image/x-icon"
            status = 404
            async with httpx.AsyncClient(timeout=10, follow_redirects=False) as client:
                next_url = url
                for _ in range(_MAX_ICON_REDIRECTS + 1):
                    assert_safe_url(next_url)
                    if not next_url.startswith("https://"):
                        raise HTTPException(status_code=404, detail="Icon unavailable")
                    # A plain fetch with a browser agent: some CDNs
                    # serve a 403 to anything that does not look like
                    # one.
                    async with client.stream(
                        "GET", next_url, headers={"User-Agent": "Mozilla/5.0"}
                    ) as response:
                        status = response.status_code
                        if status in (301, 302, 303, 307, 308):
                            location = response.headers.get("location", "")
                            if not location:
                                break
                            next_url = str(response.url.join(location))
                            continue
                        media = response.headers.get(
                            "content-type", "image/x-icon"
                        ).split(";")[0]
                        # Counted as it arrives, so an oversized or
                        # endless body is dropped rather than
                        # buffered whole and measured afterwards.
                        async for chunk in response.aiter_bytes():
                            body += chunk
                            if len(body) > _MAX_ICON_BYTES:
                                raise HTTPException(
                                    status_code=404, detail="Icon unavailable"
                                )
                        break
                else:
                    raise HTTPException(status_code=404, detail="Icon unavailable")
            if status >= 400:
                raise HTTPException(status_code=404, detail="Icon unavailable")
            if not media.startswith("image/"):
                raise HTTPException(status_code=404, detail="Not an image")
            cached = (body, media)
            _ICON_CACHE[mcp_name] = cached
        except HTTPException:
            raise
        except UnsafeURLError as exc:
            from deep_agent.utils.pylogger import get_python_logger

            get_python_logger().warning("Refusing icon fetch for %s: %s", mcp_name, exc)
            raise HTTPException(status_code=404, detail="Icon unavailable") from None
        except Exception as exc:
            from deep_agent.utils.pylogger import get_python_logger

            get_python_logger().debug("Icon fetch failed for %s: %s", mcp_name, exc)
            raise HTTPException(status_code=404, detail="Icon unavailable") from None

    body, media = cached
    return Response(
        content=body,
        media_type=media,
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/mcp/servers")
async def mcp_servers(request: Request) -> JSONResponse:
    """Every configured MCP server, enabled or not.

    `/mcp/oauth/connections` only reports servers that use OAuth and
    have a connection record, so a server that is configured and
    switched off is invisible everywhere — which looks exactly like a
    server that was never configured. An operator who has just added
    one to `mcp.json` has no way to tell the two apart.

    Secrets are omitted: no api keys, no client secrets, no tokens.
    The URL is included because a wrong URL is the usual reason a
    server does not work, and it is already in a committed file.
    """
    await _authenticated_user_id(request)

    from deep_agent.src.agent.config import agent_config

    try:
        servers = agent_config.get_mcp_servers() or {}
    except Exception as exc:
        # Module-level `logger` does not exist here — the one in this
        # file is local to another function.
        from deep_agent.utils.pylogger import get_python_logger

        get_python_logger().warning("Could not read MCP servers: %s", exc)
        servers = {}

    listed = [
        {
            "name": name,
            "url": cfg.get("url", ""),
            # For the people choosing what to connect. The agent
            # learns what a server does from its tools, not from
            # these.
            "display_name": cfg.get("display_name") or name,
            "description": cfg.get("description", ""),
            "has_icon": bool(cfg.get("icon")),
            "transport": cfg.get("transport", ""),
            "enabled": bool(cfg.get("enabled", False)),
            # Effective values, not raw ones. `auth` defaults to
            # true and `tool_prefix` falls back to the server name at
            # use time, so reporting the literal config made a
            # URL-only entry look unauthenticated and unprefixed —
            # and the UI, reading auth_mode "none", greyed out its
            # own Connect button.
            "auth_mode": cfg.get("auth_mode", "dcr")
            if cfg.get("auth", True)
            else "none",
            "tool_prefix": cfg.get("tool_prefix") or name,
        }
        for name, cfg in servers.items()
        if isinstance(cfg, dict)
    ]
    # Ordered by the name on screen. Sorting by key put "gworkspace"
    # under G when the row reads "Google Workspace", which is a list
    # that looks unsorted to the person reading it.
    listed.sort(key=lambda s: s["display_name"].lower())

    from deep_agent.src.settings import settings as _settings

    return JSONResponse(
        content={
            "servers": listed,
            "enabled_count": sum(1 for s in listed if s["enabled"]),
            # Whether re-registration is available, so the UI can
            # offer the action instead of rendering a button that
            # always 403s. Off by default: each re-registration mints
            # a new client at the provider and orphans the old one.
            "reregister_enabled": bool(_settings.ENABLE_DCR_REREGISTER_API),
        }
    )


@router.post("/mcp/{mcp_name}/reregister")
async def mcp_reregister(mcp_name: str, request: Request) -> JSONResponse:
    """Delete stale DCR client credentials and re-register with current config URLs."""
    from deep_agent.aegra.mcp_oauth_handlers import handle_mcp_reregister
    from deep_agent.src.settings import settings

    if not settings.ENABLE_DCR_REREGISTER_API:
        return JSONResponse(
            status_code=403,
            content={"detail": "DCR re-register API is disabled"},
        )

    servers = agent_config.get_mcp_servers()
    cfg = servers.get(mcp_name, {})
    if cfg.get("auth_mode") == "dcr" and not settings.MCP_DCR_ENABLED:
        return JSONResponse(
            status_code=403,
            content={"detail": "DCR is disabled"},
        )

    from deep_agent.aegra.auth_helpers import check_ldap_role

    await check_ldap_role(request, developer_only=True)
    result = await handle_mcp_reregister(mcp_name)
    return JSONResponse(content=result)


@router.get("/mcp/{mcp_name}/status")
async def mcp_status(mcp_name: str, request: Request) -> JSONResponse:
    """Return whether the current user has a valid token for the MCP."""
    from deep_agent.aegra.mcp_oauth_handlers import handle_mcp_status

    user_id = await _authenticated_user_id(request)
    result = await handle_mcp_status(user_id, mcp_name)
    return JSONResponse(content=result)


@router.delete("/mcp/{mcp_name}/disconnect")
async def mcp_disconnect(mcp_name: str, request: Request) -> JSONResponse:
    """Clear stored OAuth tokens for an MCP server for the current user."""
    from deep_agent.aegra.mcp_oauth_handlers import handle_mcp_disconnect

    user_id = await _authenticated_user_id(request)
    result = await handle_mcp_disconnect(user_id, mcp_name)
    return JSONResponse(content=result)


@router.get("/info")
async def get_agent_info() -> dict[str, Any]:
    """Return agent identity metadata from config."""
    from deep_agent.aegra.mcp_oauth_handlers import _interactive_oauth_servers

    oauth_mcps = [name for name, _ in _interactive_oauth_servers()]
    return {"name": agent_config.get_name(), "oauth_mcps": oauth_mcps}


@router.get("/mcp/{mcp_name}/capabilities")
async def mcp_capabilities(mcp_name: str, request: Request) -> JSONResponse:
    """Counts of tools, resources and prompts for a connected server.

    Reaching the server requires a working connection, so a failure
    here is reported as empty counts rather than an error: the row in
    Settings should still render, showing that the server is connected
    but did not answer.
    """
    from deep_agent.aegra.mcp_host import count_capabilities

    user_id = await _authenticated_user_id(request)
    try:
        counts = await count_capabilities(
            mcp_name,
            user_id=user_id,
            sso_token=_sso_bearer_from_request(request),
        )
    except HTTPException as exc:
        as_json = _http_exception_response(exc)
        if as_json is not None:
            return as_json
        raise
    except Exception as exc:
        from deep_agent.utils.pylogger import get_python_logger

        get_python_logger().info(
            "Could not read capabilities for '%s': %s", mcp_name, exc
        )
        counts = {"tools": None, "resources": None, "prompts": None}

    return JSONResponse(content={"mcp_name": mcp_name, **counts})


@router.get("/mcp/{mcp_name}/inventory")
async def mcp_inventory(mcp_name: str, request: Request) -> JSONResponse:
    """What a connected server actually offers, with descriptions."""
    from deep_agent.aegra.mcp_host import inventory

    user_id = await _authenticated_user_id(request)
    try:
        found = await inventory(
            mcp_name, user_id=user_id, sso_token=_sso_bearer_from_request(request)
        )
    except HTTPException as exc:
        as_json = _http_exception_response(exc)
        if as_json is not None:
            return as_json
        raise
    return JSONResponse(content={"mcp_name": mcp_name, **found})


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
