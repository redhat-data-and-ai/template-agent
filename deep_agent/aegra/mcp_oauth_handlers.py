"""OAuth connect/callback/status handlers for per-MCP DCR and oauth flows."""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
from typing import Any
from urllib.parse import urlencode

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse

from deep_agent.aegra.mcp import mcp_httpx_verify
from deep_agent.aegra.mcp_auth import (
    get_mcp_credential_resolver,
    resolve_oauth_client_secret,
)
from deep_agent.aegra.mcp_oauth_scopes import (
    parse_token_scopes,
    requested_scopes,
    validate_granted_scopes,
)
from deep_agent.aegra.mcp_token_store import McpTokenStore
from deep_agent.aegra.redis import cache_get, cache_set
from deep_agent.src.agent.config import agent_config
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_OAUTH_STATE_TTL_SECONDS = 300


def _callback_redirect_uri(request: Request) -> str:
    """Reconstruct the OAuth callback URL using the canonical public base URL.

    Behind a reverse proxy (e.g. MCP ingress with nginx rewrite-target),
    request.url.path is the rewritten pod-local path, not the original
    public path.  Use AGENT_PUBLIC_BASE_URL so the URI matches what was
    registered during DCR / authorization.
    """
    return settings.oauth_callback_url


def _pkce_pair() -> tuple[str, str]:
    """Generate a PKCE code verifier and S256 challenge pair."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


def _get_mcp_server_config(mcp_name: str) -> dict[str, Any]:
    """Return the enabled MCP server config or raise 404."""
    servers = agent_config.get_mcp_servers()
    cfg = servers.get(mcp_name)
    if not isinstance(cfg, dict) or not cfg.get("enabled", False):
        raise HTTPException(
            status_code=404, detail=f"MCP server '{mcp_name}' not found"
        )
    return cfg


async def _register_dcr_client(
    agent_name: str,
    mcp_name: str,
    oauth_cfg: dict[str, Any],
    server_cfg: dict[str, Any],
) -> tuple[str, str | None]:
    """Perform Dynamic Client Registration and persist the client credentials."""
    registration_endpoint = oauth_cfg.get("registration_endpoint")
    if not registration_endpoint:
        raise HTTPException(
            status_code=400,
            detail=f"MCP '{mcp_name}' is missing oauth.registration_endpoint",
        )

    scopes = requested_scopes(oauth_cfg)
    redirect_uri = settings.oauth_callback_url

    body = {
        "client_name": f"{agent_name}-{mcp_name}",
        "redirect_uris": [redirect_uri],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
    }
    if scopes:
        body["scope"] = " ".join(scopes)

    async with httpx.AsyncClient(verify=mcp_httpx_verify(server_cfg)) as client:
        resp = await client.post(registration_endpoint, json=body, timeout=30)
        if not resp.is_success:
            logger.error(
                "DCR registration failed for '%s' (agent '%s'): %s %s",
                mcp_name,
                agent_name,
                resp.status_code,
                resp.text,
            )
            raise HTTPException(
                status_code=502,
                detail=f"MCP client registration failed: {resp.text}",
            )
        data = resp.json()

    client_id = data.get("client_id")
    if not client_id:
        raise HTTPException(status_code=502, detail="DCR response missing client_id")

    client_secret = data.get("client_secret")
    store = McpTokenStore(settings.database_uri)
    await store.upsert_client(
        agent_name=agent_name,
        mcp_name=mcp_name,
        client_id=client_id,
        client_secret=client_secret,
        registration_data=data,
    )
    return client_id, client_secret


async def handle_mcp_connect(
    user_id: str, mcp_name: str, *, caller_origin: str | None = None
) -> dict[str, str]:
    """Start the OAuth authorization flow for *mcp_name*."""
    server_cfg = _get_mcp_server_config(mcp_name)
    auth_mode = server_cfg.get("auth_mode", "sso")
    if auth_mode not in ("oauth", "dcr"):
        raise HTTPException(
            status_code=400,
            detail=f"MCP '{mcp_name}' does not use OAuth/DCR authentication",
        )

    oauth_cfg = server_cfg.get("oauth") or {}
    if oauth_cfg.get("grant_type") == "client_credentials":
        raise HTTPException(
            status_code=400,
            detail=f"MCP '{mcp_name}' uses client_credentials grant — manual authorization is not required",
        )

    for field in ("authorization_endpoint", "token_endpoint"):
        if not oauth_cfg.get(field):
            raise HTTPException(
                status_code=400,
                detail=f"MCP '{mcp_name}' missing oauth.{field}",
            )

    redirect_uri = settings.oauth_callback_url
    current_agent_name = settings.agent_deployment_id

    store = McpTokenStore(settings.database_uri)
    if auth_mode == "dcr":
        client = await store.get_client(current_agent_name, mcp_name)
        if client is None:
            await _register_dcr_client(
                current_agent_name, mcp_name, oauth_cfg, server_cfg
            )
            client = await store.get_client(current_agent_name, mcp_name)
        client_id = client.client_id if client else None
    else:
        client_id = oauth_cfg.get("client_id")

    if not client_id:
        raise HTTPException(
            status_code=400, detail=f"No OAuth client_id for '{mcp_name}'"
        )

    scopes = requested_scopes(oauth_cfg)

    code_verifier, code_challenge = _pkce_pair()
    state = secrets.token_urlsafe(32)

    state_data_dict: dict[str, str] = {
        "user_id": user_id,
        "mcp_name": mcp_name,
        "code_verifier": code_verifier,
    }
    if caller_origin and caller_origin == settings.ui_origin:
        state_data_dict["caller_origin"] = caller_origin
    state_payload = json.dumps(state_data_dict)
    if not cache_set(
        f"mcp_oauth_state:{state}", state_payload, _OAUTH_STATE_TTL_SECONDS
    ):
        raise HTTPException(status_code=503, detail="OAuth state storage unavailable")

    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    if scopes:
        params["scope"] = " ".join(scopes)
    authorize_url = f"{oauth_cfg['authorization_endpoint']}?{urlencode(params)}"
    return {"authorize_url": authorize_url}


async def handle_mcp_oauth_callback(
    code: str | None, state: str | None, request: Request
) -> HTMLResponse:
    """Exchange authorization code and persist tokens, then notify the opener."""
    if not code or not state:
        return HTMLResponse(
            _callback_html(error="Missing code or state parameter"),
            status_code=400,
        )

    raw = cache_get(f"mcp_oauth_state:{state}")
    if not raw:
        return HTMLResponse(
            _callback_html(error="OAuth state expired or invalid"),
            status_code=400,
        )

    try:
        state_data = json.loads(raw)
        user_id = state_data["user_id"]
        mcp_name = state_data["mcp_name"]
        code_verifier = state_data["code_verifier"]
        caller_origin = state_data.get("caller_origin")
    except (json.JSONDecodeError, KeyError):
        return HTMLResponse(
            _callback_html(error="Corrupt OAuth state"),
            status_code=400,
        )

    server_cfg = _get_mcp_server_config(mcp_name)
    oauth_cfg = server_cfg.get("oauth") or {}
    token_endpoint = oauth_cfg.get("token_endpoint")
    redirect_uri = settings.oauth_callback_url
    if not token_endpoint:
        return HTMLResponse(
            _callback_html(error="MCP OAuth is not configured"),
            status_code=500,
        )

    callback_uri = _callback_redirect_uri(request)
    if callback_uri != redirect_uri:
        logger.warning(
            "OAuth callback redirect_uri mismatch for '%s': expected %r, got %r",
            mcp_name,
            redirect_uri,
            callback_uri,
        )
        return HTMLResponse(
            _callback_html(error="OAuth redirect URI mismatch"),
            status_code=400,
        )

    store = McpTokenStore(settings.database_uri)
    auth_mode = server_cfg.get("auth_mode", "sso")
    current_agent_name = settings.agent_deployment_id
    if auth_mode == "oauth":
        client_id = oauth_cfg.get("client_id")
        client_secret = resolve_oauth_client_secret(oauth_cfg, mcp_name)
    else:
        client = await store.get_client(current_agent_name, mcp_name)
        client_id = client.client_id if client else None
        client_secret = client.client_secret if client else None

    if not client_id:
        return HTMLResponse(
            _callback_html(error="OAuth client not registered"),
            status_code=400,
        )

    data: dict[str, str] = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "code_verifier": code_verifier,
    }
    if client_secret:
        data["client_secret"] = client_secret

    try:
        async with httpx.AsyncClient(verify=mcp_httpx_verify(server_cfg)) as client:
            resp = await client.post(token_endpoint, data=data, timeout=30)
            resp.raise_for_status()
            body: dict[str, Any] = resp.json()
    except Exception:
        logger.error("OAuth token exchange failed for '%s'", mcp_name, exc_info=True)
        return HTMLResponse(
            _callback_html(error="Authentication failed. Please try again."),
            status_code=502,
        )

    access_token = body.get("access_token")
    if not access_token:
        return HTMLResponse(
            _callback_html(error="Token response missing access_token"),
            status_code=502,
        )

    expires_at = McpTokenStore.expires_at_from_token_response(body)
    requested_scope_list = requested_scopes(oauth_cfg)
    scopes = validate_granted_scopes(
        parse_token_scopes(body),
        requested_scope_list,
        mcp_name,
    )
    if scopes is None and requested_scope_list:
        return HTMLResponse(
            _callback_html(
                error="Authentication failed. Required permissions were not granted."
            ),
            status_code=502,
        )

    await store.upsert_token(
        agent_name=current_agent_name,
        user_id=user_id,
        mcp_name=mcp_name,
        access_token=access_token,
        refresh_token=body.get("refresh_token"),
        expires_at=expires_at,
        scopes=scopes,
    )
    get_mcp_credential_resolver().invalidate_cache(user_id, mcp_name)
    from deep_agent.aegra.mcp import invalidate_mcp_tool_cache

    invalidate_mcp_tool_cache(user_id=user_id)
    try:
        from deep_agent.aegra.graph import invalidate_graph_cache

        invalidate_graph_cache()
    except Exception:
        logger.debug("Graph cache invalidation skipped", exc_info=True)

    opener_origin = caller_origin or settings.ui_origin
    return HTMLResponse(_callback_html(mcp_name=mcp_name, opener_origin=opener_origin))


async def handle_mcp_status(user_id: str, mcp_name: str) -> dict[str, Any]:
    """Return whether the user has a usable token for *mcp_name*."""
    server_cfg = _get_mcp_server_config(mcp_name)
    resolver = get_mcp_credential_resolver()
    connected = await resolver.has_valid_token(user_id, mcp_name, server_cfg)
    return {"mcp_name": mcp_name, "connected": connected}


def _callback_html(
    mcp_name: str | None = None,
    error: str | None = None,
    opener_origin: str | None = None,
) -> str:
    """Return the HTML page shown after OAuth callback completes."""
    if error:
        return f"""<!DOCTYPE html>
<html><head><title>MCP OAuth Error</title></head>
<body><p>{error}</p></body></html>"""

    safe_name = json.dumps(mcp_name)
    if opener_origin:
        target_origin = json.dumps(opener_origin)
        post_message_script = (
            f"if (window.opener) {{"
            f" window.opener.postMessage("
            f"{{ type: 'mcp_oauth_done', mcp_name: {safe_name} }}, {target_origin});"
            f" }}"
        )
    else:
        logger.warning(
            "No UI_ORIGIN configured — skipping postMessage for MCP '%s'",
            mcp_name,
        )
        post_message_script = ""
    return f"""<!DOCTYPE html>
<html><head><title>MCP Connected</title></head>
<body>
<p>Connected. You can close this window.</p>
<script>
  {post_message_script}
  window.close();
</script>
</body></html>"""
