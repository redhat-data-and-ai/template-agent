"""OAuth connect/callback/status handlers for per-MCP DCR and oauth flows."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import html
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
from deep_agent.aegra.redis import cache_get, cache_sadd, cache_set, cache_srem
from deep_agent.src.agent.config import agent_config
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

_MCP_AUTH_SET_TTL: int = 604800  # 7 days

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


# Discovered metadata is stable for the life of the process and a
# connect click should not pay for three round trips every time.
_METADATA_CACHE: dict[str, dict[str, Any]] = {}


async def discover_oauth_metadata(
    mcp_name: str, server_cfg: dict[str, Any]
) -> dict[str, Any]:
    """Find a server's OAuth endpoints from its published metadata.

    This is what lets a server be configured with nothing but a URL.
    Without it, `auth_mode: dcr` needs `authorization_endpoint`,
    `token_endpoint` and `registration_endpoint` written out by hand —
    three values that are not in anybody's documentation and change
    without notice. It is also why the same server works in VS Code
    from a one-line entry: the client discovers them.

    Two documents, in the order the MCP authorization spec says to try
    them. `oauth-protected-resource` lives on the MCP server and names
    its authorization server; `oauth-authorization-server` lives on
    that authorization server and lists the endpoints. A server that
    publishes neither simply returns nothing and the configured values
    stand.
    """
    url = str(server_cfg.get("url") or "")
    if not url:
        return {}
    if mcp_name in _METADATA_CACHE:
        return _METADATA_CACHE[mcp_name]

    from urllib.parse import urljoin, urlparse

    parsed = urlparse(url)
    root = f"{parsed.scheme}://{parsed.netloc}"

    async def _fetch(target: str) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(
                verify=mcp_httpx_verify(server_cfg), timeout=10
            ) as client:
                # Metadata endpoints redirect freely between a resource
                # and its issuer, so redirects are followed here.
                resp = await client.get(target, follow_redirects=True)
            if resp.is_success:
                body = resp.json()
                return body if isinstance(body, dict) else {}
        except Exception as exc:
            logger.debug("No OAuth metadata at %s: %s", target, exc)
        return {}

    # The resource document takes the same path insertion as the
    # authorization-server one. ServiceNow serves the MCP endpoint at
    # /sncapps/mcp-server/mcp/red_hat_hub_mcp and publishes its
    # metadata at
    #   /.well-known/oauth-protected-resource/sncapps/mcp-server/...
    # Probing only the host root returns 404, so the server looked
    # like it published nothing.
    server_path = parsed.path.strip("/")
    resource_urls = []
    if server_path:
        resource_urls.append(
            f"{root}/.well-known/oauth-protected-resource/{server_path}"
        )
    resource_urls.append(urljoin(root, "/.well-known/oauth-protected-resource"))

    issuer = root
    resource: dict[str, Any] = {}
    for resource_url in resource_urls:
        resource = await _fetch(resource_url)
        if resource.get("authorization_servers"):
            break

    servers = resource.get("authorization_servers") or []
    if isinstance(servers, list) and servers:
        candidate = str(servers[0])
        if urlparse(candidate).scheme in ("http", "https"):
            issuer = candidate.rstrip("/")

    # RFC 8414 §3.1: when the issuer has path components they are
    # *inserted after* `.well-known`, not appended to it. OpenID
    # Connect discovery appends instead. Both are in the wild, so both
    # are tried.
    #
    # Getting this wrong is not a subtle failure: Workato advertises
    # `https://id.workato.com/auth/servers/8SHs…`, and probing that
    # host's bare root — which is what dropping the path amounts to —
    # returns 401. The server looked like it published nothing when it
    # publishes all three endpoints, registration included.
    issuer_parsed = urlparse(issuer)
    issuer_origin = f"{issuer_parsed.scheme}://{issuer_parsed.netloc}"
    issuer_path = issuer_parsed.path.strip("/")

    candidates = []
    if issuer_path:
        candidates.append(
            f"{issuer_origin}/.well-known/oauth-authorization-server/{issuer_path}"
        )
        candidates.append(
            f"{issuer_origin}/.well-known/openid-configuration/{issuer_path}"
        )
    candidates.append(f"{issuer}/.well-known/oauth-authorization-server")
    candidates.append(f"{issuer}/.well-known/openid-configuration")

    meta: dict[str, Any] = {}
    for candidate_url in candidates:
        meta = await _fetch(candidate_url)
        if meta.get("token_endpoint"):
            break

    discovered = {
        key: meta[key]
        for key in (
            "authorization_endpoint",
            "token_endpoint",
            "registration_endpoint",
        )
        if meta.get(key)
    }

    # The scopes the *resource* says it understands, falling back to
    # the authorization server's list.
    #
    # Sending no scope at all is not a neutral default: an
    # authorization server may reject the request outright, or issue
    # a token with no scope that is then refused by the resource —
    # which surfaces as "auth failed" with nothing to go on. Harbor
    # advertises `snowflake-mcp-server` and we were asking for
    # nothing.
    scopes = resource.get("scopes_supported") or meta.get("scopes_supported") or []
    if isinstance(scopes, list) and scopes:
        discovered["scopes"] = [str(x) for x in scopes if x]

    # RFC 8707 resource indicator. The MCP authorization spec requires
    # it on both the authorize and token requests so the server knows
    # which resource the token is for; several implementations reject
    # the exchange outright without it, with a bare 400.
    canonical = resource.get("resource")
    if isinstance(canonical, str) and canonical:
        discovered["resource"] = canonical

    # Client ID Metadata Documents: the authorization server will
    # accept a URL as the client_id and fetch it. A provider that
    # supports this needs no registration endpoint, which is exactly
    # the group that was otherwise unreachable.
    if meta.get("client_id_metadata_document_supported"):
        discovered["client_id_metadata_document_supported"] = True
    if discovered:
        logger.info(
            "Discovered OAuth endpoints for '%s': %s",
            mcp_name,
            ", ".join(sorted(discovered)),
        )
    _METADATA_CACHE[mcp_name] = discovered
    return discovered


async def resolve_oauth_config(
    mcp_name: str, server_cfg: dict[str, Any]
) -> dict[str, Any]:
    """The server's `oauth` block, filled in from discovery where empty.

    Configured values win. An operator who wrote an endpoint down meant
    it, and silently replacing it with a discovered one would make the
    config file a lie.
    """
    configured = dict(server_cfg.get("oauth") or {})
    # `scopes` is in the list so a server whose endpoints are written
    # out by hand, but whose scopes are not, still gets them from
    # discovery. The metadata fetch is cached per process, so the
    # extra completeness costs one round trip per server per pod.
    needed = (
        "authorization_endpoint",
        "token_endpoint",
        "registration_endpoint",
        "scopes",
    )
    if all(configured.get(key) for key in needed):
        return configured

    for key, value in (await discover_oauth_metadata(mcp_name, server_cfg)).items():
        configured.setdefault(key, value)
    return configured


async def _register_dcr_client(
    agent_name: str,
    mcp_name: str,
    oauth_cfg: dict[str, Any],
    server_cfg: dict[str, Any],
) -> tuple[str, str | None]:
    """Perform Dynamic Client Registration and persist the client credentials."""
    registration_endpoint = oauth_cfg.get("registration_endpoint")
    if not registration_endpoint:
        # Dynamic registration is optional in the spec and plenty of
        # providers skip it. The endpoints were still discovered, so
        # the error carries them: the operator needs to register a
        # client by hand, and making them go and find the two URLs
        # again — which this process has already fetched — is a
        # pointless errand.
        discovered = " ".join(
            f'"{key}": "{oauth_cfg[key]}",'
            for key in ("authorization_endpoint", "token_endpoint")
            if oauth_cfg.get(key)
        )
        hint = (
            f" It does publish endpoints — register a client with the provider "
            f"(redirect URI {settings.oauth_callback_url}), then set "
            f'auth_mode "oauth" and oauth: {{ {discovered} '
            f'"client_id": "…", "client_secret": "…" }}.'
            if discovered
            else ""
        )
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{mcp_name}' does not support dynamic client registration." + hint
            ),
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


def _redirect_uri_is_stale(client: Any) -> bool:
    """True when *client* was registered against a different callback URL.

    The registration response records the `redirect_uris` the provider
    accepted. If the current callback is not among them the client is
    unusable, and the failure surfaces at the provider rather than
    here — which is the worst place for it, because the user sees
    someone else's error page and no way back.

    A registration that did not report its redirect URIs is treated as
    current: re-registering on every connect because the provider was
    terse would be worse than trusting it.
    """
    data = getattr(client, "registration_data", None) or {}
    registered = data.get("redirect_uris")
    if not isinstance(registered, list) or not registered:
        return False
    return settings.oauth_callback_url not in registered


async def handle_mcp_connect(
    user_id: str, mcp_name: str, *, caller_origin: str | None = None
) -> dict[str, str]:
    """Start the OAuth authorization flow for *mcp_name*."""
    server_cfg = _get_mcp_server_config(mcp_name)
    _require_interactive_oauth(mcp_name, server_cfg)
    auth_mode = server_cfg.get("auth_mode", "sso")
    # Filled in from the server's published metadata when the config
    # does not spell the endpoints out. Done here rather than at load
    # time: discovery is three HTTP calls against a third party, and
    # paying for them on every process start — for servers nobody is
    # connecting to — would make the agent's boot depend on their
    # uptime.
    oauth_cfg = await resolve_oauth_config(mcp_name, server_cfg)

    for field in ("authorization_endpoint", "token_endpoint"):
        if not oauth_cfg.get(field):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"'{mcp_name}' does not publish OAuth metadata and no "
                    f"oauth.{field} is configured. Add it to mcp.json."
                ),
            )

    redirect_uri = settings.oauth_callback_url
    current_agent_name = settings.agent_deployment_id

    store = McpTokenStore(settings.database_uri)
    # A CIMD client needs no registration and no stored record: the
    # client_id is this agent's own metadata URL. Only used when the
    # server cannot register us anyway — a provider that offers both
    # keeps the registered client, which is the more specific grant.
    use_cimd = bool(
        oauth_cfg.get("client_id_metadata_document_supported")
        and not oauth_cfg.get("registration_endpoint")
    )
    client_id: str | None
    if use_cimd and auth_mode == "dcr":
        client = None
        client_id = f"{settings.agent_public_base_url}/.well-known/oauth-client"
        logger.info(
            "Using client ID metadata document for '%s': %s", mcp_name, client_id
        )
    elif auth_mode == "dcr":
        client = await store.get_client(current_agent_name, mcp_name)
        if client is not None and _redirect_uri_is_stale(client):
            # The registration pins the redirect URI at the provider.
            # Move the agent behind a different host — a port-forward
            # to an ingress, local dev to a cluster — and every stored
            # client still names the old callback, so the provider
            # rejects the authorize request. Reusing it sends the user
            # out to the provider to be told no, with nothing on this
            # side explaining why. Re-register instead.
            logger.info(
                "Re-registering '%s': stored client's redirect URI is stale (now %s)",
                mcp_name,
                settings.oauth_callback_url,
            )
            client = None
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

    # What the client was actually *granted* beats what the server
    # advertises. RFC 7591 §3.2.1: the registration response's `scope`
    # is the set the authorization server issued to this client, and
    # it is allowed to differ from `scopes_supported`.
    #
    # Harbor is the case in point — it advertises
    # `snowflake-mcp-server` in its protected-resource document and
    # then registers the client with `read write`. Asking for the
    # advertised scope gets a bare 400 at the token exchange, with no
    # error body to explain it.
    if auth_mode == "dcr":
        registered = (getattr(client, "registration_data", None) or {}).get("scope")
        if isinstance(registered, str) and registered.strip():
            granted = registered.split()
            if granted != scopes:
                logger.info(
                    "Using registered scopes for '%s': %s (advertised: %s)",
                    mcp_name,
                    " ".join(granted),
                    " ".join(scopes) or "none",
                )
            scopes = granted

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
    # RFC 8707: name the resource the token is for. Required by the
    # MCP authorization spec and sent on both legs — an authorization
    # server that binds the code to a resource will reject an exchange
    # that does not repeat it.
    if oauth_cfg.get("resource"):
        params["resource"] = str(oauth_cfg["resource"])
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
    # Discovery again, not the raw block. The authorize step found
    # these endpoints; the exchange needs the same ones, and reading
    # the unresolved config here meant Connect opened the provider's
    # consent screen and then failed on the way back with "MCP OAuth
    # is not configured" — the worst place in the flow to fail.
    oauth_cfg = await resolve_oauth_config(mcp_name, server_cfg)
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
    elif oauth_cfg.get("client_id_metadata_document_supported") and not oauth_cfg.get(
        "registration_endpoint"
    ):
        # Mirrors the choice made at authorize time; the token
        # exchange has to present the same client_id the code was
        # issued to.
        client_id = f"{settings.agent_public_base_url}/.well-known/oauth-client"
        client_secret = None
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
    # The same resource indicator the authorize request carried.
    if oauth_cfg.get("resource"):
        data["resource"] = str(oauth_cfg["resource"])

    try:
        async with httpx.AsyncClient(verify=mcp_httpx_verify(server_cfg)) as client:
            resp = await client.post(token_endpoint, data=data, timeout=30)
            resp.raise_for_status()
            body: dict[str, Any] = resp.json()
            if body.get("ok") is False:
                error_msg = body.get("error", "unknown error")
                logger.error(
                    "OAuth token exchange failed for '%s': %s",
                    mcp_name,
                    error_msg,
                )
                return HTMLResponse(
                    _callback_html(error=f"Authentication failed: {error_msg}"),
                    status_code=502,
                )
    except httpx.HTTPStatusError as exc:
        # The provider says *why* in the response body — an OAuth
        # error response is `{"error": "...", "error_description":
        # "..."}` per RFC 6749 §5.2. Discarding it and printing
        # "Authentication failed. Please try again." leaves nothing to
        # act on, and trying again fails identically.
        detail = ""
        try:
            problem = exc.response.json()
            # Some servers nest the OAuth error object — FastAPI wraps
            # it in `detail`. Unwrap one level before giving up, and
            # if the keys still are not there fall back to the raw
            # text. The first version of this reported "(no body)" for
            # a response that said `invalid_grant` plainly, one level
            # down, which is worse than not parsing at all.
            if isinstance(problem, dict) and isinstance(problem.get("detail"), dict):
                problem = problem["detail"]
            if isinstance(problem, dict):
                detail = " ".join(
                    str(problem[key])
                    for key in ("error", "error_description")
                    if problem.get(key)
                )
        except Exception:
            detail = ""
        if not detail:
            detail = (exc.response.text or "")[:200]
        logger.error(
            "OAuth token exchange failed for '%s': %s %s",
            mcp_name,
            exc.response.status_code,
            detail or "(no body)",
        )
        return HTMLResponse(
            _callback_html(
                error=f"Authentication failed: {detail}"
                if detail
                else "Authentication failed. Please try again."
            ),
            status_code=502,
        )
    except Exception:
        logger.error("OAuth token exchange failed for '%s'", mcp_name, exc_info=True)
        return HTMLResponse(
            _callback_html(error="Authentication failed. Please try again."),
            status_code=502,
        )

    access_token = body.get("access_token")
    if not access_token:
        authed_user = body.get("authed_user")
        if isinstance(authed_user, dict):
            access_token = authed_user.get("access_token")
    if not access_token:
        logger.error(
            "Token response missing access_token for '%s': keys=%s",
            mcp_name,
            list(body.keys()),
        )
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
        # Only when the provider granted *nothing*. A partial grant is
        # kept — see validate_granted_scopes for why requiring the full
        # advertised set rejected working tokens and looped the user
        # through consent.
        return HTMLResponse(
            _callback_html(
                error=(
                    "Authentication failed. The provider granted no "
                    "permissions for this account."
                )
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
    await asyncio.to_thread(
        cache_sadd,
        f"mcp_auth_set:{current_agent_name}:{user_id}",
        mcp_name,
        ttl_seconds=_MCP_AUTH_SET_TTL,
    )
    get_mcp_credential_resolver().invalidate_cache(user_id, mcp_name)
    from deep_agent.aegra.mcp import invalidate_mcp_tool_cache

    invalidate_mcp_tool_cache(user_id=user_id)
    try:
        from deep_agent.aegra.graph import invalidate_graph_cache

        invalidate_graph_cache()
    except Exception:
        logger.warning(
            "Graph cache invalidation failed in OAuth callback", exc_info=True
        )

    opener_origin = caller_origin or settings.ui_origin
    return HTMLResponse(_callback_html(mcp_name=mcp_name, opener_origin=opener_origin))


def _interactive_oauth_servers() -> list[tuple[str, dict[str, Any]]]:
    """Enabled OAuth/DCR servers that require a user authorization-code flow."""
    servers = agent_config.get_mcp_servers()
    allowed_auth_modes = {"oauth", "dcr"} if settings.MCP_DCR_ENABLED else {"oauth"}
    result: list[tuple[str, dict[str, Any]]] = []
    for name, cfg in sorted(servers.items()):
        if not isinstance(cfg, dict) or not cfg.get("enabled"):
            continue
        if cfg.get("auth_mode") not in allowed_auth_modes:
            continue
        if (cfg.get("oauth") or {}).get("grant_type") == "client_credentials":
            continue
        result.append((name, cfg))
    return result


def _require_interactive_oauth(mcp_name: str, server_cfg: dict[str, Any]) -> None:
    """Reject MCP servers that cannot be user-connected or disconnected."""
    auth_mode = server_cfg.get("auth_mode", "sso")
    if auth_mode not in ("oauth", "dcr"):
        raise HTTPException(
            status_code=400,
            detail=f"MCP '{mcp_name}' does not use OAuth/DCR authentication",
        )
    if (server_cfg.get("oauth") or {}).get("grant_type") == "client_credentials":
        raise HTTPException(
            status_code=400,
            detail=(
                f"MCP '{mcp_name}' uses client_credentials grant — "
                "manual authorization is not required"
            ),
        )


async def handle_mcp_reregister(mcp_name: str) -> dict[str, Any]:
    """Re-register DCR client credentials with current config URLs.

    Call this when any DCR-related URL changes (authorization_endpoint,
    token_endpoint, registration_endpoint, server url, or redirect_uri)
    so that stale client_id/client_secret are replaced.

    Uses upsert semantics — the new registration atomically replaces the
    stored record only after a successful response from the authorization
    server, so existing credentials remain intact on failure.
    """
    server_cfg = _get_mcp_server_config(mcp_name)
    auth_mode = server_cfg.get("auth_mode", "sso")
    if auth_mode != "dcr":
        raise HTTPException(
            status_code=400,
            detail=f"MCP '{mcp_name}' is not using DCR authentication",
        )

    oauth_cfg = await resolve_oauth_config(mcp_name, server_cfg)
    current_agent_name = settings.agent_deployment_id

    client_id, _ = await _register_dcr_client(
        current_agent_name, mcp_name, oauth_cfg, server_cfg
    )
    logger.info(
        "DCR re-registration for '%s' (agent '%s'): new client_id=%s",
        mcp_name,
        current_agent_name,
        client_id,
    )

    return {"mcp_name": mcp_name, "re_registered": True, "client_id": client_id}


async def handle_mcp_status(user_id: str, mcp_name: str) -> dict[str, Any]:
    """Return whether the user has a usable token for *mcp_name*."""
    server_cfg = _get_mcp_server_config(mcp_name)
    resolver = get_mcp_credential_resolver()
    connected = await resolver.has_valid_token(user_id, mcp_name, server_cfg)
    return {"mcp_name": mcp_name, "connected": connected}


async def handle_mcp_connections(user_id: str) -> dict[str, Any]:
    """Return OAuth/DCR MCP connection status for the current user."""
    resolver = get_mcp_credential_resolver()
    connections: list[dict[str, Any]] = []
    for mcp_name, server_cfg in _interactive_oauth_servers():
        description = server_cfg.get("description")
        display_name = server_cfg.get("display_name")
        connected = await resolver.has_valid_token(user_id, mcp_name, server_cfg)
        connections.append(
            {
                "mcp_name": mcp_name,
                "auth_mode": server_cfg.get("auth_mode"),
                "description": description if isinstance(description, str) else "",
                "display_name": display_name if isinstance(display_name, str) else "",
                "connected": connected,
            }
        )
    return {"connections": connections}


async def handle_mcp_disconnect(user_id: str, mcp_name: str) -> dict[str, Any]:
    """Clear stored OAuth tokens for *mcp_name* for the current user."""
    server_cfg = _get_mcp_server_config(mcp_name)
    if server_cfg.get("auth_mode") == "dcr" and not settings.MCP_DCR_ENABLED:
        raise HTTPException(status_code=403, detail="DCR is disabled")
    _require_interactive_oauth(mcp_name, server_cfg)

    store = McpTokenStore(settings.database_uri)
    await store.delete_token(settings.agent_deployment_id, user_id, mcp_name)
    await asyncio.to_thread(
        cache_srem,
        f"mcp_auth_set:{settings.agent_deployment_id}:{user_id}",
        mcp_name,
    )
    get_mcp_credential_resolver().invalidate_cache(user_id, mcp_name)
    from deep_agent.aegra.mcp import invalidate_mcp_tool_cache

    invalidate_mcp_tool_cache(user_id=user_id)
    try:
        from deep_agent.aegra.graph import invalidate_graph_cache

        invalidate_graph_cache()
    except Exception:
        logger.debug("Graph cache invalidation skipped", exc_info=True)
    return {"mcp_name": mcp_name, "connected": False}


def _callback_html(
    mcp_name: str | None = None,
    error: str | None = None,
    opener_origin: str | None = None,
) -> str:
    """Return the HTML page shown after OAuth callback completes."""
    if error:
        safe_error = html.escape(error)
        return f"""<!DOCTYPE html>
<html><head><title>MCP OAuth Error</title></head>
<body><p>{safe_error}</p></body></html>"""

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
