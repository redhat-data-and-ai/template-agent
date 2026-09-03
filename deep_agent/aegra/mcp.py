"""MCP (Model Context Protocol) client for external tool integration.

This module manages connections to MCP servers that provide tools for agents.
It reads server configurations from config/agent/mcp.json, establishes parallel
connections with fault isolation, and retrieves all available tools.

Why this exists:
    MCP servers provide external capabilities (APIs, databases, etc.) as tools
    that agents can use. This module bridges the gap between our agent system
    and external MCP-compatible services.

Functions:
    refresh_access_token: Exchange a refresh token for a fresh access token
    get_mcp_tools: Connect to all MCP servers and retrieve their tools
"""

import asyncio
import base64
import contextvars
import json
import os
import time
from typing import Any

import httpx
from langchain_mcp_adapters.client import MultiServerMCPClient

from deep_agent.aegra.mcp_apps import (
    McpAppCallToolResultInterceptor,
    ensure_mcp_apps_capability_advertised,
    prepare_tools_for_model,
)
from deep_agent.src.agent.config import agent_config
from deep_agent.src.error_handling import CircuitBreaker, create_circuit_breaker
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)

# Advertise MCP Apps UI capability for langchain-mcp-adapters sessions.
ensure_mcp_apps_capability_advertised()

_SSO_TOKEN_URL: str = ""

_mcp_breaker: CircuitBreaker | None = None

_MCP_TOOL_CACHE_TTL: float = float(agent_config.get_cache_config().mcp.ttl)
_cached_tools: dict[str | None, list[Any]] = {}
_cached_tools_ts: dict[str | None, float] = {}

_current_access_token: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_access_token", default=None
)
_current_refresh_token: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_refresh_token", default=None
)
_current_user_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_user_id", default=None
)
_mcp_tool_discovery: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_mcp_tool_discovery", default=False
)


def set_mcp_auth_context(
    access_token: str | None,
    refresh_token: str | None,
    user_id: str | None = None,
) -> None:
    """Store the current request's tokens for tool-call-time auth injection.

    Called once per request in the graph factory, before the LLM may invoke
    any MCP tools. The ``_TokenInjectorInterceptor`` reads these at
    invocation time to override the cached connection's Authorization header.
    """
    _current_access_token.set(access_token)
    _current_refresh_token.set(refresh_token)
    _current_user_id.set(user_id)


class _TokenInjectorInterceptor:
    """Inject the correct per-MCP bearer token into every MCP tool call."""

    def __init__(
        self,
        mcp_name: str,
        server_cfg: dict[str, Any],
        server_key: str | None = None,
    ) -> None:
        self._mcp_name = mcp_name
        self._server_cfg = server_cfg
        self._server_key = server_key or mcp_name

    async def __call__(self, request: Any, handler: Any) -> Any:
        from deep_agent.aegra.mcp_auth import (
            NeedsAuthorization,
            get_mcp_credential_resolver,
        )

        user_id = _current_user_id.get()
        auth_mode = self._server_cfg.get("auth_mode", "sso")

        if auth_mode == "api_key":
            return await handler(request)

        try:
            if auth_mode in ("oauth", "dcr"):
                if not user_id:
                    if _mcp_tool_discovery.get():
                        access = None
                    else:
                        raise NeedsAuthorization(
                            self._server_key,
                            get_mcp_credential_resolver().connect_url(self._server_key),
                        )
                else:
                    try:
                        access = await get_mcp_credential_resolver().resolve(
                            user_id, self._server_key, self._server_cfg
                        )
                    except NeedsAuthorization:
                        if _mcp_tool_discovery.get():
                            access = None
                        else:
                            raise
            else:
                access = _current_access_token.get()
                if access:
                    access = await refresh_access_token(
                        access, _current_refresh_token.get()
                    )
        except NeedsAuthorization:
            raise
        except Exception:
            logger.error(
                "[%s] credential resolution failed", self._mcp_name, exc_info=True
            )
            access = _current_access_token.get()

        if access:
            request = request.override(headers={"Authorization": f"Bearer {access}"})
        elif self._server_cfg.get("auth", True):
            logger.warning(
                "TokenInjector: no token for MCP '%s' — call may fail auth",
                self._mcp_name,
            )
        return await handler(request)


def _get_mcp_breaker() -> CircuitBreaker:
    """Lazy-init the MCP circuit breaker (Redis auto-detected on first call)."""
    global _mcp_breaker  # noqa: PLW0603
    if _mcp_breaker is None:
        _mcp_breaker = create_circuit_breaker(
            "mcp-servers", threshold=5, reset_timeout=60.0
        )
    return _mcp_breaker


def _get_token_endpoint() -> str:
    """Derive the OIDC token endpoint from SSO_ISSUER_URL (cached)."""
    global _SSO_TOKEN_URL  # noqa: PLW0603
    if _SSO_TOKEN_URL:
        return _SSO_TOKEN_URL
    issuer: str = os.environ.get("SSO_ISSUER_URL", "").rstrip("/")
    if issuer:
        _SSO_TOKEN_URL = f"{issuer}/protocol/openid-connect/token"
    return _SSO_TOKEN_URL


def _jwt_exp(token: str) -> float:
    """Extract ``exp`` from a JWT payload without cryptographic validation."""
    try:
        payload: str = token.split(".")[1]
        payload += "=" * (4 - len(payload) % 4)
        data: dict[str, Any] = json.loads(base64.urlsafe_b64decode(payload))
        return float(data.get("exp", 0))
    except Exception:
        return 0.0


async def refresh_access_token(
    access_token: str,
    refresh_token: str | None,
) -> str:
    """Return a fresh access token, using the refresh_token grant if needed.

    If the current ``access_token`` has more than 30 seconds of remaining
    lifetime it is returned as-is.  Otherwise, if a ``refresh_token`` and
    the OIDC token endpoint are available, the token is refreshed via the
    standard ``refresh_token`` grant.

    Args:
        access_token: Current JWT access token (may be expired).
        refresh_token: OIDC refresh token (may be ``None`` or ``""``).

    Returns:
        A valid access token (refreshed if necessary, original if refresh
        is unavailable or fails).
    """
    remaining: float = _jwt_exp(access_token) - time.time()
    if remaining > 30:
        logger.debug("Access token still valid (%.0fs remaining)", remaining)
        return access_token

    if not refresh_token:
        logger.warning(
            "Access token near expiry (%.0fs) but no refresh_token available", remaining
        )
        return access_token

    token_url: str = _get_token_endpoint()
    client_id: str = os.environ.get("SSO_CLIENT_ID", "")
    client_secret: str = os.environ.get("SSO_CLIENT_SECRET", "")  # noqa: F841
    if not token_url or not client_id:
        logger.warning("Cannot refresh token — SSO_ISSUER_URL or SSO_CLIENT_ID not set")
        return access_token

    logger.info("Refreshing SSO access token (%.0fs remaining)", remaining)
    try:
        from deep_agent.aegra.auth import EVAL_TOKEN_REFRESH_ENABLED, _oidc_refresh

        new_token, new_rt = await _oidc_refresh(refresh_token)
        new_remaining: float = _jwt_exp(new_token) - time.time()
        logger.info("SSO token refreshed (%.0fs lifetime)", new_remaining)

        _current_access_token.set(new_token)
        if new_rt != refresh_token:
            _current_refresh_token.set(new_rt)
            if EVAL_TOKEN_REFRESH_ENABLED:
                sub = _current_user_id.get()
                if sub:
                    from deep_agent.aegra.mcp_crypto import encrypt_secret
                    from deep_agent.aegra.redis import cache_get, cache_set

                    if await asyncio.to_thread(cache_get, f"eval:active:{sub}"):
                        encrypted_rt = encrypt_secret(new_rt)
                        if encrypted_rt:
                            await asyncio.to_thread(
                                cache_set, f"eval:refresh:{sub}", encrypted_rt, 3600
                            )

        return new_token
    except Exception:
        logger.error("Token refresh failed — using original token", exc_info=True)
        return access_token


def mcp_httpx_verify(server_cfg: dict[str, Any]) -> bool:
    """Return the httpx ``verify`` flag for an MCP server config (default True).

    In production, SSL verification cannot be disabled.
    """
    from deep_agent.src.settings import settings

    ssl_verify = bool(server_cfg.get("ssl_verify", True))

    # Enforce SSL verification in production
    if settings.is_production and not ssl_verify:
        server_name = server_cfg.get("name", "unknown")
        logger.error(
            "MCP server '%s' has ssl_verify=false, which is not permitted in production. "
            "SSL verification will be enforced.",
            server_name,
        )
        return True

    return ssl_verify


def _get_server_configs() -> dict[str, dict[str, Any]]:
    """Get pre-loaded MCP server configurations.

    Returns:
        ``{server_name: {url, transport, enabled, auth, ssl_verify, timeout}}``
    """
    return agent_config.get_mcp_servers()


async def _resolve_connection_token(
    name: str,
    entry: dict[str, Any],
    sso_token: str | None,
    user_id: str | None,
) -> str | None:
    """Resolve the bearer token used for MCP connection/tool discovery."""
    auth_mode = entry.get("auth_mode", "sso")
    if auth_mode == "sso":
        return sso_token

    if auth_mode == "api_key":
        env_var = entry.get("auth_env_var", "")
        api_key = os.environ.get(env_var, "").strip().strip('"')
        if not api_key:
            logger.warning("MCP auth_mode=api_key but %s is not set", env_var)
        return api_key or None

    if not user_id:
        return None

    from deep_agent.aegra.mcp_auth import (
        NeedsAuthorization,
        get_mcp_credential_resolver,
    )

    try:
        return await get_mcp_credential_resolver().resolve(user_id, name, entry)
    except NeedsAuthorization:
        return None


def _build_server_config(
    entry: dict[str, Any],
    bearer_token: str | None,
) -> dict[str, Any]:
    """Build MultiServerMCPClient config from server definition.

    Args:
        entry: Server definition with url, auth, ssl_verify, transport.
        bearer_token: Optional bearer token for this MCP server.

    Returns:
        Config dict for MultiServerMCPClient.
    """
    from deep_agent.utils.pylogger import _trace_id_var

    headers: dict[str, str] = {}
    if entry.get("auth", True) and bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"

    trace_id = _trace_id_var.get()
    if trace_id:
        headers["X-Trace-ID"] = trace_id

    config: dict[str, Any] = {
        "url": entry["url"],
        "transport": entry.get("transport", "streamable_http"),
        "headers": headers,
    }

    # mcp_httpx_verify enforces True in production
    if not mcp_httpx_verify(entry):
        from deep_agent.src.settings import settings

        if settings.is_production:
            logger.warning(
                "MCP server '%s': ssl_verify forced to True in production",
                entry.get("name", entry.get("url", "unknown")),
            )
        else:
            # Only disable SSL verification in development environments
            config["httpx_client_factory"] = lambda **kw: httpx.AsyncClient(
                verify=False, **kw
            )  # nosec B501 - only in development

    return config


def _create_auth_placeholder_tool(
    mcp_name: str, server_cfg: dict[str, Any] | None = None
) -> Any:
    """Create a stub tool that triggers NeedsAuthorization when called.

    When an MCP server requires OAuth/DCR but the user hasn't authenticated yet,
    we inject this placeholder. When the LLM calls it, the stub tries to resolve
    a token (including refresh). Failed refresh raises NeedsAuthorization, which
    mcp_tool_auth wraps into a LangGraph interrupt so the UI shows the connect
    button.
    """
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel
    from pydantic import Field as PydanticField

    class _Input(BaseModel):
        query: str = PydanticField(default="", description="Your request for this tool")

    safe = mcp_name.replace("-", "_")
    svc_desc = (server_cfg or {}).get("description", f"{mcp_name} services")

    async def _require_auth(query: str = "") -> str:
        from deep_agent.aegra.mcp_auth import (
            NeedsAuthorization,
            get_mcp_credential_resolver,
        )

        user_id = _current_user_id.get()
        if user_id:
            resolver = get_mcp_credential_resolver()
            cfg = _get_server_configs().get(mcp_name, {})
            try:
                # Resolve (and refresh if needed) instead of has_valid_token().
                # A leftover refresh token must not skip re-auth after refresh fails.
                await resolver.resolve(user_id, mcp_name, cfg)
                invalidate_mcp_tool_cache(user_id)
                return (
                    f"Successfully connected to {mcp_name}. "
                    f"The tools are now available — please ask the user to "
                    f"repeat their request so the updated tools can be loaded."
                )
            except NeedsAuthorization:
                raise
            except Exception:
                pass

        raise NeedsAuthorization(
            mcp_name,
            get_mcp_credential_resolver().connect_url(mcp_name),
        )

    return StructuredTool(
        name=f"mcp__{safe}",
        description=(
            f"Call this tool to access {svc_desc}. "
            f"You MUST call this tool when the user asks about any of these services. "
            f"It will handle authentication automatically."
        ),
        func=lambda query="": "",
        coroutine=_require_auth,
        args_schema=_Input,
    )


async def _connect_single_server(
    name: str,
    config: dict[str, Any],
    server_cfg: dict[str, Any],
    timeout: int,
    *,
    required: bool = False,
    server_key: str | None = None,
    mcp_server: str | None = None,
) -> list[Any]:
    """Connect to one MCP server and return its model-visible tools.

    Failures are logged and return empty list for fault isolation.
    Updates the module-level circuit breaker on success/failure.

    Tools are annotated with ``mcp_server`` (mcp.json key) and app-only
    tools (``visibility: ["app"]``) are filtered out before return so the
    LLM never sees them. No shared Apps registry is written.

    Args:
        name: Human-readable server identifier used in log messages /
            MultiServerMCPClient connection key (may be ``tool_prefix``).
        config: MCP client connection config (url, transport, headers, etc.).
        server_cfg: Raw MCP server definition from ``mcp.json`` (auth, ssl_verify).
        timeout: Seconds before the connection attempt is cancelled.
        required: If True the server is explicitly enabled in config,
            so connection failures are logged at error level.
        server_key: Optional mcp.json key for auth token lookup and Apps
            metadata. Defaults to ``mcp_server`` or ``name`` when omitted.
        mcp_server: Alias for ``server_key`` (v2 Apps naming). Ignored when
            ``server_key`` is provided.
    """
    auth_key = server_key or mcp_server or name
    breaker = _get_mcp_breaker()
    if breaker.is_open:
        logger.warning(f"[{name}] circuit breaker open — skipping connection")
        return []

    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        try:
            async with asyncio.timeout(timeout):
                client = MultiServerMCPClient(
                    {name: config},
                    tool_interceptors=[
                        _TokenInjectorInterceptor(
                            name, server_cfg, server_key=auth_key
                        ),
                        # Capture raw MCP CallToolResult before LC conversion so
                        # UI-bound tools embed spec-faithful mcp_app.result.
                        McpAppCallToolResultInterceptor(),
                    ],
                    tool_name_prefix=bool(server_cfg.get("tool_prefix", "")),
                )
                tools: list[Any] = await client.get_tools()
            model_tools = prepare_tools_for_model(tools, auth_key)
            skipped = len(tools) - len(model_tools)
            if skipped:
                logger.info(
                    "[%s] hid %d app-only tool(s) from the model (%d model-visible)",
                    name,
                    skipped,
                    len(model_tools),
                )
            logger.info(f"[{name}] loaded {len(model_tools)} tool(s)")
            breaker.record_success()
            return model_tools
        except TimeoutError:
            if attempt < max_attempts:
                logger.warning(
                    f"[{name}] timeout after {timeout}s (attempt {attempt}/{max_attempts}), retrying"
                )
                continue
            breaker.record_failure()
            logger.error(f"[{name}] timeout after {timeout}s ({config.get('url')})")
        except Exception as exc:
            if _is_needs_authorization(exc):
                logger.info(
                    "[%s] MCP OAuth required — returning auth placeholder tool",
                    name,
                )
                return prepare_tools_for_model(
                    [_create_auth_placeholder_tool(auth_key, server_cfg)],
                    auth_key,
                )
            elif _is_auth_error(exc):
                auth_mode = server_cfg.get("auth_mode", "sso")
                if auth_mode in ("oauth", "dcr"):
                    logger.info(
                        "[%s] MCP tool discovery auth failed — returning auth placeholder tool",
                        name,
                    )
                    return prepare_tools_for_model(
                        [_create_auth_placeholder_tool(auth_key, server_cfg)],
                        auth_key,
                    )
                else:
                    logger.warning(
                        f"[{name}] MCP auth failed — {type(exc).__name__}: {exc}"
                    )
            elif _is_connection_error(exc) and not required:
                breaker.record_failure()
                logger.warning(
                    f"[{name}] not reachable ({config.get('url')}) — skipped"
                )
            else:
                breaker.record_failure()
                logger.error(
                    f"[{name}] connection failed ({config.get('url')})", exc_info=True
                )
            return []
    return []


def _is_needs_authorization(exc: BaseException) -> bool:
    """Check if an exception chain contains NeedsAuthorization."""
    from deep_agent.aegra.mcp_auth import NeedsAuthorization

    for sub in getattr(exc, "exceptions", [exc]):
        if isinstance(sub, NeedsAuthorization):
            return True
        if (
            hasattr(sub, "__cause__")
            and sub.__cause__
            and _is_needs_authorization(sub.__cause__)
        ):
            return True
    return False


def _is_auth_error(exc: BaseException) -> bool:
    """Check if an exception is caused by an HTTP 401/403 response."""
    for sub in getattr(exc, "exceptions", [exc]):
        response = getattr(sub, "response", None)
        if response is not None:
            status = getattr(response, "status_code", None)
            if status in (401, 403):
                return True
        msg: str = str(sub)
        if "401" in msg or "403" in msg or "Unauthorized" in msg or "Forbidden" in msg:
            return True
        if sub.__cause__ and _is_auth_error(sub.__cause__):
            return True
        if (
            sub.__context__
            and sub is not sub.__context__
            and _is_auth_error(sub.__context__)
        ):
            return True
    return False


def _is_connection_error(exc: BaseException) -> bool:
    """Check if an exception is a connection refused / unreachable error."""
    for sub in getattr(exc, "exceptions", [exc]):
        msg: str = str(sub).lower()
        if (
            "connecterror" in msg
            or "connection attempts failed" in msg
            or "connection refused" in msg
        ):
            return True
        if hasattr(sub, "__cause__") and sub.__cause__:
            if _is_connection_error(sub.__cause__):
                return True
    return False


def _filter_by_names(
    enabled: dict[str, dict[str, Any]],
    server_names: list[str] | None,
) -> dict[str, dict[str, Any]]:
    """Restrict *enabled* servers to only those declared in *server_names*."""
    if not server_names:
        return enabled
    requested = set(server_names)
    missing = requested - set(enabled)
    if missing:
        logger.warning(
            "Declared MCP server(s) not found or not enabled: %s",
            ", ".join(sorted(missing)),
        )
    return {k: v for k, v in enabled.items() if k in requested}


def invalidate_mcp_tool_cache(user_id: str | None = None) -> None:
    """Clear the MCP tool cache.

    When *user_id* is given, only that user's entry is removed.
    When *user_id* is ``None``, the entire cache is cleared
    (e.g. after a server-level config change).
    """
    if user_id is not None:
        prefix = f"{user_id}:"
        keys = [k for k in _cached_tools if k is not None and k.startswith(prefix)]
        for k in keys:
            _cached_tools.pop(k, None)
            _cached_tools_ts.pop(k, None)
    else:
        _cached_tools.clear()
        _cached_tools_ts.clear()


async def get_mcp_tools(
    sso_token: str | None = None,
    server_names: list[str] | None = None,
    user_id: str | None = None,
) -> list[Any]:
    """Connect to MCP server(s) and retrieve available tools.

    Results are cached for ``MCP_TOOL_CACHE_TTL`` seconds (default 300).
    Subsequent calls within the TTL window return the cached tool list
    without reconnecting, eliminating ~3-4s of overhead per request.

    Loads server definitions from ``config/agent/mcp.json``, connects to
    each enabled server in parallel, and returns a deduplicated flat list.

    When ``server_names`` is provided, only the servers whose names match
    are connected to, preventing unintended tool exposure from globally
    enabled servers that the agent did not declare.

    The ``sso_token`` should already be **refreshed** by the caller via
    ``refresh_access_token()`` before calling this function.

    Connection failures are logged but do not raise exceptions, ensuring
    the application continues with an empty tool list.

    Args:
        sso_token: Optional SSO token for authentication (pre-refreshed).
        server_names: Optional list of MCP server names to connect to.
            When provided, only these servers are used (must also be
            enabled in mcp.json). When ``None``, all enabled servers
            are connected.

    Returns:
        List of available MCP tools (empty list if all connections fail).
    """
    server_key = ",".join(sorted(server_names)) if server_names else ""
    cache_key: str | None = f"{user_id}:{server_key}" if user_id else None
    cached = _cached_tools.get(cache_key) if cache_key else None
    cached_ts = _cached_tools_ts.get(cache_key, 0.0) if cache_key else 0.0

    if cached and len(cached) > 0 and (time.time() - cached_ts) < _MCP_TOOL_CACHE_TTL:
        logger.info(
            "MCP tool cache hit (%d tools, %.0fs old, user=%s)",
            len(cached),
            time.time() - cached_ts,
            cache_key or "anonymous",
        )
        return cached

    servers: dict[str, dict[str, Any]] = _get_server_configs()
    enabled: dict[str, dict[str, Any]] = {
        k: v for k, v in servers.items() if v.get("enabled", False)
    }

    enabled = _filter_by_names(enabled, server_names)

    if not enabled:
        logger.warning("No MCP servers enabled")
        return []

    logger.warning(
        "Connecting to %d MCP server(s): %s", len(enabled), ", ".join(enabled)
    )

    has_auth: bool = bool(sso_token or user_id)
    discovery_token = _mcp_tool_discovery.set(True)
    try:
        connect_jobs = []
        placeholder_tools: list[list[Any]] = []
        for name, entry in enabled.items():
            mcp_prefix_name = entry.get("tool_prefix") or name
            bearer = await _resolve_connection_token(name, entry, sso_token, user_id)
            auth_mode = entry.get("auth_mode", "sso")
            if auth_mode in ("oauth", "dcr") and bearer is None:
                logger.info(
                    "[%s] No OAuth token for %s server — using auth placeholder",
                    mcp_prefix_name,
                    auth_mode,
                )
                placeholder_tools.append(
                    prepare_tools_for_model(
                        [_create_auth_placeholder_tool(name, entry)],
                        name,
                    )
                )
                continue
            connect_jobs.append(
                _connect_single_server(
                    name=mcp_prefix_name,
                    config=_build_server_config(entry, bearer),
                    server_cfg=entry,
                    timeout=entry.get("timeout", 30),
                    required=has_auth,
                    server_key=name,
                )
            )
        results: list[list[Any]] = await asyncio.gather(*connect_jobs)
        results.extend(placeholder_tools)
    finally:
        _mcp_tool_discovery.reset(discovery_token)

    seen: set[str] = set()
    tools: list[Any] = []
    for tool_list in results:
        for tool in tool_list:
            if tool.name not in seen:
                seen.add(tool.name)
                tools.append(tool)
            else:
                logger.warning(f"Duplicate tool '{tool.name}' skipped")

    if not tools:
        oauth_mcps = [
            name
            for name, entry in enabled.items()
            if entry.get("auth_mode") in ("oauth", "dcr")
        ]
        if oauth_mcps:
            logger.warning(
                "All MCP servers failed to load tools — connect OAuth first: %s",
                ", ".join(f"POST /mcp/{n}/connect" for n in oauth_mcps),
            )
        elif sso_token:
            logger.warning("All MCP servers failed to load tools (token present)")
        else:
            logger.warning("MCP tools deferred — no auth token at startup")
        return []

    if cache_key is not None:
        _cached_tools[cache_key] = tools
        _cached_tools_ts[cache_key] = time.time()
    logger.warning(
        "Loaded %d MCP tool(s): %s (cached for %.0fs, user=%s)",
        len(tools),
        ", ".join(seen),
        _MCP_TOOL_CACHE_TTL,
        cache_key or "anonymous",
    )
    return tools
