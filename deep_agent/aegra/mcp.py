"""MCP (Model Context Protocol) client for external tool integration.

This module manages connections to MCP servers that provide tools for agents.
It reads server configurations from config/mcp.json, establishes parallel
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

from deep_agent.src.agent.config import agent_config
from deep_agent.src.error_handling import CircuitBreaker, create_circuit_breaker
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)

_SSO_TOKEN_URL: str = ""

_mcp_breaker: CircuitBreaker | None = None

_MCP_TOOL_CACHE_TTL: float = float(agent_config.get_cache_config().mcp.ttl)
_cached_tools: list[Any] = []
_cached_tools_ts: float = 0.0

_current_access_token: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_access_token", default=None
)
_current_refresh_token: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_refresh_token", default=None
)


def set_mcp_auth_context(
    access_token: str | None,
    refresh_token: str | None,
) -> None:
    """Store the current request's tokens for tool-call-time auth injection.

    Called once per request in the graph factory, before the LLM may invoke
    any MCP tools. The ``_TokenInjectorInterceptor`` reads these at
    invocation time to override the cached connection's Authorization header.
    """
    _current_access_token.set(access_token)
    _current_refresh_token.set(refresh_token)


class _TokenInjectorInterceptor:
    """Inject the current request's SSO token into every MCP tool call.

    Reads the access token from the ``_current_access_token`` ContextVar
    (set per-request by ``set_mcp_auth_context``) and overrides the
    ``Authorization`` header on the outgoing MCP request. This ensures
    cached tool objects always use the correct user's token.
    """

    async def __call__(self, request: Any, handler: Any) -> Any:
        access = _current_access_token.get()
        if access:
            request = request.override(headers={"Authorization": f"Bearer {access}"})
        else:
            logger.warning(
                "TokenInjector: no access token in ContextVar — MCP call will use cached/anonymous auth"
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
    client_secret: str = os.environ.get("SSO_CLIENT_SECRET", "")
    if not token_url or not client_id:
        logger.warning("Cannot refresh token — SSO_ISSUER_URL or SSO_CLIENT_ID not set")
        return access_token

    logger.info("Refreshing SSO access token (%.0fs remaining)", remaining)
    try:
        async with httpx.AsyncClient() as client:
            resp: httpx.Response = await client.post(
                token_url,
                data={
                    "grant_type": "refresh_token",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                },
            )
            resp.raise_for_status()
            new_token: str = resp.json()["access_token"]
            new_remaining: float = _jwt_exp(new_token) - time.time()
            logger.info("SSO token refreshed (%.0fs lifetime)", new_remaining)
            return new_token
    except Exception:
        logger.error("Token refresh failed — using original token", exc_info=True)
        return access_token


def _get_server_configs() -> dict[str, dict[str, Any]]:
    """Get pre-loaded MCP server configurations.

    Returns:
        ``{server_name: {url, transport, enabled, auth, ssl_verify, timeout}}``
    """
    return agent_config.get_mcp_servers()


def _build_server_config(
    entry: dict[str, Any],
    sso_token: str | None,
) -> dict[str, Any]:
    """Build MultiServerMCPClient config from server definition.

    Args:
        entry: Server definition with url, auth, ssl_verify, transport.
        sso_token: Optional bearer token (should already be refreshed).

    Returns:
        Config dict for MultiServerMCPClient.
    """
    from deep_agent.utils.pylogger import _trace_id_var

    headers: dict[str, str] = {}
    if entry.get("auth", True) and sso_token:
        headers["Authorization"] = f"Bearer {sso_token}"

    trace_id = _trace_id_var.get()
    if trace_id:
        headers["X-Trace-ID"] = trace_id

    config: dict[str, Any] = {
        "url": entry["url"],
        "transport": entry.get("transport", "streamable_http"),
        "headers": headers,
    }

    if not entry.get("ssl_verify", True):
        config["httpx_client_factory"] = lambda **kw: httpx.AsyncClient(
            verify=False, **kw
        )  # nosec B501

    return config


async def _connect_single_server(
    name: str, config: dict[str, Any], timeout: int, *, required: bool = False
) -> list[Any]:
    """Connect to one MCP server and return its tools.

    Failures are logged and return empty list for fault isolation.
    Updates the module-level circuit breaker on success/failure.

    Args:
        name: Human-readable server identifier used in log messages.
        config: MCP client connection config (url, transport, headers, etc.).
        timeout: Seconds before the connection attempt is cancelled.
        required: If True the server is explicitly enabled in config,
            so connection failures are logged at error level.
    """
    breaker = _get_mcp_breaker()
    if breaker.is_open:
        logger.warning(f"[{name}] circuit breaker open — skipping connection")
        return []

    try:
        async with asyncio.timeout(timeout):
            client = MultiServerMCPClient(
                {name: config},
                tool_interceptors=[_TokenInjectorInterceptor()],
            )
            tools: list[Any] = await client.get_tools()
        logger.info(f"[{name}] loaded {len(tools)} tool(s)")
        breaker.record_success()
        return tools
    except TimeoutError:
        breaker.record_failure()
        logger.error(f"[{name}] timeout after {timeout}s ({config.get('url')})")
    except Exception as exc:
        if _is_auth_error(exc):
            logger.warning(f"[{name}] MCP auth failed — {type(exc).__name__}: {exc}")
        elif _is_connection_error(exc) and not required:
            breaker.record_failure()
            logger.warning(f"[{name}] not reachable ({config.get('url')}) — skipped")
        else:
            breaker.record_failure()
            logger.error(
                f"[{name}] connection failed ({config.get('url')})", exc_info=True
            )
    return []


def _is_auth_error(exc: BaseException) -> bool:
    """Check if an exception is caused by an HTTP 401/403 response."""
    for sub in getattr(exc, "exceptions", [exc]):
        msg: str = str(sub)
        if "401" in msg or "403" in msg or "Unauthorized" in msg or "Forbidden" in msg:
            return True
        if hasattr(sub, "__cause__") and sub.__cause__:
            if _is_auth_error(sub.__cause__):
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


async def get_mcp_tools(
    sso_token: str | None = None,
    server_names: list[str] | None = None,
) -> list[Any]:
    """Connect to MCP server(s) and retrieve available tools.

    Results are cached for ``MCP_TOOL_CACHE_TTL`` seconds (default 300).
    Subsequent calls within the TTL window return the cached tool list
    without reconnecting, eliminating ~3-4s of overhead per request.

    Loads server definitions from ``config/mcp.json``, connects to
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
    global _cached_tools, _cached_tools_ts  # noqa: PLW0603

    if _cached_tools and (time.time() - _cached_tools_ts) < _MCP_TOOL_CACHE_TTL:
        logger.info(
            "MCP tool cache hit (%d tools, %.0fs old)",
            len(_cached_tools),
            time.time() - _cached_tools_ts,
        )
        return _cached_tools

    servers: dict[str, dict[str, Any]] = _get_server_configs()
    enabled: dict[str, dict[str, Any]] = {
        k: v for k, v in servers.items() if v.get("enabled", False)
    }

    enabled = _filter_by_names(enabled, server_names)

    if not enabled:
        logger.warning("No MCP servers enabled")
        return []

    logger.warning(f"Connecting to {len(enabled)} MCP server(s): {', '.join(enabled)}")

    has_auth: bool = bool(sso_token)
    results: list[list[Any]] = await asyncio.gather(
        *[
            _connect_single_server(
                name=name,
                config=_build_server_config(entry, sso_token),
                timeout=entry.get("timeout", 30),
                required=has_auth,
            )
            for name, entry in enabled.items()
        ]
    )

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
        if sso_token:
            logger.warning("All MCP servers failed to load tools (token present)")
        else:
            logger.warning("MCP tools deferred — no auth token at startup")
        return []

    _cached_tools = tools
    _cached_tools_ts = time.time()
    logger.warning(
        f"Loaded {len(tools)} MCP tool(s): {', '.join(seen)} (cached for {_MCP_TOOL_CACHE_TTL:.0f}s)"
    )
    return tools
