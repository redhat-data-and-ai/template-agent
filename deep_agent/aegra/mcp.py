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
import math
import os
import time
from collections import OrderedDict
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

_mcp_breakers: dict[str, CircuitBreaker] = {}

# Used when a caller does not name a server; keeps the old behaviour
# rather than silently sharing one server's breaker with another.
_SHARED_BREAKER_KEY = settings.MCP_SHARED_BREAKER_KEY

_MCP_TOOL_CACHE_TTL: float = float(agent_config.get_cache_config().mcp.ttl)

# Cache keys with a background refresh already in flight, so a burst
# of requests triggers one rebuild rather than one each.
_refreshing: set[str] = set()
_MCP_AUTH_SET_TTL = settings.MCP_AUTH_SET_TTL_SECONDS

# The tool cache is keyed per user, and each entry is large: measured
# at 615,705 characters of tool description and schema for 319 tools,
# before object overhead for the tool instances themselves.
#
# It used to be an unbounded dict. The TTL is only consulted when an
# entry is *read*, so a user who stopped using the agent was never
# read again and never freed — growth tracked unique users since pod
# start, against a 2Gi limit already carrying ~1.2Gi of resident
# interpreter. That is an OOM at a user count nobody would trace back
# to a cache.
#
# Bounded and evicted least-recently-used instead. An evicted user
# pays a rediscovery, which is the right trade: slow is recoverable,
# OOMKilled is not.
_MCP_TOOL_CACHE_MAX_USERS: int = int(os.environ.get("MCP_TOOL_CACHE_MAX_USERS", "200"))
_cached_tools: OrderedDict[str | None, list[Any]] = OrderedDict()
_cached_tools_ts: dict[str | None, float] = {}
_cached_connected_set: dict[str | None, set[str]] = {}

# Bumped every time an entry is dropped.
#
# A background refresh captures the value it started with. If a
# disconnect lands while that refresh is in flight, the tools it is
# about to write were discovered *before* the disconnect, so writing
# them back silently undoes the invalidation and the disconnected
# server keeps working until the TTL runs out.
_cache_generation: dict[str | None, int] = {}

# Strong references to background refreshes; asyncio keeps only weak
# ones, so a collected task leaves its key in `_refreshing` for ever
# and no later refresh for that key can start.
_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()


def _forget_cached_tools(key: str | None) -> None:
    """Drop one cache entry from every structure that tracks it."""
    _cached_tools.pop(key, None)
    _cached_tools_ts.pop(key, None)
    _cached_connected_set.pop(key, None)
    _cache_generation[key] = _cache_generation.get(key, 0) + 1
    if isinstance(key, str):
        _refreshing.discard(key)


def _touch_cached_tools(key: str | None) -> None:
    """Mark an entry as recently used.

    Without this the OrderedDict stays in insertion order and
    "least recently used" evicts the oldest *written* entry — which
    for a long-lived user whose list refreshes rarely is exactly the
    one worth keeping.
    """
    if key in _cached_tools:
        _cached_tools.move_to_end(key)


def _store_cached_tools(key: str | None, tools: list[Any], connected: set[str]) -> None:
    """Record one user's tool list, evicting the least recently used.

    The three dicts are parallel and have to stay that way; an entry
    present in one and missing from another is how a stale auth set
    silently survives an eviction.
    """
    _cached_tools[key] = tools
    _cached_tools.move_to_end(key)
    _cached_tools_ts[key] = time.time()
    _cached_connected_set[key] = connected

    while len(_cached_tools) > _MCP_TOOL_CACHE_MAX_USERS:
        oldest, _ = _cached_tools.popitem(last=False)
        _cached_tools_ts.pop(oldest, None)
        _cached_connected_set.pop(oldest, None)
        if isinstance(oldest, str):
            _refreshing.discard(oldest)
        logger.info(
            "MCP tool cache full (%d users) — evicted %s",
            _MCP_TOOL_CACHE_MAX_USERS,
            oldest,
        )


_current_access_token: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_access_token", default=None
)
_current_refresh_token: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_refresh_token", default=None
)
_current_user_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_user_id", default=None
)

# Cross-task shared token cache keyed by user_id.
# ContextVars are task-local in asyncio, so a refresh completed in one task
# is invisible to a concurrent task for the same user.  This dict lets the
# distributed-lock peer-check work across tasks.
_user_token_cache: dict[str, tuple[str, str]] = {}
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
                        logger.info(
                            "[%s] no user_id during %s tool discovery — proceeding unauthenticated",
                            self._mcp_name,
                            auth_mode,
                        )
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
                            logger.info(
                                "[%s] NeedsAuthorization during %s tool discovery "
                                "— proceeding unauthenticated (will create placeholder)",
                                self._mcp_name,
                                auth_mode,
                            )
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
        except Exception as exc:
            logger.error(
                "[%s] credential resolution failed", self._mcp_name, exc_info=True
            )
            if auth_mode == "sso":
                access = _current_access_token.get()
            else:
                raise RuntimeError(
                    f"[{self._mcp_name}] {auth_mode} credential resolution failed"
                ) from exc

        if access:
            request = request.override(headers={"Authorization": f"Bearer {access}"})
        elif self._server_cfg.get("auth", True):
            logger.warning(
                "TokenInjector: no token for MCP '%s' — call may fail auth",
                self._mcp_name,
            )
        return await handler(request)


def _get_mcp_breaker(server_name: str | None = None) -> CircuitBreaker:
    """Lazy-init a circuit breaker for one MCP server.

    Breakers are per-server. A shared breaker lets one flaky server strip
    the tools of every other, which is the opposite of the fault isolation
    ``_connect_single_server`` is there to provide — and the agent then
    answers with no tools at all rather than a reduced set.
    """
    key = server_name or _SHARED_BREAKER_KEY
    breaker = _mcp_breakers.get(key)
    if breaker is None:
        breaker = create_circuit_breaker(
            f"mcp-server:{key}", threshold=5, reset_timeout=60.0
        )
        _mcp_breakers[key] = breaker
    return breaker


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


def _jwt_exp_or_none(token: str) -> float | None:
    """``exp`` from a JWT, or ``None`` when there is no readable claim.

    Distinct from :func:`_jwt_exp`, which collapses "unparseable" and
    "exp is 0" into the same ``0.0``. Those mean opposite things: one is a
    token we cannot measure, the other is one that expired at the epoch.
    """
    try:
        payload: str = token.split(".")[1]
        payload += "=" * (4 - len(payload) % 4)
        data: dict[str, Any] = json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return None
    exp = data.get("exp")
    return float(exp) if exp is not None else None


def _token_remaining(token: str) -> float:
    """Seconds of life left in a token, or ``inf`` if it has no readable exp.

    Opaque (non-JWT) tokens carry no parseable expiry. Treating that as
    "expired at epoch" makes every call refresh, hammering the IdP and
    adding a round trip per tool call. We cannot measure what we cannot
    read, so assume valid and let the server reject it if it is not.

    A JWT that genuinely declares ``exp: 0`` is expired, not unreadable,
    and still refreshes.
    """
    exp = _jwt_exp(token)
    # 0.0 is ambiguous: it means both "unparseable" and "exp is literally
    # 0". Only the former should be treated as immeasurable.
    if exp == 0.0 and _jwt_exp_or_none(token) is None:
        return math.inf
    return exp - time.time()


_SSO_REFRESH_BUFFER_SECS = settings.MCP_SSO_REFRESH_BUFFER_SECONDS

_sso_refresh_lock: asyncio.Lock = asyncio.Lock()


async def refresh_access_token(
    access_token: str,
    refresh_token: str | None,
    user_id: str | None = None,
) -> str:
    """Return a fresh access token, using the refresh_token grant if needed.

    Proactively refreshes when fewer than 60 seconds remain (up from 30)
    so long-running tool chains are less likely to hit expiry mid-call.

    Uses a distributed Redis lock (falling back to an in-process asyncio
    lock) to prevent concurrent refresh calls from racing — important when
    Keycloak refresh-token rotation is enabled, as the old refresh token is
    invalidated on first use.

    Args:
        access_token: Current JWT access token (may be expired).
        refresh_token: OIDC refresh token (may be ``None`` or ``""``).
        user_id: Explicit user identity for the lock key. Falls back to
            ``_current_user_id`` context var if not provided.

    Returns:
        A valid access token (refreshed if necessary, original if refresh
        is unavailable or fails).
    """
    remaining: float = _token_remaining(access_token)
    if remaining > _SSO_REFRESH_BUFFER_SECS:
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

    resolved_user_id = user_id or _current_user_id.get() or "unknown"
    return await _locked_sso_refresh(
        access_token, refresh_token, remaining, resolved_user_id
    )


async def _locked_sso_refresh(
    access_token: str,
    refresh_token: str,
    remaining: float,
    user_id: str = "unknown",
) -> str:
    """Perform the actual SSO refresh under a lock to prevent concurrent races."""
    from deep_agent.aegra.redis import distributed_lock

    lock_name = f"sso:refresh:{user_id}"

    async with distributed_lock(lock_name, ttl_seconds=15, wait_seconds=10) as state:
        cached = _user_token_cache.get(user_id)
        if state == "no_redis":
            async with _sso_refresh_lock:
                cached = _user_token_cache.get(user_id)
                if cached:
                    cached_at, cached_rt = cached
                    if cached_at != access_token:
                        check_remaining = _token_remaining(cached_at)
                        if check_remaining > _SSO_REFRESH_BUFFER_SECS:
                            logger.debug(
                                "SSO token already refreshed by peer (no-redis path)"
                            )
                            return cached_at
                latest_rt = (cached[1] if cached else None) or refresh_token
                return await _do_sso_refresh(access_token, latest_rt, remaining)
        elif state == "timeout":
            if cached:
                cached_at, _ = cached
                if cached_at != access_token:
                    check_remaining = _token_remaining(cached_at)
                    if check_remaining > _SSO_REFRESH_BUFFER_SECS:
                        logger.info(
                            "SSO refresh lock timeout — another task refreshed (%.0fs left)",
                            check_remaining,
                        )
                        return cached_at
            logger.warning("SSO refresh lock timeout — using current token")
            return access_token
        else:
            cached = _user_token_cache.get(user_id)
            if cached:
                cached_at, cached_rt = cached
                if cached_at != access_token:
                    check_remaining = _token_remaining(cached_at)
                    if check_remaining > _SSO_REFRESH_BUFFER_SECS:
                        logger.debug("SSO token already refreshed by another task")
                        return cached_at
            latest_rt = (cached[1] if cached else None) or refresh_token
            return await _do_sso_refresh(access_token, latest_rt, remaining)


async def _do_sso_refresh(
    access_token: str,
    refresh_token: str,
    remaining: float,
) -> str:
    """Execute the OIDC refresh grant and update context vars."""
    logger.info("Refreshing SSO access token (%.0fs remaining)", remaining)
    try:
        from deep_agent.aegra.auth import EVAL_TOKEN_REFRESH_ENABLED, _oidc_refresh

        new_token, new_rt = await _oidc_refresh(refresh_token)
        new_remaining: float = _token_remaining(new_token)
        logger.info("SSO token refreshed (%.0fs lifetime)", new_remaining)

        _current_access_token.set(new_token)
        if new_rt != refresh_token:
            _current_refresh_token.set(new_rt)
        sub = _current_user_id.get()
        if sub:
            _user_token_cache[sub] = (new_token, new_rt)
            if EVAL_TOKEN_REFRESH_ENABLED:
                from deep_agent.aegra.mcp_crypto import encrypt_secret
                from deep_agent.aegra.redis import cache_get, cache_set, run_redis

                if await run_redis(cache_get, f"eval:active:{sub}"):
                    encrypted_rt = encrypt_secret(new_rt)
                    if encrypted_rt:
                        await run_redis(
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
        # Renew early here. Discovery runs once per turn and has slack;
        # a tool call several seconds in does not. Without this a token
        # with forty seconds left passes discovery and then fails
        # mid-turn, and the user is told to reconnect an integration
        # that was working when they asked.
        return await get_mcp_credential_resolver().resolve(
            user_id,
            name,
            entry,
            renew_buffer_seconds=settings.MCP_TOKEN_RENEW_BUFFER_SECONDS,
        )
    except NeedsAuthorization as exc:
        logger.info(
            "[%s] no usable OAuth token during connection token resolution (%s)",
            name,
            exc.reason,
        )
        # Recorded here, not only where a tool call fails.
        #
        # The agent is told which servers are connected, so when one
        # is not it declines in prose without ever calling a tool —
        # "Jira is not connected, please connect it in Settings".
        # Nothing failed, so nothing was recorded, so the UI had
        # nothing to offer a button for. Discovery is the point where
        # we know, regardless of what the model then decides to do.
        from deep_agent.aegra.mcp_tool_auth import record_auth_request

        record_auth_request(exc)
        return None


def _bearer_refresher(
    name: str,
    entry: dict[str, Any],
    auth_mode: str,
    user_id: str | None,
) -> Any:
    """An async callable giving this server's current access token.

    Only for the per-user OAuth modes. An SSO bearer belongs to the
    request that carried it and a client-credentials token is already
    cached by the resolver, so neither needs re-resolving per request.
    """
    if auth_mode not in ("oauth", "dcr") or not user_id:
        return None

    async def _current() -> str | None:
        from deep_agent.aegra.mcp_auth import get_mcp_credential_resolver

        return await get_mcp_credential_resolver().resolve(user_id, name, entry)

    return _current


class _FreshBearer(httpx.Auth):
    """Attach the access token per request, not per graph build.

    Dataverse issues a 300-second access token. A turn routinely runs
    longer than that — the model alone stalls for 20-60s at a time —
    so a token baked into the client's headers when the graph was
    built goes stale partway through. The server answers 401, which
    the agent reports as "not connected", for an integration the user
    connected four minutes earlier.

    Asking the resolver each time costs nothing in the common case:
    it returns the cached token and only touches the network inside
    the renewal window.
    """

    def __init__(self, resolve: Any, fallback: str | None) -> None:
        self._resolve = resolve
        self._fallback = fallback

    async def async_auth_flow(self, request: httpx.Request) -> Any:
        token = self._fallback
        try:
            fresh = await self._resolve()
            if fresh:
                token = fresh
        except Exception:
            # A refresh that fails is not a reason to send nothing:
            # the token in hand may still have seconds left, and a
            # 401 is a better failure than a malformed request.
            logger.debug("Could not refresh MCP bearer", exc_info=True)
        if token:
            request.headers["Authorization"] = f"Bearer {token}"
        yield request


def _build_server_config(
    entry: dict[str, Any],
    bearer_token: str | None,
    refresh_bearer: Any = None,
) -> dict[str, Any]:
    """Build MultiServerMCPClient config from server definition.

    Args:
        entry: Server definition with url, auth, ssl_verify, transport.
        bearer_token: Optional bearer token for this MCP server.
        refresh_bearer: Optional async callable returning a current
            token. When given, the token is attached per request so a
            long turn cannot outlive it.

    Returns:
        Config dict for MultiServerMCPClient.
    """
    from deep_agent.utils.pylogger import _trace_id_var

    authenticated = bool(entry.get("auth", True)) and bearer_token is not None
    headers: dict[str, str] = {}
    if authenticated:
        # Kept as well as the per-request auth: a transport that does
        # not take an httpx client factory still gets a token.
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
    verify = mcp_httpx_verify(entry)
    if not verify:
        from deep_agent.src.settings import settings

        if settings.is_production:
            logger.warning(
                "MCP server '%s': ssl_verify forced to True in production",
                entry.get("name", entry.get("url", "unknown")),
            )
            verify = True

    ours: httpx.Auth | None = (
        _FreshBearer(refresh_bearer, bearer_token)
        if authenticated and refresh_bearer is not None
        else None
    )

    if ours is not None or not verify:

        def _factory(
            auth: httpx.Auth | None = None,
            _ours: httpx.Auth | None = ours,
            _verify: bool = verify,
            **kw: Any,
        ) -> httpx.AsyncClient:
            # The SDK passes its own `auth` (normally None); ours only
            # fills the gap so an explicit one is never overridden.
            return httpx.AsyncClient(  # nosec B501 - verify is forced True in prod
                verify=_verify,
                auth=auth or _ours,
                **kw,
            )

        config["httpx_client_factory"] = _factory

    return config


def _create_auth_placeholder_tool(
    mcp_name: str,
    server_cfg: dict[str, Any] | None = None,
    *,
    tools_present: bool = False,
) -> Any:
    """Create a stub tool standing in for a server that needs authorizing.

    When an MCP server requires OAuth/DCR but the user has not
    authenticated, this is injected in place of its tools. Calling it
    resolves a token (refreshing if it can); a failed refresh raises
    NeedsAuthorization, which `mcp_tool_auth` turns into a LangGraph
    interrupt so the UI can offer the connect button.

    ``tools_present`` says the real tools are in this run's list too.
    The placeholder is then registered only so that a *resumed* run
    can still answer a tool call checkpointed against it: resuming
    rebuilds the graph, and without this the pending call would point
    at a tool that no longer exists. In that state it never pauses —
    it just points the model at the real tools.
    """
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel
    from pydantic import Field as PydanticField

    class _Input(BaseModel):
        query: str = PydanticField(default="", description="Your request for this tool")

    safe = mcp_name.replace("-", "_")
    svc_desc = (server_cfg or {}).get("description", f"{mcp_name} services")

    async def _require_auth(query: str = "") -> str:
        from langgraph.errors import GraphBubbleUp as _GraphBubbleUp

        from deep_agent.aegra.mcp_auth import (
            NeedsAuthorization,
            get_mcp_credential_resolver,
        )

        resolver = get_mcp_credential_resolver()
        user_id = _current_user_id.get()
        if not user_id:
            raise NeedsAuthorization(mcp_name, resolver.connect_url(mcp_name))

        cfg = _get_server_configs().get(mcp_name, {})

        async def _connected() -> bool:
            # Resolve (and refresh if needed) rather than has_valid_token():
            # a leftover refresh token must not skip re-auth after refresh fails.
            try:
                await resolver.resolve(user_id, mcp_name, cfg)
                return True
            except NeedsAuthorization:
                return False
            except _GraphBubbleUp:
                raise
            except Exception:
                logger.warning(
                    "[%s] placeholder tool auth resolve failed "
                    "— treating as not connected",
                    mcp_name,
                    exc_info=True,
                )
                return False

        if not await _connected():
            # Raise, do not pause here. The wrapper this tool is handed
            # to turns NeedsAuthorization into the interrupt, and it is
            # the only place that knows whether the pause already
            # happened once in this call. Pausing in both places asked
            # twice and recorded the cooldown twice.
            raise NeedsAuthorization(mcp_name, resolver.connect_url(mcp_name))

        if tools_present:
            # Already usable in this run, so nothing to reload and
            # nothing to invalidate. Clearing the cache here undid
            # stale-while-revalidate on most turns, because the
            # placeholder's own description tells the model it must
            # call this, and the next turn then paid for a full
            # rebuild.
            return (
                f"{mcp_name} is connected and its tools are "
                f"available in this turn. Call them directly."
            )

        invalidate_mcp_tool_cache(user_id)
        return (
            f"[TOOLS_RELOADED mcp={mcp_name}] {mcp_name} is now "
            f"available in this run. Call its tools directly and "
            f"answer the question. Do not ask the user for anything."
        )

    # Two different jobs, so two different descriptions.
    #
    # When the real tools are absent this is the only way in, and the
    # model has to be told to call it. When they are present it exists
    # only so a resumed run can answer a call checkpointed against it,
    # and the "you MUST call this" wording made the model call it once
    # per relevant turn before touching the tools it already had.
    description = (
        f"Deprecated entry point for {svc_desc}. It is already connected: "
        f"call its tools directly and do not call this."
        if tools_present
        else (
            f"Call this tool to access {svc_desc}. "
            f"You MUST call this tool when the user asks about any of these "
            f"services. It will handle authentication automatically."
        )
    )

    return StructuredTool(
        name=f"mcp__{safe}",
        description=description,
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
    breaker = _get_mcp_breaker(mcp_server or name)
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
                    "[%s] MCP OAuth required — returning auth placeholder tool (%s: %s)",
                    name,
                    type(exc).__name__,
                    exc,
                )
                return prepare_tools_for_model(
                    [_create_auth_placeholder_tool(auth_key, server_cfg)],
                    auth_key,
                )
            elif _is_auth_error(exc):
                auth_mode = server_cfg.get("auth_mode", "sso")
                if auth_mode in ("oauth", "dcr"):
                    logger.info(
                        "[%s] MCP tool discovery auth failed (auth_mode=%s) "
                        "— returning auth placeholder tool (%s: %s)",
                        name,
                        auth_mode,
                        type(exc).__name__,
                        exc,
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
            _forget_cached_tools(k)
    else:
        for key in list(_cached_tools):
            _cache_generation[key] = _cache_generation.get(key, 0) + 1
        _cached_tools.clear()
        _cached_tools_ts.clear()
        _cached_connected_set.clear()
        _refreshing.clear()


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

    from deep_agent.aegra.redis import cache_expire, cache_smembers, run_redis

    auth_set_key = f"mcp_auth_set:{settings.agent_deployment_id}:{user_id}"

    # Stale-while-revalidate.
    #
    # Rebuilding the list means a session to every connected server
    # and a tools/list on each: 27 seconds for eight servers against
    # remote SaaS, measured. With a hard TTL that cost lands on
    # whichever user's request happens to arrive first after it
    # expires, and they wait half a minute for a first token through
    # no fault of their own.
    #
    # So an expired-but-present list is served immediately and
    # refreshed behind the request. The data is a tool catalogue that
    # changes when somebody connects or disconnects a server, and
    # both of those already invalidate the cache explicitly — being a
    # few seconds stale costs nothing, and the next request has the
    # fresh list.
    if (
        cached
        and len(cached) > 0
        and (time.time() - cached_ts) >= _MCP_TOOL_CACHE_TTL
        and cache_key
        and cache_key not in _refreshing
    ):
        _refreshing.add(cache_key)
        # Bound to a narrowed local: the `and cache_key` guard above
        # does not reach into a default-argument expression. The
        # default itself is the point — it captures the value now
        # rather than whatever the name holds when the task runs.
        revalidate_key: str = cache_key

        async def _revalidate(key: str = revalidate_key) -> None:
            # The old timestamp stays until the new list replaces it.
            #
            # Popping it first made the entry look *absent* rather than
            # stale for the whole refresh: the stale branch was shut
            # out by `_refreshing`, the fresh branch by a zero
            # timestamp, so every request arriving during the window
            # fell through to a full rebuild of its own — 27 seconds
            # each, in parallel, to recompute what one background task
            # was already fetching.
            generation = _cache_generation.get(key, 0)
            try:
                await get_mcp_tools(
                    sso_token=sso_token, server_names=server_names, user_id=user_id
                )
                if _cache_generation.get(key, 0) != generation:
                    # Disconnected while this was in flight. What the
                    # refresh just wrote predates the disconnect, and
                    # keeping it would re-enable a server the user has
                    # just taken away.
                    _forget_cached_tools(key)
            except Exception:
                logger.debug("Background MCP tool refresh failed", exc_info=True)
            finally:
                _refreshing.discard(key)

        _touch_cached_tools(cache_key)
        logger.info(
            "MCP tool cache stale for %s — serving %d cached tools and "
            "refreshing in the background",
            cache_key,
            len(cached),
        )
        task = asyncio.create_task(_revalidate())
        _BACKGROUND_TASKS.add(task)
        task.add_done_callback(_BACKGROUND_TASKS.discard)
        return cached

    if cached and len(cached) > 0 and (time.time() - cached_ts) < _MCP_TOOL_CACHE_TTL:
        _touch_cached_tools(cache_key)
        redis_set = await run_redis(cache_smembers, auth_set_key)
        if redis_set is None:
            return cached
        await run_redis(cache_expire, auth_set_key, _MCP_AUTH_SET_TTL)
        local_set = _cached_connected_set.get(cache_key, set())
        if redis_set == local_set:
            logger.info(
                "MCP tool cache hit (%d tools, %.0fs old, user=%s)",
                len(cached),
                time.time() - cached_ts,
                cache_key or "anonymous",
            )
            return cached
        else:
            logger.info(
                "MCP auth set changed (redis=%s, local=%s), invalidating cache for %s",
                redis_set,
                local_set,
                cache_key,
            )
            _forget_cached_tools(cache_key)

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

    pre_discovery_set = (
        await run_redis(cache_smembers, auth_set_key) if cache_key is not None else None
    )

    has_auth: bool = bool(sso_token or user_id)
    discovery_token = _mcp_tool_discovery.set(True)
    try:
        connect_jobs = []
        placeholder_tools: list[list[Any]] = []
        # Servers we are about to connect to, in job order.
        resume_placeholders: dict[str, dict[str, Any]] = {}
        connect_order: list[str] = []
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

            resume_placeholders[name] = entry
            connect_order.append(name)
            connect_jobs.append(
                _connect_single_server(
                    name=mcp_prefix_name,
                    config=_build_server_config(
                        entry,
                        bearer,
                        _bearer_refresher(name, entry, auth_mode, user_id),
                    ),
                    server_cfg=entry,
                    timeout=entry.get("timeout", 30),
                    required=has_auth,
                    server_key=name,
                )
            )
        results: list[list[Any]] = await asyncio.gather(*connect_jobs)

        # A resume placeholder, but only where the server answered.
        #
        # A run that paused on an auth placeholder resumes into a
        # freshly built graph holding the real tools; without the
        # placeholder still present the checkpointed call points at a
        # tool that no longer exists and the resume fails. Added only
        # for servers that actually connected — standing one in for a
        # server that is down would claim it is usable.
        for name, tool_list in zip(connect_order, results, strict=False):
            if not tool_list:
                continue
            resume_entry = resume_placeholders.get(name)
            if resume_entry is None:
                continue
            placeholder_tools.append(
                prepare_tools_for_model(
                    [
                        _create_auth_placeholder_tool(
                            name, resume_entry, tools_present=True
                        )
                    ],
                    name,
                )
            )

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
        _store_cached_tools(
            cache_key,
            tools,
            pre_discovery_set
            if pre_discovery_set is not None
            else _cached_connected_set.get(cache_key, set()),
        )
    logger.info(
        "Loaded %d MCP tool(s): %s (cached for %.0fs, user=%s)",
        len(tools),
        ", ".join(seen),
        _MCP_TOOL_CACHE_TTL,
        cache_key or "anonymous",
    )
    _log_tool_budget(tools)
    return tools


# Every bound tool's schema goes into the system prompt on every model
# call. Measured on this agent by PromptSizeMiddleware: 331 tools are
# 615,705 characters of schema, 98.2% of everything sent, against
# 11,040 characters of system prompt. That call recorded 119,769
# input tokens — about 362 per tool, paid per turn, per user, whether
# a tool is used or not.
#
# The count doubled from 193 to 322 without anyone noticing, because
# nothing reported what it cost. That is what this threshold is for:
# not to block a server, but to make growth visible at the moment it
# happens rather than on a bill.
_TOOL_BUDGET_WARN = int(os.environ.get("MCP_TOOL_BUDGET_WARN", "120"))
# Calibrated against recorded input tokens, not a character
# count: character/4 overestimates JSON schema by about a third.
_APPROX_TOKENS_PER_TOOL = settings.MCP_APPROX_TOKENS_PER_TOOL


def _log_tool_budget(tools: list[Any]) -> None:
    """Report the prompt cost of the bound tool set, per server."""
    if not tools:
        return

    by_server: dict[str, int] = {}
    chars_by_server: dict[str, int] = {}
    for tool in tools:
        name = getattr(tool, "name", "") or ""
        # `prepare_tools_for_model` stamps the owning server onto
        # tool.metadata, which is the authoritative attribution.
        # Parsing the name instead looked right and was wrong twice:
        # once bucketing everything under "mcp", then bucketing all
        # 319 discovered tools as built-ins, because only the auth
        # placeholders actually carry an `mcp__` prefix.
        metadata = getattr(tool, "metadata", None)
        server = None
        if isinstance(metadata, dict):
            server = metadata.get("mcp_server")
        if not server:
            parts = name.split("__")
            server = parts[1] if len(parts) >= 2 and parts[0] == "mcp" else None
        server = server or "(built-in)"
        by_server[server] = by_server.get(server, 0) + 1
        # Count is the wrong unit on its own: schemas differ by an
        # order of magnitude, so "which server to drop" is a question
        # about bytes, not tool count.
        size = len(name) + len(str(getattr(tool, "description", "") or ""))
        schema = getattr(tool, "args_schema", None)
        if schema is not None:
            size += len(str(schema))
        chars_by_server[server] = chars_by_server.get(server, 0) + size

    total = len(tools)
    estimate = total * _APPROX_TOKENS_PER_TOOL
    # Ordered by cost, so the first entry is the first thing to
    # consider dropping.
    breakdown = ", ".join(
        f"{server}={by_server[server]}t/{chars // 4}tok"
        for server, chars in sorted(chars_by_server.items(), key=lambda kv: -kv[1])
    )

    if total >= _TOOL_BUDGET_WARN:
        logger.warning(
            "Tool budget: %d tools ≈ %d input tokens per model call, "
            "paid on every turn whether used or not. Breakdown: %s. "
            "Scope servers per subagent to cut this — see "
            "docs/scalability.md.",
            total,
            estimate,
            breakdown,
        )
    else:
        logger.info(
            "Tool budget: %d tools ≈ %d input tokens per call (%s)",
            total,
            estimate,
            breakdown,
        )
