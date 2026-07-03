"""Per-MCP credential resolution for SSO pass-through and OAuth/DCR flows."""

from __future__ import annotations

import asyncio
import os
import time
from datetime import UTC, datetime
from typing import Any

import httpx

from deep_agent.aegra.mcp import (
    _current_access_token,
    _current_refresh_token,
    mcp_httpx_verify,
    refresh_access_token,
)
from deep_agent.aegra.mcp_oauth_scopes import (
    parse_token_scopes,
    requested_scopes,
    validate_granted_scopes,
)
from deep_agent.aegra.mcp_token_store import McpOAuthToken, McpTokenStore
from deep_agent.aegra.redis import distributed_lock
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_RESOLVED_TOKEN_TTL_SECONDS = 30.0
_TOKEN_EXPIRY_BUFFER_SECONDS = 30.0
_TOKEN_REFRESH_LOCK_TTL_SECONDS = 30
_TOKEN_REFRESH_LOCK_WAIT_SECONDS = 10.0
_TOKEN_REFRESH_WAIT_POLL_SECONDS = 0.1
_TOKEN_REFRESH_WAIT_ATTEMPTS = 50


def resolve_oauth_client_secret(oauth_cfg: dict[str, Any], mcp_name: str) -> str | None:
    """Resolve OAuth client secret from env (preferred) or legacy inline config."""
    env_var = oauth_cfg.get("client_secret_env")
    if env_var:
        value = os.environ.get(env_var)
        if not value:
            logger.error(
                "MCP '%s': environment variable %r is not set (oauth.client_secret_env)",
                mcp_name,
                env_var,
            )
            return None
        return value

    inline = oauth_cfg.get("client_secret")
    if inline:
        logger.warning(
            "MCP '%s': oauth.client_secret in mcp.json is insecure — "
            "set oauth.client_secret_env to an environment variable name instead",
            mcp_name,
        )
        return str(inline)

    return None


class NeedsAuthorization(Exception):
    """Raised when an oauth/dcr MCP has no usable token for the user."""

    def __init__(self, mcp_name: str, connect_url: str) -> None:
        """Store MCP name and connect URL for the authorization interrupt."""
        self.mcp_name = mcp_name
        self.connect_url = connect_url
        super().__init__(f"MCP authorization required for {mcp_name}")


class McpCredentialResolver:
    """Resolve the bearer token to send to a specific MCP server."""

    def __init__(self, token_store: McpTokenStore | None = None) -> None:
        """Initialize with an optional token store (defaults to Redis + Postgres)."""
        self._store = token_store or McpTokenStore(settings.database_uri)
        self._cache: dict[tuple[str, str], tuple[str, float]] = {}

    @staticmethod
    def connect_url(mcp_name: str) -> str:
        """Build the agent connect endpoint URL for *mcp_name*."""
        base = settings.agent_public_base_url.rstrip("/")
        return f"{base}/mcp/{mcp_name}/connect"

    async def resolve(
        self, user_id: str, mcp_name: str, server_cfg: dict[str, Any]
    ) -> str:
        """Return a bearer token for this MCP, or raise :class:`NeedsAuthorization`."""
        auth_mode = server_cfg.get("auth_mode", "sso")

        if auth_mode == "sso":
            return await self._resolve_sso()

        cache_key = (user_id, mcp_name)
        cached = self._cache.get(cache_key)
        if cached and (time.time() - cached[1]) < _RESOLVED_TOKEN_TTL_SECONDS:
            return cached[0]

        token = await self._resolve_oauth(user_id, mcp_name, server_cfg)
        self._cache[cache_key] = (token, time.time())
        return token

    async def has_valid_token(
        self, user_id: str, mcp_name: str, server_cfg: dict[str, Any]
    ) -> bool:
        """Return True when a usable token exists (or can be refreshed)."""
        auth_mode = server_cfg.get("auth_mode", "sso")
        if auth_mode == "sso":
            access = _current_access_token.get()
            return bool(access)

        stored = await self._store.get_token(user_id, mcp_name)
        if stored is None:
            return False
        if self._token_valid(stored):
            return True
        return bool(stored.refresh_token)

    def invalidate_cache(self, user_id: str, mcp_name: str) -> None:
        """Drop a cached resolved token after OAuth callback or logout."""
        self._cache.pop((user_id, mcp_name), None)

    async def _resolve_sso(self) -> str:
        """Return the refreshed SSO token from the current request context."""
        access = _current_access_token.get()
        refresh = _current_refresh_token.get()
        if not access:
            logger.warning(
                "MCP SSO auth: no access token in context — call may be anonymous"
            )
            return ""
        return await refresh_access_token(access, refresh)

    async def _resolve_oauth(
        self,
        user_id: str,
        mcp_name: str,
        server_cfg: dict[str, Any],
    ) -> str:
        stored = await self._store.get_token(user_id, mcp_name)
        if stored is None:
            raise NeedsAuthorization(mcp_name, self.connect_url(mcp_name))

        if self._token_valid(stored):
            return stored.access_token

        if not stored.refresh_token:
            raise NeedsAuthorization(mcp_name, self.connect_url(mcp_name))

        lock_name = f"mcp_token_refresh:{user_id}:{mcp_name}"
        async with distributed_lock(
            lock_name,
            ttl_seconds=_TOKEN_REFRESH_LOCK_TTL_SECONDS,
            wait_seconds=_TOKEN_REFRESH_LOCK_WAIT_SECONDS,
        ) as lock_state:
            stored = await self._store.get_token(user_id, mcp_name)
            if stored is None:
                raise NeedsAuthorization(mcp_name, self.connect_url(mcp_name))
            if self._token_valid(stored):
                return stored.access_token
            if not stored.refresh_token:
                raise NeedsAuthorization(mcp_name, self.connect_url(mcp_name))

            if lock_state == "timeout":
                refreshed_by_peer = await self._wait_for_refreshed_token(
                    user_id, mcp_name
                )
                if refreshed_by_peer:
                    return refreshed_by_peer
                raise NeedsAuthorization(mcp_name, self.connect_url(mcp_name))

            if lock_state == "no_redis":
                logger.warning(
                    "Redis unavailable; refreshing MCP token for '%s' without lock",
                    mcp_name,
                )

            refreshed = await self._refresh_mcp_token(stored, server_cfg)
            if refreshed:
                self.invalidate_cache(user_id, mcp_name)
                return refreshed

        raise NeedsAuthorization(mcp_name, self.connect_url(mcp_name))

    async def _wait_for_refreshed_token(
        self, user_id: str, mcp_name: str
    ) -> str | None:
        """Poll storage while another request holds the refresh lock."""
        for _ in range(_TOKEN_REFRESH_WAIT_ATTEMPTS):
            await asyncio.sleep(_TOKEN_REFRESH_WAIT_POLL_SECONDS)
            stored = await self._store.get_token(user_id, mcp_name)
            if stored is not None and self._token_valid(stored):
                return stored.access_token
        logger.error(
            "Timed out waiting for MCP token refresh for '%s' user '%s'",
            mcp_name,
            user_id,
        )
        return None

    @staticmethod
    def _token_valid(token: McpOAuthToken) -> bool:
        if not token.access_token:
            return False
        if token.expires_at is None:
            return True
        remaining = (token.expires_at - datetime.now(UTC)).total_seconds()
        return remaining > _TOKEN_EXPIRY_BUFFER_SECONDS

    async def _refresh_mcp_token(
        self,
        stored: McpOAuthToken,
        server_cfg: dict[str, Any],
    ) -> str | None:
        oauth_cfg = server_cfg.get("oauth") or {}
        token_endpoint = oauth_cfg.get("token_endpoint")
        if not token_endpoint:
            logger.error(
                "Cannot refresh MCP token for '%s' — token_endpoint missing in config",
                stored.mcp_name,
            )
            return None

        client_id, client_secret = await self._resolve_client_credentials(
            stored.mcp_name, server_cfg
        )
        if not client_id:
            logger.error(
                "Cannot refresh MCP token for '%s' — client_id unavailable",
                stored.mcp_name,
            )
            return None

        data: dict[str, str] = {
            "grant_type": "refresh_token",
            "refresh_token": stored.refresh_token or "",
            "client_id": client_id,
        }
        if client_secret:
            data["client_secret"] = client_secret

        try:
            async with httpx.AsyncClient(verify=mcp_httpx_verify(server_cfg)) as client:
                resp = await client.post(token_endpoint, data=data, timeout=30)
                resp.raise_for_status()
                body: dict[str, Any] = resp.json()
        except Exception:
            logger.error(
                "MCP token refresh failed for '%s'", stored.mcp_name, exc_info=True
            )
            return None

        new_access_raw = body.get("access_token")
        if not isinstance(new_access_raw, str) or not new_access_raw:
            logger.error(
                "MCP token refresh for '%s' returned no access_token", stored.mcp_name
            )
            return None
        new_access = new_access_raw

        new_refresh = body.get("refresh_token", stored.refresh_token)
        expires_at = McpTokenStore.expires_at_from_token_response(body)
        requested_scope_list = requested_scopes(oauth_cfg)
        parsed_scopes = parse_token_scopes(body)
        if parsed_scopes is not None:
            scopes = validate_granted_scopes(
                parsed_scopes, requested_scope_list, stored.mcp_name
            )
            if scopes is None and requested_scope_list:
                logger.error(
                    "MCP token refresh for '%s' returned insufficient scopes",
                    stored.mcp_name,
                )
                return None
        else:
            scopes = stored.scopes

        await self._store.upsert_token(
            user_id=stored.user_id,
            mcp_name=stored.mcp_name,
            access_token=new_access,
            refresh_token=new_refresh,
            expires_at=expires_at,
            scopes=scopes,
        )
        logger.info("Refreshed MCP OAuth token for '%s'", stored.mcp_name)
        return new_access

    async def _resolve_client_credentials(
        self, mcp_name: str, server_cfg: dict[str, Any]
    ) -> tuple[str | None, str | None]:
        auth_mode = server_cfg.get("auth_mode", "sso")
        oauth_cfg = server_cfg.get("oauth") or {}

        if auth_mode == "oauth":
            return oauth_cfg.get("client_id"), resolve_oauth_client_secret(
                oauth_cfg, mcp_name
            )

        if auth_mode == "dcr":
            client = await self._store.get_client(mcp_name)
            if client is None:
                return None, None
            return client.client_id, client.client_secret

        return None, None


_default_resolver: McpCredentialResolver | None = None


def get_mcp_credential_resolver() -> McpCredentialResolver:
    """Return the process-wide credential resolver singleton."""
    global _default_resolver  # noqa: PLW0603
    if _default_resolver is None:
        _default_resolver = McpCredentialResolver()
    return _default_resolver
