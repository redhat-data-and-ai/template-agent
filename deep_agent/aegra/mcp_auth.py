"""Per-MCP credential resolution for SSO pass-through and OAuth/DCR flows."""

from __future__ import annotations

import asyncio
import json
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

_RESOLVED_TOKEN_TTL_SECONDS = settings.MCP_RESOLVED_TOKEN_TTL_SECONDS
_TOKEN_EXPIRY_BUFFER_SECONDS = settings.MCP_TOKEN_EXPIRY_BUFFER_SECONDS
_TOKEN_REFRESH_LOCK_TTL_SECONDS = settings.MCP_TOKEN_REFRESH_LOCK_TTL_SECONDS
_TOKEN_REFRESH_LOCK_WAIT_SECONDS = settings.MCP_TOKEN_REFRESH_LOCK_WAIT_SECONDS
_TOKEN_REFRESH_WAIT_POLL_SECONDS = settings.MCP_TOKEN_REFRESH_WAIT_POLL_SECONDS
_TOKEN_REFRESH_WAIT_ATTEMPTS = settings.MCP_TOKEN_REFRESH_WAIT_ATTEMPTS


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


def _is_invalid_grant(body_text: str) -> bool:
    """True when a token response says the grant itself is dead.

    RFC 6749 §5.2 puts `error` at the top level; this provider nests
    it under `detail`. Both are checked, and a plain substring match
    is the fallback so an unparseable body still reads correctly.
    """
    if not body_text:
        return False
    try:
        body = json.loads(body_text)
    except Exception:
        return "invalid_grant" in body_text
    candidates = [body]
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict):
        candidates.append(detail)
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("error") == "invalid_grant":
            return True
    return "invalid_grant" in body_text


class NeedsAuthorization(Exception):
    """Raised when an oauth/dcr MCP has no usable token for the user.

    ``reason`` separates two situations that look identical to the
    caller and are not the same thing to the person reading the
    answer: a server never connected, and one whose token expired
    and could not be renewed. The first is "connect this"; the
    second is "reconnect this, your session lapsed". Reporting both
    as "not connected" sends someone to set up an integration they
    already set up.
    """

    def __init__(
        self,
        mcp_name: str,
        connect_url: str,
        reason: str = "not_connected",
    ) -> None:
        """Store MCP name, connect URL and why authorization is needed."""
        self.mcp_name = mcp_name
        self.connect_url = connect_url
        self.reason = reason
        super().__init__(f"MCP authorization required for {mcp_name} ({reason})")


class McpCredentialResolver:
    """Resolve the bearer token to send to a specific MCP server."""

    def __init__(self, token_store: McpTokenStore | None = None) -> None:
        """Initialize with an optional token store (defaults to Redis + Postgres)."""
        self._store = token_store or McpTokenStore(settings.database_uri)
        self._cache: dict[tuple[str, str], tuple[str, float]] = {}
        self._cc_cache: dict[str, tuple[str, float]] = {}

    @staticmethod
    def connect_url(mcp_name: str) -> str:
        """Build the agent connect endpoint URL for *mcp_name*."""
        base = settings.agent_public_base_url.rstrip("/")
        return f"{base}/mcp/{mcp_name}/connect"

    async def resolve(
        self,
        user_id: str,
        mcp_name: str,
        server_cfg: dict[str, Any],
        *,
        renew_buffer_seconds: float | None = None,
    ) -> str:
        """Return a bearer token for this MCP, or raise :class:`NeedsAuthorization`.

        ``renew_buffer_seconds`` lets a caller renew early. Discovery
        uses it so a token due to expire during the turn is replaced
        before the turn runs.
        """
        auth_mode = server_cfg.get("auth_mode", "sso")

        if auth_mode == "sso":
            return await self._resolve_sso()

        if auth_mode == "dcr" and not settings.MCP_DCR_ENABLED:
            logger.info("DCR disabled — skipping auth for MCP '%s'", mcp_name)
            return ""

        oauth_cfg = server_cfg.get("oauth") or {}
        if oauth_cfg.get("grant_type") == "client_credentials":
            return await self._resolve_client_credentials_grant(mcp_name, server_cfg)

        cache_key = (user_id, mcp_name)
        cached = self._cache.get(cache_key)
        # A caller asking for a renewal buffer is asking for a token
        # that will still be valid in N seconds, and this cache
        # cannot answer that: it holds whatever was resolved up to
        # `_RESOLVED_TOKEN_TTL_SECONDS` ago. Answering from it handed
        # back a token that expired mid-turn, which is the one thing
        # the buffer exists to prevent.
        if (
            cached
            and not renew_buffer_seconds
            and (time.time() - cached[1]) < _RESOLVED_TOKEN_TTL_SECONDS
        ):
            return cached[0]

        token = await self._resolve_oauth(
            user_id,
            mcp_name,
            server_cfg,
            renew_buffer_seconds=renew_buffer_seconds,
        )
        self._cache[cache_key] = (token, time.time())
        return token

    async def has_valid_token(
        self,
        user_id: str,
        mcp_name: str,
        server_cfg: dict[str, Any],
        *,
        verify: bool = False,
    ) -> bool:
        """Return True when a usable token exists (or can be refreshed).

        ``verify`` actually performs the refresh when the stored
        access token has expired, instead of assuming that the
        presence of a refresh token means the credential still works.

        Without it this answers a different question from the one the
        caller is asking. The settings page reported a server as
        **Connected** on the strength of a refresh token alone while
        every chat turn said the session had expired — the refresh
        was dead, and nothing here had tried it. Two surfaces, two
        answers, and the truthful one looked like the broken one.

        Default stays off: it costs a token round trip per expired
        server, so only a caller that is about to show the user a
        connection state should pay it.
        """
        auth_mode = server_cfg.get("auth_mode", "sso")
        if auth_mode == "sso":
            access = _current_access_token.get()
            return bool(access)

        oauth_cfg = server_cfg.get("oauth") or {}
        if oauth_cfg.get("grant_type") == "client_credentials":
            return True

        current_agent_name = settings.agent_deployment_id
        stored = await self._store.get_token(current_agent_name, user_id, mcp_name)
        if stored is None:
            return False
        if self._token_valid(stored):
            return True
        if not stored.refresh_token:
            return False
        if not verify:
            return True
        try:
            return bool(await self.resolve(user_id, mcp_name, server_cfg))
        except NeedsAuthorization:
            return False
        except Exception:
            # A provider outage is not a disconnected integration.
            # Report what is stored and let the next real call fail
            # with a reason, rather than telling the user to redo an
            # OAuth dance that was never the problem.
            logger.warning(
                "Could not verify %s credential for user — reporting as "
                "connected on the strength of the stored refresh token",
                mcp_name,
                exc_info=True,
            )
            return True

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

    async def _resolve_client_credentials_grant(
        self,
        mcp_name: str,
        server_cfg: dict[str, Any],
    ) -> str:
        """Acquire a token using the client_credentials grant (no user interaction)."""
        cached = self._cc_cache.get(mcp_name)
        if cached:
            token, expires_at = cached
            if (expires_at - time.time()) > _TOKEN_EXPIRY_BUFFER_SECONDS:
                return token

        lock_name = f"mcp_cc_token:{mcp_name}"
        async with distributed_lock(
            lock_name,
            ttl_seconds=_TOKEN_REFRESH_LOCK_TTL_SECONDS,
            wait_seconds=_TOKEN_REFRESH_LOCK_WAIT_SECONDS,
        ) as lock_state:
            cached = self._cc_cache.get(mcp_name)
            if cached:
                token, expires_at = cached
                if (expires_at - time.time()) > _TOKEN_EXPIRY_BUFFER_SECONDS:
                    return token

            if lock_state == "timeout":
                raise RuntimeError(
                    f"MCP '{mcp_name}': timed out waiting for client_credentials token lock"
                )

            oauth_cfg = server_cfg.get("oauth") or {}
            token_endpoint = oauth_cfg.get("token_endpoint")
            if not token_endpoint:
                raise RuntimeError(
                    f"MCP '{mcp_name}': token_endpoint missing for client_credentials grant"
                )

            client_id, client_secret = await self._resolve_client_credentials(
                mcp_name, server_cfg
            )
            if not client_id:
                raise RuntimeError(
                    f"MCP '{mcp_name}': client_id unavailable for client_credentials grant"
                )

            data: dict[str, str] = {
                "grant_type": "client_credentials",
                "client_id": client_id,
            }
            if client_secret:
                data["client_secret"] = client_secret

            scopes = requested_scopes(oauth_cfg)
            if scopes:
                data["scope"] = " ".join(scopes)

            try:
                async with httpx.AsyncClient(
                    verify=mcp_httpx_verify(server_cfg)
                ) as client:
                    resp = await client.post(token_endpoint, data=data, timeout=30)
                    resp.raise_for_status()
                    body: dict[str, Any] = resp.json()
            except Exception:
                logger.error(
                    "Client credentials token acquisition failed for '%s'",
                    mcp_name,
                    exc_info=True,
                )
                raise RuntimeError(
                    f"MCP '{mcp_name}': client_credentials token acquisition failed"
                )

            access_token = body.get("access_token")
            if not isinstance(access_token, str) or not access_token:
                raise RuntimeError(
                    f"MCP '{mcp_name}': client_credentials response missing access_token"
                )

            try:
                expires_in = int(body.get("expires_in", 3600))
            except (TypeError, ValueError):
                expires_in = 3600
            self._cc_cache[mcp_name] = (access_token, time.time() + expires_in)
            logger.info("Acquired client_credentials token for MCP '%s'", mcp_name)
            return access_token

    async def _resolve_oauth(
        self,
        user_id: str,
        mcp_name: str,
        server_cfg: dict[str, Any],
        *,
        renew_buffer_seconds: float | None = None,
    ) -> str:
        """Resolve an OAuth token, refreshing if expired or expiring soon."""
        current_agent_name = settings.agent_deployment_id
        stored = await self._store.get_token(current_agent_name, user_id, mcp_name)
        if stored is None:
            logger.info(
                "[%s] no stored OAuth token for agent=%s — authorization required",
                mcp_name,
                current_agent_name,
            )
            raise NeedsAuthorization(mcp_name, self.connect_url(mcp_name))

        if self._token_valid(stored, renew_buffer_seconds):
            return stored.access_token

        if not stored.refresh_token:
            logger.info(
                "[%s] OAuth token expired and no refresh_token — authorization required",
                mcp_name,
            )
            raise NeedsAuthorization(
                mcp_name, self.connect_url(mcp_name), reason="expired"
            )

        lock_name = f"mcp_token_refresh:{current_agent_name}:{user_id}:{mcp_name}"
        async with distributed_lock(
            lock_name,
            ttl_seconds=_TOKEN_REFRESH_LOCK_TTL_SECONDS,
            wait_seconds=_TOKEN_REFRESH_LOCK_WAIT_SECONDS,
        ) as lock_state:
            stored = await self._store.get_token(current_agent_name, user_id, mcp_name)
            if stored is None:
                logger.info(
                    "[%s] token vanished during lock — authorization required",
                    mcp_name,
                )
                raise NeedsAuthorization(mcp_name, self.connect_url(mcp_name))
            if self._token_valid(stored, renew_buffer_seconds):
                return stored.access_token
            if not stored.refresh_token:
                logger.info(
                    "[%s] token expired, no refresh_token after lock — authorization required",
                    mcp_name,
                )
                raise NeedsAuthorization(
                    mcp_name, self.connect_url(mcp_name), reason="expired"
                )

            if lock_state == "timeout":
                refreshed_by_peer = await self._wait_for_refreshed_token(
                    current_agent_name, user_id, mcp_name
                )
                if refreshed_by_peer:
                    return refreshed_by_peer
                logger.info(
                    "[%s] peer refresh timed out — authorization required",
                    mcp_name,
                )
                raise NeedsAuthorization(
                    mcp_name, self.connect_url(mcp_name), reason="expired"
                )

            if lock_state == "no_redis":
                logger.warning(
                    "Redis unavailable; refreshing MCP token for '%s' without lock",
                    mcp_name,
                )

            refreshed = await self._refresh_mcp_token(stored, server_cfg)
            if refreshed:
                self.invalidate_cache(user_id, mcp_name)
                return refreshed

        logger.info(
            "[%s] OAuth token refresh exhausted — authorization required",
            mcp_name,
        )
        raise NeedsAuthorization(mcp_name, self.connect_url(mcp_name), reason="expired")

    async def _wait_for_refreshed_token(
        self, agent_name: str, user_id: str, mcp_name: str
    ) -> str | None:
        """Poll storage while another request holds the refresh lock."""
        for _ in range(_TOKEN_REFRESH_WAIT_ATTEMPTS):
            await asyncio.sleep(_TOKEN_REFRESH_WAIT_POLL_SECONDS)
            stored = await self._store.get_token(agent_name, user_id, mcp_name)
            if stored is not None and self._token_valid(stored):
                return stored.access_token
        logger.error(
            "Timed out waiting for MCP token refresh for '%s' user '%s'",
            mcp_name,
            user_id,
        )
        return None

    @staticmethod
    def _token_valid(token: McpOAuthToken, buffer_seconds: float | None = None) -> bool:
        """Return True if the token is present and not near expiry.

        ``buffer_seconds`` widens the margin for callers that can
        afford to renew early. Tool discovery passes a large one so a
        token that would expire mid-turn is renewed before the turn
        starts, rather than failing a tool call several seconds in
        and asking the user to reconnect something that was working.
        """
        if not token.access_token:
            return False
        if token.expires_at is None:
            return True
        margin = (
            _TOKEN_EXPIRY_BUFFER_SECONDS if buffer_seconds is None else buffer_seconds
        )
        remaining = (token.expires_at - datetime.now(UTC)).total_seconds()
        return remaining > margin

    async def _refresh_mcp_token(
        self,
        stored: McpOAuthToken,
        server_cfg: dict[str, Any],
    ) -> str | None:
        """Refresh an expired token using the stored refresh_token."""
        # Discovered, like the authorize and exchange steps. A
        # discovered server has no token_endpoint in its config, so
        # reading the raw block here worked until the first access
        # token expired and then failed quietly — the connection would
        # simply stop working an hour after it was made.
        from deep_agent.aegra.mcp_oauth_handlers import resolve_oauth_config

        mcp_name = getattr(stored, "mcp_name", "")
        oauth_cfg = await resolve_oauth_config(mcp_name, server_cfg)
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
        except httpx.HTTPStatusError as exc:
            # The provider's own words. Without them a rejected
            # refresh is indistinguishable from a network blip in the
            # log, and the only visible symptom is an integration
            # that says "expired" forever while the token row still
            # looks healthy.
            body_text = ""
            try:
                body_text = exc.response.text[:500]
            except Exception:
                pass
            logger.error(
                "MCP token refresh rejected for '%s': HTTP %s %s",
                stored.mcp_name,
                exc.response.status_code,
                body_text,
            )
            if _is_invalid_grant(body_text):
                # The provider has told us this refresh token is dead.
                # Keeping it means every later check pays another round
                # trip to be told the same thing — 9s per call against
                # one server, on a settings page that lists eight
                # servers and on every chat turn that touches one.
                #
                # Clearing it (not the row) makes the next check answer
                # "not connected" from local state, which is both the
                # truth and free. Reconnecting writes a new one.
                await self._clear_dead_refresh_token(stored)
            return None
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
        # The baseline for a refresh is what this token already holds,
        # not what discovery advertises.
        #
        # A refresh request carries no `scope` parameter — see `data`
        # above — so RFC 6749 §6 keeps the scopes of the original
        # grant. Comparing the response against the discovery
        # document's `scopes_supported` therefore checks it against
        # something nobody requested. One server advertises
        # `session:role-any` and has only ever granted `read write`, so
        # every refresh was rejected for "missing" a scope the token
        # was never issued with, and the integration dropped out an
        # hour after each connect.
        #
        # Same rule as the Harbor fix on the authorize path: what the
        # provider actually granted wins over what it advertises. A
        # refresh returning *less* than the token already had is a real
        # downgrade, and still rejected.
        held_scopes = list(stored.scopes or [])
        parsed_scopes = parse_token_scopes(body)
        if parsed_scopes is not None:
            scopes = validate_granted_scopes(
                parsed_scopes, held_scopes, stored.mcp_name
            )
            if scopes is None and held_scopes:
                # Reached only when the refresh granted nothing at all.
                # A narrower grant is kept: discarding it would turn a
                # partly-capable integration into a dead one, and the
                # reconnect produces the same narrower grant.
                logger.error(
                    "MCP token refresh for '%s' granted no scopes (had: %s)",
                    stored.mcp_name,
                    held_scopes,
                )
                return None
        else:
            scopes = stored.scopes

        await self._store.upsert_token(
            agent_name=stored.agent_name,
            user_id=stored.user_id,
            mcp_name=stored.mcp_name,
            access_token=new_access,
            refresh_token=new_refresh,
            expires_at=expires_at,
            scopes=scopes,
            # A refresh renews the credential, not the person.
            account=stored.account,
        )
        logger.info(
            "Refreshed MCP OAuth token for '%s' (new expires_at=%s)",
            stored.mcp_name,
            expires_at,
        )
        return new_access

    async def _clear_dead_refresh_token(self, stored: McpOAuthToken) -> None:
        """Drop a refresh token the provider has rejected outright."""
        try:
            await self._store.upsert_token(
                agent_name=stored.agent_name,
                user_id=stored.user_id,
                mcp_name=stored.mcp_name,
                access_token=stored.access_token,
                refresh_token=None,
                expires_at=stored.expires_at,
                scopes=stored.scopes,
                account=stored.account,
            )
            self.invalidate_cache(stored.user_id, stored.mcp_name)
            logger.info(
                "[%s] cleared rejected refresh token — reconnect required",
                stored.mcp_name,
            )
        except Exception:
            # Never let bookkeeping turn a failed refresh into a
            # failed request; the caller still raises NeedsAuthorization.
            logger.warning(
                "[%s] could not clear rejected refresh token",
                stored.mcp_name,
                exc_info=True,
            )

    async def _resolve_client_credentials(
        self, mcp_name: str, server_cfg: dict[str, Any]
    ) -> tuple[str | None, str | None]:
        """Return (client_id, client_secret) from config or DCR store."""
        auth_mode = server_cfg.get("auth_mode", "sso")
        oauth_cfg = server_cfg.get("oauth") or {}

        if auth_mode == "oauth":
            return oauth_cfg.get("client_id"), resolve_oauth_client_secret(
                oauth_cfg, mcp_name
            )

        if auth_mode == "dcr":
            current_agent_name = settings.agent_deployment_id
            client = await self._store.get_client(current_agent_name, mcp_name)
            if client is not None:
                # A registered client is the more specific grant.
                return client.client_id, client.client_secret

            # No record does not mean no client.
            #
            # A CIMD server never registers us: the client_id *is*
            # this agent's metadata URL, and the authorize path knows
            # that while this one did not. So every refresh against
            # such a server failed with "client_id unavailable", and
            # the connection had to be made again by hand the first
            # time its access token expired — five minutes, for the
            # servers that prompted this.
            from deep_agent.aegra.mcp_oauth_handlers import resolve_oauth_config

            try:
                oauth = await resolve_oauth_config(mcp_name, server_cfg)
            except Exception:
                logger.debug(
                    "[%s] could not resolve OAuth config for client lookup",
                    mcp_name,
                    exc_info=True,
                )
                return None, None

            if oauth.get("client_id_metadata_document_supported") and not oauth.get(
                "registration_endpoint"
            ):
                return (
                    f"{settings.agent_public_base_url}/.well-known/oauth-client",
                    None,
                )
            return None, None

        return None, None


_default_resolver: McpCredentialResolver | None = None


def get_mcp_credential_resolver() -> McpCredentialResolver:
    """Return the process-wide credential resolver singleton."""
    global _default_resolver  # noqa: PLW0603
    if _default_resolver is None:
        _default_resolver = McpCredentialResolver()
    return _default_resolver
