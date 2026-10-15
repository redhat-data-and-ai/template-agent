"""Unit tests for MCP config validation and credential resolver."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_agent.aegra.mcp import set_mcp_auth_context
from deep_agent.aegra.mcp_auth import McpCredentialResolver, NeedsAuthorization
from deep_agent.aegra.mcp_token_store import McpOAuthToken
from deep_agent.src.agent.config.loader import AgentConfig


class TestMcpConfigValidation:
    def setup_method(self):
        AgentConfig._instance = None

    @staticmethod
    def _write_minimal_config_dir(tmp_path):
        (tmp_path / "PROMPT.md").write_text(
            """---
name: test-orchestrator
model: gemini-2.5-flash
---
Test prompt.
"""
        )

    def test_defaults_auth_mode_to_dcr(self, tmp_path):
        # Changed from "sso". Everything about a provider is now
        # discovered at connect time, so dcr is what an unqualified
        # entry should mean; "sso" is for a server that takes the
        # caller's own bearer token, which is the exception.
        self._write_minimal_config_dir(tmp_path)
        mcp_json = tmp_path / "mcp.json"
        mcp_json.write_text(
            '{"mcpServers": {"plain-mcp": {"url": "http://localhost/mcp"}}}'
        )
        cfg = AgentConfig(tmp_path)
        servers = cfg.get_mcp_servers()
        assert servers["plain-mcp"]["auth_mode"] == "dcr"

    def test_an_entry_is_enabled_without_saying_so(self, tmp_path):
        # A list where every line needs a second line saying "yes,
        # really" has a redundant column.
        self._write_minimal_config_dir(tmp_path)
        (tmp_path / "mcp.json").write_text(
            '{"mcpServers": {"plain-mcp": {"url": "http://localhost/mcp"}}}'
        )
        assert AgentConfig(tmp_path).get_mcp_servers()["plain-mcp"]["enabled"] is True

    def test_explicit_values_still_win(self, tmp_path):
        self._write_minimal_config_dir(tmp_path)
        (tmp_path / "mcp.json").write_text(
            '{"mcpServers": {"m": {"url": "http://localhost/mcp",'
            ' "enabled": false, "auth_mode": "sso"}}}'
        )
        s = AgentConfig(tmp_path).get_mcp_servers()["m"]
        assert s["enabled"] is False
        assert s["auth_mode"] == "sso"

    def test_type_http_is_read_as_a_transport(self, tmp_path):
        # So an entry can be pasted from a VS Code mcp.json unchanged.
        self._write_minimal_config_dir(tmp_path)
        (tmp_path / "mcp.json").write_text(
            '{"mcpServers": {"m": {"type": "http", "url": "http://localhost/mcp"}}}'
        )
        assert (
            AgentConfig(tmp_path).get_mcp_servers()["m"]["transport"]
            == "streamable_http"
        )

    def test_loads_jsonc_line_comments(self, tmp_path):
        self._write_minimal_config_dir(tmp_path)
        mcp_json = tmp_path / "mcp.json"
        mcp_json.write_text(
            """
            {
              "mcpServers": {
                "template-mcp-server": {
                  "url": "http://host.containers.internal:5001/mcp",
                  // "url": "http://localhost:5001/mcp",
                  "enabled": true
                }
              }
            }
            """
        )
        servers = AgentConfig(tmp_path).get_mcp_servers()
        assert (
            servers["template-mcp-server"]["url"]
            == "http://host.containers.internal:5001/mcp"
        )

    def test_loads_jsonc_with_escaped_quotes(self, tmp_path):
        self._write_minimal_config_dir(tmp_path)
        mcp_json = tmp_path / "mcp.json"
        mcp_json.write_text(
            r"""
            {
              "mcpServers": {
                "test-mcp": {
                  "url": "http://host/mcp?q=\"hello\"",
                  // comment with escaped quote: \"
                  "label": "backslash\\and-quote",
                  "enabled": true
                }
              }
            }
            """
        )
        servers = AgentConfig(tmp_path).get_mcp_servers()
        assert servers["test-mcp"]["url"] == 'http://host/mcp?q="hello"'
        assert servers["test-mcp"]["label"] == "backslash\\and-quote"

    def test_logs_error_for_oauth_without_client_id(self, tmp_path, caplog):
        self._write_minimal_config_dir(tmp_path)
        mcp_json = tmp_path / "mcp.json"
        mcp_json.write_text(
            """
            {
              "mcpServers": {
                "oauth-mcp": {
                  "url": "http://localhost/mcp",
                  "enabled": true,
                  "auth_mode": "oauth",
                  "oauth": {
                    "authorization_endpoint": "https://as.example.com/authorize",
                    "token_endpoint": "https://as.example.com/token"
                  }
                }
              }
            }
            """
        )
        with caplog.at_level("ERROR"):
            AgentConfig(tmp_path).get_mcp_servers()
        assert any("client_id is required" in r.message for r in caplog.records)

    def test_client_credentials_no_error_without_authorization_endpoint(
        self, tmp_path, caplog
    ):
        self._write_minimal_config_dir(tmp_path)
        mcp_json = tmp_path / "mcp.json"
        mcp_json.write_text(
            """
            {
              "mcpServers": {
                "cc-mcp": {
                  "url": "http://localhost/mcp",
                  "enabled": true,
                  "auth_mode": "oauth",
                  "oauth": {
                    "grant_type": "client_credentials",
                    "token_endpoint": "https://as.example.com/token",
                    "client_id": "cid"
                  }
                }
              }
            }
            """
        )
        with caplog.at_level("ERROR"):
            servers = AgentConfig(tmp_path).get_mcp_servers()
        assert not any("authorization_endpoint" in r.message for r in caplog.records)
        assert servers["cc-mcp"]["auth_mode"] == "oauth"

    def test_logs_error_for_dcr_without_registration_endpoint(self, tmp_path, caplog):
        self._write_minimal_config_dir(tmp_path)
        mcp_json = tmp_path / "mcp.json"
        mcp_json.write_text(
            """
            {
              "mcpServers": {
                "dcr-mcp": {
                  "url": "http://localhost/mcp",
                  "enabled": true,
                  "auth_mode": "dcr",
                  "oauth": {
                    "authorization_endpoint": "https://as.example.com/authorize",
                    "token_endpoint": "https://as.example.com/token"
                  }
                }
              }
            }
            """
        )
        with caplog.at_level("ERROR"):
            AgentConfig(tmp_path).get_mcp_servers()
        assert any(
            "registration_endpoint is required" in r.message for r in caplog.records
        )


@pytest.mark.asyncio
class TestMcpCredentialResolver:
    async def test_sso_returns_refreshed_token(self):
        store = AsyncMock()
        resolver = McpCredentialResolver(token_store=store)
        set_mcp_auth_context("access-token", "refresh-token")

        with patch(
            "deep_agent.aegra.mcp_auth.refresh_access_token",
            new=AsyncMock(return_value="fresh-token"),
        ) as refresh:
            token = await resolver.resolve(
                "user-1",
                "sso-mcp",
                {"auth_mode": "sso"},
            )

        assert token == "fresh-token"
        refresh.assert_awaited_once_with("access-token", "refresh-token")
        store.get_token.assert_not_called()

    async def test_oauth_raises_when_no_stored_token(self):
        store = AsyncMock()
        store.get_token = AsyncMock(return_value=None)
        resolver = McpCredentialResolver(token_store=store)

        with pytest.raises(NeedsAuthorization) as exc:
            await resolver.resolve(
                "user-1",
                "oauth-mcp",
                {
                    "auth_mode": "oauth",
                    "oauth": {"token_endpoint": "https://as.example.com/token"},
                },
            )

        assert exc.value.mcp_name == "oauth-mcp"
        assert exc.value.connect_url.endswith("/mcp/oauth-mcp/connect")

    async def test_oauth_returns_valid_stored_token(self):
        store = AsyncMock()
        store.get_token = AsyncMock(
            return_value=McpOAuthToken(
                agent_name="test-agent",
                user_id="user-1",
                mcp_name="oauth-mcp",
                access_token="stored-access",
                expires_at=datetime.now(UTC) + timedelta(hours=1),
            )
        )
        resolver = McpCredentialResolver(token_store=store)

        token = await resolver.resolve(
            "user-1",
            "oauth-mcp",
            {"auth_mode": "oauth", "oauth": {}},
        )
        assert token == "stored-access"

    async def test_oauth_refreshes_expired_token(self):
        store = AsyncMock()
        store.get_token = AsyncMock(
            return_value=McpOAuthToken(
                agent_name="test-agent",
                user_id="user-1",
                mcp_name="oauth-mcp",
                access_token="expired-access",
                refresh_token="refresh-me",
                expires_at=datetime.now(UTC) - timedelta(minutes=5),
            )
        )
        store.upsert_token = AsyncMock()
        resolver = McpCredentialResolver(token_store=store)

        with patch.object(
            resolver,
            "_refresh_mcp_token",
            new=AsyncMock(return_value="new-access"),
        ) as refresh:
            token = await resolver.resolve(
                "user-1",
                "oauth-mcp",
                {
                    "auth_mode": "oauth",
                    "oauth": {
                        "token_endpoint": "https://as.example.com/token",
                        "client_id": "cid",
                    },
                },
            )

        assert token == "new-access"
        refresh.assert_awaited_once()

    async def test_oauth_raises_when_expired_and_no_refresh_token(self):
        """Expired token with no refresh_token triggers NeedsAuthorization."""
        store = AsyncMock()
        store.get_token = AsyncMock(
            return_value=McpOAuthToken(
                agent_name="test-agent",
                user_id="user-1",
                mcp_name="oauth-mcp",
                access_token="expired-access",
                refresh_token=None,
                expires_at=datetime.now(UTC) - timedelta(minutes=5),
            )
        )
        resolver = McpCredentialResolver(token_store=store)

        with pytest.raises(NeedsAuthorization) as exc:
            await resolver.resolve(
                "user-1",
                "oauth-mcp",
                {"auth_mode": "oauth", "oauth": {}},
            )
        assert exc.value.mcp_name == "oauth-mcp"
        store.get_token.assert_awaited_once()

    async def test_resolver_caches_resolved_oauth_token(self):
        store = AsyncMock()
        store.get_token = AsyncMock(
            return_value=McpOAuthToken(
                agent_name="test-agent",
                user_id="user-1",
                mcp_name="oauth-mcp",
                access_token="stored-access",
                expires_at=datetime.now(UTC) + timedelta(hours=1),
            )
        )
        resolver = McpCredentialResolver(token_store=store)

        cfg = {"auth_mode": "oauth", "oauth": {}}
        await resolver.resolve("user-1", "oauth-mcp", cfg)
        await resolver.resolve("user-1", "oauth-mcp", cfg)

        store.get_token.assert_awaited_once()


@pytest.mark.asyncio
class TestOAuthLockEdgeCases:
    """Tests for distributed lock edge cases in _resolve_oauth."""

    @staticmethod
    def _expired_token(*, refresh_token="refresh-me"):
        return McpOAuthToken(
            agent_name="test-agent",
            user_id="user-1",
            mcp_name="oauth-mcp",
            access_token="expired-access",
            refresh_token=refresh_token,
            expires_at=datetime.now(UTC) - timedelta(minutes=5),
        )

    async def test_token_vanished_during_lock(self):
        """When token disappears after acquiring the lock, NeedsAuthorization is raised."""
        from contextlib import asynccontextmanager

        store = AsyncMock()
        # First call returns expired token, second call (inside lock) returns None
        store.get_token = AsyncMock(side_effect=[self._expired_token(), None])
        resolver = McpCredentialResolver(token_store=store)

        @asynccontextmanager
        async def _mock_lock(*a, **kw):
            yield "held"

        with patch("deep_agent.aegra.mcp_auth.distributed_lock", _mock_lock):
            with pytest.raises(NeedsAuthorization) as exc:
                await resolver.resolve(
                    "user-1", "oauth-mcp", {"auth_mode": "oauth", "oauth": {}}
                )
            assert exc.value.mcp_name == "oauth-mcp"

    async def test_no_refresh_token_after_lock(self):
        """When refresh_token vanishes after lock, NeedsAuthorization is raised."""
        from contextlib import asynccontextmanager

        no_refresh = McpOAuthToken(
            agent_name="test-agent",
            user_id="user-1",
            mcp_name="oauth-mcp",
            access_token="expired-access",
            refresh_token=None,
            expires_at=datetime.now(UTC) - timedelta(minutes=5),
        )

        store = AsyncMock()
        store.get_token = AsyncMock(side_effect=[self._expired_token(), no_refresh])
        resolver = McpCredentialResolver(token_store=store)

        @asynccontextmanager
        async def _mock_lock(*a, **kw):
            yield "held"

        with patch("deep_agent.aegra.mcp_auth.distributed_lock", _mock_lock):
            with pytest.raises(NeedsAuthorization):
                await resolver.resolve(
                    "user-1", "oauth-mcp", {"auth_mode": "oauth", "oauth": {}}
                )

    async def test_peer_refresh_timed_out(self):
        """When lock times out and peer didn't refresh, NeedsAuthorization is raised."""
        from contextlib import asynccontextmanager

        store = AsyncMock()
        store.get_token = AsyncMock(
            side_effect=[self._expired_token(), self._expired_token()]
        )
        resolver = McpCredentialResolver(token_store=store)

        @asynccontextmanager
        async def _mock_lock(*a, **kw):
            yield "timeout"

        with (
            patch("deep_agent.aegra.mcp_auth.distributed_lock", _mock_lock),
            patch.object(
                resolver,
                "_wait_for_refreshed_token",
                new=AsyncMock(return_value=None),
            ),
        ):
            with pytest.raises(NeedsAuthorization):
                await resolver.resolve(
                    "user-1", "oauth-mcp", {"auth_mode": "oauth", "oauth": {}}
                )

    async def test_refresh_exhausted_raises(self):
        """When _refresh_mcp_token returns None, NeedsAuthorization is raised."""
        from contextlib import asynccontextmanager

        store = AsyncMock()
        store.get_token = AsyncMock(
            side_effect=[self._expired_token(), self._expired_token()]
        )
        resolver = McpCredentialResolver(token_store=store)

        @asynccontextmanager
        async def _mock_lock(*a, **kw):
            yield "held"

        with (
            patch("deep_agent.aegra.mcp_auth.distributed_lock", _mock_lock),
            patch.object(
                resolver,
                "_refresh_mcp_token",
                new=AsyncMock(return_value=None),
            ),
        ):
            with pytest.raises(NeedsAuthorization):
                await resolver.resolve(
                    "user-1", "oauth-mcp", {"auth_mode": "oauth", "oauth": {}}
                )

    async def test_refresh_success_logs_and_returns(self):
        """Successful token refresh returns new access token."""
        from contextlib import asynccontextmanager

        store = AsyncMock()
        store.get_token = AsyncMock(
            side_effect=[self._expired_token(), self._expired_token()]
        )
        store.upsert_token = AsyncMock()
        resolver = McpCredentialResolver(token_store=store)

        @asynccontextmanager
        async def _mock_lock(*a, **kw):
            yield "held"

        with (
            patch("deep_agent.aegra.mcp_auth.distributed_lock", _mock_lock),
            patch.object(
                resolver,
                "_refresh_mcp_token",
                new=AsyncMock(return_value="new-access-token"),
            ),
        ):
            token = await resolver.resolve(
                "user-1", "oauth-mcp", {"auth_mode": "oauth", "oauth": {}}
            )
            assert token == "new-access-token"


@pytest.mark.asyncio
class TestRefreshMcpToken:
    """Tests for _refresh_mcp_token success path."""

    async def test_successful_refresh_returns_new_access_token(self):
        store = AsyncMock()
        store.upsert_token = AsyncMock()
        resolver = McpCredentialResolver(token_store=store)

        stored = McpOAuthToken(
            agent_name="test-agent",
            user_id="user-1",
            mcp_name="oauth-mcp",
            access_token="expired-access",
            refresh_token="refresh-me",
            expires_at=datetime.now(UTC) - timedelta(minutes=5),
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "new-access-token",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        mock_response.raise_for_status = MagicMock()

        mock_ctx = AsyncMock()
        mock_ctx.post = AsyncMock(return_value=mock_response)

        server_cfg = {
            "auth_mode": "oauth",
            "oauth": {
                "token_endpoint": "https://auth.example.com/token",
                "client_id": "cid",
                "client_secret": "csecret",
            },
        }

        with patch("deep_agent.aegra.mcp_auth.httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await resolver._refresh_mcp_token(stored, server_cfg)

        assert result == "new-access-token"
        store.upsert_token.assert_awaited_once()


@pytest.mark.asyncio
class TestClientCredentialsGrant:
    """Tests for the client_credentials grant type flow."""

    async def test_client_credentials_acquires_token(self):
        store = AsyncMock()
        resolver = McpCredentialResolver(token_store=store)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "cc-access-token",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        mock_response.raise_for_status = MagicMock()

        cfg = {
            "auth_mode": "oauth",
            "oauth": {
                "grant_type": "client_credentials",
                "token_endpoint": "https://auth.example.com/token",
                "client_id": "cid",
                "client_secret": "csecret",
            },
        }

        with patch("deep_agent.aegra.mcp_auth.httpx.AsyncClient") as mock_client:
            mock_ctx = AsyncMock(post=AsyncMock(return_value=mock_response))
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            token = await resolver.resolve("user-1", "cc-mcp", cfg)

        assert token == "cc-access-token"
        store.get_token.assert_not_called()

    async def test_client_credentials_caches_token(self):
        store = AsyncMock()
        resolver = McpCredentialResolver(token_store=store)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "cc-access-token",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = MagicMock()

        cfg = {
            "auth_mode": "oauth",
            "oauth": {
                "grant_type": "client_credentials",
                "token_endpoint": "https://auth.example.com/token",
                "client_id": "cid",
                "client_secret": "csecret",
            },
        }

        with patch("deep_agent.aegra.mcp_auth.httpx.AsyncClient") as mock_client:
            mock_ctx = AsyncMock(post=AsyncMock(return_value=mock_response))
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            await resolver.resolve("user-1", "cc-mcp", cfg)
            await resolver.resolve("user-1", "cc-mcp", cfg)

            mock_ctx.post.assert_awaited_once()

    async def test_client_credentials_has_valid_token_returns_true(self):
        store = AsyncMock()
        resolver = McpCredentialResolver(token_store=store)

        cfg = {
            "auth_mode": "oauth",
            "oauth": {
                "grant_type": "client_credentials",
                "token_endpoint": "https://auth.example.com/token",
                "client_id": "cid",
            },
        }

        result = await resolver.has_valid_token("user-1", "cc-mcp", cfg)
        assert result is True
        store.get_token.assert_not_called()

    async def test_client_credentials_raises_without_token_endpoint(self):
        store = AsyncMock()
        resolver = McpCredentialResolver(token_store=store)

        cfg = {
            "auth_mode": "oauth",
            "oauth": {
                "grant_type": "client_credentials",
                "client_id": "cid",
            },
        }

        with pytest.raises(RuntimeError, match="token_endpoint missing"):
            await resolver.resolve("user-1", "cc-mcp", cfg)

    async def test_client_credentials_raises_without_client_id(self):
        store = AsyncMock()
        resolver = McpCredentialResolver(token_store=store)

        cfg = {
            "auth_mode": "oauth",
            "oauth": {
                "grant_type": "client_credentials",
                "token_endpoint": "https://auth.example.com/token",
            },
        }

        with pytest.raises(RuntimeError, match="client_id unavailable"):
            await resolver.resolve("user-1", "cc-mcp", cfg)

    async def test_client_credentials_raises_on_http_failure(self):
        store = AsyncMock()
        resolver = McpCredentialResolver(token_store=store)

        cfg = {
            "auth_mode": "oauth",
            "oauth": {
                "grant_type": "client_credentials",
                "token_endpoint": "https://auth.example.com/token",
                "client_id": "cid",
                "client_secret": "csecret",
            },
        }

        with patch("deep_agent.aegra.mcp_auth.httpx.AsyncClient") as mock_client:
            mock_ctx = AsyncMock(
                post=AsyncMock(side_effect=Exception("connection refused")),
            )
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            with pytest.raises(RuntimeError, match="token acquisition failed"):
                await resolver.resolve("user-1", "cc-mcp", cfg)

    async def test_client_credentials_raises_on_missing_access_token(self):
        store = AsyncMock()
        resolver = McpCredentialResolver(token_store=store)

        mock_response = MagicMock()
        mock_response.json.return_value = {"token_type": "Bearer"}
        mock_response.raise_for_status = MagicMock()

        cfg = {
            "auth_mode": "oauth",
            "oauth": {
                "grant_type": "client_credentials",
                "token_endpoint": "https://auth.example.com/token",
                "client_id": "cid",
                "client_secret": "csecret",
            },
        }

        with patch("deep_agent.aegra.mcp_auth.httpx.AsyncClient") as mock_client:
            mock_ctx = AsyncMock(post=AsyncMock(return_value=mock_response))
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            with pytest.raises(RuntimeError, match="missing access_token"):
                await resolver.resolve("user-1", "cc-mcp", cfg)

    async def test_client_credentials_sends_scopes(self):
        store = AsyncMock()
        resolver = McpCredentialResolver(token_store=store)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "scoped-token",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = MagicMock()

        cfg = {
            "auth_mode": "oauth",
            "oauth": {
                "grant_type": "client_credentials",
                "token_endpoint": "https://auth.example.com/token",
                "client_id": "cid",
                "client_secret": "csecret",
                "scopes": ["read", "write"],
            },
        }

        with patch("deep_agent.aegra.mcp_auth.httpx.AsyncClient") as mock_client:
            mock_ctx = AsyncMock(post=AsyncMock(return_value=mock_response))
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            token = await resolver.resolve("user-1", "cc-mcp", cfg)

        assert token == "scoped-token"
        call_kwargs = mock_ctx.post.call_args
        data = call_kwargs.kwargs.get("data", call_kwargs[1].get("data", {}))
        assert data["scope"] == "read write"


class TestRefreshScopeBaseline:
    """What a refresh is checked against.

    A refresh request carries no `scope` parameter, so RFC 6749 §6
    keeps the scopes of the original grant. Validating the response
    against the discovery document's `scopes_supported` checks it
    against something nobody requested — which is how Dataverse, which
    advertises `session:role-any` and grants `read write`, failed every
    refresh and dropped out an hour after each connect.
    """

    @staticmethod
    def _stored(scopes):
        return McpOAuthToken(
            agent_name="test-agent",
            user_id="user-1",
            mcp_name="dataverse",
            access_token="old-access",
            refresh_token="refresh-me",
            expires_at=datetime.now(UTC) - timedelta(minutes=5),
            scopes=scopes,
        )

    async def _refresh(self, stored, response_scope, discovery_scopes):
        store = AsyncMock()
        store.upsert_token = AsyncMock()
        resolver = McpCredentialResolver(token_store=store)

        body = {"access_token": "new-access", "expires_in": 3600}
        if response_scope is not None:
            body["scope"] = response_scope

        response = MagicMock()
        response.json = MagicMock(return_value=body)
        response.raise_for_status = MagicMock()

        client = AsyncMock()
        client.post = AsyncMock(return_value=response)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "deep_agent.aegra.mcp_oauth_handlers.resolve_oauth_config",
                new=AsyncMock(
                    return_value={
                        "token_endpoint": "https://as.example.com/token",
                        "client_id": "cid",
                        "scopes": discovery_scopes,
                    }
                ),
            ),
            patch.object(
                resolver,
                "_resolve_client_credentials",
                new=AsyncMock(return_value=("cid", None)),
            ),
            patch("httpx.AsyncClient", return_value=client),
        ):
            return await resolver._refresh_mcp_token(stored, {})

    async def test_advertised_scope_the_grant_never_had_is_not_required(self):
        # The regression. Discovery says `session:role-any`; the grant
        # has always been `read write`; the refresh returns `read
        # write`. That is the grant being honoured, not a failure.
        token = await self._refresh(
            self._stored(["read", "write"]),
            response_scope="read write",
            discovery_scopes=["session:role-any"],
        )
        assert token == "new-access"

    async def test_a_narrower_refresh_is_kept(self):
        # Superseded an earlier rule that rejected any narrowing. A
        # provider may legitimately return fewer scopes (RFC 6749
        # §3.3), and discarding the token turns a partly-capable
        # integration into a dead one — while the reconnect produces
        # the same narrower grant, so the user loops. The missing
        # scope surfaces as a 403 on the one tool that needs it.
        token = await self._refresh(
            self._stored(["read", "write"]),
            response_scope="read",
            discovery_scopes=[],
        )
        assert token == "new-access"

    # The "granted nothing" rejection is not reachable from here:
    # `parse_token_scopes` normalises an empty or absent `scope` to
    # None, which RFC 6749 §5.1 treats as "same as requested". It is
    # covered directly in test_oauth_scopes.py.

    async def test_a_token_with_no_recorded_scopes_refreshes(self):
        # Tokens stored before scopes were recorded have no baseline;
        # that must not read as "lost every scope".
        token = await self._refresh(
            self._stored(None),
            response_scope="read",
            discovery_scopes=["session:role-any"],
        )
        assert token == "new-access"
