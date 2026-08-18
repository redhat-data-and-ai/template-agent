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

    def test_defaults_auth_mode_to_sso(self, tmp_path):
        self._write_minimal_config_dir(tmp_path)
        mcp_json = tmp_path / "mcp.json"
        mcp_json.write_text(
            '{"mcpServers": {"sso-mcp": {"url": "http://localhost/mcp", "enabled": true}}}'
        )
        cfg = AgentConfig(tmp_path)
        servers = cfg.get_mcp_servers()
        assert servers["sso-mcp"]["auth_mode"] == "sso"

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
