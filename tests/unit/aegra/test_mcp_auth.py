"""Unit tests for MCP config validation and credential resolver."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

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
