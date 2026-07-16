"""Unit tests for OAuth client secret resolution from environment variables."""

from __future__ import annotations

import pytest

from deep_agent.aegra.mcp_auth import resolve_oauth_client_secret
from deep_agent.src.agent.config.loader import AgentConfig


class TestResolveOauthClientSecret:
    def test_reads_from_env_var(self, monkeypatch):
        monkeypatch.setenv("TEST_MCP_CLIENT_SECRET", "from-env")
        secret = resolve_oauth_client_secret(
            {"client_secret_env": "TEST_MCP_CLIENT_SECRET"},
            "oauth-mcp",
        )
        assert secret == "from-env"

    def test_env_var_takes_precedence_over_inline(self, monkeypatch):
        monkeypatch.setenv("TEST_MCP_CLIENT_SECRET", "from-env")
        secret = resolve_oauth_client_secret(
            {
                "client_secret_env": "TEST_MCP_CLIENT_SECRET",
                "client_secret": "inline-value",
            },
            "oauth-mcp",
        )
        assert secret == "from-env"

    def test_warns_on_inline_value(self, monkeypatch, caplog):
        monkeypatch.delenv("TEST_MCP_CLIENT_SECRET", raising=False)
        with caplog.at_level("WARNING"):
            secret = resolve_oauth_client_secret(
                {"client_secret": "inline-value"},
                "oauth-mcp",
            )
        assert secret == "inline-value"
        assert any(
            "client_secret in mcp.json is insecure" in r.message for r in caplog.records
        )


class TestMcpConfigInlineSecretWarning:
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

    def test_warns_on_inline_secret_in_mcp_json(self, tmp_path, caplog):
        self._write_minimal_config_dir(tmp_path)
        (tmp_path / "mcp.json").write_text(
            """
            {
              "mcpServers": {
                "oauth-mcp": {
                  "url": "http://localhost/mcp",
                  "enabled": true,
                  "auth_mode": "oauth",
                  "oauth": {
                    "client_id": "cid",
                    "client_secret": "inline-value",
                    "authorization_endpoint": "https://as.example.com/authorize",
                    "token_endpoint": "https://as.example.com/token"
                  }
                }
              }
            }
            """
        )
        with caplog.at_level("WARNING"):
            AgentConfig(tmp_path).get_mcp_servers()
        assert any(
            "client_secret in mcp.json is insecure" in r.message for r in caplog.records
        )
