"""Tests for the agent module — _load_mcp_config helper."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from template_agent.src.core.agent import _load_mcp_config


class TestLoadMcpConfig:
    """Tests for _load_mcp_config: mcp.json prefix reading and env-var fallback."""

    def _write_mcp_json(self, tmp_path: Path, servers: dict) -> Path:
        mcp_file = tmp_path / "mcp.json"
        mcp_file.write_text(json.dumps({"mcpServers": servers}))
        return mcp_file

    # -- prefix scenarios --

    def test_reads_prefix_from_json(self, tmp_path):
        """tool_prefix in mcp.json becomes the dict key."""
        mcp_file = self._write_mcp_json(
            tmp_path,
            {
                "search-mcp-prod": {
                    "url": "http://search:9090/mcp",
                    "transport": "streamable_http",
                    "ssl_verify": False,
                    "tool_prefix": "search",
                }
            },
        )
        with patch("template_agent.src.core.agent.settings") as mock_settings:
            mock_settings.MCP_CONFIG_PATH = str(mcp_file)
            result = _load_mcp_config(sso_token=None)

        assert "search" in result
        assert "search-mcp-prod" not in result
        assert result["search"]["url"] == "http://search:9090/mcp"

    def test_no_prefix_uses_server_key(self, tmp_path):
        """Without tool_prefix, the raw server key is used."""
        mcp_file = self._write_mcp_json(
            tmp_path,
            {
                "gitlab-mcp": {
                    "url": "http://gitlab:8080/mcp",
                    "transport": "streamable_http",
                }
            },
        )
        with patch("template_agent.src.core.agent.settings") as mock_settings:
            mock_settings.MCP_CONFIG_PATH = str(mcp_file)
            result = _load_mcp_config(sso_token=None)

        assert "gitlab-mcp" in result
        assert result["gitlab-mcp"]["url"] == "http://gitlab:8080/mcp"

    def test_empty_prefix_uses_server_key(self, tmp_path):
        """Empty string tool_prefix is treated same as absent."""
        mcp_file = self._write_mcp_json(
            tmp_path,
            {
                "data-mcp": {
                    "url": "http://data:9090/mcp",
                    "transport": "streamable_http",
                    "tool_prefix": "",
                }
            },
        )
        with patch("template_agent.src.core.agent.settings") as mock_settings:
            mock_settings.MCP_CONFIG_PATH = str(mcp_file)
            result = _load_mcp_config(sso_token=None)

        assert "data-mcp" in result

    # -- env var fallback --

    def test_fallback_to_env_vars(self, tmp_path):
        """When mcp.json does not exist, falls back to env var settings."""
        missing_path = tmp_path / "nonexistent" / "mcp.json"
        with patch("template_agent.src.core.agent.settings") as mock_settings:
            mock_settings.MCP_CONFIG_PATH = str(missing_path)
            mock_settings.MCP_SERVER_URL = "http://localhost:5001/mcp/"
            mock_settings.MCP_TRANSPORT_PROTOCOL = "streamable_http"
            mock_settings.MCP_SERVER_NAME = "template-mcp-server"
            mock_settings.MCP_SSL_VERIFY = True
            result = _load_mcp_config(sso_token=None)

        assert "template-mcp-server" in result
        assert result["template-mcp-server"]["url"] == "http://localhost:5001/mcp/"
        assert result["template-mcp-server"]["transport"] == "streamable_http"

    # -- auth / SSL --

    def test_applies_sso_token(self, tmp_path):
        """Authorization header is set when sso_token is provided."""
        mcp_file = self._write_mcp_json(
            tmp_path,
            {
                "my-mcp": {
                    "url": "http://mcp:8080/mcp",
                    "transport": "streamable_http",
                }
            },
        )
        with patch("template_agent.src.core.agent.settings") as mock_settings:
            mock_settings.MCP_CONFIG_PATH = str(mcp_file)
            result = _load_mcp_config(sso_token="test-token-123")

        assert result["my-mcp"]["headers"] == {"Authorization": "Bearer test-token-123"}

    def test_no_auth_header_without_token(self, tmp_path):
        """No Authorization header when sso_token is None."""
        mcp_file = self._write_mcp_json(
            tmp_path,
            {
                "my-mcp": {
                    "url": "http://mcp:8080/mcp",
                    "transport": "streamable_http",
                }
            },
        )
        with patch("template_agent.src.core.agent.settings") as mock_settings:
            mock_settings.MCP_CONFIG_PATH = str(mcp_file)
            result = _load_mcp_config(sso_token=None)

        assert "headers" not in result["my-mcp"]

    def test_ssl_verify_false(self, tmp_path):
        """ssl_verify: false in mcp.json sets verify: False in config."""
        mcp_file = self._write_mcp_json(
            tmp_path,
            {
                "my-mcp": {
                    "url": "http://mcp:8080/mcp",
                    "transport": "streamable_http",
                    "ssl_verify": False,
                }
            },
        )
        with patch("template_agent.src.core.agent.settings") as mock_settings:
            mock_settings.MCP_CONFIG_PATH = str(mcp_file)
            result = _load_mcp_config(sso_token=None)

        assert result["my-mcp"]["verify"] is False

    def test_ssl_verify_true_omits_verify_key(self, tmp_path):
        """ssl_verify: true (default) does not add verify key."""
        mcp_file = self._write_mcp_json(
            tmp_path,
            {
                "my-mcp": {
                    "url": "http://mcp:8080/mcp",
                    "transport": "streamable_http",
                    "ssl_verify": True,
                }
            },
        )
        with patch("template_agent.src.core.agent.settings") as mock_settings:
            mock_settings.MCP_CONFIG_PATH = str(mcp_file)
            result = _load_mcp_config(sso_token=None)

        assert "verify" not in result["my-mcp"]

    # -- sub-agent consistency --

    def test_consistent_for_subagents(self, tmp_path):
        """Same mcp.json produces identical output on repeated calls."""
        mcp_file = self._write_mcp_json(
            tmp_path,
            {
                "search-mcp-prod": {
                    "url": "http://search:9090/mcp",
                    "transport": "streamable_http",
                    "tool_prefix": "search",
                }
            },
        )
        with patch("template_agent.src.core.agent.settings") as mock_settings:
            mock_settings.MCP_CONFIG_PATH = str(mcp_file)
            result1 = _load_mcp_config(sso_token="token-a")
            result2 = _load_mcp_config(sso_token="token-a")

        assert result1 == result2

    # -- env var fallback SSL --

    def test_fallback_ssl_verify_false(self, tmp_path):
        """Env var fallback respects MCP_SSL_VERIFY=False."""
        missing_path = tmp_path / "nonexistent" / "mcp.json"
        with patch("template_agent.src.core.agent.settings") as mock_settings:
            mock_settings.MCP_CONFIG_PATH = str(missing_path)
            mock_settings.MCP_SERVER_URL = "http://localhost:5001/mcp/"
            mock_settings.MCP_TRANSPORT_PROTOCOL = "streamable_http"
            mock_settings.MCP_SERVER_NAME = "template-mcp-server"
            mock_settings.MCP_SSL_VERIFY = False
            result = _load_mcp_config(sso_token=None)

        assert result["template-mcp-server"]["verify"] is False
