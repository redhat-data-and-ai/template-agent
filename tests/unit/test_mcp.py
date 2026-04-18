"""Unit tests for MCP client utilities."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from template_agent.src.core.exceptions import AppException
from template_agent.src.core.mcp import (
    _build_server_config,
    _load_server_configs,
    get_mcp_tools,
)


class TestBuildServerConfig:
    """Tests for _build_server_config function."""

    def test_config_without_sso_token(self):
        """Test server config without SSO token."""
        entry = {
            "url": "http://localhost:8000/mcp/",
            "transport": "http",
            "auth": True,
            "ssl_verify": True,
        }
        config = _build_server_config(entry, sso_token=None)

        assert config["url"] == "http://localhost:8000/mcp/"
        assert config["transport"] == "http"
        assert config["headers"] == {}
        assert "httpx_client_factory" not in config

    def test_config_with_sso_token(self):
        """Test server config with SSO token."""
        entry = {
            "url": "https://api.example.com/mcp/",
            "transport": "https",
            "auth": True,
            "ssl_verify": True,
        }
        config = _build_server_config(entry, sso_token="test_token_123")

        assert config["url"] == "https://api.example.com/mcp/"
        assert config["transport"] == "https"
        assert config["headers"] == {"Authorization": "Bearer test_token_123"}
        assert "httpx_client_factory" not in config

    def test_config_with_ssl_verify_disabled(self):
        """Test server config with SSL verification disabled."""
        entry = {
            "url": "https://api.example.com/mcp/",
            "transport": "https",
            "auth": True,
            "ssl_verify": False,
        }
        config = _build_server_config(entry, sso_token=None)

        assert "httpx_client_factory" in config
        assert callable(config["httpx_client_factory"])

        client = config["httpx_client_factory"]()
        assert hasattr(client, "get")

    def test_config_auth_disabled_ignores_token(self):
        """Test that auth=False means no Authorization header even with token."""
        entry = {
            "url": "http://localhost:8000/mcp/",
            "transport": "http",
            "auth": False,
            "ssl_verify": True,
        }
        config = _build_server_config(entry, sso_token="should_be_ignored")

        assert config["headers"] == {}

    def test_config_defaults(self):
        """Test that missing optional fields use sensible defaults."""
        entry = {"url": "http://localhost:8000/mcp/"}
        config = _build_server_config(entry, sso_token="tok")

        assert config["transport"] == "streamable_http"
        assert config["headers"] == {"Authorization": "Bearer tok"}
        assert "httpx_client_factory" not in config


class TestLoadServerConfigs:
    """Tests for _load_server_configs function."""

    def test_returns_empty_when_no_file(self, tmp_path):
        """Test returns empty dict when JSON file does not exist."""
        fake_path = tmp_path / "nonexistent.json"
        with patch("template_agent.src.core.mcp._CONFIG_PATH", fake_path):
            configs = _load_server_configs()
        assert configs == {}

    def test_loads_from_json(self, tmp_path):
        """Test loading from a valid JSON config file."""
        config_file = tmp_path / "mcp.json"
        config_file.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "server-a": {
                            "url": "http://a:5001/mcp/",
                            "transport": "streamable_http",
                            "enabled": True,
                            "auth": True,
                            "ssl_verify": False,
                            "timeout": 10,
                        }
                    }
                }
            )
        )

        with patch("template_agent.src.core.mcp._CONFIG_PATH", config_file):
            configs = _load_server_configs()

        assert "server-a" in configs
        assert configs["server-a"]["url"] == "http://a:5001/mcp/"

    def test_rejects_missing_url(self, tmp_path):
        """Test that missing 'url' field raises a validation error."""
        config_file = tmp_path / "mcp.json"
        config_file.write_text(
            json.dumps({"mcpServers": {"bad": {"transport": "streamable_http"}}})
        )

        with patch("template_agent.src.core.mcp._CONFIG_PATH", config_file):
            with pytest.raises(AppException) as exc_info:
                _load_server_configs()
            assert "missing required field" in str(exc_info.value).lower()

    def test_rejects_invalid_json(self, tmp_path):
        """Test that malformed JSON raises a validation error."""
        config_file = tmp_path / "mcp.json"
        config_file.write_text("{bad json")

        with patch("template_agent.src.core.mcp._CONFIG_PATH", config_file):
            with pytest.raises(AppException):
                _load_server_configs()


class TestGetMCPTools:
    """Tests for get_mcp_tools function."""

    @pytest.mark.asyncio
    async def test_successful_connection(self, tmp_path):
        """Test successful MCP connection with tools."""
        config_file = tmp_path / "mcp.json"
        config_file.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "test_server": {
                            "url": "http://localhost:8000/mcp/",
                            "transport": "http",
                            "enabled": True,
                            "auth": False,
                            "ssl_verify": True,
                            "timeout": 5,
                        }
                    }
                }
            )
        )

        mock_tool = MagicMock()
        mock_tool.name = "tool1"

        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=[mock_tool])

        with (
            patch("template_agent.src.core.mcp._CONFIG_PATH", config_file),
            patch(
                "template_agent.src.core.mcp.MultiServerMCPClient",
                return_value=mock_client,
            ),
        ):
            tools = await get_mcp_tools()

        assert len(tools) == 1
        assert tools[0].name == "tool1"

    @pytest.mark.asyncio
    async def test_deduplicates_tools(self, tmp_path):
        """Test that duplicate tool names are deduplicated (first wins)."""
        config_file = tmp_path / "mcp.json"
        config_file.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "server-a": {
                            "url": "http://a/mcp/",
                            "enabled": True,
                            "auth": False,
                            "ssl_verify": True,
                            "timeout": 5,
                        },
                        "server-b": {
                            "url": "http://b/mcp/",
                            "enabled": True,
                            "auth": False,
                            "ssl_verify": True,
                            "timeout": 5,
                        },
                    }
                }
            )
        )

        tool_a = MagicMock()
        tool_a.name = "shared_tool"
        tool_b = MagicMock()
        tool_b.name = "shared_tool"

        call_count = 0

        def make_client(servers):
            nonlocal call_count
            mock = MagicMock()
            if call_count == 0:
                mock.get_tools = AsyncMock(return_value=[tool_a])
            else:
                mock.get_tools = AsyncMock(return_value=[tool_b])
            call_count += 1
            return mock

        with (
            patch("template_agent.src.core.mcp._CONFIG_PATH", config_file),
            patch(
                "template_agent.src.core.mcp.MultiServerMCPClient",
                side_effect=make_client,
            ),
        ):
            tools = await get_mcp_tools()

        assert len(tools) == 1
        assert tools[0] is tool_a

    @pytest.mark.asyncio
    async def test_errors_in_development_mode(self, tmp_path):
        """Test errors return empty list in development mode."""
        config_file = tmp_path / "mcp.json"
        config_file.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "test_server": {
                            "url": "http://localhost:8000/mcp/",
                            "enabled": True,
                            "auth": False,
                            "ssl_verify": True,
                            "timeout": 1,
                        }
                    }
                }
            )
        )

        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(
            side_effect=ConnectionError("Connection refused")
        )

        with (
            patch("template_agent.src.core.mcp._CONFIG_PATH", config_file),
            patch("template_agent.src.core.mcp.settings") as mock_settings,
            patch(
                "template_agent.src.core.mcp.MultiServerMCPClient",
                return_value=mock_client,
            ),
        ):
            mock_settings.USE_INMEMORY_SAVER = True
            tools = await get_mcp_tools()

        assert tools == []

    @pytest.mark.asyncio
    async def test_errors_in_production_mode(self, tmp_path):
        """Test errors raise AppException in production mode."""
        config_file = tmp_path / "mcp.json"
        config_file.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "prod_server": {
                            "url": "http://api.example.com/mcp/",
                            "enabled": True,
                            "auth": False,
                            "ssl_verify": True,
                            "timeout": 1,
                        }
                    }
                }
            )
        )

        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(
            side_effect=ConnectionError("Connection refused")
        )

        with (
            patch("template_agent.src.core.mcp._CONFIG_PATH", config_file),
            patch("template_agent.src.core.mcp.settings") as mock_settings,
            patch(
                "template_agent.src.core.mcp.MultiServerMCPClient",
                return_value=mock_client,
            ),
        ):
            mock_settings.USE_INMEMORY_SAVER = False
            with pytest.raises(AppException) as exc_info:
                await get_mcp_tools()
            assert exc_info.value.code == "E_007"

    @pytest.mark.asyncio
    async def test_no_enabled_servers_dev_mode(self, tmp_path):
        """Test that no enabled servers returns empty list in dev mode."""
        config_file = tmp_path / "mcp.json"
        config_file.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "disabled": {
                            "url": "http://localhost/mcp/",
                            "enabled": False,
                        }
                    }
                }
            )
        )

        with (
            patch("template_agent.src.core.mcp._CONFIG_PATH", config_file),
            patch("template_agent.src.core.mcp.settings") as mock_settings,
        ):
            mock_settings.USE_INMEMORY_SAVER = True
            tools = await get_mcp_tools()

        assert tools == []
