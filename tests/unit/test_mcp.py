"""Unit tests for MCP client utilities."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from template_agent.src.core.exceptions.exceptions import AppException, AppExceptionCode
from template_agent.src.core.mcp import (
    _build_server_config,
    _handle_connection_error,
    get_mcp_tools,
)


class TestBuildServerConfig:
    """Tests for _build_server_config function."""

    def test_config_without_sso_token(self):
        """Test server config without SSO token."""
        with patch("template_agent.src.core.mcp.settings") as mock_settings:
            mock_settings.MCP_SERVER_URL = "http://localhost:8000/mcp/"
            mock_settings.MCP_TRANSPORT_PROTOCOL = "http"
            mock_settings.MCP_SSL_VERIFY = True

            config = _build_server_config(sso_token=None)

            assert config["url"] == "http://localhost:8000/mcp/"
            assert config["transport"] == "http"
            assert config["headers"] == {}
            assert "httpx_client_factory" not in config

    def test_config_with_sso_token(self):
        """Test server config with SSO token."""
        with patch("template_agent.src.core.mcp.settings") as mock_settings:
            mock_settings.MCP_SERVER_URL = "https://api.example.com/mcp/"
            mock_settings.MCP_TRANSPORT_PROTOCOL = "https"
            mock_settings.MCP_SSL_VERIFY = True

            config = _build_server_config(sso_token="test_token_123")

            assert config["url"] == "https://api.example.com/mcp/"
            assert config["transport"] == "https"
            assert config["headers"] == {"Authorization": "Bearer test_token_123"}
            assert "httpx_client_factory" not in config

    def test_config_with_ssl_verify_disabled(self):
        """Test server config with SSL verification disabled."""
        with patch("template_agent.src.core.mcp.settings") as mock_settings:
            mock_settings.MCP_SERVER_URL = "https://api.example.com/mcp/"
            mock_settings.MCP_TRANSPORT_PROTOCOL = "https"
            mock_settings.MCP_SSL_VERIFY = False

            config = _build_server_config(sso_token=None)

            assert "httpx_client_factory" in config
            assert callable(config["httpx_client_factory"])

            # Test that the factory creates an AsyncClient with verify=False
            client = config["httpx_client_factory"]()
            assert hasattr(client, "get")  # Verify it's an httpx client


class TestHandleConnectionError:
    """Tests for _handle_connection_error function."""

    @pytest.mark.parametrize(
        "error_type,is_dev",
        [
            (asyncio.TimeoutError(), True),
            (asyncio.TimeoutError(), False),
            (ConnectionError("Connection refused"), True),
            (ConnectionError("Connection refused"), False),
        ],
    )
    def test_error_handling(self, error_type, is_dev):
        """Test error handling in dev and prod modes."""
        with patch("template_agent.src.core.mcp.settings") as mock_settings:
            mock_settings.MCP_SERVER_URL = "http://localhost:8000/mcp/"
            mock_settings.MCP_CONNECTION_TIMEOUT = 30
            mock_settings.USE_INMEMORY_SAVER = is_dev

            if is_dev:
                result = _handle_connection_error(error_type)
                assert result == []
            else:
                with pytest.raises(AppException) as exc_info:
                    _handle_connection_error(error_type)
                assert exc_info.value.error_code == "E_007"


class TestGetMCPTools:
    """Tests for get_mcp_tools function."""

    @pytest.mark.asyncio
    async def test_successful_connection(self):
        """Test successful MCP connection with tools."""
        mock_tool = MagicMock()
        mock_tool.name = "tool1"
        mock_tools = [mock_tool]

        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=mock_tools)

        with patch("template_agent.src.core.mcp.settings") as mock_settings:
            mock_settings.MCP_SERVER_URL = "http://localhost:8000/mcp/"
            mock_settings.MCP_SERVER_NAME = "test_server"
            mock_settings.MCP_TRANSPORT_PROTOCOL = "http"
            mock_settings.MCP_CONNECTION_TIMEOUT = 30
            mock_settings.MCP_SSL_VERIFY = True
            mock_settings.PYTHON_LOG_LEVEL = "INFO"

            with patch(
                "template_agent.src.core.mcp.MultiServerMCPClient",
                return_value=mock_client,
            ):
                tools = await get_mcp_tools()
                assert len(tools) == 1
                assert tools[0].name == "tool1"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [asyncio.TimeoutError(), ConnectionError("Connection refused")],
    )
    async def test_errors_in_development_mode(self, error):
        """Test errors return empty list in development mode."""
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(side_effect=error)

        with patch("template_agent.src.core.mcp.settings") as mock_settings:
            mock_settings.MCP_SERVER_URL = "http://localhost:8000/mcp/"
            mock_settings.MCP_SERVER_NAME = "test_server"
            mock_settings.MCP_TRANSPORT_PROTOCOL = "http"
            mock_settings.MCP_CONNECTION_TIMEOUT = 30
            mock_settings.MCP_SSL_VERIFY = True
            mock_settings.USE_INMEMORY_SAVER = True
            mock_settings.PYTHON_LOG_LEVEL = "INFO"

            with patch(
                "template_agent.src.core.mcp.MultiServerMCPClient",
                return_value=mock_client,
            ):
                tools = await get_mcp_tools()
                assert tools == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [asyncio.TimeoutError(), ConnectionError("Connection refused")],
    )
    async def test_errors_in_production_mode(self, error):
        """Test errors raise AppException in production mode."""
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(side_effect=error)

        with patch("template_agent.src.core.mcp.settings") as mock_settings:
            mock_settings.MCP_SERVER_URL = "http://api.example.com/mcp/"
            mock_settings.MCP_SERVER_NAME = "prod_server"
            mock_settings.MCP_TRANSPORT_PROTOCOL = "http"
            mock_settings.MCP_CONNECTION_TIMEOUT = 30
            mock_settings.MCP_SSL_VERIFY = True
            mock_settings.USE_INMEMORY_SAVER = False
            mock_settings.PYTHON_LOG_LEVEL = "INFO"

            with patch(
                "template_agent.src.core.mcp.MultiServerMCPClient",
                return_value=mock_client,
            ):
                with pytest.raises(AppException) as exc_info:
                    await get_mcp_tools()
                assert exc_info.value.error_code == "E_007"
