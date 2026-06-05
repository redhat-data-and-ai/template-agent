"""Unit tests for MCP client utilities."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_agent.aegra.mcp import (
    _build_server_config,
    _connect_single_server,
    _get_server_configs,
    get_mcp_tools,
    set_mcp_auth_context,
)


class TestGetServerConfigs:
    """Tests for _get_server_configs function."""

    def test_returns_configs_from_agent_config(self):
        """Test that _get_server_configs delegates to agent_config."""
        mock_servers = {
            "server-a": {
                "url": "http://a:5001/mcp/",
                "transport": "streamable_http",
                "enabled": True,
                "auth": True,
                "ssl_verify": False,
                "timeout": 10,
            }
        }

        with patch(
            "deep_agent.aegra.mcp.agent_config.get_mcp_servers"
        ) as mock_get_servers:
            mock_get_servers.return_value = mock_servers

            result = _get_server_configs()

            assert result == mock_servers
            mock_get_servers.assert_called_once()

    def test_returns_empty_dict_when_no_servers(self):
        """Test returns empty dict when no MCP servers configured."""
        with patch(
            "deep_agent.aegra.mcp.agent_config.get_mcp_servers"
        ) as mock_get_servers:
            mock_get_servers.return_value = {}

            result = _get_server_configs()

            assert result == {}


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


class TestConnectSingleServer:
    """Tests for _connect_single_server function."""

    @pytest.mark.asyncio
    async def test_successful_connection(self):
        """Test successful connection to MCP server."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"

        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=[mock_tool])

        config = {"url": "http://localhost:8000/mcp/", "transport": "http"}

        with patch(
            "deep_agent.aegra.mcp.MultiServerMCPClient",
            return_value=mock_client,
        ):
            tools = await _connect_single_server("test_server", config, timeout=5)

            assert len(tools) == 1
            assert tools[0].name == "test_tool"

    @pytest.mark.asyncio
    async def test_connection_timeout_returns_empty_list(self):
        """Test that connection timeout returns empty list."""
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(
            side_effect=TimeoutError("Connection timed out")
        )

        config = {"url": "http://localhost:8000/mcp/", "transport": "http"}

        with patch(
            "deep_agent.aegra.mcp.MultiServerMCPClient",
            return_value=mock_client,
        ):
            tools = await _connect_single_server("slow_server", config, timeout=1)

            assert tools == []

    @pytest.mark.asyncio
    async def test_connection_error_returns_empty_list(self):
        """Test that connection errors return empty list with fault isolation."""
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(
            side_effect=ConnectionError("Connection refused")
        )

        config = {"url": "http://unreachable:8000/mcp/", "transport": "http"}

        with patch(
            "deep_agent.aegra.mcp.MultiServerMCPClient",
            return_value=mock_client,
        ):
            tools = await _connect_single_server("broken_server", config, timeout=5)

            assert tools == []

    @pytest.mark.asyncio
    async def test_generic_exception_returns_empty_list(self):
        """Test that any exception returns empty list for fault isolation."""
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(side_effect=ValueError("Unexpected error"))

        config = {"url": "http://localhost:8000/mcp/", "transport": "http"}

        with patch(
            "deep_agent.aegra.mcp.MultiServerMCPClient",
            return_value=mock_client,
        ):
            tools = await _connect_single_server("faulty_server", config, timeout=5)

            assert tools == []


def _reset_mcp_cache() -> None:
    """Clear MCP tool cache between tests."""
    from deep_agent.aegra import mcp

    mcp._cached_tools = []
    mcp._cached_tools_ts = 0.0


class TestGetMCPTools:
    """Tests for get_mcp_tools function."""

    @pytest.mark.asyncio
    async def test_successful_connection_with_tools(self):
        """Test successful MCP connection with tools."""
        _reset_mcp_cache()
        mock_servers = {
            "test_server": {
                "url": "http://localhost:8000/mcp/",
                "transport": "http",
                "enabled": True,
                "auth": False,
                "ssl_verify": True,
                "timeout": 5,
            }
        }

        mock_tool = MagicMock()
        mock_tool.name = "tool1"
        mock_tool.description = "A test tool"
        mock_tool.args_schema = None

        with (
            patch(
                "deep_agent.aegra.mcp._get_server_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.aegra.mcp._connect_single_server"
            ) as mock_connect,
        ):
            mock_get_configs.return_value = mock_servers
            mock_connect.return_value = [mock_tool]

            tools = await get_mcp_tools()

            assert len(tools) == 1
            assert tools[0].name == "tool1"
            mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_deduplicates_tools_from_multiple_servers(self):
        """Test that duplicate tool names are deduplicated (first wins)."""
        _reset_mcp_cache()
        mock_servers = {
            "server-a": {
                "url": "http://a/mcp/",
                "enabled": True,
                "auth": False,
                "timeout": 5,
            },
            "server-b": {
                "url": "http://b/mcp/",
                "enabled": True,
                "auth": False,
                "timeout": 5,
            },
        }

        tool_a1 = MagicMock()
        tool_a1.name = "shared_tool"
        tool_a1.description = "from server a"
        tool_a1.args_schema = None
        tool_a2 = MagicMock()
        tool_a2.name = "unique_a"
        tool_a2.description = "unique to a"
        tool_a2.args_schema = None

        tool_b1 = MagicMock()
        tool_b1.name = "shared_tool"
        tool_b1.description = "from server b"
        tool_b1.args_schema = None
        tool_b2 = MagicMock()
        tool_b2.name = "unique_b"
        tool_b2.description = "unique to b"
        tool_b2.args_schema = None

        with (
            patch(
                "deep_agent.aegra.mcp._get_server_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.aegra.mcp._connect_single_server"
            ) as mock_connect,
        ):
            mock_get_configs.return_value = mock_servers
            mock_connect.side_effect = [[tool_a1, tool_a2], [tool_b1, tool_b2]]

            tools = await get_mcp_tools()

            assert len(tools) == 3
            tool_names = {t.name for t in tools}
            assert tool_names == {"shared_tool", "unique_a", "unique_b"}
            assert tools[0].name == "shared_tool"
            assert tools[0].description == "from server a"

    @pytest.mark.asyncio
    async def test_no_enabled_servers_returns_empty_list(self):
        """Test that no enabled servers returns empty list."""
        _reset_mcp_cache()
        mock_servers = {
            "disabled": {
                "url": "http://localhost/mcp/",
                "enabled": False,
            }
        }

        with patch(
            "deep_agent.aegra.mcp._get_server_configs"
        ) as mock_get_configs:
            mock_get_configs.return_value = mock_servers

            tools = await get_mcp_tools()

            assert tools == []

    @pytest.mark.asyncio
    async def test_no_servers_configured_returns_empty_list(self):
        """Test that no MCP servers configured returns empty list."""
        _reset_mcp_cache()
        with patch(
            "deep_agent.aegra.mcp._get_server_configs"
        ) as mock_get_configs:
            mock_get_configs.return_value = {}

            tools = await get_mcp_tools()

            assert tools == []

    @pytest.mark.asyncio
    async def test_all_connections_fail_returns_empty_list(self):
        """Test that all connection failures return empty list gracefully."""
        _reset_mcp_cache()
        mock_servers = {
            "server-a": {
                "url": "http://a/mcp/",
                "enabled": True,
                "timeout": 1,
            },
            "server-b": {
                "url": "http://b/mcp/",
                "enabled": True,
                "timeout": 1,
            },
        }

        with (
            patch(
                "deep_agent.aegra.mcp._get_server_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.aegra.mcp._connect_single_server"
            ) as mock_connect,
        ):
            mock_get_configs.return_value = mock_servers
            mock_connect.return_value = []

            tools = await get_mcp_tools()

            assert tools == []

    @pytest.mark.asyncio
    async def test_sso_token_passed_to_build_config(self):
        """Test that SSO token is passed through to _build_server_config."""
        _reset_mcp_cache()
        mock_servers = {
            "test": {
                "url": "http://localhost/mcp/",
                "enabled": True,
                "auth": True,
                "timeout": 5,
            }
        }

        mock_tool = MagicMock()
        mock_tool.name = "tool1"
        mock_tool.description = "test"
        mock_tool.args_schema = None

        with (
            patch(
                "deep_agent.aegra.mcp._get_server_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.aegra.mcp._build_server_config"
            ) as mock_build_config,
            patch(
                "deep_agent.aegra.mcp._connect_single_server"
            ) as mock_connect,
        ):
            mock_get_configs.return_value = mock_servers
            mock_build_config.return_value = {"url": "http://localhost/mcp/"}
            mock_connect.return_value = [mock_tool]

            await get_mcp_tools(sso_token="test_token_123")

            # Verify _build_server_config was called with the token
            mock_build_config.assert_called_once()
            call_args = mock_build_config.call_args
            assert call_args[0][1] == "test_token_123"

    @pytest.mark.asyncio
    async def test_parallel_connection_to_multiple_servers(self):
        """Test that multiple servers are connected in parallel."""
        _reset_mcp_cache()
        mock_servers = {
            "server-1": {"url": "http://1/mcp/", "enabled": True, "timeout": 5},
            "server-2": {"url": "http://2/mcp/", "enabled": True, "timeout": 5},
            "server-3": {"url": "http://3/mcp/", "enabled": True, "timeout": 5},
        }

        tool1 = MagicMock()
        tool1.name = "tool1"
        tool1.description = "t1"
        tool1.args_schema = None
        tool2 = MagicMock()
        tool2.name = "tool2"
        tool2.description = "t2"
        tool2.args_schema = None
        tool3 = MagicMock()
        tool3.name = "tool3"
        tool3.description = "t3"
        tool3.args_schema = None

        with (
            patch(
                "deep_agent.aegra.mcp._get_server_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.aegra.mcp._connect_single_server"
            ) as mock_connect,
        ):
            mock_get_configs.return_value = mock_servers
            mock_connect.side_effect = [[tool1], [tool2], [tool3]]

            tools = await get_mcp_tools()

            assert mock_connect.call_count == 3
            assert len(tools) == 3

    @pytest.mark.asyncio
    async def test_server_names_filters_enabled_servers(self):
        """Test that server_names restricts which servers are connected."""
        _reset_mcp_cache()
        mock_servers = {
            "wanted": {"url": "http://w/mcp/", "enabled": True, "timeout": 5},
            "unwanted": {"url": "http://u/mcp/", "enabled": True, "timeout": 5},
        }

        tool_w = MagicMock()
        tool_w.name = "wanted_tool"
        tool_w.description = "wanted"
        tool_w.args_schema = None

        with (
            patch(
                "deep_agent.aegra.mcp._get_server_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.aegra.mcp._connect_single_server"
            ) as mock_connect,
        ):
            mock_get_configs.return_value = mock_servers
            mock_connect.return_value = [tool_w]

            tools = await get_mcp_tools(server_names=["wanted"])

            mock_connect.assert_called_once()
            assert len(tools) == 1
            assert tools[0].name == "wanted_tool"


class TestSetMcpAuthContext:
    """Tests for set_mcp_auth_context and context-var-based auth."""

    def test_sets_and_reads_access_token(self):
        """Test that set_mcp_auth_context populates access token context var."""
        from deep_agent.aegra.mcp import (
            _current_access_token,
            _current_refresh_token,
            set_mcp_auth_context,
        )

        set_mcp_auth_context("access_123", "refresh_456")

        assert _current_access_token.get() == "access_123"
        assert _current_refresh_token.get() == "refresh_456"

    def test_handles_none_tokens(self):
        """Test that None tokens are stored without error."""
        from deep_agent.aegra.mcp import (
            _current_access_token,
            _current_refresh_token,
            set_mcp_auth_context,
        )

        set_mcp_auth_context(None, None)

        assert _current_access_token.get() is None
        assert _current_refresh_token.get() is None


class TestMakeAuthRefreshingTool:
    """Tests for _make_auth_refreshing_tool wrapper."""

    def test_preserves_tool_name_and_description(self):
        """Test that wrapped tool preserves original name and description."""
        from deep_agent.aegra.mcp import _make_auth_refreshing_tool

        original = MagicMock()
        original.name = "web_search"
        original.description = "Search the web"
        original.args_schema = None

        server_entry = {"url": "http://mcp/", "auth": True, "timeout": 10}
        wrapped = _make_auth_refreshing_tool(original, "search-server", server_entry)

        assert wrapped.name == "web_search"
        assert wrapped.description == "Search the web"

    @pytest.mark.asyncio
    async def test_invocation_refreshes_token_and_reconnects(self):
        """Test that tool invocation triggers token refresh and reconnection."""
        from deep_agent.aegra.mcp import (
            _make_auth_refreshing_tool,
            set_mcp_auth_context,
        )

        original = MagicMock()
        original.name = "web_search"
        original.description = "Search the web"
        original.args_schema = None

        server_entry = {"url": "http://mcp/", "auth": True, "timeout": 10}
        wrapped = _make_auth_refreshing_tool(original, "search-server", server_entry)

        set_mcp_auth_context("stale_token", "refresh_tok")

        fresh_tool = MagicMock()
        fresh_tool.name = "web_search"
        fresh_tool.ainvoke = AsyncMock(return_value="search result")

        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=[fresh_tool])

        with (
            patch(
                "deep_agent.aegra.mcp.refresh_access_token",
                new_callable=AsyncMock,
                return_value="fresh_token",
            ) as mock_refresh,
            patch(
                "deep_agent.aegra.mcp.MultiServerMCPClient",
                return_value=mock_client,
            ) as mock_mcp_client,
        ):
            result = await wrapped.ainvoke({"query": "test"})

            mock_refresh.assert_called_once_with("stale_token", "refresh_tok")
            mock_mcp_client.assert_called_once()
            fresh_tool.ainvoke.assert_called_once_with({"query": "test"})
            assert result == "search result"
