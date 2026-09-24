"""Unit tests for MCP client utilities."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_agent.aegra.mcp import (
    _build_server_config,
    _connect_single_server,
    _create_auth_placeholder_tool,
    _get_server_configs,
    get_mcp_tools,
    mcp_httpx_verify,
    oauth_dcr_server_for_tool_name,
    placeholder_tool_name,
    record_oauth_live_names,
    resolve_declared_mcp_tools,
    rewrite_oauth_dcr_tool_names,
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


class TestMcpHttpxVerify:
    """Tests for mcp_httpx_verify helper."""

    def test_defaults_to_true(self):
        assert mcp_httpx_verify({}) is True

    def test_respects_ssl_verify_false(self):
        assert mcp_httpx_verify({"ssl_verify": False}) is False

    def test_respects_ssl_verify_true(self):
        assert mcp_httpx_verify({"ssl_verify": True}) is True


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
        config = _build_server_config(entry, None)

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
        config = _build_server_config(entry, "test_token_123")

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
        config = _build_server_config(entry, None)

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
        config = _build_server_config(entry, "should_be_ignored")

        assert config["headers"] == {}

    def test_config_defaults(self):
        """Test that missing optional fields use sensible defaults."""
        entry = {"url": "http://localhost:8000/mcp/"}
        config = _build_server_config(entry, "tok")

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
        mock_tool.metadata = None

        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=[mock_tool])

        config = {"url": "http://localhost:8000/mcp/", "transport": "http"}

        with patch(
            "deep_agent.aegra.mcp.MultiServerMCPClient",
            return_value=mock_client,
        ):
            tools = await _connect_single_server(
                "test_server",
                config,
                {},
                timeout=5,
                mcp_server="test_server",
            )

            assert len(tools) == 1
            assert tools[0].name == "test_tool"
            assert tools[0].metadata["mcp_server"] == "test_server"

    @pytest.mark.asyncio
    async def test_filters_app_only_tools_from_model_list(self):
        """App-only tools are annotated but not returned for the LLM."""
        from types import SimpleNamespace

        model_tool = SimpleNamespace(
            name="show_chart",
            metadata={
                "_meta": {
                    "ui": {
                        "resourceUri": "ui://charts/app.html",
                        "visibility": ["model", "app"],
                    }
                }
            },
        )
        app_only = SimpleNamespace(
            name="refresh_chart",
            metadata={"_meta": {"ui": {"visibility": ["app"]}}},
        )

        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=[model_tool, app_only])

        config = {"url": "http://localhost:8000/mcp/", "transport": "http"}

        with patch(
            "deep_agent.aegra.mcp.MultiServerMCPClient",
            return_value=mock_client,
        ):
            tools = await _connect_single_server(
                "charts",
                config,
                {},
                timeout=5,
                mcp_server="chart-mcp-server",
            )

        assert [t.name for t in tools] == ["show_chart"]
        assert tools[0].metadata["mcp_server"] == "chart-mcp-server"
        assert app_only.metadata["mcp_server"] == "chart-mcp-server"

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
            tools = await _connect_single_server("slow_server", config, {}, timeout=1)

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
            tools = await _connect_single_server("broken_server", config, {}, timeout=5)

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
            tools = await _connect_single_server("faulty_server", config, {}, timeout=5)

            assert tools == []

    @pytest.mark.asyncio
    async def test_needs_authorization_returns_placeholder(self):
        """NeedsAuthorization during connect returns an auth placeholder tool."""
        from deep_agent.aegra.mcp_auth import NeedsAuthorization

        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(
            side_effect=NeedsAuthorization(
                "google-workspace", "/mcp/google-workspace/connect"
            ),
        )
        server_cfg = {
            "auth_mode": "oauth",
            "description": "Google",
            "tool_prefix": "google",
        }

        with patch(
            "deep_agent.aegra.mcp.MultiServerMCPClient",
            return_value=mock_client,
        ):
            tools = await _connect_single_server(
                "google-workspace",
                {"url": "http://g/mcp/"},
                server_cfg,
                timeout=5,
            )

        assert len(tools) == 1
        assert tools[0].name == "mcp__google_workspace"

    @pytest.mark.asyncio
    async def test_auth_error_returns_placeholder_for_oauth(self):
        """HTTP 401 during connect returns auth placeholder for oauth/dcr servers."""
        exc = Exception("401 Unauthorized")
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(side_effect=exc)
        server_cfg = {"auth_mode": "dcr", "description": "Jira", "tool_prefix": "jira"}

        with patch(
            "deep_agent.aegra.mcp.MultiServerMCPClient",
            return_value=mock_client,
        ):
            tools = await _connect_single_server(
                "jira-mcp-prod",
                {"url": "http://j/mcp/"},
                server_cfg,
                timeout=5,
            )

        assert len(tools) == 1
        assert tools[0].name == "mcp__jira_mcp_prod"

    @pytest.mark.asyncio
    async def test_listing_401_drops_oauth_token(self):
        class Http401(Exception):
            def __init__(self) -> None:
                super().__init__("unauthorized")
                self.response = MagicMock(status_code=401)

        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(side_effect=Http401())
        server_cfg = {"auth_mode": "dcr", "description": "Jira"}

        with (
            patch(
                "deep_agent.aegra.mcp.MultiServerMCPClient",
                return_value=mock_client,
            ),
            patch(
                "deep_agent.aegra.mcp_tool_auth._forget_oauth_session",
                new=AsyncMock(),
            ) as mock_forget,
        ):
            tools = await _connect_single_server(
                "jira-mcp",
                {"url": "http://j/mcp/"},
                server_cfg,
                timeout=5,
                server_key="jira-mcp",
            )

        assert len(tools) == 1
        assert tools[0].name == "mcp__jira_mcp"
        mock_forget.assert_awaited_once_with("jira-mcp")

    @pytest.mark.asyncio
    async def test_listing_403_keeps_oauth_token(self):
        class Http403(Exception):
            def __init__(self) -> None:
                super().__init__("forbidden")
                self.response = MagicMock(status_code=403)

        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(side_effect=Http403())
        server_cfg = {"auth_mode": "dcr", "description": "Jira"}

        with (
            patch(
                "deep_agent.aegra.mcp.MultiServerMCPClient",
                return_value=mock_client,
            ),
            patch(
                "deep_agent.aegra.mcp_tool_auth._forget_oauth_session",
                new=AsyncMock(),
            ) as mock_forget,
        ):
            tools = await _connect_single_server(
                "jira-mcp",
                {"url": "http://j/mcp/"},
                server_cfg,
                timeout=5,
                server_key="jira-mcp",
            )

        assert len(tools) == 1
        assert tools[0].name == "mcp__jira_mcp"
        mock_forget.assert_not_called()


def _reset_mcp_cache() -> None:
    """Clear MCP tool cache between tests."""
    from deep_agent.aegra import mcp

    mcp._cached_tools.clear()
    mcp._cached_tools_ts.clear()


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

        with (
            patch("deep_agent.aegra.mcp._get_server_configs") as mock_get_configs,
            patch("deep_agent.aegra.mcp._connect_single_server") as mock_connect,
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
        tool_a2 = MagicMock()
        tool_a2.name = "unique_a"

        tool_b1 = MagicMock()
        tool_b1.name = "shared_tool"
        tool_b2 = MagicMock()
        tool_b2.name = "unique_b"

        with (
            patch("deep_agent.aegra.mcp._get_server_configs") as mock_get_configs,
            patch("deep_agent.aegra.mcp._connect_single_server") as mock_connect,
        ):
            mock_get_configs.return_value = mock_servers
            mock_connect.side_effect = [[tool_a1, tool_a2], [tool_b1, tool_b2]]

            tools = await get_mcp_tools()

            # Should have 3 tools: shared_tool (from server-a), unique_a, unique_b
            assert len(tools) == 3
            tool_names = {t.name for t in tools}
            assert tool_names == {"shared_tool", "unique_a", "unique_b"}
            # First occurrence of shared_tool wins
            assert tools[0] is tool_a1

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

        with patch("deep_agent.aegra.mcp._get_server_configs") as mock_get_configs:
            mock_get_configs.return_value = mock_servers

            tools = await get_mcp_tools()

            assert tools == []

    @pytest.mark.asyncio
    async def test_no_servers_configured_returns_empty_list(self):
        """Test that no MCP servers configured returns empty list."""
        _reset_mcp_cache()
        with patch("deep_agent.aegra.mcp._get_server_configs") as mock_get_configs:
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
            patch("deep_agent.aegra.mcp._get_server_configs") as mock_get_configs,
            patch("deep_agent.aegra.mcp._connect_single_server") as mock_connect,
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

        with (
            patch("deep_agent.aegra.mcp._get_server_configs") as mock_get_configs,
            patch("deep_agent.aegra.mcp._build_server_config") as mock_build_config,
            patch("deep_agent.aegra.mcp._connect_single_server") as mock_connect,
        ):
            mock_get_configs.return_value = mock_servers
            mock_build_config.return_value = {"url": "http://localhost/mcp/"}
            mock_connect.return_value = [mock_tool]

            await get_mcp_tools("test_token_123")

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
        tool2 = MagicMock()
        tool2.name = "tool2"
        tool3 = MagicMock()
        tool3.name = "tool3"

        with (
            patch("deep_agent.aegra.mcp._get_server_configs") as mock_get_configs,
            patch("deep_agent.aegra.mcp._connect_single_server") as mock_connect,
        ):
            mock_get_configs.return_value = mock_servers
            mock_connect.side_effect = [[tool1], [tool2], [tool3]]

            tools = await get_mcp_tools()

            # All three servers should be connected
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

        with (
            patch("deep_agent.aegra.mcp._get_server_configs") as mock_get_configs,
            patch("deep_agent.aegra.mcp._connect_single_server") as mock_connect,
        ):
            mock_get_configs.return_value = mock_servers
            mock_connect.return_value = [tool_w]

            tools = await get_mcp_tools(server_names=["wanted"])

            mock_connect.assert_called_once()
            assert len(tools) == 1
            assert tools[0].name == "wanted_tool"

    @pytest.mark.asyncio
    async def test_tool_prefix_as_connection_name(self):
        """Test that tool_prefix is derived from server_cfg inside _connect_single_server."""
        _reset_mcp_cache()
        mock_servers = {
            "jira-mcp-prod": {
                "url": "http://jira:9090/mcp",
                "enabled": True,
                "auth": False,
                "timeout": 5,
                "tool_prefix": "jira",
            }
        }

        mock_tool = MagicMock()
        mock_tool.name = "jira_search_issues"
        mock_tool.description = "Search for JIRA issues"
        mock_tool.parameters = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
            },
            "required": ["query"],
        }

        with (
            patch("deep_agent.aegra.mcp._get_server_configs") as mock_get_configs,
            patch("deep_agent.aegra.mcp._connect_single_server") as mock_connect,
        ):
            mock_get_configs.return_value = mock_servers
            mock_connect.return_value = [mock_tool]

            tools = await get_mcp_tools()

            assert len(tools) == 1
            assert tools[0].name == "jira_search_issues"
            call_kwargs = mock_connect.call_args
            assert call_kwargs[1]["name"] == "jira"

    @pytest.mark.asyncio
    async def test_no_tool_prefix_uses_server_key(self):
        """Test that without tool_prefix, server key is used as name."""
        _reset_mcp_cache()
        mock_servers = {
            "gitlab-mcp": {
                "url": "http://gitlab:8080/mcp",
                "enabled": True,
                "auth": False,
                "timeout": 5,
            }
        }
        mock_tool = MagicMock()
        mock_tool.name = "create_issue"
        with (
            patch("deep_agent.aegra.mcp._get_server_configs") as mock_get_configs,
            patch("deep_agent.aegra.mcp._connect_single_server") as mock_connect,
        ):
            mock_get_configs.return_value = mock_servers
            mock_connect.return_value = [mock_tool]

            await get_mcp_tools()

            call_kwargs = mock_connect.call_args
            assert call_kwargs[1]["name"] == "gitlab-mcp"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("auth_mode", ["oauth", "dcr"])
    async def test_auth_placeholder_uses_server_key_not_prefix(self, auth_mode):
        """OAuth/DCR server with tool_prefix should use original server key for auth."""
        _reset_mcp_cache()
        mock_servers = {
            "jira-mcp-prod": {
                "url": "http://jira:9090/mcp",
                "enabled": True,
                "auth": True,
                "auth_mode": auth_mode,
                "timeout": 5,
                "tool_prefix": "jira",
            }
        }
        with (
            patch("deep_agent.aegra.mcp._get_server_configs") as mock_get_configs,
            patch("deep_agent.aegra.mcp._resolve_connection_token") as mock_resolve,
            patch(
                "deep_agent.aegra.mcp._create_auth_placeholder_tool"
            ) as mock_placeholder,
        ):
            mock_get_configs.return_value = mock_servers
            mock_resolve.return_value = None  # no token — triggers placeholder path
            mock_tool = MagicMock()
            mock_tool.name = "mcp__jira_mcp_prod"
            mock_placeholder.return_value = mock_tool
            await get_mcp_tools()
            mock_placeholder.assert_called_once_with(
                "jira-mcp-prod", mock_servers["jira-mcp-prod"]
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("auth_mode", ["oauth", "dcr"])
    async def test_oauth_dcr_always_placeholder_even_with_token(self, auth_mode):
        """Redis token must not bind live OAuth/DCR tools at compile time."""
        _reset_mcp_cache()
        mock_servers = {
            "jira-mcp": {
                "url": "http://jira:9090/mcp",
                "enabled": True,
                "auth": True,
                "auth_mode": auth_mode,
                "timeout": 5,
                "tool_prefix": "jira",
            }
        }
        with (
            patch("deep_agent.aegra.mcp._get_server_configs") as mock_get_configs,
            patch("deep_agent.aegra.mcp._resolve_connection_token") as mock_resolve,
            patch("deep_agent.aegra.mcp._connect_single_server") as mock_connect,
            patch(
                "deep_agent.aegra.mcp._create_auth_placeholder_tool"
            ) as mock_placeholder,
        ):
            mock_get_configs.return_value = mock_servers
            mock_resolve.return_value = "redis-bearer"
            mock_tool = MagicMock()
            mock_tool.name = "mcp__jira_mcp"
            mock_placeholder.return_value = mock_tool
            tools = await get_mcp_tools(user_id="user-1")
            mock_connect.assert_not_called()
            mock_resolve.assert_not_called()
            mock_placeholder.assert_called_once_with(
                "jira-mcp", mock_servers["jira-mcp"]
            )
            assert [t.name for t in tools] == ["mcp__jira_mcp"]

    @pytest.mark.asyncio
    async def test_oauth_placeholder_does_not_block_sso_connect(self):
        """SSO servers still connect; OAuth/DCR stay placeholders."""
        _reset_mcp_cache()
        mock_servers = {
            "sso-mcp": {
                "url": "http://sso/mcp",
                "enabled": True,
                "auth": True,
                "auth_mode": "sso",
                "timeout": 5,
                "tool_prefix": "sso",
            },
            "dcr-mcp": {
                "url": "http://dcr/mcp",
                "enabled": True,
                "auth": True,
                "auth_mode": "dcr",
                "timeout": 5,
                "tool_prefix": "dcr",
            },
        }
        sso_tool = MagicMock()
        sso_tool.name = "sso_ping"
        dcr_placeholder = MagicMock()
        dcr_placeholder.name = "mcp__dcr_mcp"
        with (
            patch("deep_agent.aegra.mcp._get_server_configs") as mock_get_configs,
            patch(
                "deep_agent.aegra.mcp._resolve_connection_token",
                new=AsyncMock(return_value="sso-tok"),
            ) as mock_resolve,
            patch(
                "deep_agent.aegra.mcp._connect_single_server",
                new=AsyncMock(return_value=[sso_tool]),
            ) as mock_connect,
            patch(
                "deep_agent.aegra.mcp._create_auth_placeholder_tool",
                return_value=dcr_placeholder,
            ),
        ):
            mock_get_configs.return_value = mock_servers
            tools = await get_mcp_tools(sso_token="sso-tok", user_id="user-1")
            mock_connect.assert_called_once()
            assert mock_connect.call_args.kwargs["server_key"] == "sso-mcp"
            mock_resolve.assert_awaited_once()
            names = {t.name for t in tools}
            assert names == {"sso_ping", "mcp__dcr_mcp"}

    @pytest.mark.asyncio
    async def test_cache_is_per_user(self):
        """Different users should not share cached MCP tools."""
        _reset_mcp_cache()
        mock_servers = {
            "srv": {
                "url": "http://srv/mcp/",
                "enabled": True,
                "auth": False,
                "timeout": 5,
            }
        }
        tool_a = MagicMock()
        tool_a.name = "tool_a"
        tool_b = MagicMock()
        tool_b.name = "tool_b"

        with (
            patch("deep_agent.aegra.mcp._get_server_configs") as mock_cfg,
            patch("deep_agent.aegra.mcp._connect_single_server") as mock_conn,
        ):
            mock_cfg.return_value = mock_servers

            mock_conn.return_value = [tool_a]
            result_a = await get_mcp_tools(user_id="user-a")
            assert [t.name for t in result_a] == ["tool_a"]

            mock_conn.return_value = [tool_b]
            result_b = await get_mcp_tools(user_id="user-b")
            assert [t.name for t in result_b] == ["tool_b"]

            result_a_cached = await get_mcp_tools(user_id="user-a")
            assert [t.name for t in result_a_cached] == ["tool_a"]

    @pytest.mark.asyncio
    async def test_invalidate_cache_per_user(self):
        """Invalidating one user's cache should not affect another's."""
        _reset_mcp_cache()
        mock_servers = {
            "srv": {
                "url": "http://srv/mcp/",
                "enabled": True,
                "auth": False,
                "timeout": 5,
            }
        }
        tool = MagicMock()
        tool.name = "shared_tool"

        with (
            patch("deep_agent.aegra.mcp._get_server_configs") as mock_cfg,
            patch("deep_agent.aegra.mcp._connect_single_server") as mock_conn,
        ):
            mock_cfg.return_value = mock_servers
            mock_conn.return_value = [tool]

            await get_mcp_tools(user_id="user-a")
            await get_mcp_tools(user_id="user-a", server_names=["srv"])
            await get_mcp_tools(user_id="user-b")

            from deep_agent.aegra import mcp

            assert "user-a:" in mcp._cached_tools
            assert "user-a:srv" in mcp._cached_tools
            assert "user-b:" in mcp._cached_tools
            assert "user-a:" in mcp._cached_tools_ts
            assert "user-a:srv" in mcp._cached_tools_ts
            assert "user-b:" in mcp._cached_tools_ts

            from deep_agent.aegra.mcp import invalidate_mcp_tool_cache

            invalidate_mcp_tool_cache(user_id="user-a")

            assert "user-a:" not in mcp._cached_tools
            assert "user-a:srv" not in mcp._cached_tools
            assert "user-a:" not in mcp._cached_tools_ts
            assert "user-a:srv" not in mcp._cached_tools_ts
            assert "user-b:" in mcp._cached_tools
            assert "user-b:" in mcp._cached_tools_ts


class TestRewriteOauthDcrToolNames:
    """Builders list live OAuth/DCR names; compile still uses placeholders."""

    _SERVERS = {
        "template-mcp-server": {
            "enabled": True,
            "auth_mode": "sso",
            "tool_prefix": "template",
        },
        "template-mcp-server-dcr-open": {
            "enabled": True,
            "auth_mode": "dcr",
            "tool_prefix": "dcr_open",
        },
        "template-mcp-server-dcr-gated": {
            "enabled": True,
            "auth_mode": "dcr",
            "tool_prefix": "dcr_gated",
        },
        "jira-mcp": {
            "enabled": True,
            "auth_mode": "oauth",
            "tool_prefix": "jira",
        },
    }

    def test_placeholder_tool_name_sanitizes_hyphens(self):
        assert placeholder_tool_name("jira-mcp-prod") == "mcp__jira_mcp_prod"

    def test_rewrites_live_names_and_keeps_sso(self):
        with patch(
            "deep_agent.aegra.mcp._get_server_configs",
            return_value=self._SERVERS,
        ):
            assert rewrite_oauth_dcr_tool_names(
                [
                    "template_validate_email",
                    "dcr_gated_read_secret",
                    "dcr_open_employee_profile",
                ]
            ) == [
                "template_validate_email",
                "mcp__template_mcp_server_dcr_gated",
                "mcp__template_mcp_server_dcr_open",
            ]

    def test_longest_prefix_wins(self):
        with patch(
            "deep_agent.aegra.mcp._get_server_configs",
            return_value=self._SERVERS,
        ):
            assert rewrite_oauth_dcr_tool_names(["dcr_gated_read_secret"]) == [
                "mcp__template_mcp_server_dcr_gated"
            ]

    def test_placeholder_passthrough_and_dedupe(self):
        with patch(
            "deep_agent.aegra.mcp._get_server_configs",
            return_value=self._SERVERS,
        ):
            assert rewrite_oauth_dcr_tool_names(
                [
                    "mcp__template_mcp_server_dcr_gated",
                    "dcr_gated_read_secret",
                    "dcr_gated_vault_status",
                ]
            ) == ["mcp__template_mcp_server_dcr_gated"]

    def test_oauth_live_name_and_unknown_passthrough(self):
        with patch(
            "deep_agent.aegra.mcp._get_server_configs",
            return_value=self._SERVERS,
        ):
            assert rewrite_oauth_dcr_tool_names(["jira_search", "missing_tool"]) == [
                "mcp__jira_mcp",
                "missing_tool",
            ]

    def test_server_for_placeholder_live_and_sso(self):
        with patch(
            "deep_agent.aegra.mcp._get_server_configs",
            return_value=self._SERVERS,
        ):
            assert oauth_dcr_server_for_tool_name("mcp__jira_mcp") == "jira-mcp"
            assert oauth_dcr_server_for_tool_name("jira_search") == "jira-mcp"
            assert (
                oauth_dcr_server_for_tool_name("dcr_open_employee_profile")
                == "template-mcp-server-dcr-open"
            )
            assert oauth_dcr_server_for_tool_name("template_validate_email") is None
            assert oauth_dcr_server_for_tool_name("missing_tool") is None
            assert oauth_dcr_server_for_tool_name("") is None

    def test_empty_and_non_dict_configs(self):
        assert rewrite_oauth_dcr_tool_names([]) == []
        with patch(
            "deep_agent.aegra.mcp._get_server_configs",
            return_value=MagicMock(),
        ):
            assert rewrite_oauth_dcr_tool_names(["jira_search"]) == ["jira_search"]

    def test_rewrite_then_resolve_binds_placeholder(self):
        from deep_agent.src.agent.config.resolver import resolve_tools

        email = MagicMock()
        email.name = "template_validate_email"
        placeholder = MagicMock()
        placeholder.name = "mcp__template_mcp_server_dcr_gated"
        with patch(
            "deep_agent.aegra.mcp._get_server_configs",
            return_value=self._SERVERS,
        ):
            names = rewrite_oauth_dcr_tool_names(
                ["template_validate_email", "dcr_gated_read_secret"]
            )
        resolved = resolve_tools(names, [email, placeholder], "gated-child")
        assert [t.name for t in resolved] == [
            "template_validate_email",
            "mcp__template_mcp_server_dcr_gated",
        ]


class TestOauthDcrLiveNameCatalog:
    """Live names map to servers from tools/list, without tool_prefix."""

    _NO_PREFIX = {
        "acme-jira": {"enabled": True, "auth_mode": "dcr"},
        "acme-vault": {"enabled": True, "auth_mode": "dcr"},
        "sso-mcp": {
            "enabled": True,
            "auth_mode": "sso",
            "tool_prefix": "template",
        },
    }

    def setup_method(self):
        from deep_agent.aegra import mcp as mcp_mod

        mcp_mod._oauth_live_name_index.clear()
        mcp_mod._oauth_live_names_hydrated_at = 0.0

    def teardown_method(self):
        from deep_agent.aegra import mcp as mcp_mod

        mcp_mod._oauth_live_name_index.clear()
        mcp_mod._oauth_live_names_hydrated_at = 0.0

    def test_catalog_maps_unprefixed_live_name(self):
        with (
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value=self._NO_PREFIX,
            ),
            patch("deep_agent.aegra.redis.cache_set_persistent", return_value=True),
        ):
            record_oauth_live_names("acme-jira", ["search", "create_issue"])
            assert oauth_dcr_server_for_tool_name("search") == "acme-jira"
            assert oauth_dcr_server_for_tool_name("create_issue") == "acme-jira"
            assert rewrite_oauth_dcr_tool_names(["search"]) == ["mcp__acme_jira"]

    def test_catalog_redis_fail_does_not_index(self):
        with (
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value=self._NO_PREFIX,
            ),
            patch("deep_agent.aegra.redis.cache_set_persistent", return_value=False),
        ):
            with pytest.raises(RuntimeError, match="live tool names"):
                record_oauth_live_names("acme-jira", ["search"])
            assert oauth_dcr_server_for_tool_name("search") is None

    def test_scope_excludes_out_of_fence_server(self):
        with (
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value=self._NO_PREFIX,
            ),
            patch("deep_agent.aegra.redis.cache_set_persistent", return_value=True),
        ):
            record_oauth_live_names("acme-jira", ["search"])
            assert (
                oauth_dcr_server_for_tool_name("search", scope=["acme-jira"])
                == "acme-jira"
            )
            assert (
                oauth_dcr_server_for_tool_name("search", scope=["acme-vault"]) is None
            )

    def test_ambiguous_catalog_first_wins(self):
        with (
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value=self._NO_PREFIX,
            ),
            patch("deep_agent.aegra.redis.cache_set_persistent", return_value=True),
        ):
            record_oauth_live_names("acme-jira", ["search"])
            record_oauth_live_names("acme-vault", ["search"])
            assert oauth_dcr_server_for_tool_name("search") == "acme-jira"
            assert (
                oauth_dcr_server_for_tool_name("search", scope=["acme-jira"])
                == "acme-jira"
            )
            assert (
                oauth_dcr_server_for_tool_name("search", scope=["acme-vault"])
                == "acme-vault"
            )

    def test_placeholder_ignores_catalog_and_prefix(self):
        with patch(
            "deep_agent.aegra.mcp._get_server_configs",
            return_value=self._NO_PREFIX,
        ):
            assert oauth_dcr_server_for_tool_name("mcp__acme_jira") == "acme-jira"
            assert (
                oauth_dcr_server_for_tool_name("mcp__acme_jira", scope=["acme-vault"])
                is None
            )

    def test_sso_name_is_not_mapped(self):
        with (
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value=self._NO_PREFIX,
            ),
            patch("deep_agent.aegra.redis.cache_set_persistent", return_value=True),
        ):
            record_oauth_live_names("acme-jira", ["search"])
            assert oauth_dcr_server_for_tool_name("template_validate_email") is None

    def test_unmapped_live_name_is_not_guessed_without_catalog(self):
        with patch(
            "deep_agent.aegra.mcp._get_server_configs",
            return_value=self._NO_PREFIX,
        ):
            assert oauth_dcr_server_for_tool_name("search") is None
            assert rewrite_oauth_dcr_tool_names(["search"]) == ["search"]

    def test_optional_tool_prefix_still_rewrites(self):
        servers = {
            "acme-jira": {
                "enabled": True,
                "auth_mode": "dcr",
                "tool_prefix": "jira",
            }
        }
        with patch(
            "deep_agent.aegra.mcp._get_server_configs",
            return_value=servers,
        ):
            assert oauth_dcr_server_for_tool_name("jira_search") == "acme-jira"
            assert rewrite_oauth_dcr_tool_names(["jira_search"]) == ["mcp__acme_jira"]

    def test_hydrates_catalog_from_redis_when_process_index_empty(self):
        from deep_agent.aegra import mcp as mcp_mod

        with (
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value=self._NO_PREFIX,
            ),
            patch(
                "deep_agent.aegra.redis.cache_get",
                side_effect=lambda key: (
                    json.dumps(["search"])
                    if key == "mcp_oauth_live_names:acme-jira"
                    else None
                ),
            ),
        ):
            mcp_mod._oauth_live_name_index.clear()
            mcp_mod._oauth_live_names_hydrated_at = 0.0
            assert oauth_dcr_server_for_tool_name("search") == "acme-jira"

    def test_repeated_lookup_reads_redis_once_until_ttl(self):
        from deep_agent.aegra import mcp as mcp_mod

        getter = MagicMock(
            side_effect=lambda key: (
                json.dumps(["search"])
                if key == "mcp_oauth_live_names:acme-jira"
                else None
            )
        )
        with (
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value=self._NO_PREFIX,
            ),
            patch("deep_agent.aegra.redis.cache_get", getter),
        ):
            mcp_mod._oauth_live_name_index.clear()
            mcp_mod._oauth_live_names_hydrated_at = 0.0
            assert oauth_dcr_server_for_tool_name("search") == "acme-jira"
            first = getter.call_count
            assert first >= 1
            assert oauth_dcr_server_for_tool_name("search") == "acme-jira"
            assert getter.call_count == first
            mcp_mod._oauth_live_names_hydrated_at -= mcp_mod._OAUTH_LIVE_TOOLS_TTL + 1
            assert oauth_dcr_server_for_tool_name("search") == "acme-jira"
            assert getter.call_count > first

    def test_process_hit_still_unions_redis_owners(self):
        from deep_agent.aegra import mcp as mcp_mod

        with (
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value=self._NO_PREFIX,
            ),
            patch("deep_agent.aegra.redis.cache_set_persistent", return_value=True),
            patch(
                "deep_agent.aegra.redis.cache_get",
                side_effect=lambda key: (
                    json.dumps(["search"])
                    if key
                    in (
                        "mcp_oauth_live_names:acme-jira",
                        "mcp_oauth_live_names:acme-vault",
                    )
                    else None
                ),
            ),
        ):
            record_oauth_live_names("acme-vault", ["search"])
            assert oauth_dcr_server_for_tool_name("search") == "acme-jira"


class TestResolveDeclaredMcpTools:
    """Compile-time tools: / mcps: fence and allowlist."""

    _SERVERS = {
        "sso-mcp": {"enabled": True, "auth_mode": "sso"},
        "dcr1": {"enabled": True, "auth_mode": "dcr"},
        "dcr2": {"enabled": True, "auth_mode": "dcr"},
    }

    @staticmethod
    def _tool(name: str, server: str) -> MagicMock:
        tool = MagicMock()
        tool.name = name
        tool.metadata = {"mcp_server": server}
        return tool

    def setup_method(self):
        from deep_agent.aegra import mcp as mcp_mod

        mcp_mod._oauth_live_name_index.clear()

    def teardown_method(self):
        from deep_agent.aegra import mcp as mcp_mod

        mcp_mod._oauth_live_name_index.clear()

    def test_both_empty_returns_no_mcp_tools(self):
        email = self._tool("validate_email", "sso-mcp")
        assert resolve_declared_mcp_tools([], [], [email], "orch") == []

    def test_mcps_only_keeps_compile_tools_on_fence(self):
        email = self._tool("validate_email", "sso-mcp")
        ph1 = self._tool("mcp__dcr1", "dcr1")
        ph2 = self._tool("mcp__dcr2", "dcr2")
        with patch(
            "deep_agent.aegra.mcp._get_server_configs",
            return_value=self._SERVERS,
        ):
            names = [
                t.name
                for t in resolve_declared_mcp_tools(
                    [], ["dcr1"], [email, ph1, ph2], "child"
                )
            ]
        assert names == ["mcp__dcr1"]

    def test_tools_and_mcps_drops_sso_outside_fence_adds_placeholder(self):
        email = self._tool("validate_email", "sso-mcp")
        ph1 = self._tool("mcp__dcr1", "dcr1")
        with patch(
            "deep_agent.aegra.mcp._get_server_configs",
            return_value=self._SERVERS,
        ):
            names = [
                t.name
                for t in resolve_declared_mcp_tools(
                    ["validate_email", "employee_profile"],
                    ["dcr1"],
                    [email, ph1],
                    "child",
                )
            ]
        assert names == ["mcp__dcr1"]
        assert "validate_email" not in names

    def test_tools_and_mcps_keeps_sso_when_listed(self):
        email = self._tool("validate_email", "sso-mcp")
        ph1 = self._tool("mcp__dcr1", "dcr1")
        with patch(
            "deep_agent.aegra.mcp._get_server_configs",
            return_value=self._SERVERS,
        ):
            names = [
                t.name
                for t in resolve_declared_mcp_tools(
                    ["validate_email", "employee_profile"],
                    ["sso-mcp", "dcr1"],
                    [email, ph1],
                    "child",
                )
            ]
        assert names == ["validate_email", "mcp__dcr1"]

    def test_dcr2_tool_does_not_bind_dcr2_when_fenced_to_dcr1(self):
        ph1 = self._tool("mcp__dcr1", "dcr1")
        ph2 = self._tool("mcp__dcr2", "dcr2")
        with (
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value=self._SERVERS,
            ),
            patch("deep_agent.aegra.redis.cache_set_persistent", return_value=True),
        ):
            record_oauth_live_names("dcr2", ["read_secret"])
            names = [
                t.name
                for t in resolve_declared_mcp_tools(
                    ["read_secret"],
                    ["dcr1"],
                    [ph1, ph2],
                    "child",
                )
            ]
        assert names == ["mcp__dcr1"]
        assert "mcp__dcr2" not in names

    def test_tools_only_rewrites_via_catalog_without_adding_other_placeholders(self):
        email = self._tool("validate_email", "sso-mcp")
        ph1 = self._tool("mcp__dcr1", "dcr1")
        ph2 = self._tool("mcp__dcr2", "dcr2")
        with (
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value=self._SERVERS,
            ),
            patch("deep_agent.aegra.redis.cache_set_persistent", return_value=True),
        ):
            record_oauth_live_names("dcr1", ["search"])
            names = [
                t.name
                for t in resolve_declared_mcp_tools(
                    ["validate_email", "search"],
                    [],
                    [email, ph1, ph2],
                    "orch",
                )
            ]
        assert names == ["validate_email", "mcp__dcr1"]
        assert "mcp__dcr2" not in names

    def test_fenced_placeholder_bound_when_live_name_is_unmapped(self):
        ph1 = self._tool("mcp__dcr1", "dcr1")
        with patch(
            "deep_agent.aegra.mcp._get_server_configs",
            return_value=self._SERVERS,
        ):
            names = [
                t.name
                for t in resolve_declared_mcp_tools(
                    ["search"],
                    ["dcr1"],
                    [ph1],
                    "orch",
                )
            ]
        assert names == ["mcp__dcr1"]


class TestCreateAuthPlaceholderTool:
    """Tests for _create_auth_placeholder_tool function."""

    def test_tool_name_uses_sanitized_mcp_name(self):
        tool = _create_auth_placeholder_tool("jira-mcp-prod")
        assert tool.name == "mcp__jira_mcp_prod"

    def test_description_uses_server_cfg_description(self):
        cfg = {"description": "JIRA tickets and Confluence pages"}
        tool = _create_auth_placeholder_tool("jira-mcp", cfg)
        assert "JIRA tickets and Confluence pages" in tool.description

    def test_description_falls_back_to_mcp_name(self):
        tool = _create_auth_placeholder_tool("jira-mcp")
        assert "jira-mcp services" in tool.description

    def test_description_falls_back_when_no_description_key(self):
        tool = _create_auth_placeholder_tool("jira-mcp", {"url": "http://x"})
        assert "jira-mcp services" in tool.description

    @pytest.mark.asyncio
    async def test_require_auth_raises_needs_authorization_no_user(self):
        """Calling the placeholder with no user_id raises NeedsAuthorization."""
        from deep_agent.aegra.mcp_auth import NeedsAuthorization

        tool = _create_auth_placeholder_tool(
            "jira-mcp", {"description": "JIRA services"}
        )
        with (
            patch("deep_agent.aegra.mcp._current_user_id") as mock_ctx,
            patch(
                "deep_agent.aegra.mcp_auth.McpCredentialResolver.connect_url",
                return_value="http://localhost/mcp/jira-mcp/connect",
            ),
        ):
            mock_ctx.get.return_value = None
            with pytest.raises(NeedsAuthorization) as exc_info:
                await tool.coroutine(query="list my tickets")
            assert exc_info.value.mcp_name == "jira-mcp"

    @pytest.mark.asyncio
    async def test_require_auth_returns_success_when_resolve_succeeds(self):
        """Calling the placeholder after a usable token is resolved returns success."""
        tool = _create_auth_placeholder_tool(
            "jira-mcp", {"description": "JIRA services"}
        )
        live_tool = MagicMock()
        live_tool.name = "jira_search"
        mock_resolver = MagicMock()
        mock_resolver.resolve = AsyncMock(return_value="fresh-access-token")
        with (
            patch("deep_agent.aegra.mcp._current_user_id") as mock_ctx,
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value={"jira-mcp": {"auth_mode": "dcr"}},
            ),
            patch(
                "deep_agent.aegra.mcp_auth.get_mcp_credential_resolver",
                return_value=mock_resolver,
            ),
            patch(
                "deep_agent.aegra.mcp.get_authenticated_oauth_mcp_tools",
                new=AsyncMock(return_value=[live_tool]),
            ),
        ):
            mock_ctx.get.return_value = "user-1"
            result = await tool.coroutine(query="list my tickets")
        assert "Successfully connected to jira-mcp" in result
        assert "original request" in result
        assert "jira_search" not in result
        assert "this same run" in result
        assert "repeat their request" not in result
        mock_resolver.resolve.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_require_auth_does_not_claim_success_when_listing_empty(self):
        """Token exists but tools/list produced nothing — do not say connected."""
        tool = _create_auth_placeholder_tool(
            "jira-mcp", {"description": "JIRA services"}
        )
        mock_resolver = MagicMock()
        mock_resolver.resolve = AsyncMock(return_value="fresh-access-token")
        with (
            patch("deep_agent.aegra.mcp._current_user_id") as mock_ctx,
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value={"jira-mcp": {"auth_mode": "dcr"}},
            ),
            patch(
                "deep_agent.aegra.mcp_auth.get_mcp_credential_resolver",
                return_value=mock_resolver,
            ),
            patch(
                "deep_agent.aegra.mcp.get_authenticated_oauth_mcp_tools",
                new=AsyncMock(return_value=[]),
            ),
        ):
            mock_ctx.get.return_value = "user-1"
            result = await tool.coroutine(query="list my tickets")
        assert "Successfully connected" not in result
        assert "could not be loaded" in result
        mock_resolver.resolve.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_require_auth_raises_when_refresh_failed(self):
        """A leftover refresh token must not skip re-auth after refresh fails."""
        from deep_agent.aegra.mcp_auth import NeedsAuthorization

        tool = _create_auth_placeholder_tool(
            "jira-mcp", {"description": "JIRA services"}
        )
        mock_resolver = MagicMock()
        # has_valid_token is True whenever a refresh token remains in Redis —
        # that must not hide a failed refresh from the UI.
        mock_resolver.has_valid_token = AsyncMock(return_value=True)
        mock_resolver.resolve = AsyncMock(
            side_effect=NeedsAuthorization(
                "jira-mcp", "http://localhost/mcp/jira-mcp/connect"
            )
        )
        mock_resolver.connect_url.return_value = "http://localhost/mcp/jira-mcp/connect"
        with (
            patch("deep_agent.aegra.mcp._current_user_id") as mock_ctx,
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value={"jira-mcp": {"auth_mode": "oauth"}},
            ),
            patch(
                "deep_agent.aegra.mcp_auth.get_mcp_credential_resolver",
                return_value=mock_resolver,
            ),
        ):
            mock_ctx.get.return_value = "user-1"
            with pytest.raises(NeedsAuthorization) as exc_info:
                await tool.coroutine(query="list my tickets")
        assert exc_info.value.mcp_name == "jira-mcp"
        assert exc_info.value.connect_url == "http://localhost/mcp/jira-mcp/connect"
        mock_resolver.resolve.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_require_auth_raises_when_token_invalid(self):
        """Calling the placeholder with no usable token raises NeedsAuthorization."""
        from deep_agent.aegra.mcp_auth import NeedsAuthorization

        tool = _create_auth_placeholder_tool(
            "jira-mcp", {"description": "JIRA services"}
        )
        mock_resolver = MagicMock()
        mock_resolver.resolve = AsyncMock(
            side_effect=NeedsAuthorization(
                "jira-mcp", "http://localhost/mcp/jira-mcp/connect"
            )
        )
        mock_resolver.connect_url.return_value = "http://localhost/mcp/jira-mcp/connect"
        with (
            patch("deep_agent.aegra.mcp._current_user_id") as mock_ctx,
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value={"jira-mcp": {"auth_mode": "dcr"}},
            ),
            patch(
                "deep_agent.aegra.mcp_auth.get_mcp_credential_resolver",
                return_value=mock_resolver,
            ),
        ):
            mock_ctx.get.return_value = "user-1"
            with pytest.raises(NeedsAuthorization) as exc_info:
                await tool.coroutine(query="list my tickets")
        assert exc_info.value.connect_url == "http://localhost/mcp/jira-mcp/connect"

    @pytest.mark.asyncio
    async def test_require_auth_logs_warning_on_non_auth_exception(self, caplog):
        """Non-NeedsAuth exceptions during resolve fall through to NeedsAuthorization."""
        import logging

        from deep_agent.aegra.mcp_auth import NeedsAuthorization

        tool = _create_auth_placeholder_tool(
            "jira-mcp", {"description": "JIRA services"}
        )
        mock_resolver = MagicMock()
        mock_resolver.resolve = AsyncMock(side_effect=ConnectionError("network down"))
        mock_resolver.connect_url.return_value = "http://localhost/mcp/jira-mcp/connect"
        with (
            caplog.at_level(logging.WARNING),
            patch("deep_agent.aegra.mcp._current_user_id") as mock_ctx,
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value={"jira-mcp": {"auth_mode": "dcr"}},
            ),
            patch(
                "deep_agent.aegra.mcp_auth.get_mcp_credential_resolver",
                return_value=mock_resolver,
            ),
        ):
            mock_ctx.get.return_value = "user-1"
            with pytest.raises(NeedsAuthorization):
                await tool.coroutine(query="list my tickets")

        warning_records = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING
            and "placeholder tool auth resolve failed" in r.message
        ]
        assert len(warning_records) == 1
