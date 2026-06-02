"""Unit tests for aegra graph factory."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_runtime_mock = MagicMock()
if "langgraph_sdk.runtime" not in sys.modules:
    sys.modules["langgraph_sdk.runtime"] = _runtime_mock


def _reset_graph_state() -> None:
    from deep_agent.aegra import graph

    graph._graph_cache.clear()
    graph._graph_cache_ts.clear()


class TestAgentFactory:
    """Tests for the agent() graph factory function.

    The ``agent()`` function uses lazy imports inside its body, so
    patches must target the actual module where each symbol lives.
    """

    @pytest.mark.asyncio
    async def test_builds_agent_without_user(self):
        mock_compiled = MagicMock()
        mock_config = MagicMock()
        mock_config.get_orchestrator_config.return_value = {
            "name": "orchestrator",
            "model": "gemini-2.5-flash",
            "body": "test prompt",
            "skill_paths": [],
            "tools": [],
        }
        mock_config.resolve_tools.return_value = []
        mock_config.resolve_agent_middleware.return_value = MagicMock(
            skills_enabled=True
        )

        mock_runtime = MagicMock()
        mock_runtime.user = None

        _reset_graph_state()

        with (
            patch(
                "deep_agent.src.agent.config.agent_config",
                mock_config,
            ),
            patch(
                "deep_agent.src.infrastructure.providers.register_profiles_from_config",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.providers.resolve_model_from_config",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.aegra.mcp.get_mcp_tools",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.load_subagents",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.backend.get_configured_backend",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.infrastructure.async_tasks.build_async_middleware",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.middleware.build_middleware_list",
                return_value=[],
            ),
            patch(
                "deep_agent.src.infrastructure.middleware.resolve_memory_param",
                return_value=None,
            ),
            patch("deep_agent.aegra.graph._ensure_startup", new_callable=AsyncMock),
            patch("deepagents.create_deep_agent", return_value=mock_compiled),
        ):
            from deep_agent.aegra.graph import agent

            result = await agent(mock_runtime)
            assert result is mock_compiled

    @pytest.mark.asyncio
    async def test_builds_agent_with_sso_token(self):
        mock_compiled = MagicMock()
        mock_config = MagicMock()
        mock_config.get_orchestrator_config.return_value = {
            "name": "orchestrator",
            "model": "gemini-2.5-flash",
            "body": "test prompt",
            "skill_paths": [],
            "tools": [],
        }
        mock_config.resolve_tools.return_value = []
        mock_config.resolve_agent_middleware.return_value = MagicMock(
            skills_enabled=True
        )

        mock_user = MagicMock()
        mock_user.access_token = "test_access_token"
        mock_user.refresh_token = "test_refresh_token"
        mock_user.identity = None

        mock_runtime = MagicMock()
        mock_runtime.user = mock_user

        _reset_graph_state()

        with (
            patch(
                "deep_agent.src.agent.config.agent_config",
                mock_config,
            ),
            patch(
                "deep_agent.src.infrastructure.providers.register_profiles_from_config",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.providers.resolve_model_from_config",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.aegra.mcp.get_mcp_tools",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_agent.aegra.mcp.refresh_access_token",
                new_callable=AsyncMock,
                return_value="refreshed_token",
            ) as mock_refresh,
            patch(
                "deep_agent.src.infrastructure.subagents.load_subagents",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.backend.get_configured_backend",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.infrastructure.async_tasks.build_async_middleware",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.middleware.build_middleware_list",
                return_value=[],
            ),
            patch(
                "deep_agent.src.infrastructure.middleware.resolve_memory_param",
                return_value=None,
            ),
            patch("deep_agent.aegra.graph._ensure_startup", new_callable=AsyncMock),
            patch("deepagents.create_deep_agent", return_value=mock_compiled),
        ):
            from deep_agent.aegra.graph import agent

            result = await agent(mock_runtime)
            assert result is mock_compiled
            mock_refresh.assert_awaited_once_with(
                "test_access_token", "test_refresh_token"
            )

    @pytest.mark.asyncio
    async def test_exposes_all_mcp_tools_when_mcps_declared_without_tool_list(self):
        mock_compiled = MagicMock()
        mock_config = MagicMock()
        mock_config.get_orchestrator_config.return_value = {
            "name": "orchestrator",
            "model": "gemini-2.5-flash",
            "body": "test prompt",
            "skill_paths": [],
            "tools": [],
            "mcps": ["dataverse-mcp-prod1"],
        }
        mock_config.resolve_tools.return_value = []
        mock_config.resolve_agent_middleware.return_value = MagicMock(
            skills_enabled=True
        )

        mock_tool = MagicMock()
        mock_tool.name = "identify_dataproducts"

        mock_runtime = MagicMock()
        mock_runtime.user = None

        _reset_graph_state()

        with (
            patch(
                "deep_agent.src.agent.config.agent_config",
                mock_config,
            ),
            patch(
                "deep_agent.src.infrastructure.providers.register_profiles_from_config",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.providers.resolve_model_from_config",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.aegra.mcp.get_mcp_tools",
                new_callable=AsyncMock,
                return_value=[mock_tool],
            ) as mock_get_mcp,
            patch(
                "deep_agent.src.infrastructure.subagents.load_subagents",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.backend.get_configured_backend",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.infrastructure.async_tasks.build_async_middleware",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.middleware.build_middleware_list",
                return_value=[],
            ),
            patch(
                "deep_agent.src.infrastructure.middleware.resolve_memory_param",
                return_value=None,
            ),
            patch("deep_agent.aegra.graph._ensure_startup", new_callable=AsyncMock),
            patch("deepagents.create_deep_agent", return_value=mock_compiled) as mock_create,
        ):
            from deep_agent.aegra.graph import agent

            result = await agent(mock_runtime)

        assert result is mock_compiled
        assert mock_create.call_args.kwargs["tools"] == [mock_tool]
        mock_get_mcp.assert_awaited_once_with(
            sso_token=None, server_names=["dataverse-mcp-prod1"]
        )
