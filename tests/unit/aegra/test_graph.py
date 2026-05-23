"""Unit tests for aegra graph factory."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_runtime_mock = MagicMock()
if "langgraph_sdk.runtime" not in sys.modules:
    sys.modules["langgraph_sdk.runtime"] = _runtime_mock


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

        mock_runtime = MagicMock()
        mock_runtime.user = None

        with (
            patch(
                "deep_agent.src.agent.config.loader.agent_config",
                mock_config,
            ),
            patch(
                "deep_agent.src.agent.llm.create_model",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.infrastructure.mcp.get_mcp_tools",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.load_subagents",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.backend.get_backend",
                return_value=MagicMock(),
            ),
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

        mock_user = MagicMock()
        mock_user.access_token = "test_access_token"
        mock_user.refresh_token = "test_refresh_token"
        mock_user.identity = None

        mock_runtime = MagicMock()
        mock_runtime.user = mock_user

        with (
            patch(
                "deep_agent.src.agent.config.loader.agent_config",
                mock_config,
            ),
            patch(
                "deep_agent.src.agent.llm.create_model",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.infrastructure.mcp.get_mcp_tools",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_agent.src.infrastructure.mcp.refresh_access_token",
                new_callable=AsyncMock,
                return_value="refreshed_token",
            ) as mock_refresh,
            patch(
                "deep_agent.src.infrastructure.subagents.load_subagents",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.backend.get_backend",
                return_value=MagicMock(),
            ),
            patch("deepagents.create_deep_agent", return_value=mock_compiled),
        ):
            from deep_agent.aegra.graph import agent

            result = await agent(mock_runtime)
            assert result is mock_compiled
            mock_refresh.assert_awaited_once_with(
                "test_access_token", "test_refresh_token"
            )
