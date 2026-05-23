"""Unit tests for subagent loading."""

from unittest.mock import MagicMock, patch

import pytest

from deep_agent.src.exceptions import SubAgentError
from deep_agent.src.infrastructure.subagents import (
    VALID_AGENT_TYPES,
    _build_async_subagent,
    _build_compiled_subagent,
    _build_default_subagent,
    load_subagents,
)


class TestLoadSubagents:
    """Tests for load_subagents function."""

    def test_load_subagents_returns_none_when_no_configs(self):
        """Test that load_subagents returns None when no subagent configs exist."""
        with patch(
            "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
        ) as mock_get_configs:
            mock_get_configs.return_value = {}

            result = load_subagents(tools=[])

            assert result is None

    def test_load_subagents_raises_error_when_model_missing(self):
        """Test that load_subagents raises ValueError when model is missing."""
        with patch(
            "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
        ) as mock_get_configs:
            mock_get_configs.return_value = {
                "analyst": {
                    "name": "analyst",
                    "description": "Test analyst",
                    "body": "Test prompt",
                    # Missing 'model' field
                }
            }

            with pytest.raises(SubAgentError, match="missing required 'model' field"):
                load_subagents(tools=[])

    def test_load_single_subagent_minimal(self):
        """Test loading a single subagent with minimal config."""
        mock_model = MagicMock()
        mock_subagent = MagicMock()

        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model"
            ) as mock_create_model,
            patch("deep_agent.src.infrastructure.subagents.SubAgent") as mock_sa,
        ):
            mock_get_configs.return_value = {
                "analyst": {
                    "name": "analyst",
                    "model": "gemini-2.5-flash",
                    "description": "Test analyst",
                    "body": "Test prompt",
                }
            }
            mock_create_model.return_value = mock_model
            mock_sa.return_value = mock_subagent

            result = load_subagents(tools=[])

            assert result == [mock_subagent]
            mock_create_model.assert_called_once_with(
                model_name="gemini-2.5-flash"
            )  # now get_or_create_model
            mock_sa.assert_called_once_with(
                name="analyst",
                model=mock_model,
                description="Test analyst",
                system_prompt="Test prompt",
            )

    def test_load_subagent_with_tools(self):
        """Test loading subagent with tools that get resolved."""
        mock_tool1 = MagicMock()
        mock_tool2 = MagicMock()
        mock_model = MagicMock()
        mock_subagent = MagicMock()

        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.resolve_tools"
            ) as mock_resolve_tools,
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model"
            ) as mock_create_model,
            patch("deep_agent.src.infrastructure.subagents.SubAgent") as mock_sa,
        ):
            mock_get_configs.return_value = {
                "analyst": {
                    "name": "analyst",
                    "model": "gemini-2.5-flash",
                    "description": "Analyst",
                    "body": "Prompt",
                    "tools": ["calculate_bmi", "search_web"],
                }
            }
            mock_resolve_tools.return_value = [mock_tool1, mock_tool2]
            mock_create_model.return_value = mock_model
            mock_sa.return_value = mock_subagent

            available_tools = [mock_tool1, mock_tool2]
            result = load_subagents(tools=available_tools)

            assert result == [mock_subagent]
            mock_resolve_tools.assert_called_once_with(
                ["calculate_bmi", "search_web"], available_tools, agent_name="analyst"
            )
            mock_sa.assert_called_once_with(
                name="analyst",
                model=mock_model,
                description="Analyst",
                system_prompt="Prompt",
                tools=[mock_tool1, mock_tool2],
            )

    def test_load_subagent_with_skills(self):
        """Test loading subagent with pre-resolved skill paths."""
        mock_model = MagicMock()
        mock_subagent = MagicMock()

        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model"
            ) as mock_create_model,
            patch("deep_agent.src.infrastructure.subagents.SubAgent") as mock_sa,
        ):
            mock_get_configs.return_value = {
                "analyst": {
                    "name": "analyst",
                    "model": "gemini-2.5-flash",
                    "description": "Analyst",
                    "body": "Prompt",
                    "skill_paths": ["/path/to/bmi-report"],
                }
            }
            mock_create_model.return_value = mock_model
            mock_sa.return_value = mock_subagent

            result = load_subagents(tools=[])

            assert result == [mock_subagent]
            mock_sa.assert_called_once_with(
                name="analyst",
                model=mock_model,
                description="Analyst",
                system_prompt="Prompt",
                skills=["/path/to/bmi-report"],
            )

    def test_load_multiple_subagents(self):
        """Test loading multiple subagents."""
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()
        mock_sa1 = MagicMock()
        mock_sa2 = MagicMock()

        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model"
            ) as mock_create_model,
            patch("deep_agent.src.infrastructure.subagents.SubAgent") as mock_sa,
        ):
            mock_get_configs.return_value = {
                "analyst": {
                    "name": "analyst",
                    "model": "gemini-2.5-flash",
                    "description": "Analyst",
                    "body": "Analyst prompt",
                },
                "publisher": {
                    "name": "publisher",
                    "model": "gemini-2.5-pro",
                    "description": "Publisher",
                    "body": "Publisher prompt",
                },
            }
            mock_create_model.side_effect = [mock_model1, mock_model2]
            mock_sa.side_effect = [mock_sa1, mock_sa2]

            result = load_subagents(tools=[])

            assert result == [mock_sa1, mock_sa2]
            assert mock_create_model.call_count == 2
            assert mock_sa.call_count == 2

    def test_load_subagent_with_empty_tool_list(self):
        """Test that subagent with empty tools list doesn't call resolve_tools."""
        mock_model = MagicMock()
        mock_subagent = MagicMock()

        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.resolve_tools"
            ) as mock_resolve_tools,
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model"
            ) as mock_create_model,
            patch("deep_agent.src.infrastructure.subagents.SubAgent") as mock_sa,
        ):
            mock_get_configs.return_value = {
                "analyst": {
                    "name": "analyst",
                    "model": "gemini-2.5-flash",
                    "description": "Analyst",
                    "body": "Prompt",
                    "tools": [],
                }
            }
            mock_create_model.return_value = mock_model
            mock_sa.return_value = mock_subagent

            result = load_subagents(tools=[])

            assert result == [mock_subagent]
            mock_resolve_tools.assert_not_called()
            # SubAgent should be called without tools parameter
            mock_sa.assert_called_once_with(
                name="analyst",
                model=mock_model,
                description="Analyst",
                system_prompt="Prompt",
            )

    def test_load_subagent_uses_empty_description_when_missing(self):
        """Test that missing description defaults to empty string."""
        mock_model = MagicMock()
        mock_subagent = MagicMock()

        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model"
            ) as mock_create_model,
            patch("deep_agent.src.infrastructure.subagents.SubAgent") as mock_sa,
        ):
            mock_get_configs.return_value = {
                "analyst": {
                    "name": "analyst",
                    "model": "gemini-2.5-flash",
                    "body": "Prompt",
                    # Missing 'description'
                }
            }
            mock_create_model.return_value = mock_model
            mock_sa.return_value = mock_subagent

            result = load_subagents(tools=[])

            assert result == [mock_subagent]
            mock_sa.assert_called_once_with(
                name="analyst",
                model=mock_model,
                description="",
                system_prompt="Prompt",
            )


class TestAgentTypeSystem:
    """Tests for the type field and multi-type subagent dispatch."""

    def test_valid_agent_types_constant(self):
        assert "default" in VALID_AGENT_TYPES
        assert "compiled" in VALID_AGENT_TYPES
        assert "async" in VALID_AGENT_TYPES

    def test_invalid_type_raises_value_error(self):
        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
        ):
            mock_get_configs.return_value = {
                "bad": {
                    "name": "bad",
                    "type": "invalid_type",
                    "model": "gemini-2.5-pro",
                    "description": "Bad agent",
                    "body": "Prompt",
                }
            }
            with pytest.raises(SubAgentError, match="invalid type 'invalid_type'"):
                load_subagents(tools=[])

    def test_missing_type_defaults_to_default(self):
        """No type field means SubAgent (default)."""
        mock_model = MagicMock()
        mock_subagent = MagicMock()

        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model"
            ) as mock_create_model,
            patch("deep_agent.src.infrastructure.subagents.SubAgent") as mock_sa,
        ):
            mock_get_configs.return_value = {
                "analyst": {
                    "name": "analyst",
                    "model": "gemini-2.5-flash",
                    "description": "Analyst",
                    "body": "Prompt",
                    # No 'type' field
                }
            }
            mock_create_model.return_value = mock_model
            mock_sa.return_value = mock_subagent

            result = load_subagents(tools=[])
            assert result == [mock_subagent]
            mock_sa.assert_called_once()

    def test_type_default_builds_subagent(self):
        mock_model = MagicMock()
        mock_subagent = MagicMock()

        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model"
            ) as mock_create_model,
            patch("deep_agent.src.infrastructure.subagents.SubAgent") as mock_sa,
        ):
            mock_get_configs.return_value = {
                "publisher": {
                    "name": "publisher",
                    "type": "default",
                    "model": "gemini-2.5-pro",
                    "description": "Publisher",
                    "body": "Prompt",
                }
            }
            mock_create_model.return_value = mock_model
            mock_sa.return_value = mock_subagent

            result = load_subagents(tools=[])
            assert result == [mock_subagent]

    def test_type_compiled_builds_compiled_subagent(self):
        mock_model = MagicMock()
        mock_graph = MagicMock()

        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model"
            ) as mock_create_model,
            patch(
                "deep_agent.src.infrastructure.subagents.create_deep_agent"
            ) as mock_create_agent,
            patch(
                "deep_agent.src.infrastructure.subagents.get_backend"
            ) as mock_get_backend,
            patch(
                "deep_agent.src.infrastructure.subagents.CompiledSubAgent"
            ) as mock_compiled,
        ):
            mock_get_configs.return_value = {
                "analyst": {
                    "name": "analyst",
                    "type": "compiled",
                    "model": "gemini-2.5-pro",
                    "description": "Fast analyst",
                    "body": "Prompt",
                }
            }
            mock_create_model.return_value = mock_model
            mock_create_agent.return_value = mock_graph
            mock_get_backend.return_value = MagicMock()
            mock_compiled.return_value = MagicMock()

            result = load_subagents(tools=[])
            assert len(result) == 1
            mock_create_agent.assert_called_once()
            mock_compiled.assert_called_once_with(
                name="analyst",
                description="Fast analyst",
                runnable=mock_graph,
            )

    def test_type_async_builds_async_subagent(self):
        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.src.infrastructure.subagents.AsyncSubAgent"
            ) as mock_async_sa,
        ):
            mock_get_configs.return_value = {
                "researcher": {
                    "name": "researcher",
                    "type": "async",
                    "description": "Remote researcher",
                    "body": "",
                    "graph_id": "researcher-graph",
                    "url": "http://research-agent:8000",
                }
            }
            mock_async_sa.return_value = MagicMock()

            result = load_subagents(tools=[])
            assert len(result) == 1
            mock_async_sa.assert_called_once_with(
                name="researcher",
                description="Remote researcher",
                graph_id="researcher-graph",
                url="http://research-agent:8000",
            )

    def test_type_async_raises_without_graph_id(self):
        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
        ):
            mock_get_configs.return_value = {
                "bad_async": {
                    "name": "bad_async",
                    "type": "async",
                    "description": "Missing graph_id",
                    "body": "",
                    # No graph_id
                }
            }
            with pytest.raises(SubAgentError, match="missing required 'graph_id'"):
                load_subagents(tools=[])
