"""Unit tests for subagent loading."""

from unittest.mock import MagicMock, patch

import pytest

from template_agent.src.infrastructure.subagents import load_subagents


class TestLoadSubagents:
    """Tests for load_subagents function."""

    def test_load_subagents_returns_none_when_no_configs(self):
        """Test that load_subagents returns None when no subagent configs exist."""
        with patch(
            "template_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
        ) as mock_get_configs:
            mock_get_configs.return_value = {}

            result = load_subagents(tools=[])

            assert result is None

    def test_load_subagents_raises_error_when_model_missing(self):
        """Test that load_subagents raises ValueError when model is missing."""
        with patch(
            "template_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
        ) as mock_get_configs:
            mock_get_configs.return_value = {
                "analyst": {
                    "name": "analyst",
                    "description": "Test analyst",
                    "body": "Test prompt",
                    # Missing 'model' field
                }
            }

            with pytest.raises(ValueError, match="missing required 'model' field"):
                load_subagents(tools=[])

    def test_load_single_subagent_minimal(self):
        """Test loading a single subagent with minimal config."""
        mock_model = MagicMock()
        mock_subagent = MagicMock()

        with (
            patch(
                "template_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "template_agent.src.infrastructure.subagents.create_model"
            ) as mock_create_model,
            patch("template_agent.src.infrastructure.subagents.SubAgent") as mock_sa,
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
            mock_create_model.assert_called_once_with(model_name="gemini-2.5-flash")
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
                "template_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "template_agent.src.infrastructure.subagents.agent_config.resolve_tools"
            ) as mock_resolve_tools,
            patch(
                "template_agent.src.infrastructure.subagents.create_model"
            ) as mock_create_model,
            patch("template_agent.src.infrastructure.subagents.SubAgent") as mock_sa,
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
                "template_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "template_agent.src.infrastructure.subagents.create_model"
            ) as mock_create_model,
            patch("template_agent.src.infrastructure.subagents.SubAgent") as mock_sa,
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
                "template_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "template_agent.src.infrastructure.subagents.create_model"
            ) as mock_create_model,
            patch("template_agent.src.infrastructure.subagents.SubAgent") as mock_sa,
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
                "template_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "template_agent.src.infrastructure.subagents.agent_config.resolve_tools"
            ) as mock_resolve_tools,
            patch(
                "template_agent.src.infrastructure.subagents.create_model"
            ) as mock_create_model,
            patch("template_agent.src.infrastructure.subagents.SubAgent") as mock_sa,
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
                "template_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "template_agent.src.infrastructure.subagents.create_model"
            ) as mock_create_model,
            patch("template_agent.src.infrastructure.subagents.SubAgent") as mock_sa,
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
