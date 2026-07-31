"""Unit tests for subagent loading."""

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from deep_agent.src.agent.config.model import ModelSpec, Provider
from deep_agent.src.exceptions import SubAgentError
from deep_agent.src.infrastructure.subagents import VALID_AGENT_TYPES, load_subagents


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
        """Test that load_subagents uses default model when none configured."""
        mock_model = MagicMock()

        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_orchestrator_config",
                return_value={},  # No orchestrator model either
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model_from_spec",
                return_value=mock_model,
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.SubAgent",
                return_value=MagicMock(),
            ),
        ):
            mock_get_configs.return_value = {
                "analyst": {
                    "name": "analyst",
                    "description": "Test analyst",
                    "body": "Test prompt",
                    # Missing 'model' field - will use default
                }
            }

            result = load_subagents(tools=[])
            assert result is not None  # Successfully creates with default model

    def test_load_single_subagent_minimal(self):
        """Test loading a single subagent with minimal config."""
        mock_model = MagicMock()
        mock_subagent = MagicMock()

        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_orchestrator_config",
                return_value={},  # No orchestrator config
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model_from_spec"
            ) as mock_create_model,
            patch("deep_agent.src.infrastructure.subagents.SubAgent") as mock_sa,
            patch(
                "deep_agent.src.infrastructure.subagents.build_audit_middleware",
                return_value=None,
            ),
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
            mock_create_model.assert_called_once()
            # Should be called without middleware when no fallback
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
                "deep_agent.src.infrastructure.subagents.agent_config.get_orchestrator_config",
                return_value={},  # No orchestrator config
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.resolve_tools"
            ) as mock_resolve_tools,
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model_from_spec"
            ) as mock_create_model,
            patch("deep_agent.src.infrastructure.subagents.SubAgent") as mock_sa,
            patch(
                "deep_agent.src.infrastructure.subagents.build_audit_middleware",
                return_value=None,
            ),
        ):
            mock_get_configs.return_value = {
                "analyst": {
                    "name": "analyst",
                    "model": "gemini-2.5-flash",
                    "description": "Analyst",
                    "body": "Prompt",
                    "allowed_tools": ["calculate_bmi", "search_web"],
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
                "deep_agent.src.infrastructure.subagents.agent_config.get_orchestrator_config",
                return_value={},  # No orchestrator config
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model_from_spec"
            ) as mock_create_model,
            patch("deep_agent.src.infrastructure.subagents.SubAgent") as mock_sa,
            patch(
                "deep_agent.src.infrastructure.subagents.build_audit_middleware",
                return_value=None,
            ),
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
                skills=["/skills/bmi-report"],
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
                "deep_agent.src.infrastructure.subagents.agent_config.get_orchestrator_config",
                return_value={},  # No orchestrator config
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model_from_spec"
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
                "deep_agent.src.infrastructure.subagents.agent_config.get_orchestrator_config",
                return_value={},  # No orchestrator config
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.resolve_tools"
            ) as mock_resolve_tools,
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model_from_spec"
            ) as mock_create_model,
            patch("deep_agent.src.infrastructure.subagents.SubAgent") as mock_sa,
            patch(
                "deep_agent.src.infrastructure.subagents.build_audit_middleware",
                return_value=None,
            ),
        ):
            mock_get_configs.return_value = {
                "analyst": {
                    "name": "analyst",
                    "model": "gemini-2.5-flash",
                    "description": "Analyst",
                    "body": "Prompt",
                    "allowed_tools": [],
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
                "deep_agent.src.infrastructure.subagents.agent_config.get_orchestrator_config",
                return_value={},  # No orchestrator config
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model_from_spec"
            ) as mock_create_model,
            patch("deep_agent.src.infrastructure.subagents.SubAgent") as mock_sa,
            patch(
                "deep_agent.src.infrastructure.subagents.build_audit_middleware",
                return_value=None,
            ),
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
                "deep_agent.src.infrastructure.subagents.get_or_create_model_from_spec"
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
                "deep_agent.src.infrastructure.subagents.get_or_create_model_from_spec"
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
                "deep_agent.src.infrastructure.subagents.get_or_create_model_from_spec"
            ) as mock_create_model,
            patch("deepagents.create_deep_agent") as mock_create_agent,
            patch(
                "deep_agent.src.infrastructure.backend.get_configured_backend"
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
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_orchestrator_config",
                return_value={},
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.AsyncSubAgent", None
            ),  # Simulate async support not available
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
            with pytest.raises(
                SubAgentError, match="requires deepagents with async support"
            ):
                load_subagents(tools=[])


class TestSubagentProviderConfig:
    """Tests for provider-aware model configuration."""

    def test_inherits_orchestrator_string_model(self):
        mock_model = MagicMock()

        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_orchestrator_config",
                return_value={"model": "gemini-2.5-flash"},
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model_from_spec",
                return_value=mock_model,
            ) as mock_from_spec,
            patch(
                "deep_agent.src.infrastructure.subagents.SubAgent",
                return_value=MagicMock(),
            ),
        ):
            mock_get_configs.return_value = {
                "analyst": {
                    "name": "analyst",
                    "description": "Analyst",
                    "body": "Prompt",
                }
            }

            load_subagents(tools=[])

            spec = mock_from_spec.call_args[0][0]
            assert spec.name == "gemini-2.5-flash"

    def test_orchestrator_as_fallback_when_subagent_has_string_model(self):
        """Subagent with string model and no fallback → orchestrator becomes fallback."""
        mock_model = MagicMock()

        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_orchestrator_config",
                return_value={"model": "gemini-2.5-flash"},
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model_from_spec",
                return_value=mock_model,
            ) as mock_from_spec,
            patch(
                "langchain.agents.middleware.ModelFallbackMiddleware"
            ) as mock_middleware,
            patch(
                "deep_agent.src.infrastructure.subagents.SubAgent",
                return_value=MagicMock(),
            ) as mock_sa,
        ):
            mock_get_configs.return_value = {
                "analyst": {
                    "name": "analyst",
                    "description": "Analyst",
                    "body": "Prompt",
                    "model": "gpt-4",  # String model, no fallback
                }
            }

            load_subagents(tools=[])

            # Verify middleware was created and passed to SubAgent
            assert mock_middleware.called
            call_kwargs = mock_sa.call_args[1]
            assert "middleware" in call_kwargs
            assert mock_middleware.return_value in call_kwargs["middleware"]

    def test_orchestrator_as_fallback_when_subagent_has_dict_model_no_fallback(self):
        """Subagent with dict model and no fallback → orchestrator becomes fallback."""
        mock_model = MagicMock()

        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_orchestrator_config",
                return_value={"model": "gemini-2.5-flash"},
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model_from_spec",
                return_value=mock_model,
            ) as mock_from_spec,
            patch(
                "langchain.agents.middleware.ModelFallbackMiddleware"
            ) as mock_middleware,
            patch(
                "deep_agent.src.infrastructure.subagents.SubAgent",
                return_value=MagicMock(),
            ) as mock_sa,
        ):
            mock_get_configs.return_value = {
                "analyst": {
                    "name": "analyst",
                    "description": "Analyst",
                    "body": "Prompt",
                    "model": {"provider": "openai", "name": "gpt-4"},
                }
            }

            load_subagents(tools=[])

            # Verify middleware was created and passed to SubAgent
            assert mock_middleware.called
            call_kwargs = mock_sa.call_args[1]
            assert "middleware" in call_kwargs
            assert mock_middleware.return_value in call_kwargs["middleware"]

    def test_keeps_explicit_fallback_when_provided(self):
        """Subagent with explicit fallback → keep as-is (don't override)."""
        mock_model = MagicMock()

        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_orchestrator_config",
                return_value={"model": "gemini-2.5-flash"},
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model_from_spec",
                return_value=mock_model,
            ) as mock_from_spec,
            patch(
                "langchain.agents.middleware.ModelFallbackMiddleware"
            ) as mock_middleware,
            patch(
                "deep_agent.src.infrastructure.subagents.SubAgent",
                return_value=MagicMock(),
            ) as mock_sa,
        ):
            mock_get_configs.return_value = {
                "analyst": {
                    "name": "analyst",
                    "description": "Analyst",
                    "body": "Prompt",
                    "model": {
                        "provider": "openai",
                        "name": "gpt-4",
                        "fallback": {"provider": "vertex", "name": "gemini-3.1-pro"},
                    },
                }
            }

            load_subagents(tools=[])

            # Verify middleware was created and passed to SubAgent
            assert mock_middleware.called
            call_kwargs = mock_sa.call_args[1]
            assert "middleware" in call_kwargs
            assert mock_middleware.return_value in call_kwargs["middleware"]

    def test_no_fallback_when_no_orchestrator_model(self):
        """Subagent with model but orchestrator has no model → no fallback added."""
        mock_model = MagicMock()

        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_orchestrator_config",
                return_value={},  # No orchestrator model
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model_from_spec",
                return_value=mock_model,
            ) as mock_from_spec,
            patch(
                "deep_agent.src.infrastructure.subagents.SubAgent",
                return_value=MagicMock(),
            ),
        ):
            mock_get_configs.return_value = {
                "analyst": {
                    "name": "analyst",
                    "description": "Analyst",
                    "body": "Prompt",
                    "model": "gpt-4",
                }
            }

            load_subagents(tools=[])

            spec = mock_from_spec.call_args[0][0]
            assert spec.name == "gpt-4"
            # No orchestrator model → no fallback
            assert spec.fallback is None

    def test_strips_nested_fallback_from_orchestrator(self):
        """Orchestrator with fallback → strip when using as subagent fallback."""
        mock_model = MagicMock()

        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_orchestrator_config",
                return_value={
                    "model": {
                        "provider": "vertex",
                        "name": "gemini-2.5-flash",
                        "fallback": {"provider": "openai", "name": "gpt-4o-mini"},
                    }
                },
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model_from_spec",
                return_value=mock_model,
            ) as mock_from_spec,
            patch(
                "langchain.agents.middleware.ModelFallbackMiddleware"
            ) as mock_middleware,
            patch(
                "deep_agent.src.infrastructure.subagents.SubAgent",
                return_value=MagicMock(),
            ) as mock_sa,
            patch(
                "deep_agent.src.infrastructure.subagents.build_audit_middleware",
                return_value=None,
            ),
        ):
            mock_get_configs.return_value = {
                "analyst": {
                    "name": "analyst",
                    "description": "Analyst",
                    "body": "Prompt",
                    "model": "gpt-4",
                }
            }

            load_subagents(tools=[])

            # Verify middleware was created and passed to SubAgent
            assert mock_middleware.called
            call_kwargs = mock_sa.call_args[1]
            assert "middleware" in call_kwargs
            assert len(call_kwargs["middleware"]) == 1
            assert mock_middleware.return_value in call_kwargs["middleware"]


class TestToolAccessControl:
    """Tests for allowed_tools, denied_tools, and tool_approval in subagent building."""

    def test_denied_tools_filtered_from_resolved(self):
        """Subagent with denied_tools has those tools removed."""
        mock_tool_a = MagicMock()
        mock_tool_a.name = "tool_a"
        mock_tool_b = MagicMock()
        mock_tool_b.name = "tool_b"
        mock_model = MagicMock()

        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_orchestrator_config",
                return_value={},
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.resolve_tools"
            ) as mock_resolve,
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model_from_spec",
                return_value=mock_model,
            ),
            patch("deep_agent.src.infrastructure.subagents.SubAgent") as mock_sa,
            patch(
                "deep_agent.src.infrastructure.subagents.filter_denied_tools"
            ) as mock_filter,
        ):
            mock_get_configs.return_value = {
                "agent1": {
                    "name": "agent1",
                    "model": "gemini-2.5-flash",
                    "description": "Test",
                    "body": "Prompt",
                    "allowed_tools": ["tool_a", "tool_b"],
                    "denied_tools": ["tool_b"],
                }
            }
            mock_resolve.return_value = [mock_tool_a, mock_tool_b]
            mock_filter.return_value = [mock_tool_a]
            mock_sa.return_value = MagicMock()

            load_subagents(tools=[mock_tool_a, mock_tool_b])

            mock_filter.assert_called_once_with(
                [mock_tool_a, mock_tool_b], ["tool_b"], agent_name="agent1"
            )
            mock_sa.assert_called_once()
            call_kwargs = mock_sa.call_args[1]
            assert call_kwargs["tools"] == [mock_tool_a]

    def test_default_subagent_rejects_tool_approval(self):
        """Default subagent with tool_approval raises error — must use compiled."""
        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_orchestrator_config",
                return_value={},
            ),
        ):
            mock_get_configs.return_value = {
                "agent1": {
                    "name": "agent1",
                    "type": "default",
                    "model": "gemini-2.5-flash",
                    "description": "Test",
                    "body": "Prompt",
                    "allowed_tools": ["sensitive_tool"],
                    "tool_approval": ["sensitive_tool"],
                }
            }
            with pytest.raises(SubAgentError, match="does not support tool_approval"):
                load_subagents(tools=[])

    def test_default_subagent_with_tool_approval_is_rejected_by_loader(self):
        """Default subagent with tool_approval is rejected by load_subagents."""
        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_orchestrator_config",
                return_value={},
            ),
        ):
            mock_get_configs.return_value = {
                "default_agent": {
                    "name": "default_agent",
                    "type": "default",
                    "model": "gemini-2.5-flash",
                    "description": "Test",
                    "body": "Prompt",
                    "allowed_tools": ["send_email"],
                    "tool_approval": ["send_email"],
                }
            }
            with pytest.raises(SubAgentError, match="does not support tool_approval"):
                load_subagents(tools=[])

    def test_async_subagent_rejects_tool_approval(self):
        """Async subagent with tool_approval raises error."""
        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_orchestrator_config",
                return_value={},
            ),
        ):
            mock_get_configs.return_value = {
                "remote": {
                    "name": "remote",
                    "type": "async",
                    "description": "Remote agent",
                    "body": "",
                    "graph_id": "remote-graph",
                    "tool_approval": ["some_tool"],
                }
            }
            with pytest.raises(SubAgentError, match="does not support tool_approval"):
                load_subagents(tools=[])

    def test_denied_tools_with_mcp_inheritance(self):
        """Subagent inheriting all MCP tools still filters denied ones."""
        mock_tool_a = MagicMock()
        mock_tool_a.name = "safe_tool"
        mock_tool_b = MagicMock()
        mock_tool_b.name = "dangerous_tool"
        mock_model = MagicMock()

        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_orchestrator_config",
                return_value={"mcps": ["my-mcp"]},
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model_from_spec",
                return_value=mock_model,
            ),
            patch("deep_agent.src.infrastructure.subagents.SubAgent") as mock_sa,
            patch(
                "deep_agent.src.infrastructure.subagents.filter_denied_tools"
            ) as mock_filter,
        ):
            mock_get_configs.return_value = {
                "admin": {
                    "name": "admin",
                    "model": "gemini-2.5-flash",
                    "description": "Admin",
                    "body": "Prompt",
                    # No allowed_tools — inherits all via mcps
                    "denied_tools": ["dangerous_tool"],
                }
            }
            mock_filter.return_value = [mock_tool_a]
            mock_sa.return_value = MagicMock()

            load_subagents(tools=[mock_tool_a, mock_tool_b])

            mock_filter.assert_called_once()

    def test_compiled_subagent_gets_denied_tools_filtered(self):
        """Compiled subagent also filters denied tools (same as default)."""
        mock_tool = MagicMock()
        mock_tool.name = "tool_a"
        mock_model = MagicMock()
        mock_graph = MagicMock()

        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_orchestrator_config",
                return_value={},
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.resolve_tools"
            ) as mock_resolve,
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model_from_spec",
                return_value=mock_model,
            ),
            patch("deepagents.create_deep_agent") as mock_create_agent,
            patch(
                "deep_agent.src.infrastructure.backend.get_configured_backend"
            ) as mock_backend,
            patch(
                "deep_agent.src.infrastructure.subagents.CompiledSubAgent"
            ) as mock_compiled,
            patch(
                "deep_agent.src.infrastructure.subagents.filter_denied_tools"
            ) as mock_filter,
        ):
            mock_get_configs.return_value = {
                "analyst": {
                    "name": "analyst",
                    "type": "compiled",
                    "model": "gemini-2.5-pro",
                    "description": "Analyst",
                    "body": "Prompt",
                    "allowed_tools": ["tool_a", "tool_b"],
                    "denied_tools": ["tool_b"],
                }
            }
            mock_resolve.return_value = [mock_tool]
            mock_filter.return_value = [mock_tool]
            mock_create_agent.return_value = mock_graph
            mock_backend.return_value = MagicMock()
            mock_compiled.return_value = MagicMock()

            load_subagents(tools=[mock_tool])

            mock_filter.assert_called_once()

    def test_two_subagents_get_different_tool_sets(self):
        """Two subagents with different allowed_tools get isolated tool sets."""
        mock_tool_a = MagicMock()
        mock_tool_a.name = "tool_a"
        mock_tool_b = MagicMock()
        mock_tool_b.name = "tool_b"
        mock_model = MagicMock()

        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_orchestrator_config",
                return_value={},
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.resolve_tools"
            ) as mock_resolve,
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model_from_spec",
                return_value=mock_model,
            ),
            patch("deep_agent.src.infrastructure.subagents.SubAgent") as mock_sa,
        ):
            mock_get_configs.return_value = {
                "agent_x": {
                    "name": "agent_x",
                    "model": "gemini-2.5-flash",
                    "description": "Agent X",
                    "body": "Prompt",
                    "allowed_tools": ["tool_a"],
                },
                "agent_y": {
                    "name": "agent_y",
                    "model": "gemini-2.5-flash",
                    "description": "Agent Y",
                    "body": "Prompt",
                    "allowed_tools": ["tool_b"],
                },
            }
            # resolve_tools returns different results per call
            mock_resolve.side_effect = [[mock_tool_a], [mock_tool_b]]
            mock_sa.return_value = MagicMock()

            load_subagents(tools=[mock_tool_a, mock_tool_b])

            assert mock_sa.call_count == 2
            calls = mock_sa.call_args_list
            # First subagent gets tool_a only
            assert calls[0][1]["tools"] == [mock_tool_a]
            # Second subagent gets tool_b only
            assert calls[1][1]["tools"] == [mock_tool_b]

    def _mock_tac_disabled(self):
        """Return a mock middleware config with tool_access_control.enabled=False."""
        mw_config = MagicMock()
        mw_config.defaults.tool_access_control.enabled = False
        return mw_config

    def test_denied_tools_skipped_when_flag_disabled(self):
        """When tool_access_control.enabled=False, denied_tools are not filtered."""
        mock_tool_a = MagicMock()
        mock_tool_a.name = "tool_a"
        mock_tool_b = MagicMock()
        mock_tool_b.name = "tool_b"
        mock_model = MagicMock()

        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_orchestrator_config",
                return_value={},
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_middleware_config",
                return_value=self._mock_tac_disabled(),
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.resolve_tools"
            ) as mock_resolve,
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model_from_spec",
                return_value=mock_model,
            ),
            patch("deep_agent.src.infrastructure.subagents.SubAgent") as mock_sa,
            patch(
                "deep_agent.src.infrastructure.subagents.filter_denied_tools"
            ) as mock_filter,
        ):
            mock_get_configs.return_value = {
                "agent1": {
                    "name": "agent1",
                    "model": "gemini-2.5-flash",
                    "description": "Test",
                    "body": "Prompt",
                    "allowed_tools": ["tool_a", "tool_b"],
                    "denied_tools": ["tool_b"],
                }
            }
            mock_resolve.return_value = [mock_tool_a, mock_tool_b]
            mock_sa.return_value = MagicMock()

            load_subagents(tools=[mock_tool_a, mock_tool_b])

            mock_filter.assert_not_called()
            call_kwargs = mock_sa.call_args[1]
            assert call_kwargs["tools"] == [mock_tool_a, mock_tool_b]

    def test_tool_approval_skipped_when_flag_disabled(self):
        """When tool_access_control.enabled=False, tool_approval does not raise for default subagent."""
        mock_model = MagicMock()

        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_orchestrator_config",
                return_value={},
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_middleware_config",
                return_value=self._mock_tac_disabled(),
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model_from_spec",
                return_value=mock_model,
            ),
            patch("deep_agent.src.infrastructure.subagents.SubAgent") as mock_sa,
        ):
            mock_get_configs.return_value = {
                "agent1": {
                    "name": "agent1",
                    "type": "default",
                    "model": "gemini-2.5-flash",
                    "description": "Test",
                    "body": "Prompt",
                    "allowed_tools": ["sensitive_tool"],
                    "tool_approval": ["sensitive_tool"],
                }
            }
            mock_sa.return_value = MagicMock()

            load_subagents(tools=[])

            mock_sa.assert_called_once()

    def test_compiled_denied_tools_skipped_when_flag_disabled(self):
        """When tool_access_control.enabled=False, compiled subagent skips denied_tools and interrupt_on."""
        mock_tool = MagicMock()
        mock_tool.name = "tool_a"
        mock_model = MagicMock()
        mock_graph = MagicMock()

        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs"
            ) as mock_get_configs,
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_orchestrator_config",
                return_value={},
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_middleware_config",
                return_value=self._mock_tac_disabled(),
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.resolve_tools"
            ) as mock_resolve,
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model_from_spec",
                return_value=mock_model,
            ),
            patch("deepagents.create_deep_agent") as mock_create_agent,
            patch(
                "deep_agent.src.infrastructure.backend.get_configured_backend"
            ) as mock_backend,
            patch(
                "deep_agent.src.infrastructure.subagents.CompiledSubAgent"
            ) as mock_compiled,
            patch(
                "deep_agent.src.infrastructure.subagents.filter_denied_tools"
            ) as mock_filter,
        ):
            mock_get_configs.return_value = {
                "analyst": {
                    "name": "analyst",
                    "type": "compiled",
                    "model": "gemini-2.5-pro",
                    "description": "Analyst",
                    "body": "Prompt",
                    "allowed_tools": ["tool_a", "tool_b"],
                    "denied_tools": ["tool_b"],
                    "tool_approval": ["tool_a"],
                }
            }
            mock_resolve.return_value = [mock_tool]
            mock_create_agent.return_value = mock_graph
            mock_backend.return_value = MagicMock()
            mock_compiled.return_value = MagicMock()

            load_subagents(tools=[mock_tool])

            mock_filter.assert_not_called()
            create_kwargs = mock_create_agent.call_args[1]
            assert "interrupt_on" not in create_kwargs
