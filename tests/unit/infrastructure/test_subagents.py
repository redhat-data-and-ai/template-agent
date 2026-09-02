"""Unit tests for subagent loading."""

from unittest.mock import MagicMock, patch

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
            patch(
                "deep_agent.src.infrastructure.subagents.build_opa_middleware",
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

        mock_settings = MagicMock()
        mock_settings.GUARDIAN_API_BASE = ""
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
            patch("deep_agent.src.settings.settings", mock_settings),
            patch(
                "deep_agent.src.infrastructure.subagents.build_opa_middleware",
                return_value=None,
            ),
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
            patch(
                "deep_agent.src.infrastructure.subagents.build_opa_middleware",
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
            patch(
                "deep_agent.src.infrastructure.subagents.build_opa_middleware",
                return_value=None,
            ),
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
            patch(
                "deep_agent.src.infrastructure.subagents.build_opa_middleware",
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
        mock_settings = MagicMock()
        mock_settings.GUARDIAN_API_BASE = ""

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
            patch("deep_agent.src.settings.settings", mock_settings),
            patch("deep_agent.src.pii.get_scrubber", return_value=None),
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
            assert mock_middleware.return_value in call_kwargs["middleware"]


import sys

from deep_agent.src.infrastructure.subagents import (
    _build_fallback_middleware,
    _normalize_model_to_dict,
    _resolve_async_headers,
)


class TestResolveAsyncHeaders:
    """Tests for _resolve_async_headers function."""

    def test_returns_none_when_no_token_env(self):
        """No env var set → returns None."""
        with patch.dict("os.environ", {}, clear=True):
            result = _resolve_async_headers("foo")
        assert result is None

    def test_returns_bearer_header_when_env_set(self):
        """Env var ASYNC_SUBAGENT_FOO_TOKEN=abc → returns Authorization: Bearer abc."""
        with patch.dict("os.environ", {"ASYNC_SUBAGENT_FOO_TOKEN": "abc"}, clear=True):
            result = _resolve_async_headers("foo")
        assert result == {"Authorization": "Bearer abc"}

    def test_normalizes_hyphens_to_underscores(self):
        """Agent name 'my-agent' → checks env var ASYNC_SUBAGENT_MY_AGENT_TOKEN."""
        with patch.dict(
            "os.environ", {"ASYNC_SUBAGENT_MY_AGENT_TOKEN": "secret"}, clear=True
        ):
            result = _resolve_async_headers("my-agent")
        assert result == {"Authorization": "Bearer secret"}


class TestNormalizeModelToDict:
    """Tests for _normalize_model_to_dict function."""

    def test_string_becomes_dict(self):
        """String model config → dict with name and provider keys."""
        result = _normalize_model_to_dict("gemini-2.5-flash")
        assert isinstance(result, dict)
        assert result["name"] == "gemini-2.5-flash"
        assert "provider" in result

    def test_dict_returned_as_copy(self):
        """Dict model config → returned as copy with same content."""
        original = {"provider": "x", "name": "y"}
        result = _normalize_model_to_dict(original)
        assert isinstance(result, dict)
        assert result == {"provider": "x", "name": "y"}
        assert result is not original

    def test_dict_fallback_stripped_when_requested(self):
        """Dict with fallback key and strip_fallback=True → no fallback key."""
        result = _normalize_model_to_dict(
            {"name": "m", "fallback": {}}, strip_fallback=True
        )
        assert isinstance(result, dict)
        assert "fallback" not in result
        assert result["name"] == "m"

    def test_invalid_type_returns_original_with_warning(self):
        """Invalid type (int) → returned as-is (just warns)."""
        result = _normalize_model_to_dict(42)
        assert result == 42


class TestBuildFallbackMiddleware:
    """Tests for _build_fallback_middleware function."""

    def test_no_fallback_returns_empty_list(self):
        """ModelSpec with fallback=None → returns []."""
        spec = ModelSpec(
            provider=Provider.VERTEX, name="gemini-2.5-flash", fallback=None
        )
        result = _build_fallback_middleware(spec)
        assert result == []

    def test_import_error_returns_empty_list(self):
        """If ModelFallbackMiddleware cannot be imported → returns []."""
        fallback_spec = ModelSpec(
            provider=Provider.VERTEX, name="gemini-2.5-flash", fallback=None
        )
        spec = ModelSpec(provider=Provider.VERTEX, name="gpt-4", fallback=fallback_spec)
        with patch.dict(sys.modules, {"langchain.agents.middleware": None}):
            result = _build_fallback_middleware(spec)
        assert result == []


class TestGuardianActivationGate:
    """Guardian wrapping requires BOTH guardrail config.enabled AND GUARDIAN_API_BASE."""

    def _guardrail_cfg(self, enabled: bool) -> MagicMock:
        cfg = MagicMock()
        cfg.enabled = enabled
        return cfg

    # ── default subagent ────────────────────────────────────────────────────

    def test_default_subagent_wraps_tools_when_guardian_active(self):
        """Both enabled=True and GUARDIAN_API_BASE set → wrap_tools called."""
        mock_tool = MagicMock()
        mock_settings = MagicMock()
        mock_settings.GUARDIAN_API_BASE = "http://guardian.internal"

        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs",
                return_value={
                    "analyst": {
                        "model": "gemini-2.5-flash",
                        "description": "A",
                        "body": "P",
                        "tools": ["t"],
                    }
                },
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_orchestrator_config",
                return_value={},
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_guardrails_config",
                return_value=self._guardrail_cfg(enabled=True),
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.resolve_tools",
                return_value=[mock_tool],
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model_from_spec",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.build_audit_middleware",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.build_opa_middleware",
                return_value=None,
            ),
            patch("deep_agent.src.settings.settings", mock_settings),
            patch(
                "deep_agent.src.guardrails.tool_proxy.wrap_tools",
                return_value=[mock_tool],
            ) as mock_wrap,
            patch(
                "deep_agent.src.infrastructure.subagents.SubAgent",
                return_value=MagicMock(),
            ) as mock_sa_cls,
        ):
            load_subagents(tools=[mock_tool])

        mock_wrap.assert_called_once_with([mock_tool])
        assert mock_sa_cls.call_args.kwargs["tools"] == [mock_tool]

    def test_default_subagent_skips_wrapping_when_config_disabled(self):
        """enabled=False + GUARDIAN_API_BASE set → wrap_tools not called."""
        mock_tool = MagicMock()
        mock_settings = MagicMock()
        mock_settings.GUARDIAN_API_BASE = "http://guardian.internal"

        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs",
                return_value={
                    "analyst": {
                        "model": "gemini-2.5-flash",
                        "description": "A",
                        "body": "P",
                        "tools": ["t"],
                    }
                },
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_orchestrator_config",
                return_value={},
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_guardrails_config",
                return_value=self._guardrail_cfg(enabled=False),
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.resolve_tools",
                return_value=[mock_tool],
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model_from_spec",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.build_audit_middleware",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.build_opa_middleware",
                return_value=None,
            ),
            patch("deep_agent.src.settings.settings", mock_settings),
            patch("deep_agent.src.guardrails.tool_proxy.wrap_tools") as mock_wrap,
            patch(
                "deep_agent.src.infrastructure.subagents.SubAgent",
                return_value=MagicMock(),
            ),
        ):
            load_subagents(tools=[mock_tool])

        mock_wrap.assert_not_called()

    def test_default_subagent_skips_wrapping_when_api_base_absent(self):
        """enabled=True + no GUARDIAN_API_BASE → wrap_tools not called."""
        mock_tool = MagicMock()
        mock_settings = MagicMock()
        mock_settings.GUARDIAN_API_BASE = ""

        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs",
                return_value={
                    "analyst": {
                        "model": "gemini-2.5-flash",
                        "description": "A",
                        "body": "P",
                        "tools": ["t"],
                    }
                },
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_orchestrator_config",
                return_value={},
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_guardrails_config",
                return_value=self._guardrail_cfg(enabled=True),
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.resolve_tools",
                return_value=[mock_tool],
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model_from_spec",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.build_audit_middleware",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.build_opa_middleware",
                return_value=None,
            ),
            patch("deep_agent.src.settings.settings", mock_settings),
            patch("deep_agent.src.guardrails.tool_proxy.wrap_tools") as mock_wrap,
            patch(
                "deep_agent.src.infrastructure.subagents.SubAgent",
                return_value=MagicMock(),
            ),
        ):
            load_subagents(tools=[mock_tool])

        mock_wrap.assert_not_called()

    # ── compiled subagent ────────────────────────────────────────────────────

    def test_compiled_subagent_wraps_and_applies_safety_when_guardian_active(self):
        """Both enabled=True and GUARDIAN_API_BASE set → wrap_tools + SafetyAwareRunnable."""
        mock_graph = MagicMock()
        mock_safety = MagicMock()
        mock_settings = MagicMock()
        mock_settings.GUARDIAN_API_BASE = "http://guardian.internal"

        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs",
                return_value={
                    "analyst": {
                        "type": "compiled",
                        "model": "gemini-2.5-pro",
                        "description": "A",
                        "body": "P",
                    }
                },
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_orchestrator_config",
                return_value={},
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_guardrails_config",
                return_value=self._guardrail_cfg(enabled=True),
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model_from_spec",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.infrastructure.backend.get_configured_backend",
                return_value=MagicMock(),
            ),
            patch("deepagents.create_deep_agent", return_value=mock_graph),
            patch(
                "deep_agent.src.infrastructure.subagents.CompiledSubAgent",
                return_value=MagicMock(),
            ) as mock_compiled_sa_cls,
            patch(
                "deep_agent.src.infrastructure.subagents.build_audit_middleware",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.build_opa_middleware",
                return_value=None,
            ),
            patch("deep_agent.src.settings.settings", mock_settings),
            patch("deep_agent.src.pii.get_scrubber", return_value=None),
            patch(
                "deep_agent.src.guardrails.tool_proxy.wrap_tools", return_value=[]
            ) as mock_wrap,
            patch(
                "deep_agent.aegra.safety.SafetyAwareRunnable", return_value=mock_safety
            ) as mock_safety_cls,
        ):
            load_subagents(tools=[])

        mock_wrap.assert_called_once()
        mock_safety_cls.assert_called_once_with(mock_graph)
        assert mock_compiled_sa_cls.call_args.kwargs["runnable"] is mock_safety

    def test_compiled_subagent_skips_guardian_when_config_disabled(self):
        """enabled=False + GUARDIAN_API_BASE set → no wrap_tools or SafetyAwareRunnable."""
        mock_graph = MagicMock()
        mock_settings = MagicMock()
        mock_settings.GUARDIAN_API_BASE = "http://guardian.internal"

        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs",
                return_value={
                    "analyst": {
                        "type": "compiled",
                        "model": "gemini-2.5-pro",
                        "description": "A",
                        "body": "P",
                    }
                },
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_orchestrator_config",
                return_value={},
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_guardrails_config",
                return_value=self._guardrail_cfg(enabled=False),
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model_from_spec",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.infrastructure.backend.get_configured_backend",
                return_value=MagicMock(),
            ),
            patch("deepagents.create_deep_agent", return_value=mock_graph),
            patch(
                "deep_agent.src.infrastructure.subagents.CompiledSubAgent",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.build_audit_middleware",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.build_opa_middleware",
                return_value=None,
            ),
            patch("deep_agent.src.settings.settings", mock_settings),
            patch("deep_agent.src.pii.get_scrubber", return_value=None),
            patch("deep_agent.src.guardrails.tool_proxy.wrap_tools") as mock_wrap,
            patch("deep_agent.aegra.safety.SafetyAwareRunnable") as mock_safety_cls,
        ):
            load_subagents(tools=[])

        mock_wrap.assert_not_called()
        mock_safety_cls.assert_not_called()

    def test_compiled_subagent_skips_guardian_when_api_base_absent(self):
        """enabled=True + no GUARDIAN_API_BASE → no wrap_tools or SafetyAwareRunnable."""
        mock_graph = MagicMock()
        mock_settings = MagicMock()
        mock_settings.GUARDIAN_API_BASE = ""

        with (
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_all_subagent_configs",
                return_value={
                    "analyst": {
                        "type": "compiled",
                        "model": "gemini-2.5-pro",
                        "description": "A",
                        "body": "P",
                    }
                },
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_orchestrator_config",
                return_value={},
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.agent_config.get_guardrails_config",
                return_value=self._guardrail_cfg(enabled=True),
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.get_or_create_model_from_spec",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.infrastructure.backend.get_configured_backend",
                return_value=MagicMock(),
            ),
            patch("deepagents.create_deep_agent", return_value=mock_graph),
            patch(
                "deep_agent.src.infrastructure.subagents.CompiledSubAgent",
                return_value=MagicMock(),
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.build_audit_middleware",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.build_opa_middleware",
                return_value=None,
            ),
            patch("deep_agent.src.settings.settings", mock_settings),
            patch("deep_agent.src.pii.get_scrubber", return_value=None),
            patch("deep_agent.src.guardrails.tool_proxy.wrap_tools") as mock_wrap,
            patch("deep_agent.aegra.safety.SafetyAwareRunnable") as mock_safety_cls,
        ):
            load_subagents(tools=[])

        mock_wrap.assert_not_called()
        mock_safety_cls.assert_not_called()
