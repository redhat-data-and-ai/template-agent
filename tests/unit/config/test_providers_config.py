"""Unit tests for providers configuration and profile registration."""

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from deep_agent.src.agent.config.providers import (
    AsyncTaskConfig,
    GeneralPurposeSubagentConfig,
    HarnessProfileConfig,
    ProviderConfig,
    ProvidersFileConfig,
    load_providers_config,
)
from deep_agent.src.infrastructure.async_tasks import (
    _extract_async_subagents,
    build_async_middleware,
)
from deep_agent.src.infrastructure.providers import (
    _register_harness_profiles,
    _register_provider_profiles,
    resolve_model_from_config,
)


class TestProviderModels:
    """Test Pydantic model defaults."""

    def test_default_strategy_is_legacy(self):
        config = ProvidersFileConfig()
        assert config.resolve_strategy == "legacy"

    def test_default_async_tasks_enabled(self):
        config = ProvidersFileConfig()
        assert config.async_tasks.enabled is True
        assert config.async_tasks.system_prompt is None

    def test_default_general_purpose_subagent(self):
        gp = GeneralPurposeSubagentConfig()
        assert gp.enabled is True
        assert gp.description is None
        assert gp.system_prompt is None

    def test_harness_profile_defaults(self):
        hp = HarnessProfileConfig()
        assert hp.system_prompt_suffix == ""
        assert hp.excluded_tools == []
        assert hp.excluded_middleware == []
        assert hp.general_purpose_subagent.enabled is True

    def test_provider_config_defaults(self):
        pc = ProviderConfig()
        assert pc.init_kwargs == {}


class TestLoadProvidersConfig:
    """Test loading providers.yaml from disk."""

    def test_returns_defaults_when_missing(self, tmp_path):
        config = load_providers_config(tmp_path / "nope.yaml")
        assert config.resolve_strategy == "legacy"
        assert config.providers == {}
        assert config.harness_profiles == {}

    def test_loads_valid_yaml(self, tmp_path):
        content = """
resolve_strategy: deepagents

providers:
  google_genai:
    init_kwargs:
      temperature: 0.0
  openai:
    init_kwargs:
      api_key: test

harness_profiles:
  gemini-2.5-pro:
    system_prompt_suffix: "Think step by step."
    excluded_tools: [execute]
    general_purpose_subagent:
      enabled: false

async_tasks:
  enabled: false
  system_prompt: "Custom async prompt"
"""
        config_file = tmp_path / "providers.yaml"
        config_file.write_text(content)

        config = load_providers_config(config_file)
        assert config.resolve_strategy == "deepagents"
        assert len(config.providers) == 2
        assert config.providers["openai"].init_kwargs == {"api_key": "test"}
        assert len(config.harness_profiles) == 1
        hp = config.harness_profiles["gemini-2.5-pro"]
        assert hp.system_prompt_suffix == "Think step by step."
        assert hp.excluded_tools == ["execute"]
        assert hp.general_purpose_subagent.enabled is False
        assert config.async_tasks.enabled is False
        assert config.async_tasks.system_prompt == "Custom async prompt"

    def test_returns_defaults_on_invalid_yaml(self, tmp_path):
        config_file = tmp_path / "providers.yaml"
        config_file.write_text("{{invalid")
        config = load_providers_config(config_file)
        assert config.resolve_strategy == "legacy"


class TestResolveModel:
    """Test model resolution dispatch."""

    def test_legacy_strategy_uses_cache(self):
        config = ProvidersFileConfig(resolve_strategy="legacy")
        with patch(
            "deep_agent.src.infrastructure.providers.get_or_create_model",
            return_value="mock_model",
        ) as mock:
            result = resolve_model_from_config("gemini-2.5-pro", config)
        assert result == "mock_model"
        mock.assert_called_once_with(
            model_name="gemini-2.5-pro",
            temperature=0.0,
            max_output_tokens=None,
        )

    def test_deepagents_strategy_calls_resolve_model(self):
        config = ProvidersFileConfig(resolve_strategy="deepagents")
        with patch(
            "deep_agent.src.infrastructure.providers.resolve_model",
            return_value="da_model",
        ):
            result = resolve_model_from_config("openai:gpt-5.4", config)
        assert result == "da_model"


class TestRegisterProfiles:
    """Test profile registration functions."""

    def test_register_provider_profiles(self):
        config = ProvidersFileConfig(
            providers={
                "google_genai": ProviderConfig(init_kwargs={"temperature": 0.0}),
            }
        )
        mock_profile_cls = MagicMock()
        mock_register = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "deepagents": MagicMock(
                    ProviderProfile=mock_profile_cls,
                    register_provider_profile=mock_register,
                )
            },
        ):
            _register_provider_profiles(config)
        mock_register.assert_called_once()

    def test_register_harness_profiles(self):
        config = ProvidersFileConfig(
            harness_profiles={
                "gemini-2.5-pro": HarnessProfileConfig(
                    system_prompt_suffix="Think.",
                    excluded_tools=["execute"],
                    general_purpose_subagent=GeneralPurposeSubagentConfig(
                        enabled=False
                    ),
                ),
            }
        )
        mock_hp_cls = MagicMock()
        mock_gp_cls = MagicMock()
        mock_register = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "deepagents": MagicMock(
                    HarnessProfile=mock_hp_cls,
                    GeneralPurposeSubagentProfile=mock_gp_cls,
                    register_harness_profile=mock_register,
                )
            },
        ):
            _register_harness_profiles(config)
        mock_register.assert_called_once()
        mock_gp_cls.assert_called_once_with(
            enabled=False, description=None, system_prompt=None
        )


class TestAsyncMiddleware:
    """Test async middleware builder."""

    def test_returns_none_when_disabled(self):
        config = AsyncTaskConfig(enabled=False)
        result = build_async_middleware([MagicMock()], config)
        assert result is None

    def test_returns_none_when_no_subagents(self):
        config = AsyncTaskConfig(enabled=True)
        result = build_async_middleware(None, config)
        assert result is None

    def test_returns_none_when_no_async_subagents(self):
        config = AsyncTaskConfig(enabled=True)
        regular_sub = MagicMock(spec=[])
        with patch(
            "deep_agent.src.infrastructure.async_tasks._extract_async_subagents",
            return_value=[],
        ):
            result = build_async_middleware([regular_sub], config)
        assert result is None

    def test_builds_middleware_for_async_subagents(self):
        config = AsyncTaskConfig(enabled=True, system_prompt="Custom prompt")
        async_sub = MagicMock()
        mock_mw = MagicMock()

        with (
            patch(
                "deep_agent.src.infrastructure.async_tasks._extract_async_subagents",
                return_value=[async_sub],
            ),
            patch(
                "deep_agent.src.infrastructure.async_tasks.AsyncSubAgentMiddleware",
                return_value=mock_mw,
            ) as mock_cls,
        ):
            result = build_async_middleware([async_sub], config)

        assert result is mock_mw
        mock_cls.assert_called_once_with(
            async_subagents=[async_sub],
            system_prompt="Custom prompt",
        )
