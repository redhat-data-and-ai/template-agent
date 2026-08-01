"""Unit tests for cache warming."""

from unittest.mock import MagicMock, patch

from deep_agent.src.cache import warming
from deep_agent.src.cache.config import CacheSettings


class TestWarmCaches:
    def test_skips_when_disabled(self):
        disabled = CacheSettings(CACHE_ENABLED=False)
        with patch.object(warming, "cache_settings", disabled):
            result = warming.warm_caches()
            assert result == {}

    def test_warms_models_when_enabled(self):
        enabled = CacheSettings(
            CACHE_ENABLED=True,
            CACHE_WARMING_ENABLED=True,
            CACHE_MODEL_ENABLED=True,
        )
        mock_config = MagicMock()
        mock_config.get_orchestrator_config.return_value = {"model": "gemini-2.5-flash"}
        mock_config.get_all_subagent_configs.return_value = {
            "sub1": {
                "model": {"provider": "vertex", "name": "gemini-2.5-pro"},
            },
        }

        with (
            patch.object(warming, "cache_settings", enabled),
            patch("deep_agent.src.agent.config.agent_config", mock_config),
            patch(
                "deep_agent.src.cache.model_cache.get_or_create_model_from_spec"
            ) as mock_from_spec,
        ):
            result = warming.warm_caches()
            assert result["models"] is True
            # Both orchestrator and subagent use get_or_create_model_from_spec
            assert mock_from_spec.call_count == 2

    def test_handles_model_warming_failure(self):
        enabled = CacheSettings(
            CACHE_ENABLED=True,
            CACHE_WARMING_ENABLED=True,
            CACHE_MODEL_ENABLED=True,
        )
        mock_config = MagicMock()
        mock_config.get_orchestrator_config.side_effect = Exception("boom")

        with (
            patch.object(warming, "cache_settings", enabled),
            patch("deep_agent.src.agent.config.agent_config", mock_config),
        ):
            result = warming.warm_caches()
            assert result["models"] is False

    def test_skips_models_when_model_cache_disabled(self):
        enabled = CacheSettings(
            CACHE_ENABLED=True,
            CACHE_WARMING_ENABLED=True,
            CACHE_MODEL_ENABLED=False,
        )
        with patch.object(warming, "cache_settings", enabled):
            result = warming.warm_caches()
            assert result["models"] is False

    def test_parses_orchestrator_model_with_provider(self):
        """Orchestrator models support provider specification."""
        enabled = CacheSettings(
            CACHE_ENABLED=True,
            CACHE_WARMING_ENABLED=True,
            CACHE_MODEL_ENABLED=True,
        )
        mock_config = MagicMock()
        # Orchestrator with explicit provider
        mock_config.get_orchestrator_config.return_value = {
            "model": {
                "provider": "vertex",
                "name": "gemini-2.5-pro",
            }
        }
        mock_config.get_all_subagent_configs.return_value = {}

        with (
            patch.object(warming, "cache_settings", enabled),
            patch("deep_agent.src.agent.config.agent_config", mock_config),
            patch(
                "deep_agent.src.cache.model_cache.get_or_create_model_from_spec"
            ) as mock_from_spec,
        ):
            result = warming.warm_caches()
            assert result["models"] is True
            assert mock_from_spec.call_count == 1

            # Verify the spec has correct provider
            spec = mock_from_spec.call_args[0][0]
            assert spec.name == "gemini-2.5-pro"
            assert spec.provider.value == "vertex"
