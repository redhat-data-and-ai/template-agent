"""Unit tests for model cache."""

from unittest.mock import MagicMock, patch

from deep_agent.src.agent.config.model import ModelSpec, Provider
from deep_agent.src.cache import model_cache
from deep_agent.src.cache.config import CacheSettings


class TestModelCache:
    def setup_method(self):
        model_cache._legacy_cache = None
        model_cache._spec_cache = None

    def test_passthrough_when_disabled(self):
        disabled = CacheSettings(CACHE_ENABLED=False)
        mock_model = MagicMock()

        with (
            patch.object(model_cache, "cache_settings", disabled),
            patch("deep_agent.src.agent.llm.create_model", return_value=mock_model),
        ):
            result = model_cache.get_or_create_model("gemini-2.5-pro")
            assert result is mock_model

    def test_cache_hit(self):
        enabled = CacheSettings(CACHE_ENABLED=True, CACHE_MODEL_ENABLED=True)
        mock_model = MagicMock()

        with (
            patch.object(model_cache, "cache_settings", enabled),
            patch(
                "deep_agent.src.agent.llm.create_model", return_value=mock_model
            ) as create,
        ):
            m1 = model_cache.get_or_create_model("gemini-2.5-pro", 0.0, 8192)
            m2 = model_cache.get_or_create_model("gemini-2.5-pro", 0.0, 8192)

            assert m1 is mock_model
            assert m2 is mock_model
            assert create.call_count == 1

    def test_different_params_different_entries(self):
        enabled = CacheSettings(CACHE_ENABLED=True, CACHE_MODEL_ENABLED=True)
        model_a = MagicMock(name="model_a")
        model_b = MagicMock(name="model_b")

        with (
            patch.object(model_cache, "cache_settings", enabled),
            patch(
                "deep_agent.src.agent.llm.create_model", side_effect=[model_a, model_b]
            ),
        ):
            r1 = model_cache.get_or_create_model("gemini-2.5-pro", 0.0, 8192)
            r2 = model_cache.get_or_create_model("gemini-2.5-pro", 0.5, 8192)

            assert r1 is model_a
            assert r2 is model_b

    def test_invalidate_all(self):
        enabled = CacheSettings(CACHE_ENABLED=True, CACHE_MODEL_ENABLED=True)
        with patch.object(model_cache, "cache_settings", enabled):
            model_cache._get_cache()[("test", 0.0, 8192)] = MagicMock()
            assert model_cache.cached_count() == 1

            model_cache.invalidate()
            assert model_cache.cached_count() == 0

    def test_invalidate_by_name(self):
        enabled = CacheSettings(CACHE_ENABLED=True, CACHE_MODEL_ENABLED=True)
        with patch.object(model_cache, "cache_settings", enabled):
            cache = model_cache._get_cache()
            cache[("gemini", 0.0, 8192)] = MagicMock()
            cache[("claude", 0.0, 8192)] = MagicMock()
            assert model_cache.cached_count() == 2

            model_cache.invalidate("gemini")
            assert model_cache.cached_count() == 1


class TestModelCacheFromSpec:
    """Tests for provider-aware spec cache."""

    def setup_method(self):
        model_cache._legacy_cache = None
        model_cache._spec_cache = None

    def test_spec_cache_hit(self):
        enabled = CacheSettings(CACHE_ENABLED=True, CACHE_MODEL_ENABLED=True)
        mock_model = MagicMock()
        spec = ModelSpec(provider=Provider.VERTEX, name="gemini-2.5-pro")

        with (
            patch.object(model_cache, "cache_settings", enabled),
            patch(
                "deep_agent.src.agent.provider_factory.create_model_from_spec",
                return_value=mock_model,
            ) as create,
        ):
            m1 = model_cache.get_or_create_model_from_spec(spec)
            m2 = model_cache.get_or_create_model_from_spec(spec)

            assert m1 is mock_model
            assert m2 is mock_model
            assert create.call_count == 1

    def test_different_providers_same_name_different_entries(self):
        enabled = CacheSettings(CACHE_ENABLED=True, CACHE_MODEL_ENABLED=True)
        vertex_model = MagicMock(name="vertex_model")
        openai_model = MagicMock(name="openai_model")
        vertex_spec = ModelSpec(provider=Provider.VERTEX, name="shared-name")
        openai_spec = ModelSpec(provider=Provider.OPENAI, name="shared-name")

        with (
            patch.object(model_cache, "cache_settings", enabled),
            patch(
                "deep_agent.src.agent.provider_factory.create_model_from_spec",
                side_effect=[vertex_model, openai_model],
            ),
        ):
            r1 = model_cache.get_or_create_model_from_spec(vertex_spec)
            r2 = model_cache.get_or_create_model_from_spec(openai_spec)

            assert r1 is vertex_model
            assert r2 is openai_model

    def test_fallback_changes_cache_key(self):
        enabled = CacheSettings(CACHE_ENABLED=True, CACHE_MODEL_ENABLED=True)
        model_a = MagicMock(name="model_a")
        model_b = MagicMock(name="model_b")
        spec_no_fb = ModelSpec(provider=Provider.VERTEX, name="gemini-2.5-pro")
        spec_with_fb = ModelSpec(
            provider=Provider.VERTEX,
            name="gemini-2.5-pro",
            fallback=ModelSpec(provider=Provider.OPENAI, name="gpt-4o-mini"),
        )

        with (
            patch.object(model_cache, "cache_settings", enabled),
            patch(
                "deep_agent.src.agent.provider_factory.create_model_from_spec",
                side_effect=[model_a, model_b],
            ),
        ):
            r1 = model_cache.get_or_create_model_from_spec(spec_no_fb)
            r2 = model_cache.get_or_create_model_from_spec(spec_with_fb)

            assert r1 is model_a
            assert r2 is model_b
