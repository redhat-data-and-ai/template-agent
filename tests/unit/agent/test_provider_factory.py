"""Unit tests for model config parsing and provider factory."""

from unittest.mock import MagicMock, patch

import pytest

from deep_agent.src.agent.config.model import (
    ModelSpec,
    Provider,
    infer_provider,
    model_spec_cache_key,
    parse_model_config,
)
from deep_agent.src.agent.provider_factory import (
    _create_by_provider,
    create_model_from_spec,
)


class TestProviderAliases:
    """Tests for alternate provider spellings accepted in config."""

    @pytest.mark.parametrize("alias", ["vertex_ai", "vertex-ai", "VERTEX_AI"])
    def test_alias_resolves_to_canonical_member(self, alias):
        assert Provider(alias) is Provider.VERTEX

    def test_canonical_values_unaffected(self):
        assert Provider("vertex") is Provider.VERTEX
        assert Provider("openai") is Provider.OPENAI
        assert Provider("openai-compatibility") is Provider.OPENAI_COMPATIBILITY

    def test_unknown_value_still_raises(self):
        with pytest.raises(ValueError):
            Provider("azure")

    def test_non_string_value_still_raises(self):
        with pytest.raises(ValueError):
            Provider(42)


@pytest.fixture
def no_declared_providers():
    """Isolate name-based inference from whatever models/*.yaml is on disk."""
    with patch(
        "deep_agent.src.agent.config.model._declared_provider", return_value=None
    ):
        yield


@pytest.mark.usefixtures("no_declared_providers")
class TestInferProvider:
    """Name-based inference, with no models/*.yaml declaration to defer to."""

    def test_gemini_models_infer_vertex(self):
        assert infer_provider("gemini-2.5-pro") == Provider.VERTEX
        assert infer_provider("gemini-2.5-flash") == Provider.VERTEX

    def test_claude_models_infer_vertex(self):
        assert infer_provider("claude-sonnet-4") == Provider.VERTEX

    def test_gpt_models_infer_openai(self):
        assert infer_provider("gpt-4o-mini") == Provider.OPENAI
        assert infer_provider("gpt-4") == Provider.OPENAI
        assert infer_provider("gpt-3.5-turbo") == Provider.OPENAI
        # Case-insensitive matching
        assert infer_provider("GPT-4") == Provider.OPENAI
        assert infer_provider("Gpt-4o") == Provider.OPENAI

    def test_unknown_models_infer_openai_compatibility(self):
        """Names off the known Vertex lists are assumed OpenAI-compatible."""
        assert infer_provider("mistral-7b") == Provider.OPENAI_COMPATIBILITY
        assert infer_provider("llama-3-70b") == Provider.OPENAI_COMPATIBILITY
        assert infer_provider("custom-model") == Provider.OPENAI_COMPATIBILITY


class TestInferProviderFromModelConfigs:
    """A models/*.yaml declaration is consulted before any name heuristic."""

    def test_declaration_supplies_provider_for_unknown_name(self):
        with patch(
            "deep_agent.src.agent.config.model._declared_provider",
            return_value=Provider.OPENAI_COMPATIBILITY,
        ):
            assert infer_provider("granite-3.3-8b") == Provider.OPENAI_COMPATIBILITY

    def test_declaration_beats_gpt_name_heuristic(self):
        """A gpt-* model served via a proxy can be declared as such."""
        with patch(
            "deep_agent.src.agent.config.model._declared_provider",
            return_value=Provider.OPENAI_COMPATIBILITY,
        ):
            assert infer_provider("gpt-4o-mini") == Provider.OPENAI_COMPATIBILITY

    def test_falls_through_when_no_declaration(self):
        with patch(
            "deep_agent.src.agent.config.model._declared_provider", return_value=None
        ):
            assert infer_provider("granite-3.3-8b") == Provider.OPENAI_COMPATIBILITY
            assert infer_provider("gemini-2.5-pro") == Provider.VERTEX


@pytest.mark.usefixtures("no_declared_providers")
class TestParseModelConfig:
    """Tests for parse_model_config()."""

    def test_parses_legacy_string_vertex(self):
        spec = parse_model_config("gemini-2.5-pro")
        assert spec.provider == Provider.VERTEX
        assert spec.name == "gemini-2.5-pro"
        assert spec.fallback is None

    def test_parses_legacy_string_openai(self):
        spec = parse_model_config("gpt-4o-mini")
        assert spec.provider == Provider.OPENAI
        assert spec.name == "gpt-4o-mini"

    def test_parses_legacy_string_openai_compatibility(self):
        spec = parse_model_config("mistral-7b")
        assert spec.provider == Provider.OPENAI_COMPATIBILITY
        assert spec.name == "mistral-7b"

    def test_parses_object_without_provider_infers(self):
        """Provider is optional in dict format - infers from name."""
        spec = parse_model_config({"name": "gpt-4"})
        assert spec.provider == Provider.OPENAI  # Inferred
        assert spec.name == "gpt-4"

        spec2 = parse_model_config({"name": "gemini-2.5-pro"})
        assert spec2.provider == Provider.VERTEX  # Inferred

        spec3 = parse_model_config({"name": "mistral-7b"})
        assert spec3.provider == Provider.OPENAI_COMPATIBILITY  # Inferred

    def test_parses_object_form(self):
        spec = parse_model_config({"provider": "vertex", "name": "gemini-2.5-pro"})
        assert spec.provider == Provider.VERTEX
        assert spec.name == "gemini-2.5-pro"

    def test_parses_object_with_fallback(self):
        spec = parse_model_config(
            {
                "provider": "vertex",
                "name": "gemini-2.5-pro",
                "fallback": {"provider": "openai", "name": "gpt-4o-mini"},
            }
        )
        assert spec.fallback is not None
        assert spec.fallback.provider == Provider.OPENAI
        assert spec.fallback.name == "gpt-4o-mini"
        assert spec.fallback.fallback is None

    def test_parses_fallback_without_provider_infers(self):
        """Fallback can omit provider - infers from name."""
        spec = parse_model_config(
            {
                "provider": "vertex",
                "name": "gemini-2.5-pro",
                "fallback": {"name": "gpt-4"},  # No provider - inferred
            }
        )
        assert spec.fallback is not None
        assert spec.fallback.provider == Provider.OPENAI  # Inferred from "gpt-4"
        assert spec.fallback.name == "gpt-4"

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_model_config("")

    @pytest.mark.parametrize(
        "alias", ["vertex_ai", "vertex-ai", "VERTEX_AI", "  vertex_ai  "]
    )
    def test_accepts_vertex_aliases(self, alias):
        """Alternate Vertex spellings normalise to the canonical member."""
        spec = parse_model_config({"provider": alias, "name": "gemini-2.5-pro"})
        assert spec.provider is Provider.VERTEX
        assert spec.provider.value == "vertex"
        # Normalising at the enum keeps derived identities canonical.
        assert model_spec_cache_key(spec) == "vertex:gemini-2.5-pro"

    def test_rejects_invalid_provider(self):
        with pytest.raises(ValueError, match="invalid provider"):
            parse_model_config({"provider": "azure", "name": "gpt-4"})

    def test_rejects_missing_name(self):
        with pytest.raises(ValueError, match="requires non-empty 'name'"):
            parse_model_config({"provider": "vertex"})

    def test_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match="unknown model config keys"):
            parse_model_config(
                {"provider": "vertex", "name": "gemini-2.5-pro", "extra": "x"}
            )

    def test_rejects_nested_fallback(self):
        with pytest.raises(ValueError, match="nested fallback"):
            parse_model_config(
                {
                    "provider": "vertex",
                    "name": "gemini-2.5-pro",
                    "fallback": {
                        "provider": "openai",
                        "name": "gpt-4o-mini",
                        "fallback": {"provider": "vertex", "name": "gemini-2.5-flash"},
                    },
                }
            )

    def test_display_name_with_fallback(self):
        spec = parse_model_config(
            {
                "provider": "vertex",
                "name": "gemini-2.5-pro",
                "fallback": {"provider": "openai", "name": "gpt-4o-mini"},
            }
        )
        assert "fallback" in spec.display_name()


class TestModelSpecCacheKey:
    """Tests for model_spec_cache_key()."""

    def test_key_without_fallback(self):
        spec = ModelSpec(provider=Provider.VERTEX, name="gemini-2.5-pro")
        assert model_spec_cache_key(spec) == "vertex:gemini-2.5-pro"

    def test_key_with_fallback(self):
        spec = ModelSpec(
            provider=Provider.VERTEX,
            name="gemini-2.5-pro",
            fallback=ModelSpec(provider=Provider.OPENAI, name="gpt-4o-mini"),
        )
        assert model_spec_cache_key(spec) == "vertex:gemini-2.5-pro→openai:gpt-4o-mini"


class TestCreateModelFromSpec:
    """Tests for create_model_from_spec() routing."""

    def test_routes_vertex_provider(self):
        mock_model = MagicMock()
        spec = ModelSpec(provider=Provider.VERTEX, name="gemini-2.5-pro")

        with patch(
            "deep_agent.src.agent.provider_factory._create_by_provider",
            return_value=mock_model,
        ) as mock_create:
            result = create_model_from_spec(spec)

        assert result is mock_model
        mock_create.assert_called_once()
        assert mock_create.call_args[0][0] == Provider.VERTEX

    def test_routes_openai_provider(self):
        mock_model = MagicMock()
        spec = ModelSpec(provider=Provider.OPENAI, name="gpt-4o-mini")

        with patch(
            "deep_agent.src.agent.provider_factory._create_by_provider",
            return_value=mock_model,
        ) as mock_create:
            result = create_model_from_spec(spec)

        assert result is mock_model
        assert mock_create.call_args[0][0] == Provider.OPENAI

    def test_routes_openai_compatibility_provider(self):
        mock_model = MagicMock()
        spec = ModelSpec(provider=Provider.OPENAI_COMPATIBILITY, name="mistral-7b")

        with patch(
            "deep_agent.src.agent.provider_factory._create_by_provider",
            return_value=mock_model,
        ) as mock_create:
            result = create_model_from_spec(spec)

        assert result is mock_model
        assert mock_create.call_args[0][0] == Provider.OPENAI_COMPATIBILITY


class TestFallbackChain:
    """Tests for primary → secondary fallback chaining."""

    def test_with_fallbacks_called_when_fallback_present(self):
        primary = MagicMock()
        secondary = MagicMock()
        chained = MagicMock()
        primary.with_fallbacks.return_value = chained

        spec = ModelSpec(
            provider=Provider.VERTEX,
            name="gemini-2.5-pro",
            fallback=ModelSpec(provider=Provider.OPENAI, name="gpt-4o-mini"),
        )

        with patch(
            "deep_agent.src.agent.provider_factory._create_by_provider",
            side_effect=[primary, secondary],
        ):
            result = create_model_from_spec(spec)

        assert result is chained
        primary.with_fallbacks.assert_called_once_with([secondary])

    def test_no_with_fallbacks_when_no_fallback(self):
        primary = MagicMock()
        spec = ModelSpec(provider=Provider.VERTEX, name="gemini-2.5-pro")

        with patch(
            "deep_agent.src.agent.provider_factory._create_by_provider",
            return_value=primary,
        ):
            result = create_model_from_spec(spec)

        assert result is primary
        primary.with_fallbacks.assert_not_called()


class TestCreateByProvider:
    """Tests for _create_by_provider() delegation."""

    def test_vertex_delegates_to_vertex_model(self):
        mock_model = MagicMock()
        with patch(
            "deep_agent.src.agent.provider_factory._create_vertex_model",
            return_value=mock_model,
        ) as mock_vertex:
            result = _create_by_provider(
                Provider.VERTEX,
                "gemini-2.5-pro",
                temperature=0.0,
                max_output_tokens=8192,
            )
        assert result is mock_model
        mock_vertex.assert_called_once_with("gemini-2.5-pro", 0.0, 8192, location=None)

    def test_openai_delegates_to_vllm_model(self):
        mock_model = MagicMock()
        with patch(
            "deep_agent.src.agent.provider_factory._create_vllm_model",
            return_value=mock_model,
        ) as mock_vllm:
            result = _create_by_provider(
                Provider.OPENAI, "gpt-4o-mini", temperature=0.0, max_output_tokens=4096
            )
        assert result is mock_model
        mock_vllm.assert_called_once_with("gpt-4o-mini", 0.0, 4096)

    def test_openai_compatibility_delegates_to_vllm_model(self):
        mock_model = MagicMock()
        with patch(
            "deep_agent.src.agent.provider_factory._create_vllm_model",
            return_value=mock_model,
        ) as mock_vllm:
            result = _create_by_provider(
                Provider.OPENAI_COMPATIBILITY,
                "mistral-7b",
                temperature=0.0,
                max_output_tokens=4096,
            )
        assert result is mock_model
        mock_vllm.assert_called_once_with("mistral-7b", 0.0, 4096)

    def test_rejects_unsupported_provider(self):
        with pytest.raises(ValueError, match="unsupported provider"):
            _create_by_provider(
                "azure",  # type: ignore[arg-type]
                "some-model",
                temperature=0.0,
                max_output_tokens=4096,
            )


class TestModelNamePassthrough:
    """Model names reach the endpoint verbatim — no prefixing or rewriting.

    Deployments that namespace their models (e.g. a serving platform exposing
    ``/data/<model>``) must spell the prefix out in the config.
    """

    @pytest.mark.parametrize(
        "provider",
        [Provider.OPENAI, Provider.OPENAI_COMPATIBILITY],
    )
    @pytest.mark.parametrize(
        "model_name",
        [
            "granite-guardian-3.1-8b",
            "/data/granite-guardian-3.1-8b",
            "data/granite-guardian-3.1-8b",
            "gpt-4o-mini",
        ],
    )
    def test_model_name_passed_through_unchanged(self, provider, model_name):
        mock_model = MagicMock()
        with patch(
            "deep_agent.src.agent.provider_factory._create_vllm_model",
            return_value=mock_model,
        ) as mock_vllm:
            _create_by_provider(
                provider,
                model_name,
                temperature=0.0,
                max_output_tokens=4096,
            )
        mock_vllm.assert_called_once_with(model_name, 0.0, 4096)
