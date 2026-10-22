"""Unit tests for models/*.yaml loading and provider-scoped lookup."""

import pytest

from deep_agent.src.agent.config import AgentConfig
from deep_agent.src.agent.config.model import Provider
from deep_agent.src.exceptions import AppException


class TestModelConfigKeys:
    """Entries are keyed provider:model_id, or bare model_id when unscoped."""

    def setup_method(self):
        AgentConfig._instance = None

    def _make_config_dir(self, tmp_path, models: dict[str, str]):
        config_dir = tmp_path / "agent"
        (config_dir / "runtime").mkdir(parents=True)
        (config_dir / "PROMPT.md").write_text(
            "---\nname: test\nmodel: gemini-2.5-flash\n---\nPrompt.\n"
        )
        models_dir = config_dir / "models"
        models_dir.mkdir()
        for filename, body in models.items():
            (models_dir / filename).write_text(body)
        return config_dir

    def test_no_models_dir_returns_empty(self, tmp_path):
        config_dir = tmp_path / "agent"
        (config_dir / "runtime").mkdir(parents=True)
        (config_dir / "PROMPT.md").write_text(
            "---\nname: test\nmodel: gemini-2.5-flash\n---\nPrompt.\n"
        )
        cfg = AgentConfig(config_dir)
        assert cfg.get_model_params("anything") == {}

    def test_declared_provider_produces_scoped_key(self, tmp_path):
        config_dir = self._make_config_dir(
            tmp_path,
            {
                "g.yaml": "model_id: gemini-2.5-pro\nprovider: vertex\ntemperature: 0.5\n"
            },
        )
        cfg = AgentConfig(config_dir)
        cfg._ensure_loaded()
        assert list(cfg._model_configs) == [("vertex", "gemini-2.5-pro")]

    def test_missing_provider_produces_bare_key(self, tmp_path):
        config_dir = self._make_config_dir(
            tmp_path, {"g.yaml": "model_id: gemini-2.5-pro\ntemperature: 0.5\n"}
        )
        cfg = AgentConfig(config_dir)
        cfg._ensure_loaded()
        assert list(cfg._model_configs) == [(None, "gemini-2.5-pro")]

    def test_provider_alias_normalised_in_key(self, tmp_path):
        """vertex_ai and vertex must not produce two different keys."""
        config_dir = self._make_config_dir(
            tmp_path,
            {"g.yaml": "model_id: gemini-2.5-pro\nprovider: vertex_ai\n"},
        )
        cfg = AgentConfig(config_dir)
        cfg._ensure_loaded()
        assert list(cfg._model_configs) == [("vertex", "gemini-2.5-pro")]

    @pytest.mark.parametrize(
        "bad", ["watsonx", "openai-compatible", '""', "[1, 2]", "123"]
    )
    def test_unsupported_provider_is_fatal(self, tmp_path, bad):
        """A wrong provider reroutes the model, so it must not be ignored."""
        config_dir = self._make_config_dir(
            tmp_path, {"g.yaml": f"model_id: some-model\nprovider: {bad}\n"}
        )
        cfg = AgentConfig(config_dir)
        with pytest.raises(AppException, match="Unsupported provider"):
            cfg._ensure_loaded()

    def test_error_names_the_file_and_accepted_values(self, tmp_path):
        config_dir = self._make_config_dir(
            tmp_path, {"g.yaml": "model_id: some-model\nprovider: watsonx\n"}
        )
        cfg = AgentConfig(config_dir)
        with pytest.raises(AppException) as exc:
            cfg._ensure_loaded()
        assert "g.yaml" in exc.value.detail
        assert "openai-compatibility" in exc.value.detail
        assert "vertex_ai" in exc.value.detail


class TestGetModelParamsLookup:
    """Lookup order: provider-scoped key first, then bare model_id."""

    def setup_method(self):
        AgentConfig._instance = None

    def _make_config_dir(self, tmp_path, models: dict[str, str]):
        config_dir = tmp_path / "agent"
        (config_dir / "runtime").mkdir(parents=True)
        (config_dir / "PROMPT.md").write_text(
            "---\nname: test\nmodel: gemini-2.5-flash\n---\nPrompt.\n"
        )
        models_dir = config_dir / "models"
        models_dir.mkdir()
        for filename, body in models.items():
            (models_dir / filename).write_text(body)
        return config_dir

    def test_same_model_name_two_providers_stay_distinct(self, tmp_path):
        """The reason provider-scoped keys exist."""
        config_dir = self._make_config_dir(
            tmp_path,
            {
                "a.yaml": "model_id: shared\nprovider: vertex\ntemperature: 0.1\n",
                "b.yaml": (
                    "model_id: shared\nprovider: openai-compatibility\n"
                    "temperature: 0.9\n"
                ),
            },
        )
        cfg = AgentConfig(config_dir)
        assert cfg.get_model_params("shared", Provider.VERTEX)["temperature"] == 0.1
        assert (
            cfg.get_model_params("shared", Provider.OPENAI_COMPATIBILITY)["temperature"]
            == 0.9
        )

    def test_falls_back_to_bare_key_when_no_scoped_entry(self, tmp_path):
        config_dir = self._make_config_dir(
            tmp_path, {"a.yaml": "model_id: shared\ntemperature: 0.4\n"}
        )
        cfg = AgentConfig(config_dir)
        assert cfg.get_model_params("shared", Provider.VERTEX)["temperature"] == 0.4

    def test_scoped_entry_wins_over_bare(self, tmp_path):
        config_dir = self._make_config_dir(
            tmp_path,
            {
                "a.yaml": "model_id: shared\ntemperature: 0.4\n",
                "b.yaml": "model_id: shared\nprovider: vertex\ntemperature: 0.7\n",
            },
        )
        cfg = AgentConfig(config_dir)
        assert cfg.get_model_params("shared", Provider.VERTEX)["temperature"] == 0.7
        assert cfg.get_model_params("shared")["temperature"] == 0.4

    def test_scoped_entry_not_reachable_without_provider(self, tmp_path):
        """A provider-scoped file only answers provider-scoped lookups."""
        config_dir = self._make_config_dir(
            tmp_path, {"a.yaml": "model_id: shared\nprovider: vertex\n"}
        )
        cfg = AgentConfig(config_dir)
        assert cfg.get_model_params("shared") == {}
        assert cfg.get_model_params("shared", Provider.OPENAI) == {}

    @pytest.mark.parametrize("provider", ["vertex", "vertex_ai", Provider.VERTEX])
    def test_lookup_accepts_enum_or_string_including_alias(self, tmp_path, provider):
        config_dir = self._make_config_dir(
            tmp_path,
            {"a.yaml": "model_id: shared\nprovider: vertex\ntemperature: 0.2\n"},
        )
        cfg = AgentConfig(config_dir)
        assert cfg.get_model_params("shared", provider)["temperature"] == 0.2

    def test_unknown_model_returns_empty(self, tmp_path):
        config_dir = self._make_config_dir(
            tmp_path, {"a.yaml": "model_id: shared\nprovider: vertex\n"}
        )
        cfg = AgentConfig(config_dir)
        assert cfg.get_model_params("nope", Provider.VERTEX) == {}


class TestDeclaredProviderIndex:
    """Frontmatter without `provider:` picks it up from models/*.yaml."""

    def setup_method(self):
        AgentConfig._instance = None

    def _make_config_dir(self, tmp_path, models: dict[str, str]):
        config_dir = tmp_path / "agent"
        (config_dir / "runtime").mkdir(parents=True)
        (config_dir / "PROMPT.md").write_text(
            "---\nname: test\nmodel: gemini-2.5-flash\n---\nPrompt.\n"
        )
        models_dir = config_dir / "models"
        models_dir.mkdir()
        for filename, body in models.items():
            (models_dir / filename).write_text(body)
        return config_dir

    def test_declared_provider_is_exposed(self, tmp_path):
        config_dir = self._make_config_dir(
            tmp_path,
            {"a.yaml": "model_id: granite\nprovider: openai-compatibility\n"},
        )
        cfg = AgentConfig(config_dir)
        assert cfg.get_declared_provider("granite") == "openai-compatibility"

    def test_alias_is_canonicalised_in_index(self, tmp_path):
        config_dir = self._make_config_dir(
            tmp_path, {"a.yaml": "model_id: gem\nprovider: vertex_ai\n"}
        )
        cfg = AgentConfig(config_dir)
        assert cfg.get_declared_provider("gem") == "vertex"

    def test_bare_entry_declares_nothing(self, tmp_path):
        config_dir = self._make_config_dir(
            tmp_path, {"a.yaml": "model_id: granite\ntemperature: 0.2\n"}
        )
        cfg = AgentConfig(config_dir)
        assert cfg.get_declared_provider("granite") is None

    def test_ambiguous_model_declares_nothing(self, tmp_path):
        """Two providers for one model_id gives no basis to choose."""
        config_dir = self._make_config_dir(
            tmp_path,
            {
                "a.yaml": "model_id: shared\nprovider: vertex\n",
                "b.yaml": "model_id: shared\nprovider: openai-compatibility\n",
            },
        )
        cfg = AgentConfig(config_dir)
        assert cfg.get_declared_provider("shared") is None
        # Both remain reachable for provider-scoped lookups.
        assert cfg.get_model_params("shared", Provider.VERTEX) != {}
        assert cfg.get_model_params("shared", Provider.OPENAI_COMPATIBILITY) != {}

    def test_unknown_model_declares_nothing(self, tmp_path):
        config_dir = self._make_config_dir(
            tmp_path, {"a.yaml": "model_id: granite\nprovider: vertex\n"}
        )
        cfg = AgentConfig(config_dir)
        assert cfg.get_declared_provider("nope") is None


class TestEmptyProviderKey:
    """`provider:` with no value is YAML null — indistinguishable from absent."""

    def setup_method(self):
        AgentConfig._instance = None

    def test_valueless_provider_key_treated_as_undeclared(self, tmp_path):
        config_dir = tmp_path / "agent"
        (config_dir / "runtime").mkdir(parents=True)
        (config_dir / "PROMPT.md").write_text(
            "---\nname: test\nmodel: gemini-2.5-flash\n---\nPrompt.\n"
        )
        (config_dir / "models").mkdir()
        (config_dir / "models" / "a.yaml").write_text(
            "model_id: some-model\nprovider:\ntemperature: 0.3\n"
        )
        cfg = AgentConfig(config_dir)
        assert cfg.get_declared_provider("some-model") is None
        assert cfg.get_model_params("some-model")["temperature"] == 0.3


class TestSupportedProviderValues:
    """The single source of truth shared by both error messages."""

    def test_lists_canonical_values_and_aliases(self):
        from deep_agent.src.agent.config.model import supported_provider_values

        values = supported_provider_values()
        assert values == sorted(values), "must be stable for error messages"
        for expected in ("vertex", "openai", "openai-compatibility", "vertex_ai"):
            assert expected in values


class TestDuplicateKeys:
    """Two files claiming the same provider:model_id key."""

    def setup_method(self):
        AgentConfig._instance = None

    def test_same_model_and_provider_last_file_wins(self, tmp_path):
        """sorted() filename order decides; b.yaml overrides a.yaml."""
        config_dir = tmp_path / "agent"
        (config_dir / "runtime").mkdir(parents=True)
        (config_dir / "PROMPT.md").write_text(
            "---\nname: test\nmodel: gemini-2.5-flash\n---\nPrompt.\n"
        )
        (config_dir / "models").mkdir()
        (config_dir / "models" / "a.yaml").write_text(
            "model_id: dup\nprovider: vertex\ntemperature: 0.1\n"
        )
        (config_dir / "models" / "b.yaml").write_text(
            "model_id: dup\nprovider: vertex\ntemperature: 0.9\n"
        )
        cfg = AgentConfig(config_dir)
        cfg._ensure_loaded()
        assert list(cfg._model_configs) == [("vertex", "dup")]
        assert cfg.get_model_params("dup", Provider.VERTEX)["temperature"] == 0.9
        # One provider, so still unambiguous for inference.
        assert cfg.get_declared_provider("dup") == "vertex"


class TestLookupWithBadProvider:
    """get_model_params validates a caller-supplied provider string."""

    def setup_method(self):
        AgentConfig._instance = None

    def test_invalid_provider_string_raises(self, tmp_path):
        config_dir = tmp_path / "agent"
        (config_dir / "runtime").mkdir(parents=True)
        (config_dir / "PROMPT.md").write_text(
            "---\nname: test\nmodel: gemini-2.5-flash\n---\nPrompt.\n"
        )
        cfg = AgentConfig(config_dir)
        with pytest.raises(AppException, match="Unsupported provider"):
            cfg.get_model_params("anything", "watsonx")


class TestFrontmatterInheritsDeclaredProvider:
    """End-to-end: no provider in frontmatter, provider declared in models/."""

    def setup_method(self):
        AgentConfig._instance = None

    def _config_dir(self, tmp_path, frontmatter_model):
        config_dir = tmp_path / "agent"
        (config_dir / "runtime").mkdir(parents=True)
        (config_dir / "PROMPT.md").write_text(
            f"---\nname: test\nmodel: {frontmatter_model}\n---\nPrompt.\n"
        )
        (config_dir / "models").mkdir()
        (config_dir / "models" / "granite.yaml").write_text(
            "model_id: granite-3.3-8b-instruct\n"
            "provider: openai-compatibility\n"
            "temperature: 0.25\n"
        )
        # Deliberately not a name the known-model lists recognise, so VERTEX
        # can only come from this declaration and not from name inference.
        (config_dir / "models" / "gem.yaml").write_text(
            "model_id: custom-gem\nprovider: vertex_ai\ntemperature: 0.5\n"
        )
        return config_dir

    def test_spec_picks_up_declared_provider(self, tmp_path, monkeypatch):
        from deep_agent.src.agent.config import loader as loader_mod
        from deep_agent.src.agent.config.model import parse_model_config

        cfg = AgentConfig(self._config_dir(tmp_path, "granite-3.3-8b-instruct"))
        monkeypatch.setattr(loader_mod, "agent_config", cfg)

        spec = parse_model_config(cfg.get_orchestrator_config()["model"])
        assert spec.provider is Provider.OPENAI_COMPATIBILITY
        # ...and the params lookup then resolves under that same provider.
        assert cfg.get_model_params(spec.name, spec.provider)["temperature"] == 0.25

    def test_declared_alias_resolves_to_canonical_member(self, tmp_path, monkeypatch):
        from deep_agent.src.agent.config import loader as loader_mod
        from deep_agent.src.agent.config.model import infer_provider, parse_model_config

        cfg = AgentConfig(self._config_dir(tmp_path, "custom-gem"))
        monkeypatch.setattr(loader_mod, "agent_config", cfg)

        spec = parse_model_config(cfg.get_orchestrator_config()["model"])
        assert spec.provider is Provider.VERTEX

        # Guard: without the declaration this name would not infer VERTEX, so
        # the assertion above can only be satisfied by the vertex_ai alias.
        monkeypatch.setattr(
            "deep_agent.src.agent.config.model._declared_provider", lambda _: None
        )
        assert infer_provider("custom-gem") is Provider.OPENAI_COMPATIBILITY

    def test_explicit_frontmatter_provider_overrides_declaration(
        self, tmp_path, monkeypatch
    ):
        from deep_agent.src.agent.config import loader as loader_mod
        from deep_agent.src.agent.config.model import parse_model_config

        cfg = AgentConfig(self._config_dir(tmp_path, "granite-3.3-8b-instruct"))
        monkeypatch.setattr(loader_mod, "agent_config", cfg)

        spec = parse_model_config(
            {"provider": "openai", "name": "granite-3.3-8b-instruct"}
        )
        assert spec.provider is Provider.OPENAI


class TestModelIdContainingColon:
    """model_id has no character restrictions — e.g. Ollama's "llama3:8b"."""

    def setup_method(self):
        AgentConfig._instance = None

    def _config_dir(self, tmp_path, models: dict[str, str]):
        config_dir = tmp_path / "agent"
        (config_dir / "runtime").mkdir(parents=True)
        (config_dir / "PROMPT.md").write_text(
            "---\nname: test\nmodel: gemini-2.5-flash\n---\nPrompt.\n"
        )
        (config_dir / "models").mkdir()
        for filename, body in models.items():
            (config_dir / "models" / filename).write_text(body)
        return config_dir

    def test_colon_in_model_id_is_not_split_into_a_provider(self, tmp_path):
        cfg = AgentConfig(
            self._config_dir(
                tmp_path, {"a.yaml": "model_id: llama3:8b\ntemperature: 0.1\n"}
            )
        )
        cfg._ensure_loaded()
        assert list(cfg._model_configs) == [(None, "llama3:8b")]
        # "llama3" must not be mistaken for a declared provider of "8b".
        assert cfg.get_declared_provider("8b") is None
        assert cfg.get_declared_provider("llama3:8b") is None
        assert cfg.get_model_params("llama3:8b")["temperature"] == 0.1

    def test_bare_colon_name_does_not_collide_with_scoped_entry(self, tmp_path):
        """(None, "vertex:8b") and ("vertex", "8b") are distinct keys."""
        cfg = AgentConfig(
            self._config_dir(
                tmp_path,
                {
                    "a.yaml": "model_id: vertex:8b\ntemperature: 0.1\n",
                    "b.yaml": "model_id: 8b\nprovider: vertex\ntemperature: 0.9\n",
                },
            )
        )
        cfg._ensure_loaded()
        assert len(cfg._model_configs) == 2
        assert cfg.get_model_params("vertex:8b")["temperature"] == 0.1
        assert cfg.get_model_params("8b", Provider.VERTEX)["temperature"] == 0.9
        # The bare entry must not pollute or ambiguate the real declaration.
        assert cfg.get_declared_provider("8b") == "vertex"

    def test_colon_name_still_scopes_by_provider(self, tmp_path):
        cfg = AgentConfig(
            self._config_dir(
                tmp_path,
                {
                    "a.yaml": (
                        "model_id: llama3:8b\nprovider: openai-compatibility\n"
                        "temperature: 0.2\n"
                    )
                },
            )
        )
        cfg._ensure_loaded()
        assert list(cfg._model_configs) == [("openai-compatibility", "llama3:8b")]
        assert cfg.get_declared_provider("llama3:8b") == "openai-compatibility"
        assert (
            cfg.get_model_params("llama3:8b", Provider.OPENAI_COMPATIBILITY)[
                "temperature"
            ]
            == 0.2
        )

    def test_non_string_model_id_is_coerced(self, tmp_path):
        """A YAML-numeric model_id must stay reachable by its string form."""
        cfg = AgentConfig(
            self._config_dir(tmp_path, {"a.yaml": "model_id: 123\nprovider: vertex\n"})
        )
        cfg._ensure_loaded()
        assert list(cfg._model_configs) == [("vertex", "123")]
        assert cfg.get_declared_provider("123") == "vertex"
