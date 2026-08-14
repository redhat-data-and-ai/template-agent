"""Unit tests for middleware configuration resolution."""

from pathlib import Path
from unittest.mock import patch

import pytest

from deep_agent.src.agent.config.middleware import (
    MemoryConfig,
    MiddlewareDefaults,
    MiddlewareFileConfig,
    PatchToolCallsConfig,
    ProfileConfig,
    ResolvedMiddlewareConfig,
    SkillsConfig,
    SummarizationToolConfig,
    load_middleware_config,
    resolve_middleware,
)


class TestMiddlewareModels:
    """Test Pydantic model defaults and validation."""

    def test_defaults_all_enabled(self):
        defaults = MiddlewareDefaults()
        assert defaults.summarization_tool.enabled is True
        assert defaults.memory.enabled is True
        assert defaults.patch_tool_calls.enabled is True
        assert defaults.skills.enabled is True
        assert defaults.extra == []

    def test_memory_default_namespaces(self):
        config = MemoryConfig()
        assert config.namespaces == ["memories"]

    def test_profile_defaults_empty(self):
        profile = ProfileConfig()
        assert profile.excluded_middleware == []
        assert profile.excluded_tools == []
        assert profile.system_prompt_suffix == ""

    def test_file_config_defaults(self):
        config = MiddlewareFileConfig()
        assert config.defaults.summarization_tool.enabled is True
        assert config.profiles == {}


class TestLoadMiddlewareConfig:
    """Test loading middleware.yaml from disk."""

    def test_returns_defaults_when_file_missing(self, tmp_path):
        config = load_middleware_config(tmp_path / "nonexistent.yaml")
        assert config.defaults.summarization_tool.enabled is True
        assert config.profiles == {}

    def test_loads_valid_yaml(self, tmp_path):
        yaml_content = """
defaults:
  summarization_tool:
    enabled: false
  memory:
    enabled: true
    namespaces:
      - user_memories
      - shared
profiles:
  gemini-2.5-pro:
    excluded_middleware:
      - patch_tool_calls
    system_prompt_suffix: "Be helpful."
"""
        config_file = tmp_path / "middleware.yaml"
        config_file.write_text(yaml_content)

        config = load_middleware_config(config_file)
        assert config.defaults.summarization_tool.enabled is False
        assert config.defaults.memory.namespaces == ["user_memories", "shared"]
        assert "gemini-2.5-pro" in config.profiles
        assert config.profiles["gemini-2.5-pro"].system_prompt_suffix == "Be helpful."

    def test_returns_defaults_on_invalid_yaml(self, tmp_path):
        config_file = tmp_path / "middleware.yaml"
        config_file.write_text("not: [valid: yaml: {{")

        config = load_middleware_config(config_file)
        assert config.defaults.summarization_tool.enabled is True


class TestResolveMiddleware:
    """Test the resolution logic: defaults → profile → overrides."""

    def test_all_defaults_no_profile_no_overrides(self):
        config = MiddlewareFileConfig()
        resolved = resolve_middleware(config, "unknown-model")

        assert resolved.summarization_tool_enabled is True
        assert resolved.memory_enabled is True
        assert resolved.patch_tool_calls_enabled is True
        assert resolved.skills_enabled is True
        assert resolved.memory_namespaces == ["memories"]

    def test_profile_excludes_patch_tool_calls(self):
        config = MiddlewareFileConfig(
            profiles={
                "claude-sonnet": ProfileConfig(excluded_middleware=["patch_tool_calls"])
            }
        )
        resolved = resolve_middleware(config, "claude-sonnet")
        assert resolved.patch_tool_calls_enabled is False

    def test_agent_override_disables_memory(self):
        config = MiddlewareFileConfig()
        resolved = resolve_middleware(config, "gemini-2.5-pro", {"memory": False})
        assert resolved.memory_enabled is False

    def test_agent_override_dict_with_enabled(self):
        config = MiddlewareFileConfig()
        overrides = {"summarization_tool": {"enabled": False}}
        resolved = resolve_middleware(config, "gemini-2.5-pro", overrides)
        assert resolved.summarization_tool_enabled is False

    def test_agent_override_memory_namespaces(self):
        config = MiddlewareFileConfig()
        overrides = {"memory": {"enabled": True, "namespaces": ["custom_ns"]}}
        resolved = resolve_middleware(config, "gemini-2.5-pro", overrides)
        assert resolved.memory_enabled is True
        assert resolved.memory_namespaces == ["custom_ns"]

    def test_extra_middleware_merged(self):
        config = MiddlewareFileConfig(
            defaults=MiddlewareDefaults(extra=["module_a:ClassA"])
        )
        overrides = {"extra": ["module_b:ClassB"]}
        resolved = resolve_middleware(config, "model", overrides)
        assert resolved.extra_middleware == ["module_a:ClassA", "module_b:ClassB"]

    def test_global_disabled_respected(self):
        config = MiddlewareFileConfig(
            defaults=MiddlewareDefaults(
                summarization_tool=SummarizationToolConfig(enabled=False),
                memory=MemoryConfig(enabled=False),
            )
        )
        resolved = resolve_middleware(config, "model")
        assert resolved.summarization_tool_enabled is False
        assert resolved.memory_enabled is False
