"""Unit tests for filesystem configuration and permissions builder."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deep_agent.src.agent.config.filesystem import (
    BackendConfig,
    FilesystemFileConfig,
    FilesystemSettings,
    LocalShellConfig,
    PermissionRule,
    StateConfig,
    load_filesystem_config,
)
from deep_agent.src.infrastructure.permissions import build_permissions


class TestFilesystemModels:
    """Test Pydantic model defaults."""

    def test_default_backend_is_state(self):
        config = FilesystemFileConfig()
        assert config.backend.type == "state"

    def test_default_permissions_empty(self):
        config = FilesystemFileConfig()
        assert config.permissions == []

    def test_default_settings(self):
        settings = FilesystemSettings()
        assert settings.tool_token_limit_before_evict == 20_000
        assert settings.human_message_token_limit_before_evict == 50_000
        assert settings.max_execute_timeout == 3600

    def test_local_shell_defaults(self):
        ls = LocalShellConfig()
        assert ls.timeout == 120
        assert ls.max_output_bytes == 100_000

    def test_state_default_disabled(self):
        state = StateConfig()
        assert state.enabled is False

    def test_permission_rule_defaults_to_allow(self):
        rule = PermissionRule(operations=["read"], paths=["**"])
        assert rule.mode == "allow"


class TestLoadFilesystemConfig:
    """Test loading filesystem.yaml from disk."""

    def test_returns_defaults_when_missing(self, tmp_path):
        config = load_filesystem_config(tmp_path / "nope.yaml")
        assert config.backend.type == "state"
        assert config.permissions == []

    def test_loads_valid_yaml(self, tmp_path):
        content = """
backend:
  type: composite
  local_shell:
    timeout: 60
    max_output_bytes: 50000
  routes:
    "/scratch/": state
    "/": local_shell

permissions:
  - operations: [read, glob]
    paths: ["config/**"]
    mode: allow
  - operations: [write]
    paths: ["**/*.py"]
    mode: deny

settings:
  tool_token_limit_before_evict: 10000
  max_execute_timeout: 1800
"""
        config_file = tmp_path / "filesystem.yaml"
        config_file.write_text(content)

        config = load_filesystem_config(config_file)
        assert config.backend.type == "composite"
        assert config.backend.local_shell.timeout == 60
        assert config.backend.routes == {"/scratch/": "state", "/": "local_shell"}
        assert len(config.permissions) == 2
        assert config.permissions[0].operations == ["read", "glob"]
        assert config.permissions[1].mode == "deny"
        assert config.settings.tool_token_limit_before_evict == 10_000

    def test_returns_defaults_on_invalid_yaml(self, tmp_path):
        config_file = tmp_path / "filesystem.yaml"
        config_file.write_text("{{invalid")
        config = load_filesystem_config(config_file)
        assert config.backend.type == "state"


class TestBuildPermissions:
    """Test FilesystemPermission construction from config."""

    def test_returns_none_when_no_rules(self):
        config = FilesystemFileConfig()
        result = build_permissions(config)
        assert result is None

    def test_builds_permission_objects(self):
        config = FilesystemFileConfig(
            permissions=[
                PermissionRule(
                    operations=["read", "glob", "grep"],
                    paths=["config/**"],
                    mode="allow",
                ),
                PermissionRule(
                    operations=["write", "edit"],
                    paths=["**/*.py"],
                    mode="deny",
                ),
            ]
        )

        mock_perm = MagicMock()
        with patch(
            "deep_agent.src.infrastructure.permissions.FilesystemPermission",
            return_value=mock_perm,
        ) as mock_cls:
            result = build_permissions(config)

        assert result is not None
        assert len(result) == 2
        assert mock_cls.call_count == 2
        mock_cls.assert_any_call(
            operations=["read", "glob", "grep"],
            paths=["config/**"],
            mode="allow",
        )

    def test_skips_invalid_rules_gracefully(self):
        config = FilesystemFileConfig(
            permissions=[
                PermissionRule(operations=["read"], paths=["ok/**"], mode="allow"),
            ]
        )

        with patch(
            "deep_agent.src.infrastructure.permissions.FilesystemPermission",
            side_effect=ValueError("bad rule"),
        ):
            result = build_permissions(config)

        assert result is None
