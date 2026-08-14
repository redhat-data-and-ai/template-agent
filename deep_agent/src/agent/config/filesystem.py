"""Filesystem configuration models.

Provides validated Pydantic models for the ``filesystem:`` section of
config/agent/runtime/agent.yaml:
- Backend type selection (local_shell / state / composite)
- Filesystem permissions (operations + paths + mode)
- FilesystemMiddleware tuning (eviction thresholds, timeouts)

The template-agent user only touches YAML. This module converts
declarative config into parameters for the backend and create_deep_agent().
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


class LocalShellConfig(BaseModel):
    """Configuration for LocalShellBackend."""

    timeout: int = 120
    max_output_bytes: int = 100_000


class StateConfig(BaseModel):
    """Configuration for StateBackend (ephemeral in-memory)."""

    enabled: bool = False


class StoreConfig(BaseModel):
    """Configuration for StoreBackend (cross-thread persistent)."""

    enabled: bool = False
    scope: Literal["user", "assistant", "org"] = "user"


class BackendConfig(BaseModel):
    """Backend selection and configuration."""

    type: Literal["state", "composite", "store", "local_shell"] = "state"
    local_shell: LocalShellConfig = Field(default_factory=LocalShellConfig)
    state: StateConfig = Field(default_factory=StateConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    routes: dict[str, str] = Field(default_factory=dict)


class PermissionRule(BaseModel):
    """A single filesystem permission rule."""

    operations: list[str]
    paths: list[str]
    mode: Literal["allow", "deny"] = "allow"


class FilesystemSettings(BaseModel):
    """FilesystemMiddleware tuning parameters."""

    tool_token_limit_before_evict: int = 20_000
    human_message_token_limit_before_evict: int = 50_000
    max_execute_timeout: int = 3600


class FilesystemFileConfig(BaseModel):
    """Structure of the ``filesystem:`` section in runtime/agent.yaml."""

    backend: BackendConfig = Field(default_factory=BackendConfig)
    permissions: list[PermissionRule] = Field(default_factory=list)
    permission_inheritance: bool = False
    settings: FilesystemSettings = Field(default_factory=FilesystemSettings)


def load_filesystem_config(config_path: Path) -> FilesystemFileConfig:
    """Load and validate filesystem.yaml from disk.

    Args:
        config_path: Path to filesystem.yaml.

    Returns:
        Validated FilesystemFileConfig. Returns defaults if file is missing.
    """
    if not config_path.is_file():
        logger.info("No filesystem.yaml found — using defaults (local_shell)")
        return FilesystemFileConfig()

    try:
        raw = yaml.safe_load(config_path.read_text()) or {}
        config: FilesystemFileConfig = FilesystemFileConfig.model_validate(raw)
        logger.info(
            "Loaded filesystem config: backend=%s, %d permission rule(s)",
            config.backend.type,
            len(config.permissions),
        )
        return config
    except Exception as e:
        logger.warning("Failed to parse filesystem.yaml, using defaults: %s", e)
        return FilesystemFileConfig()
