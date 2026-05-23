"""Provider and harness profile configuration models.

Provides Pydantic models for the ``providers:``, ``harness_profiles:``,
and ``async_tasks:`` sections of config/agent/runtime/agent.yaml:
- Model resolution strategy (legacy vs deepagents)
- ProviderProfile registration (init_chat_model kwargs per provider)
- HarnessProfile registration (runtime adjustments per model)
- Async task middleware configuration

Users edit YAML. This module validates and converts to typed config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


class ProviderConfig(BaseModel):
    """Configuration for a single provider (maps to ProviderProfile)."""

    init_kwargs: dict[str, Any] = Field(default_factory=dict)


class GeneralPurposeSubagentConfig(BaseModel):
    """Config for the auto-added general-purpose subagent."""

    enabled: bool = True
    description: str | None = None
    system_prompt: str | None = None


class HarnessProfileConfig(BaseModel):
    """Configuration for a single harness profile (maps to HarnessProfile)."""

    system_prompt_suffix: str = ""
    excluded_tools: list[str] = Field(default_factory=list)
    excluded_middleware: list[str] = Field(default_factory=list)
    general_purpose_subagent: GeneralPurposeSubagentConfig = Field(
        default_factory=GeneralPurposeSubagentConfig,
    )


class AsyncTaskConfig(BaseModel):
    """Configuration for AsyncSubAgentMiddleware."""

    enabled: bool = True
    system_prompt: str | None = None


class ProvidersFileConfig(BaseModel):
    """Structure of the providers + harness_profiles sections in runtime/agent.yaml."""

    resolve_strategy: Literal["legacy", "deepagents"] = "legacy"
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    harness_profiles: dict[str, HarnessProfileConfig] = Field(default_factory=dict)
    async_tasks: AsyncTaskConfig = Field(default_factory=AsyncTaskConfig)


def load_providers_config(config_path: Path) -> ProvidersFileConfig:
    """Load and validate providers.yaml from disk.

    Args:
        config_path: Path to providers.yaml.

    Returns:
        Validated ProvidersFileConfig. Returns defaults if file is missing.
    """
    if not config_path.is_file():
        logger.info("No providers.yaml found — using defaults (legacy resolution)")
        return ProvidersFileConfig()

    try:
        raw = yaml.safe_load(config_path.read_text()) or {}
        config: ProvidersFileConfig = ProvidersFileConfig.model_validate(raw)
        logger.info(
            "Loaded providers config: strategy=%s, %d provider(s), %d harness profile(s)",
            config.resolve_strategy,
            len(config.providers),
            len(config.harness_profiles),
        )
        return config
    except Exception as e:
        logger.warning("Failed to parse providers.yaml, using defaults: %s", e)
        return ProvidersFileConfig()
