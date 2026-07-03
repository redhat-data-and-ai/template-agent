"""Middleware configuration models and resolution logic.

Provides Pydantic models for the ``middleware:`` and ``harness_profiles:``
sections of config/agent/runtime/agent.yaml and resolves the final
middleware configuration for each agent by merging:

    global defaults → profile (matched from model field) → per-agent overrides

The template-agent user only touches YAML config. This module converts
declarative config into the parameters needed by the middleware builder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


class SummarizationToolConfig(BaseModel):
    """Config for SummarizationToolMiddleware."""

    enabled: bool = True


class HumanApprovalConfig(BaseModel):
    """Config for human-in-the-loop tool approval.

    When enabled, the agent pauses before executing any tool call and
    waits for the user to approve, reject, or always-allow it.
    Backed by deepagents HumanInTheLoopMiddleware via interrupt_on.
    """

    enabled: bool = False
    mode: Literal["all", "none"] = "all"
    exclude: list[str] = Field(default_factory=list)


class MemoryConfig(BaseModel):
    """Config for MemoryMiddleware (activated via memory= param)."""

    enabled: bool = True
    namespaces: list[str] = Field(default_factory=lambda: ["memories"])


class PatchToolCallsConfig(BaseModel):
    """Config for PatchToolCallsMiddleware (auto-included by deepagents)."""

    enabled: bool = True


class SkillsConfig(BaseModel):
    """Config for SkillsMiddleware (auto-included when skills= provided)."""

    enabled: bool = True


class ModelCallLimitConfig(BaseModel):
    """Config for ModelCallLimitMiddleware — cap LLM calls per run."""

    enabled: bool = True
    run_limit: int = 50


class ToolCallLimitConfig(BaseModel):
    """Config for ToolCallLimitMiddleware — cap tool calls per run."""

    enabled: bool = True
    run_limit: int = 200


class ModelRetryConfig(BaseModel):
    """Config for ModelRetryMiddleware — retry on transient failures."""

    enabled: bool = True
    max_retries: int = 3
    backoff_factor: float = 2.0
    initial_delay: float = 1.0


class ModelFallbackConfig(BaseModel):
    """Config for ModelFallbackMiddleware — switch to backup model."""

    enabled: bool = False
    fallback_model: str = ""


class ToolRetryConfig(BaseModel):
    """Config for ToolRetryMiddleware — retry specific tools."""

    enabled: bool = False
    max_retries: int = 2
    tools: list[str] = Field(default_factory=list)


class PIIRule(BaseModel):
    """A single PII detection rule."""

    type: str
    strategy: str = "redact"


class PIIConfig(BaseModel):
    """Config for PIIMiddleware — detect and handle PII."""

    enabled: bool = False
    rules: list[PIIRule] = Field(default_factory=list)


class MiddlewareDefaults(BaseModel):
    """Global middleware defaults from middleware.yaml."""

    summarization_tool: SummarizationToolConfig = Field(
        default_factory=SummarizationToolConfig
    )
    human_approval: HumanApprovalConfig = Field(default_factory=HumanApprovalConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    patch_tool_calls: PatchToolCallsConfig = Field(default_factory=PatchToolCallsConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    model_call_limit: ModelCallLimitConfig = Field(default_factory=ModelCallLimitConfig)
    tool_call_limit: ToolCallLimitConfig = Field(default_factory=ToolCallLimitConfig)
    model_retry: ModelRetryConfig = Field(default_factory=ModelRetryConfig)
    model_fallback: ModelFallbackConfig = Field(default_factory=ModelFallbackConfig)
    tool_retry: ToolRetryConfig = Field(default_factory=ToolRetryConfig)
    pii: PIIConfig = Field(default_factory=PIIConfig)
    extra: list[str] = Field(default_factory=list)


class ProfileConfig(BaseModel):
    """Per-model profile configuration for HarnessProfile registration."""

    excluded_middleware: list[str] = Field(default_factory=list)
    excluded_tools: list[str] = Field(default_factory=list)
    system_prompt_suffix: str = ""
    general_purpose_subagent: dict[str, Any] = Field(default_factory=dict)


class MiddlewareFileConfig(BaseModel):
    """Structure of the middleware + harness_profiles sections in runtime/agent.yaml."""

    defaults: MiddlewareDefaults = Field(default_factory=MiddlewareDefaults)
    profiles: dict[str, ProfileConfig] = Field(default_factory=dict)


class ResolvedMiddlewareConfig(BaseModel):
    """Final resolved config for a single agent after merge."""

    summarization_tool_enabled: bool = True
    human_approval: HumanApprovalConfig = Field(default_factory=HumanApprovalConfig)
    memory_enabled: bool = True
    memory_namespaces: list[str] = Field(default_factory=lambda: ["memories"])
    patch_tool_calls_enabled: bool = True
    skills_enabled: bool = True
    model_call_limit: ModelCallLimitConfig = Field(default_factory=ModelCallLimitConfig)
    tool_call_limit: ToolCallLimitConfig = Field(default_factory=ToolCallLimitConfig)
    model_retry: ModelRetryConfig = Field(default_factory=ModelRetryConfig)
    model_fallback: ModelFallbackConfig = Field(default_factory=ModelFallbackConfig)
    tool_retry: ToolRetryConfig = Field(default_factory=ToolRetryConfig)
    pii: PIIConfig = Field(default_factory=PIIConfig)
    extra_middleware: list[str] = Field(default_factory=list)
    excluded_middleware: list[str] = Field(default_factory=list)


def load_middleware_config(config_path: Path) -> MiddlewareFileConfig:
    """Load and validate middleware.yaml from disk.

    Args:
        config_path: Path to middleware.yaml.

    Returns:
        Validated MiddlewareFileConfig. Returns defaults if file is missing.
    """
    if not config_path.is_file():
        logger.info("No middleware.yaml found — using defaults")
        return MiddlewareFileConfig()

    try:
        raw = yaml.safe_load(config_path.read_text()) or {}
        config: MiddlewareFileConfig = MiddlewareFileConfig.model_validate(raw)
        logger.info("Loaded middleware config: %d profile(s)", len(config.profiles))
        return config
    except Exception as e:
        logger.warning("Failed to parse middleware.yaml, using defaults: %s", e)
        return MiddlewareFileConfig()


def resolve_middleware(
    file_config: MiddlewareFileConfig,
    model_name: str,
    agent_overrides: dict[str, Any] | None = None,
) -> ResolvedMiddlewareConfig:
    """Resolve final middleware config for an agent.

    Merge order: global defaults → profile (from model name) → agent overrides.

    Args:
        file_config: Parsed middleware.yaml config.
        model_name: Model name from agent frontmatter (used for profile lookup).
        agent_overrides: Optional middleware: block from agent frontmatter.

    Returns:
        Fully resolved middleware configuration for this agent.
    """
    defaults = file_config.defaults
    profile = file_config.profiles.get(model_name, ProfileConfig())
    overrides = agent_overrides or {}

    summarization_enabled = _resolve_bool(
        defaults.summarization_tool.enabled,
        overrides.get("summarization_tool"),
    )
    memory_enabled = _resolve_bool(
        defaults.memory.enabled,
        overrides.get("memory"),
    )
    patch_enabled = _resolve_bool(
        defaults.patch_tool_calls.enabled,
        overrides.get("patch_tool_calls"),
    )
    skills_enabled = _resolve_bool(
        defaults.skills.enabled,
        overrides.get("skills"),
    )

    memory_namespaces = defaults.memory.namespaces
    if isinstance(overrides.get("memory"), dict):
        memory_namespaces = overrides["memory"].get("namespaces", memory_namespaces)

    extra = list(defaults.extra)
    if "extra" in overrides:
        extra.extend(overrides["extra"])

    if "patch_tool_calls" in profile.excluded_middleware:
        patch_enabled = False

    human_approval = defaults.human_approval
    if isinstance(overrides.get("human_approval"), dict):
        human_approval = HumanApprovalConfig.model_validate(overrides["human_approval"])
    elif isinstance(overrides.get("human_approval"), bool):
        human_approval = HumanApprovalConfig(enabled=overrides["human_approval"])

    return ResolvedMiddlewareConfig(
        summarization_tool_enabled=summarization_enabled,
        human_approval=human_approval,
        memory_enabled=memory_enabled,
        memory_namespaces=memory_namespaces,
        patch_tool_calls_enabled=patch_enabled,
        skills_enabled=skills_enabled,
        model_call_limit=defaults.model_call_limit,
        tool_call_limit=defaults.tool_call_limit,
        model_retry=defaults.model_retry,
        model_fallback=defaults.model_fallback,
        tool_retry=defaults.tool_retry,
        pii=defaults.pii,
        extra_middleware=extra,
        excluded_middleware=profile.excluded_middleware,
    )


def _resolve_bool(default: bool, override: Any) -> bool:
    """Resolve a boolean config with potential override.

    Override can be: bool, dict with 'enabled' key, or None (use default).
    """
    if override is None:
        return default
    if isinstance(override, bool):
        return override
    if isinstance(override, dict):
        return bool(override.get("enabled", default))
    return default
