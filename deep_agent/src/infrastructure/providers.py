"""Provider and harness profile registration.

Reads the validated ProvidersFileConfig and registers ProviderProfile
and HarnessProfile instances with the deepagents profile registry.

Also provides resolve_model_from_config() which picks between the
legacy create_model() path and deepagents resolve_model() based on
the resolve_strategy setting.

Template-agent users never call this directly — it's wired by graph.py
and factory.py at agent creation time.
"""

from __future__ import annotations

from typing import Any

from deep_agent.src.agent.config.providers import ProvidersFileConfig
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_profiles_registered: bool = False


def register_profiles_from_config(config: ProvidersFileConfig) -> None:
    """Register ProviderProfile and HarnessProfile instances from config.

    Idempotent — only registers once per process lifetime.

    Args:
        config: Validated providers.yaml config.
    """
    global _profiles_registered  # noqa: PLW0603
    if _profiles_registered:
        return

    _register_provider_profiles(config)
    _register_harness_profiles(config)
    _profiles_registered = True


def resolve_model_from_config(
    model_name: str,
    config: ProvidersFileConfig,
    *,
    temperature: float = 0.0,
    max_output_tokens: int | None = None,
) -> Any:
    """Resolve a model string to a BaseChatModel using the configured strategy.

    Args:
        model_name: Model name (e.g., "gemini-2.5-pro" or "openai:gpt-5.4").
        config: Validated providers config.
        temperature: Model temperature.
        max_output_tokens: Maximum output tokens.

    Returns:
        A BaseChatModel instance.
    """
    if config.resolve_strategy == "deepagents":
        return _resolve_via_deepagents(model_name)

    return _resolve_via_legacy(model_name, temperature, max_output_tokens)


def _resolve_via_legacy(
    model_name: str,
    temperature: float,
    max_output_tokens: int | None,
) -> Any:
    """Legacy resolution — use our hardcoded create_model() factory."""
    from deep_agent.src.cache.model_cache import get_or_create_model

    return get_or_create_model(
        model_name=model_name,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


def _resolve_via_deepagents(model_name: str) -> Any:
    """Deepagents resolution — use resolve_model() with registered profiles."""
    try:
        from deepagents import resolve_model

        logger.info("Resolving model via deepagents: %s", model_name)
        return resolve_model(model_name)
    except ImportError:
        logger.warning(
            "deepagents.resolve_model not available — falling back to legacy"
        )
        from deep_agent.src.cache.model_cache import get_or_create_model

        return get_or_create_model(model_name=model_name)


def _register_provider_profiles(config: ProvidersFileConfig) -> None:
    """Register ProviderProfile instances for each configured provider."""
    if not config.providers:
        return

    try:
        from deepagents import ProviderProfile, register_provider_profile
    except ImportError:
        logger.debug("deepagents profiles API not available — skipping registration")
        return

    for provider_key, provider_cfg in config.providers.items():
        try:
            profile = ProviderProfile(init_kwargs=provider_cfg.init_kwargs)
            register_provider_profile(provider_key, profile)
            logger.info("Registered ProviderProfile: %s", provider_key)
        except Exception as e:
            logger.warning(
                "Failed to register ProviderProfile '%s': %s", provider_key, e
            )


def _register_harness_profiles(config: ProvidersFileConfig) -> None:
    """Register HarnessProfile instances for each configured model."""
    if not config.harness_profiles:
        return

    try:
        from deepagents import (
            GeneralPurposeSubagentProfile,
            HarnessProfile,
            register_harness_profile,
        )
    except ImportError:
        logger.debug("deepagents profiles API not available — skipping registration")
        return

    for model_key, harness_cfg in config.harness_profiles.items():
        try:
            gp_config = harness_cfg.general_purpose_subagent
            gp_profile = GeneralPurposeSubagentProfile(
                enabled=gp_config.enabled,
                description=gp_config.description,
                system_prompt=gp_config.system_prompt,
            )

            profile = HarnessProfile(
                system_prompt_suffix=harness_cfg.system_prompt_suffix or None,
                excluded_tools=frozenset(harness_cfg.excluded_tools),
                excluded_middleware=frozenset(harness_cfg.excluded_middleware),
                general_purpose_subagent=gp_profile,
            )
            register_harness_profile(model_key, profile)
            logger.info("Registered HarnessProfile: %s", model_key)
        except Exception as e:
            logger.warning("Failed to register HarnessProfile '%s': %s", model_key, e)
