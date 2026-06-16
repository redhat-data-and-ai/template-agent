"""Cache warming — pre-populate caches at startup.

When ``CACHE_WARMING_ENABLED`` is true, ``warm_caches()`` pre-creates
the default orchestrator and subagent LLM model instances so the first
user request doesn't pay the cold-start penalty.

Feature flag: ``CACHE_WARMING_ENABLED`` (+ ``CACHE_ENABLED``).
"""

from deep_agent.src.cache.config import cache_settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


def warm_caches() -> dict[str, bool]:
    """Pre-populate caches. Returns a status dict per cache layer.

    Safe to call even when caching is disabled — returns immediately.
    """
    results: dict[str, bool] = {}

    if not cache_settings.is_enabled("warming"):
        logger.debug("Cache warming disabled — skipping")
        return results

    logger.info("Warming caches...")

    results["models"] = _warm_models()

    logger.info("Cache warming complete: %s", results)
    return results


def _warm_models() -> bool:
    """Pre-create LLM model instances for orchestrator + subagents."""
    if not cache_settings.is_enabled("model"):
        return False

    try:
        from deep_agent.src.agent.config import agent_config
        from deep_agent.src.agent.config.model import parse_model_config
        from deep_agent.src.cache.model_cache import get_or_create_model_from_spec

        orch = agent_config.get_orchestrator_config()
        orch_model = orch.get("model", "gemini-3.1-pro-preview")

        # Parse orchestrator model to ModelSpec (supports provider)
        orch_spec = parse_model_config(orch_model)
        get_or_create_model_from_spec(orch_spec)
        logger.info(
            "Warmed orchestrator model: %s (provider: %s)",
            orch_spec.name,
            orch_spec.provider.value,
        )

        for name, cfg in agent_config.get_all_subagent_configs().items():
            sub_model = cfg.get("model")
            if sub_model:
                spec = parse_model_config(sub_model)
                get_or_create_model_from_spec(spec)
                logger.info(
                    "Warmed subagent '%s' model: %s", name, spec.display_name()
                )

        return True
    except Exception:
        logger.warning("Model cache warming failed", exc_info=True)
        return False
