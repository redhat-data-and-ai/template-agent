"""LLM model instance cache.

Caches ``BaseChatModel`` instances by ``(model_name, temperature,
max_output_tokens)`` so repeated per-request calls to
``create_model()`` reuse the same client handle.

Model instances are **not** serialisable, so this is L1 (in-memory)
only — no Redis layer.

Feature flag: ``CACHE_MODEL_ENABLED`` (+ master ``CACHE_ENABLED``).
"""

import threading

from cachetools import TTLCache  # type: ignore[import-untyped]

from deep_agent.src.cache import metrics
from deep_agent.src.cache.config import cache_settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_CacheKey = tuple[str, float, int]

_lock = threading.Lock()
_cache: TTLCache[_CacheKey, object] | None = None


def _get_cache() -> TTLCache[_CacheKey, object]:
    global _cache  # noqa: PLW0603
    if _cache is None:
        _cache = TTLCache(
            maxsize=cache_settings.CACHE_MODEL_MAX_SIZE,
            ttl=cache_settings.CACHE_MODEL_TTL,
        )
    return _cache


def get_or_create_model(
    model_name: str,
    temperature: float = 0.0,
    max_output_tokens: int | None = None,
) -> object:
    """Return a cached model or create a new one.

    When the cache is disabled (flag off), this is a straight
    passthrough to ``create_model()``.

    Returns:
        A ``BaseChatModel`` instance.
    """
    from deep_agent.src.agent.llm import create_model
    from deep_agent.src.settings import settings

    tokens = max_output_tokens or settings.MAX_OUTPUT_TOKENS

    if not cache_settings.is_enabled("model"):
        return create_model(model_name, temperature, tokens)

    key: _CacheKey = (model_name, temperature, tokens)

    with _lock:
        cache = _get_cache()
        model = cache.get(key)
        if model is not None:
            metrics.record_hit("model")
            logger.debug("Model cache HIT: %s", model_name)
            return model

    metrics.record_miss("model")
    logger.debug("Model cache MISS: %s — creating", model_name)
    model = create_model(model_name, temperature, tokens)

    with _lock:
        cache = _get_cache()
        cache[key] = model
        metrics.record_set("model")

    return model


def invalidate(model_name: str | None = None) -> None:
    """Drop cached model(s).

    Args:
        model_name: If given, remove only entries for this model.
            If None, clear the entire model cache.
    """
    with _lock:
        cache = _get_cache()
        if model_name is None:
            cache.clear()
            logger.info("Model cache cleared")
            return
        keys_to_remove = [k for k in cache if k[0] == model_name]
        for k in keys_to_remove:
            del cache[k]
        if keys_to_remove:
            logger.info(
                "Model cache: evicted %d entry(s) for '%s'",
                len(keys_to_remove),
                model_name,
            )


def cached_count() -> int:
    """Return the number of currently cached models."""
    with _lock:
        return len(_get_cache())
