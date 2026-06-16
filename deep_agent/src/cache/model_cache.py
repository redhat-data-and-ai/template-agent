"""LLM model instance cache.

Caches ``BaseChatModel`` instances by ``(model_name, temperature,
max_output_tokens)`` or by ``(spec_cache_key, temperature, tokens)``
for provider-aware specs so repeated per-request calls reuse the same
client handle.

Model instances are **not** serialisable, so this is L1 (in-memory)
only — no Redis layer.

**Memory usage**: Two separate caches exist (legacy string-based and spec-based),
each limited to ``CACHE_MODEL_MAX_SIZE`` entries. Maximum total memory usage is
2x the configured limit (e.g., if limit is 100, up to 200 models may be cached).

Feature flag: ``CACHE_MODEL_ENABLED`` (+ master ``CACHE_ENABLED``).
"""

import threading

from cachetools import TTLCache  # type: ignore[import-untyped]

from deep_agent.src.agent.config.model import ModelSpec, model_spec_cache_key
from deep_agent.src.cache import metrics
from deep_agent.src.cache.config import cache_settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_LegacyCacheKey = tuple[str, float, int]
_SpecCacheKey = tuple[str, float, int]  # (cache_id, temperature, tokens)

_lock = threading.Lock()
_legacy_cache: TTLCache[_LegacyCacheKey, object] | None = None
_spec_cache: TTLCache[_SpecCacheKey, object] | None = None


def _get_legacy_cache() -> TTLCache[_LegacyCacheKey, object]:
    global _legacy_cache  # noqa: PLW0603
    if _legacy_cache is None:
        _legacy_cache = TTLCache(
            maxsize=cache_settings.CACHE_MODEL_MAX_SIZE,
            ttl=cache_settings.CACHE_MODEL_TTL,
        )
    return _legacy_cache


def _get_spec_cache() -> TTLCache[_SpecCacheKey, object]:
    global _spec_cache  # noqa: PLW0603
    if _spec_cache is None:
        _spec_cache = TTLCache(
            maxsize=cache_settings.CACHE_MODEL_MAX_SIZE,
            ttl=cache_settings.CACHE_MODEL_TTL,
        )
    return _spec_cache


def _get_cache() -> TTLCache[_LegacyCacheKey, object]:
    """Backward-compatible alias for legacy cache getter.

    Used by both tests and production code that still uses string-based model names.
    """
    return _get_legacy_cache()


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

    key: _LegacyCacheKey = (model_name, temperature, tokens)

    with _lock:
        cache = _get_legacy_cache()
        model = cache.get(key)
        if model is not None:
            metrics.record_hit("model")
            logger.debug("Model cache HIT: %s", model_name)
            return model

    metrics.record_miss("model")
    logger.debug("Model cache MISS: %s — creating", model_name)
    model = create_model(model_name, temperature, tokens)

    with _lock:
        cache = _get_legacy_cache()
        cache[key] = model
        metrics.record_set("model")

    return model


def get_or_create_model_from_spec(
    spec: ModelSpec,
    temperature: float = 0.0,
    max_output_tokens: int | None = None,
) -> object:
    """Return a cached model for a :class:`ModelSpec` or create a new one.

    Cache key includes provider, model name, and fallback chain so
    different provider configurations never collide.

    Returns:
        A ``BaseChatModel`` instance.
    """
    from deep_agent.src.agent.provider_factory import create_model_from_spec
    from deep_agent.src.settings import settings

    tokens = max_output_tokens or settings.MAX_OUTPUT_TOKENS
    cache_id = model_spec_cache_key(spec)

    if not cache_settings.is_enabled("model"):
        return create_model_from_spec(
            spec, temperature=temperature, max_output_tokens=tokens
        )

    key: _SpecCacheKey = (cache_id, temperature, tokens)

    with _lock:
        cache = _get_spec_cache()
        model = cache.get(key)
        if model is not None:
            metrics.record_hit("model")
            logger.debug("Model cache HIT: %s", cache_id)
            return model

    metrics.record_miss("model")
    logger.debug("Model cache MISS: %s — creating", cache_id)
    model = create_model_from_spec(
        spec, temperature=temperature, max_output_tokens=tokens
    )

    with _lock:
        cache = _get_spec_cache()
        cache[key] = model
        metrics.record_set("model")

    return model


def invalidate(model_name: str | None = None) -> None:
    """Drop cached model(s).

    Args:
        model_name: If given, remove only legacy-cache entries for this model.
            If None, clear both legacy and spec caches.
    """
    with _lock:
        legacy = _get_legacy_cache()
        spec = _get_spec_cache()
        if model_name is None:
            legacy.clear()
            spec.clear()
            logger.info("Model cache cleared")
            return
        keys_to_remove = [k for k in legacy if k[0] == model_name]
        for k in keys_to_remove:
            del legacy[k]
        if keys_to_remove:
            logger.info(
                "Model cache: evicted %d legacy entry(s) for '%s'",
                len(keys_to_remove),
                model_name,
            )


def cached_count() -> int:
    """Return the number of currently cached models (legacy + spec)."""
    with _lock:
        return len(_get_legacy_cache()) + len(_get_spec_cache())
