"""Multi-layer caching for the template agent.

All cache layers are **disabled by default** and activated via
environment variables. Set ``CACHE_ENABLED=true`` plus individual
layer flags (``CACHE_MODEL_ENABLED``, ``CACHE_PERSONALIZATION_ENABLED``,
etc.) to opt in.

Exports:
    cache_settings:         Configuration singleton (feature flags + TTLs)
    get_or_create_model:    Cached LLM model factory
    warm_caches:            Startup cache warming
    metrics:                Hit/miss/set counters
"""

from deep_agent.src.cache.config import cache_settings

__all__ = [
    "cache_settings",
]
