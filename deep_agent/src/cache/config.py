"""Cache configuration with feature flags.

Every cache layer is disabled by default. Enable via environment
variables — the master ``CACHE_ENABLED`` switch must be ``true``
for any individual cache to activate.

Environment variables:
    CACHE_ENABLED:                 Master switch (default: false)
    CACHE_MODEL_ENABLED:           LLM model instance cache (default: false)
    CACHE_MODEL_TTL:               Model cache TTL in seconds (default: 600)
    CACHE_MODEL_MAX_SIZE:          Max cached model instances (default: 10)
    CACHE_PERSONALIZATION_ENABLED: User personalization cache (default: false)
    CACHE_PERSONALIZATION_TTL:     Personalization TTL in seconds (default: 120)
    CACHE_METRICS_ENABLED:         Log cache hit/miss counters (default: false)
    CACHE_WARMING_ENABLED:         Pre-create models at startup (default: false)
    CACHE_REDIS_ENABLED:           Enable Redis as L2 cache layer (default: false)

Note:
    MCP tool cache TTL and compiled graph cache TTL are configured via
    config/agent/runtime/agent.yaml (cache.mcp.ttl, cache.graph.ttl),
    NOT via environment variables.
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class CacheSettings(BaseSettings):
    """Feature-flagged cache configuration loaded from environment."""

    CACHE_ENABLED: bool = Field(default=False)

    CACHE_MODEL_ENABLED: bool = Field(default=False)
    CACHE_MODEL_TTL: int = Field(default=600, ge=10, le=7200)
    CACHE_MODEL_MAX_SIZE: int = Field(default=10, ge=1, le=100)

    CACHE_PERSONALIZATION_ENABLED: bool = Field(default=False)
    CACHE_PERSONALIZATION_TTL: int = Field(default=120, ge=10, le=3600)

    CACHE_METRICS_ENABLED: bool = Field(default=False)
    CACHE_WARMING_ENABLED: bool = Field(default=False)
    CACHE_REDIS_ENABLED: bool = Field(default=False)

    def is_enabled(self, layer: str) -> bool:
        """Check if a specific cache layer is active.

        Both the master switch and the layer-specific flag must be true.
        """
        if not self.CACHE_ENABLED:
            return False
        flag = getattr(self, f"CACHE_{layer.upper()}_ENABLED", False)
        return bool(flag)


cache_settings = CacheSettings()
