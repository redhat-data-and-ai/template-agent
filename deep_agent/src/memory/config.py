"""Memory management configuration with feature flags.

All memory background processing is disabled by default.
Enable via environment variables.

Environment variables:
    MEMORY_CONSOLIDATION_ENABLED:    Master switch (default: false)
    MEMORY_DECAY_ENABLED:            Exponential decay scoring (default: false)
    MEMORY_CLUSTERING_ENABLED:       Semantic clustering (default: false)
    MEMORY_RELATIONSHIPS_ENABLED:    Relationship inference (default: false)
    MEMORY_SCHEDULER_INTERVAL_HOURS: Job run interval (default: 6)
    MEMORY_MAX_INJECT:               Max memories injected into prompt (default: 20)
    MEMORY_DECAY_LAMBDA:             Decay rate — higher = faster fade (default: 0.05)
    MEMORY_CLUSTER_THRESHOLD:        Similarity threshold for clustering (default: 0.4)
    MEMORY_CONSOLIDATION_MIN_CLUSTER: Min cluster size to consolidate (default: 3)
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class MemorySettings(BaseSettings):
    """Feature-flagged memory management configuration."""

    MEMORY_CONSOLIDATION_ENABLED: bool = Field(default=False)
    MEMORY_DECAY_ENABLED: bool = Field(default=False)
    MEMORY_CLUSTERING_ENABLED: bool = Field(default=False)
    MEMORY_RELATIONSHIPS_ENABLED: bool = Field(default=False)

    MEMORY_SCHEDULER_INTERVAL_HOURS: int = Field(default=6, ge=1, le=168)
    MEMORY_MAX_INJECT: int = Field(default=20, ge=1, le=200)
    MEMORY_DECAY_LAMBDA: float = Field(default=0.05, ge=0.001, le=1.0)
    MEMORY_CLUSTER_THRESHOLD: float = Field(default=0.4, ge=0.1, le=0.95)
    MEMORY_CONSOLIDATION_MIN_CLUSTER: int = Field(default=3, ge=2, le=20)

    def is_enabled(self, layer: str) -> bool:
        """Check if a specific memory layer is active.

        Master switch must be on for any layer to activate.
        """
        if not self.MEMORY_CONSOLIDATION_ENABLED:
            return False
        flag = getattr(self, f"MEMORY_{layer.upper()}_ENABLED", False)
        return bool(flag)


memory_settings = MemorySettings()
