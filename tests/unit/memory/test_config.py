"""Unit tests for memory configuration."""

from deep_agent.src.memory.config import MemorySettings


class TestMemorySettings:
    def test_defaults_all_disabled(self):
        s = MemorySettings(
            MEMORY_CONSOLIDATION_ENABLED=False,
            MEMORY_DECAY_ENABLED=False,
            MEMORY_CLUSTERING_ENABLED=False,
            MEMORY_RELATIONSHIPS_ENABLED=False,
        )
        assert s.MEMORY_CONSOLIDATION_ENABLED is False
        assert s.MEMORY_DECAY_ENABLED is False

    def test_is_enabled_requires_master(self):
        s = MemorySettings(
            MEMORY_CONSOLIDATION_ENABLED=False,
            MEMORY_DECAY_ENABLED=True,
        )
        assert s.is_enabled("decay") is False

    def test_is_enabled_with_master_on(self):
        s = MemorySettings(
            MEMORY_CONSOLIDATION_ENABLED=True,
            MEMORY_DECAY_ENABLED=True,
        )
        assert s.is_enabled("decay") is True

    def test_is_enabled_unknown_layer(self):
        s = MemorySettings(MEMORY_CONSOLIDATION_ENABLED=True)
        assert s.is_enabled("nonexistent") is False

    def test_defaults(self):
        s = MemorySettings()
        assert s.MEMORY_MAX_INJECT == 20
        assert s.MEMORY_DECAY_LAMBDA == 0.05
        assert s.MEMORY_SCHEDULER_INTERVAL_HOURS == 6
