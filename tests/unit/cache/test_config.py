"""Unit tests for cache configuration."""

from deep_agent.src.cache.config import CacheSettings


class TestCacheSettings:
    def test_defaults_all_disabled(self):
        s = CacheSettings(
            CACHE_ENABLED=False,
            CACHE_MODEL_ENABLED=False,
            CACHE_PERSONALIZATION_ENABLED=False,
            CACHE_METRICS_ENABLED=False,
            CACHE_WARMING_ENABLED=False,
            CACHE_REDIS_ENABLED=False,
        )
        assert s.CACHE_ENABLED is False
        assert s.CACHE_MODEL_ENABLED is False
        assert s.CACHE_PERSONALIZATION_ENABLED is False

    def test_is_enabled_requires_master_switch(self):
        s = CacheSettings(CACHE_ENABLED=False, CACHE_MODEL_ENABLED=True)
        assert s.is_enabled("model") is False

    def test_is_enabled_with_master_on(self):
        s = CacheSettings(CACHE_ENABLED=True, CACHE_MODEL_ENABLED=True)
        assert s.is_enabled("model") is True

    def test_is_enabled_unknown_layer(self):
        s = CacheSettings(CACHE_ENABLED=True)
        assert s.is_enabled("nonexistent") is False

    def test_ttl_defaults(self):
        s = CacheSettings()
        assert s.CACHE_MODEL_TTL == 600
        assert s.CACHE_PERSONALIZATION_TTL == 120

    def test_max_size_defaults(self):
        s = CacheSettings()
        assert s.CACHE_MODEL_MAX_SIZE == 10
