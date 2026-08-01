"""Unit tests for cache metrics."""

from unittest.mock import patch

from deep_agent.src.cache import metrics
from deep_agent.src.cache.config import CacheSettings


class TestCacheMetrics:
    def setup_method(self):
        metrics.reset()

    def test_record_and_snapshot(self):
        enabled = CacheSettings(CACHE_ENABLED=True, CACHE_METRICS_ENABLED=True)
        with patch.object(metrics, "cache_settings", enabled):
            metrics.record_hit("test")
            metrics.record_hit("test")
            metrics.record_miss("test")
            metrics.record_set("test")
            metrics.record_delete("test")

            snap = metrics.snapshot()
            assert snap["test"]["hits"] == 2
            assert snap["test"]["misses"] == 1
            assert snap["test"]["sets"] == 1
            assert snap["test"]["deletes"] == 1

    def test_disabled_does_not_record(self):
        disabled = CacheSettings(CACHE_ENABLED=False)
        with patch.object(metrics, "cache_settings", disabled):
            metrics.record_hit("test")
            assert metrics.snapshot() == {}

    def test_reset_clears_all(self):
        enabled = CacheSettings(CACHE_ENABLED=True, CACHE_METRICS_ENABLED=True)
        with patch.object(metrics, "cache_settings", enabled):
            metrics.record_hit("test")
            metrics.reset()
            assert metrics.snapshot() == {}

    def test_get_stats(self):
        enabled = CacheSettings(CACHE_ENABLED=True, CACHE_METRICS_ENABLED=True)
        with patch.object(metrics, "cache_settings", enabled):
            metrics.record_hit("x")
            metrics.record_miss("x")
            stats = metrics.get_stats()
            assert stats["x"]["total"] == 2
            assert stats["x"]["hit_rate"] == 50.0

    def test_log_summary_does_not_raise(self):
        enabled = CacheSettings(CACHE_ENABLED=True, CACHE_METRICS_ENABLED=True)
        with patch.object(metrics, "cache_settings", enabled):
            metrics.record_hit("x")
            metrics.log_summary()

    def test_log_summary_skips_when_disabled(self):
        disabled = CacheSettings(CACHE_ENABLED=False)
        with patch.object(metrics, "cache_settings", disabled):
            metrics.log_summary()
