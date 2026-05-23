"""Unit tests for multi-layer cache."""

from unittest.mock import patch

from deep_agent.src.cache.backend import InMemoryCache, NullCache
from deep_agent.src.cache.multi_layer import MultiLayerCache, create_null_layer


class TestMultiLayerCache:
    def test_l1_hit(self):
        l1 = InMemoryCache(max_size=10, default_ttl=60)
        l1.set("k", "v1")
        ml = MultiLayerCache("test", l1=l1)
        assert ml.get("k") == "v1"

    def test_l2_hit_backfills_l1(self):
        l1 = InMemoryCache(max_size=10, default_ttl=60)
        l2 = InMemoryCache(max_size=10, default_ttl=60)
        l2.set("k", "from-l2")
        ml = MultiLayerCache("test", l1=l1, l2=l2)

        assert ml.get("k") == "from-l2"
        assert l1.get("k") == "from-l2"

    def test_miss_both_layers(self):
        l1 = InMemoryCache(max_size=10, default_ttl=60)
        l2 = InMemoryCache(max_size=10, default_ttl=60)
        ml = MultiLayerCache("test", l1=l1, l2=l2)
        assert ml.get("missing") is None

    def test_set_writes_both(self):
        l1 = InMemoryCache(max_size=10, default_ttl=60)
        l2 = InMemoryCache(max_size=10, default_ttl=60)
        ml = MultiLayerCache("test", l1=l1, l2=l2)

        ml.set("k", "v")
        assert l1.get("k") == "v"
        assert l2.get("k") == "v"

    def test_delete_both(self):
        l1 = InMemoryCache(max_size=10, default_ttl=60)
        l2 = InMemoryCache(max_size=10, default_ttl=60)
        ml = MultiLayerCache("test", l1=l1, l2=l2)

        ml.set("k", "v")
        ml.delete("k")
        assert l1.get("k") is None
        assert l2.get("k") is None

    def test_clear_only_l1(self):
        l1 = InMemoryCache(max_size=10, default_ttl=60)
        l2 = InMemoryCache(max_size=10, default_ttl=60)
        ml = MultiLayerCache("test", l1=l1, l2=l2)
        ml.set("k", "v")
        ml.clear()
        assert l1.get("k") is None
        assert l2.get("k") == "v"

    def test_name(self):
        ml = MultiLayerCache("my-cache", l1=NullCache())
        assert ml.name == "my-cache"

    def test_no_l2(self):
        l1 = InMemoryCache(max_size=10, default_ttl=60)
        ml = MultiLayerCache("test", l1=l1, l2=None)
        ml.set("k", "v")
        assert ml.get("k") == "v"


class TestCreateNullLayer:
    def test_returns_noop(self):
        ml = create_null_layer("disabled")
        assert ml.get("k") is None
        assert ml.set("k", "v") is False
