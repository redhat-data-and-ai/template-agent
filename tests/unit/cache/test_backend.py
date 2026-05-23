"""Unit tests for cache backend implementations."""

from unittest.mock import MagicMock, patch

import pytest

from deep_agent.src.cache.backend import (
    CacheBackend,
    InMemoryCache,
    NullCache,
    RedisCache,
)


class TestNullCache:
    def test_get_always_none(self):
        c = NullCache()
        assert c.get("any-key") is None

    def test_set_always_false(self):
        c = NullCache()
        assert c.set("k", "v") is False

    def test_delete_always_false(self):
        c = NullCache()
        assert c.delete("k") is False

    def test_clear_is_noop(self):
        NullCache().clear()

    def test_name(self):
        assert NullCache().name == "null"

    def test_implements_protocol(self):
        assert isinstance(NullCache(), CacheBackend)


class TestInMemoryCache:
    def test_set_and_get(self):
        c = InMemoryCache(max_size=10, default_ttl=60)
        c.set("k1", "v1")
        assert c.get("k1") == "v1"

    def test_get_miss(self):
        c = InMemoryCache()
        assert c.get("missing") is None

    def test_delete_existing(self):
        c = InMemoryCache()
        c.set("k1", "v1")
        assert c.delete("k1") is True
        assert c.get("k1") is None

    def test_delete_missing(self):
        c = InMemoryCache()
        assert c.delete("nope") is False

    def test_clear(self):
        c = InMemoryCache()
        c.set("a", "1")
        c.set("b", "2")
        c.clear()
        assert c.size == 0

    def test_size(self):
        c = InMemoryCache(max_size=10, default_ttl=60)
        c.set("a", "1")
        c.set("b", "2")
        assert c.size == 2

    def test_name(self):
        assert InMemoryCache().name == "memory"

    def test_implements_protocol(self):
        assert isinstance(InMemoryCache(), CacheBackend)


class TestRedisCache:
    def test_name(self):
        assert RedisCache().name == "redis"

    def test_get_returns_none_when_no_client(self):
        c = RedisCache()
        with patch.object(c, "_get_client", return_value=None):
            assert c.get("key") is None

    def test_set_returns_false_when_no_client(self):
        c = RedisCache()
        with patch.object(c, "_get_client", return_value=None):
            assert c.set("k", "v") is False

    def test_delete_returns_false_when_no_client(self):
        c = RedisCache()
        with patch.object(c, "_get_client", return_value=None):
            assert c.delete("k") is False

    def test_get_with_client(self):
        c = RedisCache(key_prefix="test:")
        mock = MagicMock()
        mock.get.return_value = "cached"
        c._client = mock
        c._checked = True
        assert c.get("k") == "cached"
        mock.get.assert_called_once_with("test:k")

    def test_set_with_client(self):
        c = RedisCache(default_ttl=60, key_prefix="test:")
        mock = MagicMock()
        c._client = mock
        c._checked = True
        assert c.set("k", "v") is True
        mock.setex.assert_called_once_with("test:k", 60, "v")

    def test_get_handles_exception(self):
        c = RedisCache()
        mock = MagicMock()
        mock.get.side_effect = Exception("redis down")
        c._client = mock
        c._checked = True
        assert c.get("k") is None

    def test_clear_is_noop(self):
        RedisCache().clear()
