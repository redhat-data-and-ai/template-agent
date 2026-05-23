"""Unit tests for aegra redis module."""

from unittest.mock import MagicMock, patch

import pytest

import deep_agent.aegra.redis as redis_mod
from deep_agent.aegra.redis import (
    cache_delete,
    cache_get,
    cache_set,
    get_redis_client,
    get_redis_config,
)


@pytest.fixture(autouse=True)
def _reset_client():
    """Reset the module-level singleton before each test."""
    redis_mod._client = None
    yield
    redis_mod._client = None


class TestGetRedisConfig:
    def test_returns_all_keys(self):
        cfg = get_redis_config()
        assert "url" in cfg
        assert "max_connections" in cfg
        assert "socket_timeout" in cfg
        assert "retry_on_timeout" in cfg
        assert "key_prefix" in cfg


class TestGetRedisClient:
    def test_returns_cached_client(self):
        mock_client = MagicMock()
        redis_mod._client = mock_client
        assert get_redis_client() is mock_client

    def test_returns_none_when_redis_unavailable(self):
        mock_redis = MagicMock()
        mock_redis.from_url.side_effect = ConnectionError("refused")
        with patch.dict("sys.modules", {"redis": mock_redis}):
            result = get_redis_client()
            assert result is None

    def test_returns_none_when_redis_not_installed(self):
        with patch.dict("sys.modules", {"redis": None}):
            with patch("builtins.__import__", side_effect=ImportError("no redis")):
                redis_mod._client = None
                result = get_redis_client()
                assert result is None


class TestCacheGet:
    def test_returns_none_when_no_client(self):
        with patch("deep_agent.aegra.redis.get_redis_client", return_value=None):
            assert cache_get("key") is None

    def test_returns_value_from_redis(self):
        mock_client = MagicMock()
        mock_client.get.return_value = "cached_value"
        redis_mod._client = mock_client
        assert cache_get("key") == "cached_value"

    def test_returns_none_on_error(self):
        mock_client = MagicMock()
        mock_client.get.side_effect = Exception("redis error")
        redis_mod._client = mock_client
        assert cache_get("key") is None


class TestCacheSet:
    def test_returns_false_when_no_client(self):
        with patch("deep_agent.aegra.redis.get_redis_client", return_value=None):
            assert cache_set("key", "value") is False

    def test_returns_true_on_success(self):
        mock_client = MagicMock()
        redis_mod._client = mock_client
        assert cache_set("key", "value", ttl_seconds=60) is True
        mock_client.setex.assert_called_once()

    def test_returns_false_on_error(self):
        mock_client = MagicMock()
        mock_client.setex.side_effect = Exception("write fail")
        redis_mod._client = mock_client
        assert cache_set("key", "value") is False


class TestCacheDelete:
    def test_returns_false_when_no_client(self):
        with patch("deep_agent.aegra.redis.get_redis_client", return_value=None):
            assert cache_delete("key") is False

    def test_returns_true_on_success(self):
        mock_client = MagicMock()
        redis_mod._client = mock_client
        assert cache_delete("key") is True

    def test_returns_false_on_error(self):
        mock_client = MagicMock()
        mock_client.delete.side_effect = Exception("fail")
        redis_mod._client = mock_client
        assert cache_delete("key") is False
