"""Unit tests for personalization cache."""

import json
from unittest.mock import MagicMock, patch

import pytest

from deep_agent.src.cache import personalization_cache
from deep_agent.src.cache.config import CacheSettings


class TestPersonalizationCache:
    def setup_method(self):
        personalization_cache._redis = None

    async def test_get_returns_none_when_disabled(self):
        disabled = CacheSettings(CACHE_ENABLED=False)
        with patch.object(personalization_cache, "cache_settings", disabled):
            result = await personalization_cache.get_personalization("user-1")
            assert result is None

    async def test_set_is_noop_when_disabled(self):
        disabled = CacheSettings(CACHE_ENABLED=False)
        with patch.object(personalization_cache, "cache_settings", disabled):
            await personalization_cache.set_personalization("user-1", [], [])

    async def test_cache_roundtrip(self):
        enabled = CacheSettings(CACHE_ENABLED=True, CACHE_PERSONALIZATION_ENABLED=True)
        mock_redis = MagicMock()
        store: dict[str, str] = {}

        def fake_get(key: str) -> str | None:
            return store.get(key)

        def fake_set(key: str, value: str, ttl: int | None = None) -> bool:
            store[key] = value
            return True

        mock_redis.get = fake_get
        mock_redis.set = fake_set

        with (
            patch.object(personalization_cache, "cache_settings", enabled),
            patch.object(personalization_cache, "_get_redis", return_value=mock_redis),
        ):
            memories = [{"content": "likes pizza"}]
            rules = [{"content": "be brief"}]
            await personalization_cache.set_personalization("user-1", memories, rules)

            result = await personalization_cache.get_personalization("user-1")
            assert result is not None
            assert result[0] == memories
            assert result[1] == rules

    async def test_get_handles_corrupt_data(self):
        enabled = CacheSettings(CACHE_ENABLED=True, CACHE_PERSONALIZATION_ENABLED=True)
        mock_redis = MagicMock()
        mock_redis.get.return_value = "not-valid-json{{"
        mock_redis.delete.return_value = True

        with (
            patch.object(personalization_cache, "cache_settings", enabled),
            patch.object(personalization_cache, "_get_redis", return_value=mock_redis),
        ):
            result = await personalization_cache.get_personalization("user-1")
            assert result is None

    async def test_invalidate(self):
        enabled = CacheSettings(CACHE_ENABLED=True, CACHE_PERSONALIZATION_ENABLED=True)
        mock_redis = MagicMock()

        with (
            patch.object(personalization_cache, "cache_settings", enabled),
            patch.object(personalization_cache, "_get_redis", return_value=mock_redis),
        ):
            await personalization_cache.invalidate("user-1")
            mock_redis.delete.assert_called_once()

    async def test_invalidate_none_is_noop(self):
        await personalization_cache.invalidate(None)
