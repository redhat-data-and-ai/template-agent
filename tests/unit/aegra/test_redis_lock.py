"""Unit tests for Redis distributed locks."""

from unittest.mock import MagicMock, patch

import deep_agent.aegra.redis as redis_mod
from deep_agent.aegra.redis import (
    acquire_distributed_lock,
    distributed_lock,
    release_distributed_lock,
)


class TestDistributedLock:
    def test_acquire_and_release(self):
        mock_client = MagicMock()
        mock_client.set.return_value = True
        mock_client.eval.return_value = 1
        redis_mod._client = mock_client

        token = acquire_distributed_lock(
            "refresh:user:mcp", ttl_seconds=30, wait_seconds=1
        )
        assert token is not None
        assert release_distributed_lock("refresh:user:mcp", token) is True
        mock_client.set.assert_called_once()
        mock_client.eval.assert_called_once()

    def test_acquire_returns_none_when_redis_unavailable(self):
        with patch("deep_agent.aegra.redis.get_redis_client", return_value=None):
            assert acquire_distributed_lock("refresh:user:mcp") is None


class TestDistributedLockAsync:
    async def test_yields_no_redis_when_client_missing(self):
        with patch("deep_agent.aegra.redis.get_redis_client", return_value=None):
            async with distributed_lock("refresh:user:mcp") as state:
                assert state == "no_redis"

    async def test_yields_held_when_lock_acquired(self):
        with (
            patch(
                "deep_agent.aegra.redis.acquire_distributed_lock",
                return_value="lock-token",
            ),
            patch(
                "deep_agent.aegra.redis.release_distributed_lock",
                return_value=True,
            ) as release,
            patch(
                "deep_agent.aegra.redis.get_redis_client",
                return_value=MagicMock(),
            ),
        ):
            async with distributed_lock("refresh:user:mcp") as state:
                assert state == "held"
            release.assert_called_once_with("refresh:user:mcp", "lock-token")
