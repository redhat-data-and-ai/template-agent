"""Unit tests for Redis distributed locks."""

from unittest.mock import AsyncMock, MagicMock, patch

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
    """The async lock talks to the asyncio client directly.

    It used to hand the blocking helper to a thread, where it polled
    with ``time.sleep`` for up to ``wait_seconds``. These tests patch
    ``get_async_redis_client`` for that reason — the sync client is no
    longer on this path.
    """

    async def test_yields_no_redis_when_client_missing(self):
        with patch(
            "deep_agent.aegra.redis.get_async_redis_client",
            AsyncMock(return_value=None),
        ):
            async with distributed_lock("refresh:user:mcp") as state:
                assert state == "no_redis"

    async def test_yields_held_when_lock_acquired(self):
        client = AsyncMock()
        client.set = AsyncMock(return_value=True)
        client.eval = AsyncMock(return_value=1)
        with patch(
            "deep_agent.aegra.redis.get_async_redis_client",
            AsyncMock(return_value=client),
        ):
            async with distributed_lock("refresh:user:mcp") as state:
                assert state == "held"
        client.set.assert_awaited_once()
        # Released through the compare-and-delete script, so a lock
        # whose TTL already expired is not deleted out from under
        # whoever holds it now.
        client.eval.assert_awaited_once()

    async def test_yields_timeout_when_never_acquired(self):
        client = AsyncMock()
        client.set = AsyncMock(return_value=None)
        with patch(
            "deep_agent.aegra.redis.get_async_redis_client",
            AsyncMock(return_value=client),
        ):
            async with distributed_lock("refresh:user:mcp", wait_seconds=0) as state:
                assert state == "timeout"
        client.eval.assert_not_awaited()
