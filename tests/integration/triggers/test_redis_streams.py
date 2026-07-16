"""Integration tests for RedisStreamsConsumer with a real Redis server.

Requires Redis running at ``redis://localhost:6379/0``.  Each test uses
a unique stream/group name to avoid interference between tests.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

pytestmark = pytest.mark.integration

_REDIS_URL = "redis://localhost:6379/0"


def _unique_name(prefix: str = "test") -> str:
    """Return a collision-free stream/group name."""
    return f"{prefix}-{uuid4().hex[:8]}"


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
async def redis_client():
    """Yield a real ``redis.asyncio`` client, skip if Redis is unavailable."""
    aioredis = pytest.importorskip("redis.asyncio")
    client = aioredis.from_url(_REDIS_URL, decode_responses=True)
    try:
        await client.ping()
    except Exception:
        pytest.skip("Redis not available at localhost:6379")

    yield client
    await client.aclose()


@pytest.fixture()
def stream_name() -> str:
    return _unique_name("stream")


@pytest.fixture()
def group_name() -> str:
    return _unique_name("group")


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestRedisStreamsConsumerIntegration:
    """Integration tests exercising RedisStreamsConsumer against real Redis."""

    async def test_creates_group_and_stream_on_first_consume(
        self, redis_client, stream_name, group_name
    ):
        """Consumer creates the stream and consumer group automatically."""
        from deep_agent.src.triggers.sources.queue import RedisStreamsConsumer

        consumer = RedisStreamsConsumer(
            stream=stream_name,
            consumer_group=group_name,
            consumer_name="worker-1",
            redis_url=_REDIS_URL,
            block_ms=100,
        )

        # _ensure_client() creates the group and stream via XGROUP CREATE MKSTREAM.
        await consumer._ensure_client()

        # Stream and group should now exist in Redis.
        groups = await redis_client.xinfo_groups(stream_name)
        group_names = [g["name"] for g in groups]
        assert group_name in group_names

        await consumer.close()
        await redis_client.delete(stream_name)

    async def test_produce_consume_ack_cycle(
        self, redis_client, stream_name, group_name
    ):
        """XADD -> consume -> ack round-trip works end-to-end."""
        from deep_agent.src.triggers.sources.queue import RedisStreamsConsumer

        consumer = RedisStreamsConsumer(
            stream=stream_name,
            consumer_group=group_name,
            consumer_name="worker-1",
            redis_url=_REDIS_URL,
            block_ms=100,
        )

        # Produce a message before consuming.
        await redis_client.xadd(
            stream_name, {"name": "integration-event", "data": "hello"}
        )

        received = []
        async for msg in consumer.consume():
            received.append(msg)
            await consumer.ack(msg)
            # After receiving one message, stop.
            consumer._running = False

        assert len(received) == 1
        assert received[0].data["name"] == "integration-event"
        assert received[0].data["data"] == "hello"
        assert received[0].id  # should be a valid Redis stream ID

        # Verify the message was acknowledged (pending count should be 0).
        groups = await redis_client.xinfo_groups(stream_name)
        target_group = [g for g in groups if g["name"] == group_name][0]
        assert target_group["pending"] == 0

        await consumer.close()
        await redis_client.delete(stream_name)

    async def test_handles_existing_group_busygroup(
        self, redis_client, stream_name, group_name
    ):
        """Creating a consumer when the group already exists does not raise."""
        from deep_agent.src.triggers.sources.queue import RedisStreamsConsumer

        # Pre-create the group to trigger BUSYGROUP.
        await redis_client.xgroup_create(stream_name, group_name, id="0", mkstream=True)

        consumer = RedisStreamsConsumer(
            stream=stream_name,
            consumer_group=group_name,
            consumer_name="worker-1",
            redis_url=_REDIS_URL,
            block_ms=100,
        )

        # Should not raise.
        consumer._running = False
        async for _ in consumer.consume():
            pass  # pragma: no cover

        # Group still exists and is intact.
        groups = await redis_client.xinfo_groups(stream_name)
        assert any(g["name"] == group_name for g in groups)

        await consumer.close()
        await redis_client.delete(stream_name)

    async def test_multiple_messages_consumed_in_order(
        self, redis_client, stream_name, group_name
    ):
        """Multiple messages are consumed in the order they were added."""
        from deep_agent.src.triggers.sources.queue import RedisStreamsConsumer

        consumer = RedisStreamsConsumer(
            stream=stream_name,
            consumer_group=group_name,
            consumer_name="worker-1",
            redis_url=_REDIS_URL,
            block_ms=100,
        )

        # Add several messages.
        for i in range(5):
            await redis_client.xadd(stream_name, {"name": f"event-{i}", "seq": str(i)})

        received = []
        async for msg in consumer.consume():
            received.append(msg)
            await consumer.ack(msg)
            if len(received) >= 5:
                consumer._running = False

        assert len(received) == 5
        names = [m.data["name"] for m in received]
        assert names == [f"event-{i}" for i in range(5)]

        await consumer.close()
        await redis_client.delete(stream_name)

    async def test_close_cleans_up_connection(
        self, redis_client, stream_name, group_name
    ):
        """close() sets running=False and tears down the client."""
        from deep_agent.src.triggers.sources.queue import RedisStreamsConsumer

        consumer = RedisStreamsConsumer(
            stream=stream_name,
            consumer_group=group_name,
            consumer_name="worker-1",
            redis_url=_REDIS_URL,
            block_ms=100,
        )

        # Force client creation.
        await consumer._ensure_client()
        assert consumer._client is not None

        await consumer.close()

        assert consumer._running is False
        assert consumer._client is None

        await redis_client.delete(stream_name)
