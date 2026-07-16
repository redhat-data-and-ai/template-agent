"""Integration tests for RedisSink with a real Redis server.

Requires Redis running at ``redis://localhost:6379/0``.  Each test uses
a unique stream name to avoid interference between tests.
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from deep_agent.src.triggers.sinks.protocol import TriggerResult
from deep_agent.src.triggers.sinks.redis import RedisSink
from deep_agent.src.triggers.sources.protocol import TriggerEvent

pytestmark = pytest.mark.integration

_REDIS_URL = "redis://localhost:6379/0"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _unique_stream() -> str:
    return f"test-results-{uuid4().hex[:8]}"


def _make_event(**overrides) -> TriggerEvent:
    defaults = {
        "name": "test-event",
        "payload": {"key": "value"},
        "source": "integration-test",
    }
    defaults.update(overrides)
    return TriggerEvent(**defaults)


def _make_result(**overrides) -> TriggerResult:
    defaults = {
        "event": _make_event(),
        "output": {"answer": 42},
        "duration_ms": 50.0,
        "success": True,
    }
    defaults.update(overrides)
    return TriggerResult(**defaults)


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


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestRedisSinkIntegration:
    """Integration tests verifying RedisSink writes to a real Redis stream."""

    async def test_emit_writes_to_stream(self, redis_client):
        """A single emit() writes a JSONL entry that XRANGE can read back."""
        stream = _unique_stream()
        sink = RedisSink(stream=stream, redis_url=_REDIS_URL)

        try:
            result = _make_result()
            await sink.emit(result)

            entries = await redis_client.xrange(stream)
            assert len(entries) == 1

            _msg_id, fields = entries[0]
            parsed = json.loads(fields["result"])
            assert parsed["success"] is True
            assert parsed["event"]["name"] == "test-event"
            assert parsed["output"] == {"answer": 42}
        finally:
            await sink.close()
            await redis_client.delete(stream)

    async def test_multiple_emits_create_multiple_entries(self, redis_client):
        """Multiple emit() calls create corresponding stream entries in order."""
        stream = _unique_stream()
        sink = RedisSink(stream=stream, redis_url=_REDIS_URL)

        try:
            for i in range(3):
                await sink.emit(_make_result(output=f"result-{i}"))

            entries = await redis_client.xrange(stream)
            assert len(entries) == 3

            outputs = [json.loads(e[1]["result"])["output"] for e in entries]
            assert outputs == ["result-0", "result-1", "result-2"]
        finally:
            await sink.close()
            await redis_client.delete(stream)

    async def test_close_cleans_up(self, redis_client):
        """close() tears down the internal Redis client."""
        stream = _unique_stream()
        sink = RedisSink(stream=stream, redis_url=_REDIS_URL)

        try:
            # Force client creation by emitting.
            await sink.emit(_make_result())
            assert sink._client is not None

            await sink.close()
            assert sink._client is None
        finally:
            await redis_client.delete(stream)
