"""Unit tests for the Redis output sink."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from deep_agent.src.triggers.sinks.protocol import TriggerResult
from deep_agent.src.triggers.sinks.redis import RedisSink
from deep_agent.src.triggers.sources.protocol import TriggerEvent


def _make_event(**overrides) -> TriggerEvent:
    defaults = {
        "name": "test-event",
        "payload": {"key": "value"},
        "source": "unit-test",
    }
    defaults.update(overrides)
    return TriggerEvent(**defaults)


def _make_result(**overrides) -> TriggerResult:
    defaults = {
        "event": _make_event(),
        "output": {"answer": 42},
        "duration_ms": 75.0,
        "success": True,
    }
    defaults.update(overrides)
    return TriggerResult(**defaults)


class TestRedisSink:
    """Test RedisSink publishes to a Redis Stream via XADD."""

    async def test_emit_calls_xadd_with_serialized_result(self):
        sink = RedisSink(stream="test-stream")
        mock_client = AsyncMock()
        sink._client = mock_client

        result = _make_result()
        await sink.emit(result)

        mock_client.xadd.assert_awaited_once()
        call_args = mock_client.xadd.call_args
        assert call_args[0][0] == "test-stream"

        payload_json = call_args[0][1]["result"]
        parsed = json.loads(payload_json)
        assert parsed["success"] is True
        assert parsed["event"]["name"] == "test-event"

    async def test_close_closes_redis_client(self):
        sink = RedisSink(stream="test-stream")
        mock_client = AsyncMock()
        sink._client = mock_client

        await sink.close()

        mock_client.aclose.assert_awaited_once()
        assert sink._client is None

    async def test_redis_error_in_emit_is_caught_and_logged(self):
        sink = RedisSink(stream="test-stream")
        mock_client = AsyncMock()
        mock_client.xadd.side_effect = ConnectionError("Redis down")
        sink._client = mock_client

        with patch("deep_agent.src.triggers.sinks.redis.logger") as mock_logger:
            await sink.emit(_make_result())  # must not raise

        mock_logger.exception.assert_called_once()
        assert "test-stream" in mock_logger.exception.call_args[0][1]
