"""End-to-end integration tests for the trigger pipeline.

Exercises the full path: trigger source -> EventTriggerMiddleware ->
(mocked graph) -> output sink.  The graph is always mocked — only the
trigger sources, middleware orchestration, and sinks use real I/O.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from deep_agent.src.triggers.config import (
    HeadlessConfig,
    OutputSinkConfig,
    QueueTriggerConfig,
    TriggerConfig,
    WebhookTriggerConfig,
)
from deep_agent.src.triggers.middleware import EventTriggerMiddleware

pytestmark = pytest.mark.integration

_REDIS_URL = "redis://localhost:6379/0"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file and return parsed lines."""
    if not path.exists():
        return []
    lines = path.read_text().strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _mock_graph(return_value: dict | None = None) -> AsyncMock:
    """Create a mock graph whose ainvoke returns a simple dict."""
    graph = AsyncMock()
    graph.ainvoke.return_value = return_value or {
        "messages": [{"role": "assistant", "content": "ok"}]
    }
    return graph


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestWebhookToFileSink:
    """Webhook POST -> middleware -> mocked graph -> file sink."""

    async def test_webhook_event_produces_file_output(self, tmp_path: Path):
        """POST to webhook -> graph invoked -> result written to JSONL file."""
        output_file = tmp_path / "output.jsonl"

        config = HeadlessConfig(
            triggers=TriggerConfig(
                webhook=WebhookTriggerConfig(
                    enabled=True, host="127.0.0.1", port=0, path="/trigger"
                ),
            ),
            output_sinks=[OutputSinkConfig(type="file", path=str(output_file))],
            drain_timeout=5.0,
        )
        graph = _mock_graph({"result": "webhook-ok"})
        mw = EventTriggerMiddleware(config=config, graph=graph, redis_url=_REDIS_URL)

        await mw.start()

        # Resolve the actual port the webhook source bound to.
        webhook_source = mw._sources[0]
        port = webhook_source._server.sockets[0].getsockname()[1]

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://127.0.0.1:{port}/trigger",
                    json={"event": "e2e-test", "input": "hello"},
                )
                assert resp.status_code == 200

            # Allow the middleware processing loop time to invoke the graph and emit.
            await asyncio.sleep(0.5)
        finally:
            await mw.stop()

        # Verify the graph was called with the event payload.
        graph.ainvoke.assert_awaited_once()
        call_args = graph.ainvoke.call_args[0][0]
        content = json.loads(call_args["messages"][0]["content"])
        assert content["input"] == "hello"

        # Verify the output file contains one JSONL entry.
        results = _read_jsonl(output_file)
        assert len(results) == 1
        assert results[0]["success"] is True
        assert results[0]["event"]["name"] == "e2e-test"
        assert results[0]["output"] == {"result": "webhook-ok"}


class TestQueueToFileSink:
    """Redis Stream -> middleware -> mocked graph -> file sink."""

    async def test_queue_event_produces_file_output(self, tmp_path: Path):
        """Push message to Redis Stream -> graph invoked -> result in JSONL."""
        aioredis = pytest.importorskip("redis.asyncio")
        client = aioredis.from_url(_REDIS_URL, decode_responses=True)
        try:
            await client.ping()
        except Exception:
            pytest.skip("Redis not available at localhost:6379")

        from uuid import uuid4

        stream = f"e2e-tasks-{uuid4().hex[:8]}"
        group = f"e2e-workers-{uuid4().hex[:8]}"
        output_file = tmp_path / "queue-output.jsonl"

        config = HeadlessConfig(
            triggers=TriggerConfig(
                queue=QueueTriggerConfig(
                    enabled=True,
                    backend="redis_streams",
                    stream=stream,
                    consumer_group=group,
                    consumer_name="e2e-worker",
                ),
            ),
            output_sinks=[OutputSinkConfig(type="file", path=str(output_file))],
            drain_timeout=5.0,
        )
        graph = _mock_graph({"result": "queue-ok"})
        mw = EventTriggerMiddleware(config=config, graph=graph, redis_url=_REDIS_URL)

        await mw.start()

        try:
            # Produce a message into the stream.
            await client.xadd(stream, {"name": "queue-e2e", "data": "world"})

            # Wait for consumption + graph invocation + sink write.
            await asyncio.sleep(1.0)
        finally:
            await mw.stop()
            await client.delete(stream)
            await client.aclose()

        graph.ainvoke.assert_awaited_once()

        results = _read_jsonl(output_file)
        assert len(results) == 1
        assert results[0]["success"] is True
        assert results[0]["event"]["name"] == "queue-e2e"
        assert results[0]["output"] == {"result": "queue-ok"}


class TestGraphErrorProducesFailedResult:
    """Graph raises -> TriggerResult with success=False."""

    async def test_graph_error_results_in_failure(self, tmp_path: Path):
        """When the graph raises, the result has success=False and error message."""
        output_file = tmp_path / "error-output.jsonl"

        config = HeadlessConfig(
            triggers=TriggerConfig(
                webhook=WebhookTriggerConfig(
                    enabled=True, host="127.0.0.1", port=0, path="/trigger"
                ),
            ),
            output_sinks=[OutputSinkConfig(type="file", path=str(output_file))],
            drain_timeout=5.0,
        )
        graph = AsyncMock()
        graph.ainvoke.side_effect = RuntimeError("model unavailable")
        mw = EventTriggerMiddleware(config=config, graph=graph, redis_url=_REDIS_URL)

        await mw.start()

        webhook_source = mw._sources[0]
        port = webhook_source._server.sockets[0].getsockname()[1]

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://127.0.0.1:{port}/trigger",
                    json={"event": "error-test", "x": 1},
                )
                assert resp.status_code == 200

            await asyncio.sleep(0.5)
        finally:
            await mw.stop()

        results = _read_jsonl(output_file)
        assert len(results) == 1
        assert results[0]["success"] is False
        assert "model unavailable" in results[0]["error"]
        assert results[0]["event"]["name"] == "error-test"
