"""Integration tests for WebhookTriggerSource with real HTTP connections.

No external services required — the webhook listener binds to a random
port on localhost and tests communicate via ``httpx.AsyncClient``.
"""

from __future__ import annotations

import httpx
import pytest

from deep_agent.src.triggers.config import WebhookTriggerConfig
from deep_agent.src.triggers.sources.webhook import WebhookTriggerSource

pytestmark = pytest.mark.integration


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


async def _start_source(
    path: str = "/trigger",
) -> tuple[WebhookTriggerSource, int]:
    """Create and start a webhook source on a random OS-assigned port.

    Returns ``(source, actual_port)``.
    """
    config = WebhookTriggerConfig(
        enabled=True,
        host="127.0.0.1",
        port=0,
        path=path,
    )
    source = WebhookTriggerSource(config)
    await source.start()
    assert source._server is not None
    port = source._server.sockets[0].getsockname()[1]
    return source, port


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestWebhookListenerIntegration:
    """Integration tests sending real HTTP requests to the webhook listener."""

    async def test_valid_post_returns_200_and_produces_event(self):
        """POST valid JSON to the trigger path produces a TriggerEvent."""
        source, port = await _start_source()
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://127.0.0.1:{port}/trigger",
                    json={"event": "integration-test", "key": "value"},
                )

            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "accepted"

            event = source._queue.get_nowait()
            assert event.name == "integration-test"
            assert event.payload == {"key": "value"}
            assert event.source == "webhook"
        finally:
            await source.stop()

    async def test_invalid_json_returns_400(self):
        """POST with invalid JSON body returns 400 and enqueues nothing."""
        source, port = await _start_source()
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://127.0.0.1:{port}/trigger",
                    content=b"<<<not json>>>",
                    headers={"Content-Type": "application/json"},
                )

            assert response.status_code == 400
            body = response.json()
            assert "invalid JSON" in body["error"]
            assert source._queue.empty()
        finally:
            await source.stop()

    async def test_wrong_path_returns_404(self):
        """POST to a non-configured path returns 404."""
        source, port = await _start_source()
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://127.0.0.1:{port}/not-the-trigger",
                    json={"event": "lost"},
                )

            assert response.status_code == 404
            assert source._queue.empty()
        finally:
            await source.stop()

    async def test_multiple_posts_produce_events_in_order(self):
        """Sequential POST requests produce events in FIFO order."""
        source, port = await _start_source()
        try:
            async with httpx.AsyncClient() as client:
                for i in range(4):
                    resp = await client.post(
                        f"http://127.0.0.1:{port}/trigger",
                        json={"event": f"evt-{i}", "seq": i},
                    )
                    assert resp.status_code == 200

            assert source._queue.qsize() == 4
            names = [source._queue.get_nowait().name for _ in range(4)]
            assert names == [f"evt-{i}" for i in range(4)]
        finally:
            await source.stop()
