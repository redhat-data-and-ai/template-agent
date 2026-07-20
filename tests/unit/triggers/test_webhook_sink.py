"""Unit tests for the webhook output sink."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from deep_agent.src.triggers.sinks.protocol import TriggerResult
from deep_agent.src.triggers.sinks.webhook import WebhookSink
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
        "duration_ms": 50.0,
        "success": True,
    }
    defaults.update(overrides)
    return TriggerResult(**defaults)


def _mock_response(status_code: int) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    return resp


class TestWebhookSink:
    """Test WebhookSink POSTs JSON to a URL with retry logic."""

    async def test_emit_posts_json_to_url(self):
        sink = WebhookSink(url="https://example.com/hook")
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = _mock_response(200)
        sink._client = mock_client

        await sink.emit(_make_result())

        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[0][0] == "https://example.com/hook"
        assert "Content-Type" in call_kwargs[1]["headers"]
        assert call_kwargs[1]["headers"]["Content-Type"] == "application/json"

    async def test_custom_headers_included_in_request(self):
        custom_headers = {"X-Api-Key": "secret-123", "X-Custom": "value"}
        sink = WebhookSink(url="https://example.com/hook", headers=custom_headers)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = _mock_response(200)
        sink._client = mock_client

        await sink.emit(_make_result())

        sent_headers = mock_client.post.call_args[1]["headers"]
        assert sent_headers["X-Api-Key"] == "secret-123"
        assert sent_headers["X-Custom"] == "value"
        assert sent_headers["Content-Type"] == "application/json"

    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_5xx_triggers_retry(self, mock_sleep: AsyncMock):
        sink = WebhookSink(url="https://example.com/hook", max_retries=2)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = _mock_response(503)
        sink._client = mock_client

        await sink.emit(_make_result())

        # 1 initial + 2 retries = 3 total calls
        assert mock_client.post.call_count == 3

    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_4xx_does_not_retry(self, mock_sleep: AsyncMock):
        sink = WebhookSink(url="https://example.com/hook", max_retries=3)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = _mock_response(422)
        sink._client = mock_client

        await sink.emit(_make_result())

        mock_client.post.assert_called_once()

    async def test_close_closes_httpx_client(self):
        sink = WebhookSink(url="https://example.com/hook")
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        sink._client = mock_client

        await sink.close()

        mock_client.aclose.assert_awaited_once()
        assert sink._client is None
