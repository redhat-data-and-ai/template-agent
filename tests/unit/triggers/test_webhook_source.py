"""Unit tests for WebhookTriggerSource."""

import asyncio
import json

from deep_agent.src.triggers.config import WebhookTriggerConfig
from deep_agent.src.triggers.sources.webhook import WebhookTriggerSource


async def _send_raw_http(
    host: str,
    port: int,
    method: str,
    path: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Send a raw HTTP request and return (status_code, response_body)."""
    reader, writer = await asyncio.open_connection(host, port)
    try:
        request_line = f"{method} {path} HTTP/1.1\r\n"
        header_lines = f"Host: {host}:{port}\r\n"
        if headers:
            for k, v in headers.items():
                header_lines += f"{k}: {v}\r\n"
        if body is not None:
            header_lines += f"Content-Length: {len(body)}\r\n"
        header_lines += "\r\n"

        writer.write(request_line.encode() + header_lines.encode())
        if body is not None:
            writer.write(body)
        await writer.drain()

        # Read response status line.
        status_line = await reader.readline()
        parts = status_line.decode().strip().split(" ", 2)
        status_code = int(parts[1])

        # Read headers until blank line.
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break

        # Read remaining response body.
        response_body = await reader.read(4096)
        return status_code, response_body.decode()
    finally:
        writer.close()
        await writer.wait_closed()


class TestWebhookTriggerSource:
    """Test the async HTTP webhook listener."""

    async def _start_source(
        self,
        path: str = "/trigger",
    ) -> tuple[WebhookTriggerSource, int]:
        """Create and start a source on a random port, return (source, port)."""
        config = WebhookTriggerConfig(
            enabled=True,
            host="127.0.0.1",
            port=0,  # OS assigns a free port.
            path=path,
        )
        source = WebhookTriggerSource(config)
        await source.start()
        # Extract the actual bound port.
        assert source._server is not None
        port = source._server.sockets[0].getsockname()[1]
        return source, port

    async def test_start_binds_server(self):
        source, port = await self._start_source()
        try:
            assert source._server is not None
            assert port > 0
        finally:
            await source.stop()

    async def test_stop_closes_server(self):
        source, _ = await self._start_source()
        await source.stop()
        assert source._server is None

    async def test_stop_when_not_started_is_noop(self):
        config = WebhookTriggerConfig()
        source = WebhookTriggerSource(config)
        # Should not raise.
        await source.stop()
        assert source._server is None

    async def test_valid_post_enqueues_trigger_event(self):
        source, port = await self._start_source()
        try:
            payload = {"event": "test-event", "key": "value"}
            status, body = await _send_raw_http(
                "127.0.0.1",
                port,
                "POST",
                "/trigger",
                body=json.dumps(payload).encode(),
            )

            assert status == 200
            response = json.loads(body)
            assert response["status"] == "accepted"

            # Event should be in the queue.
            event = source._queue.get_nowait()
            assert event.name == "test-event"
            assert event.payload == {"key": "value"}
            assert event.source == "webhook"
        finally:
            await source.stop()

    async def test_invalid_json_returns_400(self):
        source, port = await self._start_source()
        try:
            status, body = await _send_raw_http(
                "127.0.0.1",
                port,
                "POST",
                "/trigger",
                body=b"not valid json{{{",
            )
            assert status == 400
            response = json.loads(body)
            assert "invalid JSON" in response["error"]
            assert source._queue.empty()
        finally:
            await source.stop()

    async def test_wrong_path_returns_404(self):
        source, port = await self._start_source()
        try:
            status, _ = await _send_raw_http(
                "127.0.0.1",
                port,
                "POST",
                "/wrong-path",
                body=json.dumps({"event": "x"}).encode(),
            )
            assert status == 404
            assert source._queue.empty()
        finally:
            await source.stop()

    async def test_get_method_returns_405(self):
        source, port = await self._start_source()
        try:
            status, body = await _send_raw_http(
                "127.0.0.1",
                port,
                "GET",
                "/trigger",
            )
            assert status == 405
            response = json.loads(body)
            assert "not allowed" in response["error"]
        finally:
            await source.stop()

    async def test_put_method_returns_405(self):
        source, port = await self._start_source()
        try:
            status, _ = await _send_raw_http(
                "127.0.0.1",
                port,
                "PUT",
                "/trigger",
                body=json.dumps({"event": "x"}).encode(),
            )
            assert status == 405
        finally:
            await source.stop()

    async def test_trigger_event_source_is_webhook(self):
        source, port = await self._start_source()
        try:
            await _send_raw_http(
                "127.0.0.1",
                port,
                "POST",
                "/trigger",
                body=json.dumps({"data": 1}).encode(),
            )
            event = source._queue.get_nowait()
            assert event.source == "webhook"
        finally:
            await source.stop()

    async def test_event_name_defaults_to_webhook(self):
        source, port = await self._start_source()
        try:
            # Payload without an "event" key defaults name to "webhook".
            await _send_raw_http(
                "127.0.0.1",
                port,
                "POST",
                "/trigger",
                body=json.dumps({"key": "val"}).encode(),
            )
            event = source._queue.get_nowait()
            assert event.name == "webhook"
        finally:
            await source.stop()

    async def test_multiple_events_can_be_queued(self):
        source, port = await self._start_source()
        try:
            for i in range(3):
                await _send_raw_http(
                    "127.0.0.1",
                    port,
                    "POST",
                    "/trigger",
                    body=json.dumps({"event": f"e{i}"}).encode(),
                )

            assert source._queue.qsize() == 3
            names = [source._queue.get_nowait().name for _ in range(3)]
            assert names == ["e0", "e1", "e2"]
        finally:
            await source.stop()

    async def test_custom_path(self):
        source, port = await self._start_source(path="/custom/webhook")
        try:
            status, _ = await _send_raw_http(
                "127.0.0.1",
                port,
                "POST",
                "/custom/webhook",
                body=json.dumps({"event": "custom"}).encode(),
            )
            assert status == 200
            event = source._queue.get_nowait()
            assert event.name == "custom"
        finally:
            await source.stop()

    async def test_non_object_json_body_returns_400(self):
        source, port = await self._start_source()
        try:
            status, body = await _send_raw_http(
                "127.0.0.1",
                port,
                "POST",
                "/trigger",
                body=json.dumps([1, 2, 3]).encode(),
            )
            assert status == 400
            response = json.loads(body)
            assert "JSON object" in response["error"]
        finally:
            await source.stop()
