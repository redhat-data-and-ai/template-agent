"""Webhook trigger source — lightweight HTTP listener using asyncio.

Accepts POST requests at a configurable path, parses the JSON body,
and yields ``TriggerEvent`` instances through the async-iterator
protocol.  No third-party HTTP framework is required; the server is
built on top of ``asyncio.start_server`` and raw HTTP parsing.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

from deep_agent.src.triggers.config import WebhookTriggerConfig
from deep_agent.src.triggers.sources.protocol import TriggerEvent
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

# Maximum request body size (1 MiB) to prevent unbounded memory usage.
_MAX_BODY_SIZE = 1_048_576

# Read timeout for an individual HTTP request (seconds).
_READ_TIMEOUT = 30.0


def _build_response(status: int, reason: str, body: dict[str, Any]) -> bytes:
    """Build a minimal HTTP/1.1 response with a JSON body."""
    payload = json.dumps(body).encode()
    lines = [
        f"HTTP/1.1 {status} {reason}",
        "Content-Type: application/json",
        f"Content-Length: {len(payload)}",
        "Connection: close",
        "",
        "",
    ]
    return "\r\n".join(lines).encode() + payload


class WebhookTriggerSource:
    """Async HTTP listener that emits ``TriggerEvent`` for each POST.

    Implements the ``TriggerSource`` protocol — ``start``/``stop`` control
    the server lifecycle, and ``__aiter__``/``__anext__`` pull events from
    an internal ``asyncio.Queue``.
    """

    def __init__(self, config: WebhookTriggerConfig) -> None:
        """Initialize the webhook trigger source with the given configuration."""
        self._config = config
        self._queue: asyncio.Queue[TriggerEvent] = asyncio.Queue()
        self._server: asyncio.Server | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Bind and start the HTTP server."""
        self._server = await asyncio.start_server(
            self._handle_connection,
            host=self._config.host,
            port=self._config.port,
        )
        addrs = [s.getsockname() for s in self._server.sockets]
        logger.info(
            "webhook trigger listening",
            host=self._config.host,
            port=self._config.port,
            path=self._config.path,
            addresses=addrs,
        )

    async def stop(self) -> None:
        """Shut down the HTTP server gracefully."""
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None
        logger.info("webhook trigger stopped")

    # ------------------------------------------------------------------
    # Async-iterator protocol
    # ------------------------------------------------------------------

    def __aiter__(self) -> AsyncIterator[TriggerEvent]:
        """Return the async iterator."""
        return self

    async def __anext__(self) -> TriggerEvent:
        """Return the next webhook trigger event."""
        return await self._queue.get()

    # ------------------------------------------------------------------
    # Connection handler
    # ------------------------------------------------------------------

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single TCP connection (one HTTP request)."""
        try:
            await asyncio.wait_for(
                self._process_request(reader, writer),
                timeout=_READ_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("webhook request timed out")
        except Exception:
            logger.error("webhook request error", exc_info=True)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _process_request(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Parse the HTTP request and dispatch based on method/path."""
        # Read the request line.
        request_line = await reader.readline()
        if not request_line:
            return

        parts = request_line.decode("utf-8", errors="replace").strip().split(" ", 2)
        if len(parts) < 2:
            await self._send(
                writer, 400, "Bad Request", {"error": "malformed request line"}
            )
            return

        method, path = parts[0], parts[1]

        # Read headers.
        content_length = 0
        while True:
            header_line = await reader.readline()
            if header_line in (b"\r\n", b"\n", b""):
                break
            header = header_line.decode("utf-8", errors="replace").strip()
            if header.lower().startswith("content-length:"):
                try:
                    content_length = int(header.split(":", 1)[1].strip())
                except (ValueError, IndexError):
                    pass

        # Route: only POST to the configured path is accepted.
        if path != self._config.path:
            await self._send(writer, 404, "Not Found", {"error": "not found"})
            return

        if method.upper() != "POST":
            await self._send(
                writer,
                405,
                "Method Not Allowed",
                {"error": f"method {method} not allowed"},
            )
            return

        # Guard against oversized bodies.
        if content_length > _MAX_BODY_SIZE:
            await self._send(
                writer,
                413,
                "Payload Too Large",
                {"error": f"body exceeds {_MAX_BODY_SIZE} bytes"},
            )
            return

        # Read body.
        body_bytes = await reader.read(content_length) if content_length > 0 else b""

        # Parse JSON.
        try:
            payload: Any = json.loads(body_bytes) if body_bytes else {}
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            await self._send(
                writer,
                400,
                "Bad Request",
                {"error": f"invalid JSON: {exc}"},
            )
            return

        if not isinstance(payload, dict):
            await self._send(
                writer,
                400,
                "Bad Request",
                {"error": "request body must be a JSON object"},
            )
            return

        # Build event and enqueue.
        event_name = payload.pop("event", "webhook")
        event = TriggerEvent(
            name=str(event_name),
            payload=payload,
            source="webhook",
        )
        await self._queue.put(event)
        logger.debug("webhook event enqueued", event_name=event.name)

        await self._send(writer, 200, "OK", {"status": "accepted"})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _send(
        writer: asyncio.StreamWriter,
        status: int,
        reason: str,
        body: dict[str, Any],
    ) -> None:
        """Write an HTTP response and drain."""
        writer.write(_build_response(status, reason, body))
        await writer.drain()
