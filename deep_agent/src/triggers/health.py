"""Health check endpoint for headless worker — serves /healthz and /readyz."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from deep_agent.utils.pylogger import get_python_logger

if TYPE_CHECKING:
    from deep_agent.src.triggers.middleware import EventTriggerMiddleware

logger = get_python_logger()


async def start_health_server(
    host: str,
    port: int,
    middleware: EventTriggerMiddleware,
) -> asyncio.Server:
    """Start a minimal HTTP server for liveness and readiness probes."""

    async def _handle(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            request_line = await reader.readline()
            if not request_line:
                return
            parts = request_line.decode("utf-8", errors="replace").strip().split(" ", 2)
            path = parts[1] if len(parts) >= 2 else "/"

            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break

            if path in ("/healthz", "/health"):
                body = {"status": "ok"}
                status = 200
            elif path == "/readyz":
                has_sources = len(middleware._sources) > 0
                loop_running = (
                    middleware._loop_task is not None
                    and not middleware._loop_task.done()
                )
                ready = has_sources and loop_running
                body = {
                    "status": "ready" if ready else "not_ready",
                    "sources": str(len(middleware._sources)),
                    "sinks": str(len(middleware._sinks)),
                    "loop_running": str(loop_running),
                }
                status = 200 if ready else 503
            else:
                body = {"error": "not found"}
                status = 404

            payload = json.dumps(body).encode()
            phrases = {200: "OK", 404: "Not Found", 503: "Service Unavailable"}
            header = (
                f"HTTP/1.1 {status} {phrases.get(status, 'Error')}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(payload)}\r\n"
                f"Connection: close\r\n"
                f"\r\n"
            )
            writer.write(header.encode() + payload)
            await writer.drain()
        except Exception:
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(_handle, host, port)
    logger.info("health check listening on %s:%d (/healthz, /readyz)", host, port)
    return server
