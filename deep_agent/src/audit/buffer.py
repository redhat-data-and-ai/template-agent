"""Local audit event buffer — in-memory queue for transient failures."""

from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Any

from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_lock = Lock()
_queue: deque[dict[str, Any]] = deque()
_dropped = 0


def enqueue(envelope: dict[str, Any]) -> None:
    """Append envelope to in-memory buffer."""
    global _dropped  # noqa: PLW0603
    buffer_max = settings.PLATFORM_AUDIT_BUFFER_MAX
    with _lock:
        if len(_queue) >= buffer_max:
            _dropped += 1
            if _dropped == 1 or _dropped % 100 == 0:
                logger.warning(
                    "platform_audit_buffer_full",
                    dropped=_dropped,
                    max=buffer_max,
                )
            return
        _queue.append(envelope)


def drain() -> list[dict[str, Any]]:
    """Return and clear all buffered envelopes."""
    with _lock:
        items = list(_queue)
        _queue.clear()
    return items
