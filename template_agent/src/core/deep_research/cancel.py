"""Cancel support for deep research runs."""

import asyncio
from typing import Set


class CancelStore:
    """In-memory store for tracking cancelled research runs by thread_id."""

    def __init__(self) -> None:
        """Initialize the cancel store with an empty set of cancelled thread IDs."""
        self._cancelled: Set[str] = set()
        self._lock = asyncio.Lock()

    async def request_cancel(self, thread_id: str) -> None:
        """Mark the given thread_id as cancelled."""
        async with self._lock:
            self._cancelled.add(thread_id)

    async def is_cancelled(self, thread_id: str) -> bool:
        """Return True if the given thread_id has been requested for cancellation."""
        async with self._lock:
            return thread_id in self._cancelled

    async def clear(self, thread_id: str) -> None:
        """Remove the thread_id from the cancelled set."""
        async with self._lock:
            self._cancelled.discard(thread_id)


_cancel_store: CancelStore | None = None


def get_cancel_store() -> CancelStore:
    """Return the singleton CancelStore instance."""
    global _cancel_store
    if _cancel_store is None:
        _cancel_store = CancelStore()
    return _cancel_store
