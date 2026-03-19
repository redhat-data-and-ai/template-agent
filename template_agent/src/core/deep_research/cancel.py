"""Cancel support for deep research runs.

Uses a two-tier architecture (identical to ``plan_store``):

* **L1** -- an in-memory ``set`` for fast, same-pod look-ups.
* **L2** -- an optional LangGraph ``BaseStore`` (Postgres-backed in
  production, ``InMemoryStore`` in dev) that survives pod restarts and
  is visible to every pod.

When no backing store is configured (e.g. in unit tests) the store
degrades gracefully to L1-only behaviour -- which matches the
original single-pod semantics.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from template_agent.utils.pylogger import get_python_logger

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore

logger = get_python_logger()

_CANCEL_NAMESPACE: tuple[str, ...] = ("deep_research", "cancellations")


class CancelStore:
    """Two-tier store for tracking cancelled research runs by thread_id.

    L1 is an in-memory ``set`` (fast, per-pod).  L2 is an optional
    ``BaseStore`` that provides cross-pod persistence.
    """

    def __init__(self) -> None:
        """Initialize the cancel store with an empty set and no backing store."""
        self._cancelled: set[str] = set()
        self._lock = asyncio.Lock()
        self._store: BaseStore | None = None

    def configure_backing_store(self, store: Any) -> None:
        """Attach a LangGraph ``BaseStore`` for cross-pod persistence.

        Safe to call multiple times; the last store wins.  Passing
        ``None`` disables the backing store and reverts to L1-only mode.
        """
        self._store = store
        if store is not None:
            logger.info(
                "CancelStore: backing store configured (%s)",
                type(store).__name__,
            )
        else:
            logger.info("CancelStore: backing store removed (L1-only mode)")

    async def request_cancel(self, thread_id: str) -> None:
        """Mark *thread_id* as cancelled in both L1 and L2."""
        async with self._lock:
            self._cancelled.add(thread_id)

        await self._store_put(thread_id)

    async def is_cancelled(self, thread_id: str) -> bool:
        """Return ``True`` if *thread_id* has been cancelled.

        Checks L1 first; on a miss, falls back to L2 and back-fills L1
        so subsequent checks on the same pod are fast.
        """
        async with self._lock:
            if thread_id in self._cancelled:
                return True

        found = await self._store_get(thread_id)
        if found:
            async with self._lock:
                self._cancelled.add(thread_id)
        return found

    async def clear(self, thread_id: str) -> None:
        """Remove *thread_id* from both L1 and L2."""
        async with self._lock:
            self._cancelled.discard(thread_id)

        await self._store_delete(thread_id)

    # ------------------------------------------------------------------
    # L2 helpers -- every call is wrapped so a store failure never blocks
    # the cancellation path.
    # ------------------------------------------------------------------

    async def _store_put(self, thread_id: str) -> None:
        if self._store is None:
            return
        try:
            await self._store.aput(
                _CANCEL_NAMESPACE,
                thread_id,
                {"cancelled": True},
                index=False,
            )
        except Exception:
            logger.warning(
                "CancelStore: failed to persist cancellation for %s to backing store",
                thread_id,
                exc_info=True,
            )

    async def _store_get(self, thread_id: str) -> bool:
        if self._store is None:
            return False
        try:
            item = await self._store.aget(_CANCEL_NAMESPACE, thread_id)
            return item is not None and bool(item.value.get("cancelled"))
        except Exception:
            logger.warning(
                "CancelStore: failed to read cancellation for %s from backing store",
                thread_id,
                exc_info=True,
            )
            return False

    async def _store_delete(self, thread_id: str) -> None:
        if self._store is None:
            return
        try:
            await self._store.adelete(_CANCEL_NAMESPACE, thread_id)
        except Exception:
            logger.warning(
                "CancelStore: failed to delete cancellation for %s from backing store",
                thread_id,
                exc_info=True,
            )


_cancel_store: CancelStore | None = None


def get_cancel_store() -> CancelStore:
    """Return the singleton CancelStore instance."""
    global _cancel_store
    if _cancel_store is None:
        _cancel_store = CancelStore()
    return _cancel_store
