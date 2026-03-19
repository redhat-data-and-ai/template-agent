"""Global storage management for the template agent system.

This module provides:

* A single global checkpoint (``InMemorySaver``) that persists across the
  entire application lifecycle when using in-memory storage mode.
* A shared ``BaseStore`` instance (``InMemoryStore`` for dev,
  ``AsyncPostgresStore`` for production) that cross-cutting concerns
  like the ``CancelStore`` can use for multi-pod persistence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from langgraph.checkpoint.memory import InMemorySaver

from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)

# ---------------------------------------------------------------------------
# Checkpoint (checkpointer) singleton
# ---------------------------------------------------------------------------

_global_checkpoint: Optional[InMemorySaver] = None

_thread_registry: dict[str, set[str]] = {}


def get_global_checkpoint() -> InMemorySaver:
    """Get the global in-memory checkpoint instance.

    This creates a single checkpoint instance that persists for the entire
    application lifecycle, ensuring all components use the same storage.
    The same instance serves as both checkpointer and store.

    Returns:
        The global InMemorySaver instance.
    """
    global _global_checkpoint
    if _global_checkpoint is None:
        _global_checkpoint = InMemorySaver()
        logger.info("Created global InMemorySaver checkpoint instance")
    return _global_checkpoint


def register_thread(user_id: str, thread_id: str) -> None:
    """Register a thread for a user."""
    global _thread_registry
    if user_id not in _thread_registry:
        _thread_registry[user_id] = set()
    _thread_registry[user_id].add(thread_id)
    logger.info(f"Registered thread {thread_id} for user {user_id}")


def get_user_threads(user_id: str) -> list[str]:
    """Get all threads for a user."""
    global _thread_registry
    threads = list(_thread_registry.get(user_id, set()))
    logger.info(f"Retrieved {len(threads)} threads for user {user_id}: {threads}")
    return threads


get_shared_checkpointer = get_global_checkpoint

# ---------------------------------------------------------------------------
# Shared BaseStore singleton (used by CancelStore and similar cross-cutting
# concerns that need multi-pod visibility).
# ---------------------------------------------------------------------------

_shared_store: BaseStore | None = None
_pg_store_ctx: Any = None


async def initialize_shared_store() -> BaseStore | None:
    """Create and return the shared ``BaseStore``.

    * **In-memory mode** (``USE_INMEMORY_SAVER=True``): returns an
      ``InMemoryStore`` -- sufficient for single-pod / dev deployments.
    * **Production mode**: returns an ``AsyncPostgresStore`` backed by
      the same Postgres instance used for checkpoints.

    This function is idempotent -- calling it again returns the existing
    store.
    """
    global _shared_store, _pg_store_ctx

    if _shared_store is not None:
        return _shared_store

    if settings.USE_INMEMORY_SAVER:
        from langgraph.store.memory import InMemoryStore

        _shared_store = InMemoryStore()
        logger.info("Shared store: InMemoryStore (dev mode)")
    else:
        try:
            from langgraph.store.postgres.aio import AsyncPostgresStore

            _pg_store_ctx = AsyncPostgresStore.from_conn_string(
                settings.database_uri,
            )
            _shared_store = await _pg_store_ctx.__aenter__()
            if hasattr(_shared_store, "setup"):
                await _shared_store.setup()
            logger.info("Shared store: AsyncPostgresStore (production)")
        except Exception:
            from langgraph.store.memory import InMemoryStore

            logger.warning(
                "Failed to create AsyncPostgresStore for shared store; "
                "falling back to InMemoryStore (cancellations will be "
                "pod-local only)",
                exc_info=True,
            )
            _shared_store = InMemoryStore()
            _pg_store_ctx = None

    return _shared_store


async def shutdown_shared_store() -> None:
    """Cleanly close the shared store (release Postgres pool if applicable)."""
    global _shared_store, _pg_store_ctx

    if _pg_store_ctx is not None:
        try:
            await _pg_store_ctx.__aexit__(None, None, None)
        except Exception:
            logger.warning("Error closing shared Postgres store", exc_info=True)
        _pg_store_ctx = None

    _shared_store = None
    logger.info("Shared store shut down")


def get_shared_store() -> BaseStore | None:
    """Return the shared ``BaseStore``, or ``None`` if not yet initialised."""
    return _shared_store
