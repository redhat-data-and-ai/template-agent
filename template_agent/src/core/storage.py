"""Global storage management for the template agent system.

This module provides a single global checkpoint instance that persists across
the entire application lifecycle when using in-memory storage mode.
"""

from typing import Optional

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.memory import InMemoryStore

from template_agent.src.core.exceptions.exceptions import AppException, AppExceptionCode
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)

# Global singletons for the entire application lifecycle
_global_checkpoint: Optional[InMemorySaver] = None
_global_store: Optional[InMemoryStore] = None
_thread_registry: dict[str, set[str]] = {}


def get_global_checkpoint() -> InMemorySaver:
    """Get the global in-memory checkpoint instance.

    Returns:
        The global InMemorySaver instance.
    """
    global _global_checkpoint
    if _global_checkpoint is None:
        _global_checkpoint = InMemorySaver()
        logger.info("Created global InMemorySaver checkpoint instance")
    return _global_checkpoint


def get_global_store() -> InMemoryStore:
    """Get the global in-memory store instance.

    Returns:
        The global InMemoryStore instance.
    """
    global _global_store
    if _global_store is None:
        _global_store = InMemoryStore()
        logger.info("Created global InMemoryStore instance")
    return _global_store


def register_thread(user_id: str, thread_id: str) -> None:
    """Register a thread for a user.

    Args:
        user_id: The user ID
        thread_id: The thread ID to register
    """
    global _thread_registry
    if user_id not in _thread_registry:
        _thread_registry[user_id] = set()
    _thread_registry[user_id].add(thread_id)
    logger.info(f"Registered thread {thread_id} for user {user_id}")


def get_user_threads(user_id: str) -> list[str]:
    """Get all threads for a user.

    Args:
        user_id: The user ID

    Returns:
        List of thread IDs for the user
    """
    global _thread_registry
    threads = list(_thread_registry.get(user_id, set()))
    logger.info(f"Retrieved {len(threads)} threads for user {user_id}: {threads}")
    return threads


def reset_global_storage() -> None:
    """Reset the global checkpoint instance.

    This is useful for testing or when you want to clear all data.
    """
    global _global_checkpoint, _global_store, _thread_registry
    _global_checkpoint = None
    _global_store = None
    _thread_registry = {}
    logger.info("Reset global storage instances and thread registry")


async def initialize_database() -> None:
    """Initialize PostgreSQL database schema on application startup.

    Ensures the checkpoints table and related schema are created
    before any requests are processed. Only runs when using PostgreSQL
    storage (USE_INMEMORY_SAVER=False).

    Raises:
        AppException: If database connection or schema creation fails.
    """
    if settings.USE_INMEMORY_SAVER:
        logger.info("Using in-memory storage - skipping database initialization")
        return

    try:
        logger.info("Initializing PostgreSQL database schema")
        async with AsyncPostgresSaver.from_conn_string(
            settings.database_uri
        ) as checkpoint:
            if hasattr(checkpoint, "setup"):
                await checkpoint.setup()
                logger.info("Database schema initialized successfully")
            else:
                logger.warning(
                    "AsyncPostgresSaver does not have setup method"
                    " - schema may need manual creation"
                )
    except Exception as e:
        logger.error(f"Failed to initialize database schema: {e}", exc_info=True)
        raise AppException(
            f"Database initialization failed: {str(e)}",
            AppExceptionCode.CONFIGURATION_INITIALIZATION_ERROR,
        )


# Backward compatibility aliases
get_shared_checkpointer = get_global_checkpoint
get_shared_store = get_global_store
reset_shared_storage = reset_global_storage
