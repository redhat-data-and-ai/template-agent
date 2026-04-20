"""PostgreSQL checkpointer for persistent agent conversation state.

This module manages the PostgreSQL-based checkpointer that stores agent state
across requests. It enables conversation persistence, state recovery, and
time-travel debugging by saving agent state snapshots to the database.

Why this exists:
    Agents need to maintain state across multiple API requests. The checkpointer
    saves each conversation turn to PostgreSQL, enabling resume, replay, and
    branching conversations.

Functions:
    get_checkpointer: Get a checkpointer instance (async context manager)
    initialize_checkpointer: One-time setup at application startup
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from template_agent.src.exceptions import AppException, ErrorCodes
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


async def initialize_checkpointer() -> None:
    """Initialize PostgreSQL database schema at application startup.

    Creates checkpoints table and related schema before processing requests.

    Raises:
        AppException: If database connection or schema creation fails.
    """
    try:
        logger.info("Initializing PostgreSQL schema")
        async with AsyncPostgresSaver.from_conn_string(
            settings.database_uri
        ) as checkpointer:
            await checkpointer.setup()
            logger.info("Database schema initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        raise AppException(
            f"Database initialization failed: {e}",
            ErrorCodes.CONFIGURATION_INITIALIZATION_ERROR,
        ) from e


@asynccontextmanager
async def get_checkpointer() -> AsyncGenerator[AsyncPostgresSaver, None]:
    """Get PostgreSQL checkpointer with automatic lifecycle management.

    Yields:
        AsyncPostgresSaver instance connected to PostgreSQL.

    Example:
        async with get_checkpointer() as checkpointer:
            agent = create_deep_agent(checkpointer=checkpointer, ...)
    """
    logger.info("Connecting to PostgreSQL checkpointer")
    async with AsyncPostgresSaver.from_conn_string(
        settings.database_uri
    ) as checkpointer:
        logger.info(f"Checkpointer ready: {type(checkpointer).__name__}")
        yield checkpointer
