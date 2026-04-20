"""Application lifecycle management for FastAPI.

This module manages the application startup and shutdown lifecycle, including
initialization of the PostgreSQL checkpointer, backend environment, and Langfuse
observability client.

Why this exists:
    Proper initialization order is critical - database schema must be created before
    requests arrive, the backend venv should be prepared to avoid first-request delay,
    and cleanup must happen gracefully on shutdown. This module centralizes that logic.

Functions:
    lifespan: Async context manager for application lifecycle
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langfuse import Langfuse

from template_agent.src.infrastructure.backend import initialize_backend
from template_agent.src.infrastructure.checkpointer import initialize_checkpointer
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Configure application lifespan.

    This context manager handles the application startup and shutdown
    lifecycle. Checkpointer schema is initialized on startup, while agent
    initialization is deferred to per-request handling to allow for
    authenticated MCP connections.

    Args:
        app: The FastAPI application instance to manage.

    Yields:
        None: The lifespan context for the application.

    Raises:
        AppException: If checkpointer initialization fails on startup.
    """
    logger.info("agent_server_starting")

    # Initialize checkpointer schema on startup
    try:
        await initialize_checkpointer()
    except Exception as e:
        logger.critical("checkpointer_initialization_failed", error=str(e))
        raise

    # Initialize the shell backend (venv + deps) so the first request is fast
    try:
        initialize_backend()
    except Exception as e:
        logger.critical("backend_initialization_failed", error=str(e))
        raise

    # Initialize Langfuse client on startup
    try:
        app.state.langfuse_client = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            base_url=settings.LANGFUSE_BASE_URL,
        )
        logger.info("langfuse_client_initialized")
    except Exception as e:
        logger.warning(f"langfuse_initialization_failed: {e}")
        app.state.langfuse_client = None

    logger.info("agent_server_ready")
    yield
    logger.info("agent_server_shutting_down")

    # Flush pending Langfuse traces on shutdown
    if app.state.langfuse_client:
        try:
            app.state.langfuse_client.shutdown()
            logger.info("langfuse_traces_flushed")
        except Exception as e:
            logger.warning(f"langfuse_shutdown_failed: {e}")
