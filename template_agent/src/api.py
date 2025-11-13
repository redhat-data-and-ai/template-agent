"""FastAPI server implementation for the template agent.

This module provides the main FastAPI application setup, including
middleware configuration, route registration, and application lifecycle
management for the template agent service.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from template_agent.src.core.exceptions.exceptions import AppException, AppExceptionCode
from template_agent.src.routes.feedback import router as feedback_router
from template_agent.src.routes.health import router as health_router
from template_agent.src.routes.history import router as history_router
from template_agent.src.routes.stream import router as stream_router
from template_agent.src.routes.threads import router as threads_router
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Configure application lifespan.

    This context manager handles the application startup and shutdown
    lifecycle. Agent initialization is deferred to per-request handling
    to allow for authenticated MCP connections.

    Args:
        app: The FastAPI application instance to manage.

    Yields:
        None: The lifespan context for the application.
    """
    logger.info(
        "Agent server starting - MCP connection will be established per-request"
    )
    yield
    logger.info("Agent server shutting down")


# Create FastAPI application with lifespan management
app = FastAPI(lifespan=lifespan)

# Configure CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure application logger
app.logger = get_python_logger(settings.PYTHON_LOG_LEVEL)

# Register all route handlers
app.include_router(health_router)
app.include_router(stream_router)
app.include_router(feedback_router)
app.include_router(history_router)
app.include_router(threads_router)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Generic exception handler for unhandled exceptions."""
    logger.exception(
        f"Unhandled exception occurred for request_method={request.method}, request_path={request.url.path}, error={exc}"
    )
    logger.debug(f"Unhandled exception occurred for request={request}, error={exc}")
    return JSONResponse(
        status_code=AppExceptionCode.INTERNAL_SERVER_ERROR.response_code,
        content={
            "detail_message": str(exc),
            "message": AppExceptionCode.INTERNAL_SERVER_ERROR.message,
            "error_code": AppExceptionCode.INTERNAL_SERVER_ERROR.error_code,
        },
    )


@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    """App exception handler for unhandled exceptions."""
    logger.warn(
        f"App exception occurred for request_method={request.method}, request_path={request.url.path}, error={exc}"
    )
    logger.debug(f"App exception occurred for request={request}, error={exc}")
    return JSONResponse(
        status_code=exc.response_code,
        content={
            "detail_message": exc.detail_message,
            "message": exc.message,
            "error_code": exc.error_code,
        },
    )
