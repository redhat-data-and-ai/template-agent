"""FastAPI application factory.

This module provides the function to create and configure the FastAPI application
with all routes, middleware, exception handlers, and lifecycle management.

Why this exists:
    Centralizes FastAPI app creation, making it easy to instantiate the app for
    different contexts (production, testing, etc.) with consistent configuration.

Functions:
    create_app: Create and configure the FastAPI application
"""

from fastapi import FastAPI, Request
from starlette.responses import JSONResponse

from template_agent.src.api.lifecycle import lifespan
from template_agent.src.api.middleware import configure_middleware
from template_agent.src.api.routes.agent.feedback import router as feedback_router
from template_agent.src.api.routes.agent.stream import router as stream_router
from template_agent.src.api.routes.health import router as health_router
from template_agent.src.api.routes.memory.history import router as history_router
from template_agent.src.api.routes.memory.threads import router as threads_router
from template_agent.src.exceptions import AppException, ErrorCodes
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application ready to serve requests.
    """
    # Create FastAPI application with lifespan management
    app = FastAPI(lifespan=lifespan)

    # Configure middleware (logging, CORS, etc.)
    configure_middleware(app)

    # Configure application logger
    app.logger = logger

    # Register all route handlers
    app.include_router(health_router)
    app.include_router(stream_router)
    app.include_router(feedback_router)
    app.include_router(history_router)
    app.include_router(threads_router)

    # Register exception handlers
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """Generic exception handler for unhandled exceptions."""
        logger.exception(
            "unhandled_exception",
            request_method=request.method,
            request_path=request.url.path,
            error=str(exc),
        )
        return JSONResponse(
            status_code=ErrorCodes.INTERNAL_SERVER_ERROR.status,
            content={
                "detail_message": str(exc),
                "message": ErrorCodes.INTERNAL_SERVER_ERROR.message,
                "error_code": ErrorCodes.INTERNAL_SERVER_ERROR.code,
            },
        )

    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        """App exception handler for handled application exceptions."""
        logger.warning(
            "app_exception",
            request_method=request.method,
            request_path=request.url.path,
            error=str(exc),
            error_code=exc.code,
        )
        return JSONResponse(
            status_code=exc.status,
            content={
                "detail_message": exc.detail,
                "message": exc.message,
                "error_code": exc.code,
            },
        )

    return app


# Create the application instance
app = create_app()
