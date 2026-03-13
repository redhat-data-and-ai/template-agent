"""FastAPI server implementation for the template agent.

This module provides the main FastAPI application setup, including
middleware configuration, route registration, and application lifecycle
management for the template agent service.
"""

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from template_agent.src.core.agent import initialize_database
from template_agent.src.core.deep_research.cancel import get_cancel_store
from template_agent.src.core.exceptions.exceptions import AppException, AppExceptionCode
from template_agent.src.routes.deep_research_plan import (
    router as deep_research_plan_router,
)
from template_agent.src.routes.feedback import router as feedback_router
from template_agent.src.routes.health import router as health_router
from template_agent.src.routes.history import router as history_router
from template_agent.src.routes.stream import router as stream_router
from template_agent.src.routes.threads import router as threads_router
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger
from template_agent.utils.tracing import AgentTracer

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


class RequestLoggingMiddleware:
    """Pure ASGI middleware for request/response logging.

    Unlike BaseHTTPMiddleware, this does NOT buffer streaming response bodies,
    allowing SSE and other streaming responses to flow through uninterrupted.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not settings.REQUEST_LOGGING_ENABLED:
            await self.app(scope, receive, send)
            return

        start_time = time.time()
        request = Request(scope)

        request_data: dict[str, Any] = {
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host if request.client else None,
            "query_params": dict(request.query_params)
            if request.query_params
            else None,
        }
        if settings.REQUEST_LOG_HEADERS:
            request_data["headers"] = dict(request.headers)

        logger.info("incoming_request", **request_data)

        response_status: int | None = None

        async def send_wrapper(message: Message) -> None:
            nonlocal response_status
            if message["type"] == "http.response.start":
                response_status = message["status"]
                duration_ms = (time.time() - start_time) * 1000
                response_data: dict[str, Any] = {
                    "method": request_data["method"],
                    "path": request_data["path"],
                    "status_code": response_status,
                    "duration_ms": round(duration_ms, 2),
                }
                if settings.REQUEST_LOG_HEADERS:
                    headers = dict(
                        (
                            k.decode() if isinstance(k, bytes) else k,
                            v.decode() if isinstance(v, bytes) else v,
                        )
                        for k, v in message.get("headers", [])
                    )
                    response_data["headers"] = headers
                logger.info("outgoing_response", **response_data)
            await send(message)

        await self.app(scope, receive, send_wrapper)


_SKIP_TRACE_PATHS = frozenset({"/health", "/docs", "/openapi.json", "/redoc"})


class RequestTracingMiddleware:
    """Pure ASGI middleware that creates a Langfuse AgentTracer per request.

    For non-health/docs paths, an ``AgentTracer`` is attached to
    ``request.state.root_tracer`` so downstream handlers can record
    spans and generations.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        path = request.url.path

        if path in _SKIP_TRACE_PATHS:
            await self.app(scope, receive, send)
            return

        trace_id = request.headers.get("X-Trace-ID") or str(uuid4())
        root_tracer = AgentTracer(
            name=f"{request.method} {path}",
            trace_id=trace_id,
            metadata={
                "method": request.method,
                "path": path,
                "client_host": request.client.host if request.client else None,
            },
        )
        request.state.root_tracer = root_tracer

        await self.app(scope, receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Configure application lifespan.

    This context manager handles the application startup and shutdown
    lifecycle. Database schema is initialized on startup, while agent
    initialization is deferred to per-request handling to allow for
    authenticated MCP connections.

    Args:
        app: The FastAPI application instance to manage.

    Yields:
        None: The lifespan context for the application.

    Raises:
        AppException: If database initialization fails on startup.
    """
    logger.info("Agent server starting up")

    # Initialize database schema on startup
    try:
        await initialize_database()
    except Exception as e:
        logger.critical(f"Failed to initialize database on startup: {e}")
        raise

    logger.info("Agent server ready - MCP connection will be established per-request")
    yield
    logger.info("Agent server shutting down")


# Create FastAPI application with lifespan management
app = FastAPI(lifespan=lifespan)

# Register request logging middleware first to capture all requests
app.add_middleware(RequestLoggingMiddleware)

# Register tracing middleware to create per-request Langfuse traces
app.add_middleware(RequestTracingMiddleware)

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
app.include_router(deep_research_plan_router)


@app.delete("/v1/cancel/{thread_id}")
async def cancel_deep_research(thread_id: str):
    """Cancel an in-progress deep research run."""
    store = get_cancel_store()
    await store.request_cancel(thread_id)
    logger.info(f"Cancel requested for thread_id={thread_id}")
    return {"status": "cancelled", "thread_id": thread_id}


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
    logger.warning(
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
