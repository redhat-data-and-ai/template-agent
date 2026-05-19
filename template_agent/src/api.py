"""FastAPI server implementation for the template agent.

This module provides the main FastAPI application setup, including
middleware configuration, route registration, and application lifecycle
management for the template agent service.
"""

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Callable
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from template_agent.src.core.agent import initialize_database
from template_agent.src.core.exceptions.exceptions import AppException, AppExceptionCode
from template_agent.src.routes.feedback import router as feedback_router
from template_agent.src.routes.health import router as health_router
from template_agent.src.routes.history import router as history_router
from template_agent.src.routes.stream import router as stream_router
from template_agent.src.routes.threads import router as threads_router
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all incoming requests and outgoing responses."""

    async def dispatch(self, request: Request, call_next: Callable):
        """Process and log incoming requests and outgoing responses."""
        request.state.request_id = request.headers.get("X-Request-ID") or str(uuid4())

        if not settings.REQUEST_LOGGING_ENABLED:
            return await call_next(request)

        start_time = time.time()

        # Capture request details
        request_data = {
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host if request.client else None,
            "query_params": dict(request.query_params)
            if request.query_params
            else None,
        }

        # Optionally log headers
        if settings.REQUEST_LOG_HEADERS:
            request_data["headers"] = dict(request.headers)

        # Optionally log request body
        if settings.REQUEST_LOG_BODY:
            try:
                body_bytes = await request.body()
                body_size = len(body_bytes)

                if body_size > 0:
                    request_data["body_size"] = body_size
                    if (
                        settings.REQUEST_LOG_BODY_MAX_SIZE == 0
                        or body_size <= settings.REQUEST_LOG_BODY_MAX_SIZE
                    ):
                        try:
                            body_str = body_bytes.decode("utf-8")
                            request_data["body"] = body_str
                        except UnicodeDecodeError:
                            request_data["body"] = "<binary data>"
                    else:
                        request_data["body"] = f"<truncated: {body_size} bytes>"

                # Rebuild request with body
                async def receive():
                    return {"type": "http.request", "body": body_bytes}

                request = Request(request.scope, receive)
            except Exception as e:
                logger.warning("Failed to read request body", error=str(e))

        logger.info("incoming_request", **request_data)

        # Process request
        response = await call_next(request)

        # Capture response details
        duration_ms = (time.time() - start_time) * 1000
        response_data = {
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
        }

        # Optionally log response headers
        if settings.REQUEST_LOG_HEADERS:
            response_data["headers"] = dict(response.headers)

        logger.info("outgoing_response", **response_data)

        return response


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

# =============================================================================
# Middleware registration order (Starlette executes LAST registered = OUTERMOST):
# Execution order: CORS → RequestLogging → A2AIdentity → Route
# A2AIdentityMiddleware enforces X-Calling-Agent-ID on /a2a/* and (when
# PROTECT_STREAM_ENDPOINT is True) /v1/stream.  Rejected callers get HTTP 403
# per A2A spec Section 7.4.  The context builder then reads the validated
# identity from request.state — it does not re-enforce.
# Registration order: A2AIdentity → RequestLogging → CORS
# =============================================================================

# 1. A2A Identity middleware (innermost — protects /a2a and optionally /v1/stream)
if settings.A2A_ENABLED:
    from template_agent.src.middleware.a2a_identity import A2AIdentityMiddleware

    app.add_middleware(A2AIdentityMiddleware)

# 2. Request logging middleware (sets request.state.request_id, logs requests)
app.add_middleware(RequestLoggingMiddleware)

# 3. CORS middleware (outermost — handles preflight before anything else)
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

# --- A2A Protocol (mounted as a Starlette sub-application) ---
if settings.A2A_ENABLED:
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.routes import (
        create_agent_card_routes,
        create_jsonrpc_routes,
        create_rest_routes,
    )
    from a2a.server.tasks import DatabaseTaskStore, InMemoryTaskStore, TaskStore
    from starlette.applications import Starlette

    from template_agent.src.a2a.agent_card import build_agent_card
    from template_agent.src.a2a.auth import A2AServerCallContextBuilder
    from template_agent.src.a2a.executor import TemplateAgentExecutor

    agent_card = build_agent_card()

    task_store: TaskStore
    if settings.USE_INMEMORY_SAVER:
        task_store = InMemoryTaskStore()
        logger.info("A2A task store: in-memory (development mode)")
    else:
        from sqlalchemy.ext.asyncio import create_async_engine

        a2a_engine = create_async_engine(
            settings.async_database_uri,
            echo=False,
            pool_pre_ping=True,
        )
        task_store = DatabaseTaskStore(engine=a2a_engine)
        logger.info("A2A task store: PostgreSQL (persistent)")

    a2a_request_handler = DefaultRequestHandler(
        agent_executor=TemplateAgentExecutor(
            supported_output_modes=list(agent_card.default_output_modes),
        ),
        task_store=task_store,
        agent_card=agent_card,
    )

    a2a_context_builder = A2AServerCallContextBuilder()

    # Collect all SDK-managed routes into a single Starlette sub-app.
    # This keeps A2A error handling self-contained (JSON-RPC errors stay
    # JSON-RPC shaped) and avoids FastAPI's generic exception handlers
    # interfering with A2A responses.
    a2a_routes = []

    # Agent Card discovery (public, no auth)
    a2a_routes.extend(create_agent_card_routes(agent_card))

    # JSON-RPC binding at /a2a with v0.3 backward compatibility
    a2a_routes.extend(
        create_jsonrpc_routes(
            request_handler=a2a_request_handler,
            rpc_url="/a2a",
            context_builder=a2a_context_builder,
            enable_v0_3_compat=True,
        )
    )

    # HTTP+JSON/REST binding (spec Section 11 / Section 5.3)
    # Provides: POST /message:send, POST /message:stream,
    #           GET /tasks, GET /tasks/{id}, POST /tasks/{id}:cancel,
    #           POST /tasks/{id}:subscribe, push notification CRUD,
    #           GET /extendedAgentCard
    a2a_routes.extend(
        create_rest_routes(
            request_handler=a2a_request_handler,
            context_builder=a2a_context_builder,
            enable_v0_3_compat=True,
            path_prefix="/a2a",
        )
    )

    a2a_app = Starlette(routes=a2a_routes)
    app.mount("/", a2a_app)

    logger.info(
        "A2A protocol enabled: card at /.well-known/agent-card.json, "
        "JSON-RPC at /a2a (v1.0 + v0.3 compat), "
        "REST at /message:send, /tasks, etc."
    )


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
