"""Aegra custom FastAPI application (``http.app`` entry point).

Registers route modules on a single app that Aegra loads as the base
application and merges core LangGraph Platform routes onto.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from deep_agent.aegra.eval_routes import eval_mgmt_router
from deep_agent.aegra.eval_routes import router as eval_router
from deep_agent.aegra.feedback import feedback_router
from deep_agent.aegra.mcp_routes import router as mcp_router
from deep_agent.aegra.personalization_routes import personalization_router
from deep_agent.aegra.security_middleware import (
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
)
from deep_agent.aegra.thread_cleanup import thread_cleanup_router
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import (
    bind_request_context,
    clear_request_context,
    get_python_logger,
)

logger = get_python_logger()


def _patch_aegra_persistence_if_inmemory() -> None:
    """Disable aegra_api database initialization when running in-memory mode.

    The aegra_api lifespan unconditionally calls db_manager.initialize() which
    opens a PostgreSQL connection pool. When deploying without a database
    (USE_INMEMORY_SAVER=true), this causes the pod to crash. This patch
    replaces initialize() with a no-op so the lifespan completes cleanly.
    """
    import os

    disable_persistence = os.environ.get("AEGRA_DISABLE_PERSISTENCE", "").lower() in (
        "true",
        "1",
    )
    use_inmemory = os.environ.get("USE_INMEMORY_SAVER", "").lower() in ("true", "1")

    if not (disable_persistence or use_inmemory):
        return

    try:
        from aegra_api.core.database import db_manager

        async def _noop_initialize() -> None:
            logger.info("aegra_db_initialize_skipped_inmemory_mode")

        db_manager.initialize = _noop_initialize
        logger.info("aegra_persistence_patched_for_inmemory_mode")
    except ImportError:
        pass


_patch_aegra_persistence_if_inmemory()

from deep_agent.aegra.shutdown import register_atexit  # noqa: E402

register_atexit()


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    from deep_agent.aegra.startup import run_startup
    from deep_agent.src.observability.otel_setup import (
        setup_otel_metrics,
        setup_otel_traces,
    )

    await run_startup()
    setup_otel_metrics(settings, logger)
    setup_otel_traces(_app, settings, logger)
    _verify_custom_thread_delete(_app)
    yield


def _verify_custom_thread_delete(application: FastAPI) -> None:
    """Verify our custom DELETE /threads/{id} takes priority over Aegra's built-in.

    Aegra's _include_core_routers adds a threads_router after our custom
    app routes. FastAPI matches the first registered route, so ours wins.
    This check fails loudly if that assumption ever breaks.
    """
    from deep_agent.aegra.thread_cleanup import delete_thread_with_cleanup

    for route in application.routes:
        if (
            hasattr(route, "path")
            and route.path == "/threads/{thread_id}"
            and hasattr(route, "methods")
            and "DELETE" in route.methods
        ):
            if route.endpoint is delete_thread_with_cleanup:
                logger.info("custom_thread_delete_route_verified")
                return
            raise RuntimeError(
                "custom_thread_delete_route_overridden: "
                "Aegra's built-in DELETE /threads/{thread_id} is taking "
                "priority over our custom cleanup handler. "
                "Check router registration order in http_app.py."
            )
    logger.warning("thread_delete_route_not_found")


app = FastAPI(title="template-agent-custom", lifespan=_lifespan)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler with PII scrubbing for production."""
    from deep_agent.src.pii_scrubber import scrub_error_response

    logger.error(
        "Unhandled exception: %s",
        exc,
        exc_info=True,
        extra={"path": request.url.path, "method": request.method},
    )

    error_response = scrub_error_response(
        detail="An unexpected error occurred. Please try again later.",
        exc=exc,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response,
    )


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Extract correlation headers and bind them to the structured log context.

    Headers consumed:
    - ``X-Trace-ID``: UI-originated trace identifier
    - ``X-Request-ID``: gateway-originated request correlation ID
    - ``X-Org-ID``: organisation owning the agent
    - ``X-Agent-ID``: ``org/name`` agent identifier
    """

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        """Extract correlation headers and bind to structured log context."""
        trace_id = request.headers.get("x-trace-id") or uuid4().hex
        request_id = request.headers.get("x-request-id") or uuid4().hex
        org_id = request.headers.get("x-org-id")
        agent_id = request.headers.get("x-agent-id")
        bind_request_context(
            trace_id=trace_id,
            request_id=request_id,
            org_id=org_id,
            agent_id=agent_id,
        )
        try:
            from opentelemetry import trace as otel_trace

            span = otel_trace.get_current_span()
            if span is not None and span.is_recording():
                span.set_attribute("app.trace_id", trace_id)
        except ImportError:
            pass
        try:
            response = await call_next(request)
            response.headers["X-Trace-ID"] = trace_id
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            clear_request_context()


app.add_middleware(RequestContextMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    RequestSizeLimitMiddleware, max_size_bytes=settings.REQUEST_BODY_MAX_SIZE
)
app.include_router(thread_cleanup_router)
app.include_router(personalization_router)
app.include_router(mcp_router)
app.include_router(feedback_router)
app.include_router(eval_router)
app.include_router(eval_mgmt_router)


@app.get("/version")
def version() -> dict[str, str]:
    """Return service name and version."""
    from deep_agent.aegra import __version__

    return {"service": "template-agent", "version": __version__}
