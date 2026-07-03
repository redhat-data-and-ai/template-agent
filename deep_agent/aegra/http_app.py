"""Aegra custom FastAPI application (``http.app`` entry point).

Registers route modules on a single app that Aegra loads as the base
application and merges core LangGraph Platform routes onto.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

from deep_agent.aegra.feedback import feedback_router
from deep_agent.aegra.mcp_routes import router as mcp_router
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
    yield


app = FastAPI(title="template-agent-custom", lifespan=_lifespan)


class TraceIDMiddleware(BaseHTTPMiddleware):
    """Propagate X-Trace-ID from incoming requests into the logging context.

    Every log line emitted during a request will include the trace_id,
    enabling end-to-end correlation across UI → BFF → Agent.
    """

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        """Bind trace ID to logging context and echo it on the response."""
        trace_id = request.headers.get("x-trace-id") or uuid4().hex
        bind_request_context(trace_id=trace_id)
        response = await call_next(request)
        response.headers["X-Trace-ID"] = trace_id
        clear_request_context()
        return response


app.add_middleware(TraceIDMiddleware)
app.include_router(mcp_router)
app.include_router(feedback_router)
