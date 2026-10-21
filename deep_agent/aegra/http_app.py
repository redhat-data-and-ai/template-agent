"""Aegra custom FastAPI application (``http.app`` entry point).

Registers route modules on a single app that Aegra loads as the base
application and merges core LangGraph Platform routes onto.
"""

from __future__ import annotations

import copy
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from deep_agent.aegra.admin_routes import router as admin_router
from deep_agent.aegra.admission import AdmissionControlMiddleware
from deep_agent.aegra.analytics_routes import router as analytics_router
from deep_agent.aegra.attachment_routes import router as attachment_router
from deep_agent.aegra.auth_helpers import authenticated_user_id
from deep_agent.aegra.channel_routes import router as channel_router
from deep_agent.aegra.document_routes import router as document_router
from deep_agent.aegra.eval_routes import eval_mgmt_router
from deep_agent.aegra.eval_routes import router as eval_router
from deep_agent.aegra.feedback import feedback_router
from deep_agent.aegra.fork_routes import router as fork_router
from deep_agent.aegra.mcp_crypto import McpKeyMissingError
from deep_agent.aegra.mcp_routes import router as mcp_router
from deep_agent.aegra.personalization_routes import router as personalization_router
from deep_agent.aegra.projects_routes import projects_router
from deep_agent.aegra.search_routes import router as search_router
from deep_agent.aegra.security_middleware import (
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
    StreamingBodyLimitMiddleware,
)
from deep_agent.aegra.share_routes import router as share_router
from deep_agent.aegra.skill_routes import router as skill_router
from deep_agent.aegra.tag_routes import router as tag_router
from deep_agent.aegra.thread_cleanup import thread_cleanup_router
from deep_agent.aegra.upload_routes import upload_router
from deep_agent.src.agent.config import agent_config
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


@app.exception_handler(McpKeyMissingError)
async def mcp_key_missing_handler(
    request: Request, exc: McpKeyMissingError
) -> JSONResponse:
    """Answer a missing token-encryption key with what is missing.

    Without this it reached the catch-all below and came back as
    "An unexpected error occurred. Please try again later." — advice
    to wait for a condition that will never change on its own, for a
    fault the server had already identified precisely. Connecting an
    MCP server then looks like a broken integration rather than an
    unfinished deployment.

    503 rather than 500: the dependency the request needs is not
    configured, and retrying is pointless until it is. The message
    names an environment variable and how to generate a value — no
    secret, and nothing an operator does not already have.
    """
    logger.error("MCP token encryption key missing: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "detail": str(exc),
            "error_type": "configuration_error",
        },
    )


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


_TRACEPARENT = re.compile(r"^00-([0-9a-f]{32})-[0-9a-f]{16}-[0-9a-f]{2}$")


def _inbound_trace_id(request: Request) -> str | None:
    """The W3C trace id for this request, if there is one.

    Minting a fresh uuid4 when no ``X-Trace-ID`` arrives gives the log
    line an identifier that exists in no other system — the trace of
    the same request, wherever it was exported, is filed under the W3C
    id instead, and nothing joins the two. Preferring the W3C id costs
    nothing when both are present (``X-Trace-ID`` still wins, so the
    gateway's contract is unchanged) and makes logs and traces land on
    one key when only ``traceparent`` is.

    The active span is the better source because the instrumentation
    has already validated and, where sampling demanded it, replaced
    the inbound context. The header is parsed only as a fallback for
    deployments running with OTEL disabled, which is the default.
    """
    from deep_agent.utils.pylogger import current_otel_ids

    otel_trace_id, _ = current_otel_ids()
    if otel_trace_id:
        return otel_trace_id

    match = _TRACEPARENT.match(request.headers.get("traceparent", "").strip().lower())
    if match and match.group(1) != "0" * 32:
        return match.group(1)
    return None


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Extract correlation headers and bind them to the structured log context.

    Headers consumed:
    - ``X-Trace-ID``: UI-originated trace identifier
    - ``traceparent``: W3C trace context, used when X-Trace-ID is absent
    - ``X-Request-ID``: gateway-originated request correlation ID
    - ``X-Org-ID``: organisation owning the agent
    - ``X-Agent-ID``: ``org/name`` agent identifier
    - ``X-User-ID``: BFF-forwarded session username (prefs/rules/store)
    """

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        """Extract correlation headers and bind to structured log context."""
        trace_id = (
            request.headers.get("x-trace-id")
            or _inbound_trace_id(request)
            or uuid4().hex
        )
        request_id = request.headers.get("x-request-id") or uuid4().hex
        org_id = request.headers.get("x-org-id")
        agent_id = request.headers.get("x-agent-id")
        user_id = request.headers.get("x-user-id")
        bind_request_context(
            trace_id=trace_id,
            request_id=request_id,
            org_id=org_id,
            agent_id=agent_id,
            user_id=user_id,
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
# Multipart framing (boundary, part headers) sits on top of the archive
# bytes; 64KB is ample for a handful of parts and keeps the body cap from
# rejecting an archive that is itself within UPLOAD_MAX_ARCHIVE_BYTES.
from deep_agent.src.attachments.config import AttachmentsConfig  # noqa: E402

_MULTIPART_OVERHEAD_ALLOWANCE = 64 * 1024

# /upload legitimately carries more than the global cap, but it is never
# exempt: FastAPI parses and spools a multipart body before the route can
# check the feature flag or authorization, so an unlimited route would let
# an unauthenticated caller fill the disk. It gets a higher limit instead,
# with headroom over UPLOAD_MAX_ARCHIVE_BYTES for multipart framing --
# the archive itself is still capped exactly while it is buffered.
_UPLOAD_BODY_LIMIT = settings.UPLOAD_MAX_ARCHIVE_BYTES + _MULTIPART_OVERHEAD_ALLOWANCE
_BODY_LIMITS = {"/upload": _UPLOAD_BODY_LIMIT}

# Counts bytes as they stream in, so an oversized body is cut off before
# the multipart parser ever sees it. Added first so it wraps outermost.
# One number for both, sized from the attachment caps rather than a
# multiple of the JSON limit: `/threads/{id}/state` is how an
# attachment is written, and it is also a route any authenticated
# client can post plain JSON to.
_ATTACHMENT_BODY_LIMIT = (
    int(
        AttachmentsConfig().max_file_bytes * 4 / 3  # base64
    )
    + _MULTIPART_OVERHEAD_ALLOWANCE
)

app.add_middleware(
    StreamingBodyLimitMiddleware,
    max_size_bytes=settings.REQUEST_BODY_MAX_SIZE,
    path_limits=_BODY_LIMITS,
    upload_max_bytes=_ATTACHMENT_BODY_LIMIT,
)
# Cheap pre-check on the declared Content-Length, so a well-behaved client
# is rejected before sending anything at all.
app.add_middleware(
    RequestSizeLimitMiddleware,
    max_size_bytes=settings.REQUEST_BODY_MAX_SIZE,
    path_limits=_BODY_LIMITS,
    upload_max_bytes=_ATTACHMENT_BODY_LIMIT,
)
app.add_middleware(
    AdmissionControlMiddleware, max_concurrent=settings.MAX_CONCURRENT_RUNS
)
app.include_router(thread_cleanup_router)
app.include_router(skill_router)
app.include_router(document_router)
app.include_router(personalization_router)
app.include_router(projects_router)
app.include_router(mcp_router)
app.include_router(feedback_router)
app.include_router(admin_router)
app.include_router(eval_router)
app.include_router(eval_mgmt_router)
# Externally reachable — see deep_agent/aegra/channel_routes.py.
app.include_router(channel_router)
app.include_router(search_router)
app.include_router(fork_router)
app.include_router(attachment_router)
app.include_router(analytics_router)
app.include_router(tag_router)
app.include_router(share_router)
app.include_router(upload_router)


@app.get("/version")
def version() -> dict[str, str]:
    """Return service name and version."""
    from deep_agent.aegra import __version__

    return {"service": "template-agent", "version": __version__}


@app.get("/config/agent-structure")
async def agent_structure(request: Request) -> dict:
    """Return the agent structure in a dict from config."""
    await authenticated_user_id(request)
    name: str = agent_config.get_name()

    orchestrator: dict[str, Any] = copy.deepcopy(agent_config.get_orchestrator_config())
    orchestrator.pop("body", None)
    orchestrator.pop("skill_paths", None)

    subagents: dict[str, dict[str, Any]] = copy.deepcopy(
        agent_config.get_all_subagent_configs()
    )
    for key in subagents:
        subagents[key].pop("body", None)
        subagents[key].pop("skill_paths", None)

    mcp_servers: dict[str, Any] = copy.deepcopy(agent_config.get_mcp_servers())
    for key in mcp_servers:
        mcp_servers[key].pop("ssl_verify", None)
        mcp_servers[key].pop("oauth", None)
        mcp_servers[key].pop("auth_env_var", None)

    # Names only here; the accessor returns name -> directory and
    # the structure endpoint reports what exists, not where it is.
    # Insertion order, not sorted — the order is the config's.
    available_skills: list[str] = list(agent_config.get_available_skills())

    return {
        "name": name,
        "orchestrator": orchestrator,
        "subagents": subagents,
        "mcp_servers": mcp_servers,
        "available_skills": available_skills,
    }
