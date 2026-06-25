"""User feedback HTTP endpoint for Langfuse scores (B-1).

Registers ``POST /feedback`` on the Aegra custom FastAPI app (see
``http.app`` in ``aegra.json``). Aegra loads this app as the base
application and merges core LangGraph Platform routes onto it.

When Langfuse credentials are absent, submissions are logged and accepted
without contacting Langfuse.

When ``thread_id`` and ``message_id`` are present, feedback is also
persisted to Postgres for cross-session history.
"""

from __future__ import annotations

import json
from typing import Any, Literal
from uuid import UUID, uuid4

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.middleware.base import BaseHTTPMiddleware

from deep_agent.aegra.telemetry import get_langfuse_client
from deep_agent.src.agent.config import agent_config
from deep_agent.src.feedback.repository import FeedbackRepository
from deep_agent.src.schema import FeedbackRequest, FeedbackResponse
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

        db_manager.initialize = _noop_initialize  # type: ignore[method-assign]
        logger.info("aegra_persistence_patched_for_inmemory_mode")
    except ImportError:
        pass


_patch_aegra_persistence_if_inmemory()

from deep_agent.aegra.shutdown import register_atexit

register_atexit()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    from deep_agent.aegra.startup import run_startup
    from deep_agent.src.observability.otel_setup import setup_otel_metrics, setup_otel_traces

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
        trace_id = request.headers.get("x-trace-id") or uuid4().hex
        bind_request_context(trace_id=trace_id)
        response = await call_next(request)
        response.headers["X-Trace-ID"] = trace_id
        clear_request_context()
        return response


app.add_middleware(TraceIDMiddleware)


def _score_to_feedback_polarity(req: FeedbackRequest) -> Literal["up", "down"]:
    """Map request name/value to stored feedback polarity."""
    name_lower = (req.name or "").lower()
    if "down" in name_lower or "negative" in name_lower:
        return "down"
    if "up" in name_lower or "positive" in name_lower:
        return "up"
    return "up" if req.value >= 0.5 else "down"


async def _persist_feedback_to_postgres(req: FeedbackRequest) -> None:
    if not req.thread_id or not req.message_id:
        return
    if not settings.database_uri:
        logger.warning(
            "feedback_postgres_skipped_no_database_uri",
            thread_id=req.thread_id,
            message_id=req.message_id,
        )
        return
    polarity = _score_to_feedback_polarity(req)
    user_id = req.user_id if req.user_id else "anonymous"
    repo = FeedbackRepository(settings.database_uri)
    await repo.upsert_feedback(
        req.thread_id,
        req.message_id,
        user_id,
        polarity,
        req.trace_id,
    )
    logger.info(
        "feedback_recorded_postgres",
        thread_id=req.thread_id,
        message_id=req.message_id,
        user_id=user_id,
        feedback=polarity,
    )


def _resolve_langfuse_trace_id(client: Any, thread_id: str | None) -> str | None:
    """Look up the real Langfuse trace_id by session_id (thread_id).

    The Langfuse SDK auto-generates trace IDs that differ from LangGraph run_ids.
    This queries the Langfuse API to find the latest trace in the session so
    feedback scores attach to the correct trace in the dashboard.
    """
    if not thread_id:
        return None
    try:
        traces = client.api.trace.list(session_id=thread_id, limit=1)
        if traces.data:
            return str(traces.data[0].id)
    except Exception as exc:
        logger.debug(
            "langfuse_trace_lookup_failed",
            session_id=thread_id,
            error=str(exc),
        )
    return None


async def record_feedback(request_data: dict[str, Any]) -> FeedbackResponse:
    """Validate feedback input, optionally record a Langfuse score, return success.

    Args:
        request_data: Raw JSON object (mapping) from the client.

    Returns:
        ``FeedbackResponse`` with status ``success``.

    Raises:
        ValidationError: If the payload does not satisfy ``FeedbackRequest``.
        RuntimeError: If Langfuse is configured but score submission fails.
    """
    req = FeedbackRequest.model_validate(request_data)

    logger.info(
        "feedback_received",
        trace_id=req.trace_id,
        name=req.name,
        value=req.value,
        kwargs_keys=sorted(req.kwargs.keys()) if req.kwargs else [],
    )

    langfuse_client = get_langfuse_client()
    if langfuse_client is None:
        logger.info(
            "feedback_skipped_langfuse_unconfigured",
            trace_id=req.trace_id,
            name=req.name,
        )
        await _persist_feedback_to_postgres(req)
        return FeedbackResponse()

    resolved_trace_id = _resolve_langfuse_trace_id(langfuse_client, req.thread_id)
    effective_trace_id = resolved_trace_id or req.trace_id

    if resolved_trace_id and resolved_trace_id != req.trace_id:
        logger.info(
            "feedback_trace_id_resolved",
            original=req.trace_id,
            resolved=resolved_trace_id,
            thread_id=req.thread_id,
        )

    try:
        langfuse_client.create_score(
            trace_id=effective_trace_id,
            name=req.name,
            value=req.value,
            data_type="BOOLEAN",
            **(req.kwargs or {}),
        )
        logger.info(
            "feedback_recorded_langfuse",
            trace_id=effective_trace_id,
            name=req.name,
        )
    except Exception as exc:
        logger.warning(
            "feedback_langfuse_score_failed",
            trace_id=effective_trace_id,
            name=req.name,
            error=str(exc),
        )

    await _persist_feedback_to_postgres(req)
    return FeedbackResponse()


async def feedback_handler(request: Request) -> JSONResponse:
    """ASGI/Starlette handler: read JSON, validate, record feedback."""
    try:
        body_bytes = await request.body()
        if not body_bytes.strip():
            return JSONResponse(
                status_code=422,
                content={"detail": [{"msg": "Empty body", "type": "value_error"}]},
            )
        payload = json.loads(body_bytes.decode("utf-8"))
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=422,
            content={
                "detail": [{"msg": "Invalid JSON body", "type": "json_invalid"}],
            },
        )

    if not isinstance(payload, dict):
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {
                        "msg": "JSON body must be an object",
                        "type": "type_error",
                    },
                ],
            },
        )

    try:
        resp = await record_feedback(payload)
    except ValidationError as exc:
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors(include_url=False)},
        )
    except Exception:
        logger.exception("feedback_handler_error")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return JSONResponse(
        status_code=200,
        content=resp.model_dump(),
    )


def _validate_thread_id(thread_id: str) -> str:
    """Validate that thread_id is a well-formed UUID."""
    try:
        return str(UUID(thread_id))
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid thread_id format (expected UUID)"
        ) from None


@app.get("/feedback/{thread_id}")
async def get_thread_feedback(
    thread_id: str, user_id: str = "anonymous"
) -> dict[str, Any]:
    """Return all feedback for a thread."""
    thread_id = _validate_thread_id(thread_id)
    if not settings.database_uri:
        return {"feedback": []}
    repo = FeedbackRepository(settings.database_uri)
    items = await repo.list_feedback(thread_id, user_id)
    return {"feedback": items}


@app.get("/threads/{thread_id}/token-usage")
async def get_thread_token_usage_endpoint(thread_id: str) -> dict[str, Any]:
    """Return cumulative token usage for a thread."""
    thread_id = _validate_thread_id(thread_id)
    from dataclasses import asdict

    from deep_agent.src.token_budget.service import (
        TokenUsageNotFoundError,
        TokenUsageUnavailableError,
        get_thread_token_usage,
    )

    try:
        usage = await get_thread_token_usage(thread_id)
    except TokenUsageNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No token usage for thread {thread_id}",
        ) from None
    except TokenUsageUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="Token usage storage unavailable",
        ) from None

    return asdict(usage)


@app.get("/info")
async def get_agent_info() -> dict[str, str]:
    """Return agent identity metadata from config."""
    return {"name": agent_config.get_name()}


app.add_api_route("/feedback", feedback_handler, methods=["POST"])
