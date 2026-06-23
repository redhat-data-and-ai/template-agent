"""OTEL emission for per-call and daily token usage."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import Any

from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_counters_initialized = False
_token_counter: Any | None = None
_thread_total_counter: Any | None = None
_daily_total_counter: Any | None = None
_counters_lock = threading.Lock()


def token_budget_otel_enabled() -> bool:
    """Return True when token usage metrics export is enabled."""
    return bool(settings.ENABLE_OTEL_METRICS and settings.OTEL_EXPORTER_OTLP_ENDPOINT)


def token_budget_traces_enabled() -> bool:
    """Return True when token usage span events can be exported."""
    return bool(
        settings.otel_traces_active() and settings.resolved_otel_traces_endpoint()
    )


def _agent_name() -> str:
    try:
        from deep_agent.src.agent.config import agent_config

        return agent_config.get_name()
    except Exception:
        return settings.OTEL_SERVICE_NAME


def _format_timestamp(value: Any | None = None) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC).isoformat()
        return value.isoformat()
    if isinstance(value, str) and value:
        return value
    return datetime.now(UTC).isoformat()


def _ensure_counters() -> None:
    global _counters_initialized, _token_counter, _thread_total_counter, _daily_total_counter  # noqa: PLW0603

    if _counters_initialized or not token_budget_otel_enabled():
        return

    with _counters_lock:
        if _counters_initialized or not token_budget_otel_enabled():
            return
        _counters_initialized = True

        try:
            from opentelemetry import metrics

            meter = metrics.get_meter("template-agent.token-budget")
            _token_counter = meter.create_counter(
                "token_budget.tokens",
                description="Billable LLM tokens recorded per call",
            )
            _thread_total_counter = meter.create_counter(
                "token_budget.thread_total",
                description="Cumulative thread token totals after each LLM call",
            )
            _daily_total_counter = meter.create_counter(
                "token_budget.daily_total",
                description="Cumulative per-user daily token totals after each LLM call",
            )
        except Exception:
            logger.warning("token_budget_otel_counter_init_failed", exc_info=True)


def emit_token_usage(
    *,
    thread_id: str,
    user_id: str | None,
    input_tokens: int,
    output_tokens: int,
    cumulative_total: int,
    cumulative_input: int,
    cumulative_output: int,
    timestamp: Any | None = None,
) -> None:
    """Emit OTEL metrics and optional span events for a single LLM usage record."""
    if not token_budget_otel_enabled():
        return

    _ensure_counters()

    recorded_at = _format_timestamp(timestamp)
    agent_name = _agent_name()
    attributes = {
        "thread_id": thread_id,
        "agent.name": agent_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cumulative_total_tokens": cumulative_total,
        "cumulative_input_tokens": cumulative_input,
        "cumulative_output_tokens": cumulative_output,
        "timestamp": recorded_at,
    }
    if user_id:
        attributes["user_id"] = user_id

    logger.info("token_budget_usage", **attributes)

    if token_budget_traces_enabled():
        try:
            from opentelemetry import trace as otel_trace

            span = otel_trace.get_current_span()
            if span is not None and span.is_recording():
                span.add_event("token_budget.usage", attributes=attributes)
        except ImportError:
            pass

    metric_attrs = {
        "agent.name": agent_name,
        "thread_id": thread_id,
    }
    if user_id:
        metric_attrs["user_id"] = user_id

    if _token_counter is not None:
        if input_tokens > 0:
            _token_counter.add(input_tokens, {**metric_attrs, "token.type": "input"})
        if output_tokens > 0:
            _token_counter.add(output_tokens, {**metric_attrs, "token.type": "output"})

    if _thread_total_counter is not None and cumulative_total > 0:
        _thread_total_counter.add(
            cumulative_total,
            {**metric_attrs, "aggregation": "cumulative"},
        )


def emit_daily_token_usage(
    *,
    user_id: str,
    total_tokens: int,
    date: str,
    timestamp: Any | None = None,
) -> None:
    """Emit OTEL metrics and optional span events for a user's daily token rollup."""
    if not token_budget_otel_enabled():
        return

    _ensure_counters()

    recorded_at = _format_timestamp(timestamp)
    attributes = {
        "user_id": user_id,
        "total_tokens": total_tokens,
        "date": date,
        "timestamp": recorded_at,
        "agent.name": _agent_name(),
    }

    logger.info("token_budget_daily_usage", **attributes)

    if token_budget_traces_enabled():
        try:
            from opentelemetry import trace as otel_trace

            span = otel_trace.get_current_span()
            if span is not None and span.is_recording():
                span.add_event("token_budget.daily_usage", attributes=attributes)
        except ImportError:
            pass

    if _daily_total_counter is not None and total_tokens > 0:
        _daily_total_counter.add(
            total_tokens,
            {
                "agent.name": _agent_name(),
                "user_id": user_id,
                "date": date,
            },
        )
