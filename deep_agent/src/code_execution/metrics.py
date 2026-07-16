"""Observability for code execution — reuses existing otel.py infrastructure.

Four layers:
1. OTEL Metrics — instruments on MetricsContainer in aegra/otel.py
2. OTEL Tracing — spans via get_tracer() from aegra/otel.py
3. Platform Audit — structured JSON via audit/emitter.py
4. Structured Logs — JSON lines to stderr via stdlib logging
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger("deep_agent.src.code_execution")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.DEBUG)


def _log_json(level: int, event: str, **fields: Any) -> None:
    """Emit a structured JSON log line to stderr."""
    record = {
        "event": event,
        "logger": "deep_agent.src.code_execution",
        "level": logging.getLevelName(level).lower(),
        "timestamp": datetime.now(UTC).isoformat(),
        "service": "template-agent",
        **{k: v for k, v in fields.items() if v is not None},
    }
    logger.log(level, json.dumps(record, default=str))


def _get_otel_metrics() -> Any:
    """Return the global MetricsContainer from otel.py, or None."""
    try:
        from deep_agent.aegra.otel import get_metrics

        return get_metrics()
    except Exception:
        return None


def _get_tracer() -> Any:
    """Return an OTEL tracer from otel.py, or None."""
    try:
        from deep_agent.aegra.otel import get_tracer

        return get_tracer("code_execution")
    except Exception:
        return None


def _resolve_user_identity() -> tuple[str | None, str | None, str | None]:
    """Resolve user, org, trace_id from audit context and OTEL span."""
    user = None
    org = os.environ.get("AI_PLATFORM_AGENT_ORG")
    trace_id = None

    try:
        from deep_agent.src.audit.context import get_audit_context

        ctx = get_audit_context()
        user = ctx.get("user")
        org = ctx.get("org") or org
        trace_id = ctx.get("trace_id")
    except ImportError:
        pass

    if trace_id is None:
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            if span and span.get_span_context().is_valid:
                trace_id = format(span.get_span_context().trace_id, "032x")
        except Exception:
            pass

    return user, org, trace_id


def emit_audit_event(event_type: str, **details: Any) -> None:
    """Emit a platform audit event via the audit emitter."""
    try:
        from deep_agent.src.audit.emitter import emit_audit_event as _emit

        _emit(event_type, **details)
    except Exception as exc:
        _log_json(logging.DEBUG, "audit_emit_failed", error=str(exc))


def compute_code_hash(code: str) -> str:
    """Return a SHA-256 hash of the code string."""
    return f"sha256:{hashlib.sha256(code.encode()).hexdigest()}"


class CodeExecutionMetrics:
    """Observability using existing otel.py MetricsContainer and get_tracer()."""

    def __init__(self) -> None:
        """Initialize with tracer from otel.py."""
        self._tracer = _get_tracer()

    def _mc(self) -> Any:
        """Get MetricsContainer (lazy — may not be initialized at import time)."""
        return _get_otel_metrics()

    # --- Execution metrics (OTEL Layer 1 + Log Layer 4) ---

    def record_execution(
        self,
        *,
        language: str,
        org: str,
        exit_code: int,
        status: str,
        duration: float,
    ) -> None:
        """Record execution via MetricsContainer + structured log."""
        mc = self._mc()
        attrs = {
            "language": language,
            "org": org,
            "exit_code": str(exit_code),
            "status": status,
        }
        if mc:
            mc.code_execution_duration_seconds.record(duration, attrs)
            mc.code_executions_total.add(1, attrs)
        _log_json(
            logging.INFO,
            "code_execution_metric",
            language=language,
            org=org,
            exit_code=exit_code,
            status=status,
            duration_seconds=round(duration, 3),
        )

    def record_error(self, *, language: str, org: str, error_type: str) -> None:
        """Record error via MetricsContainer + structured log."""
        mc = self._mc()
        if mc:
            mc.code_execution_errors_total.add(
                1, {"language": language, "org": org, "error_type": error_type}
            )
        _log_json(
            logging.WARNING,
            "code_execution_error_metric",
            language=language,
            org=org,
            error_type=error_type,
        )

    def record_scheduling_latency(self, *, org: str, duration: float) -> None:
        """Record pod scheduling latency via MetricsContainer + structured log."""
        mc = self._mc()
        if mc:
            mc.code_execution_scheduling_seconds.record(duration, {"org": org})
        _log_json(
            logging.INFO,
            "code_execution_scheduling_latency",
            org=org,
            scheduling_seconds=round(duration, 3),
        )

    def increment_active(self, *, org: str) -> None:
        """Increment active execution gauge."""
        mc = self._mc()
        if mc:
            mc.code_execution_active.add(1, {"org": org})
        _log_json(logging.DEBUG, "code_execution_active_increment", org=org)

    def decrement_active(self, *, org: str) -> None:
        """Decrement active execution gauge."""
        mc = self._mc()
        if mc:
            mc.code_execution_active.add(-1, {"org": org})
        _log_json(logging.DEBUG, "code_execution_active_decrement", org=org)

    # --- Queue metrics (OTEL Layer 1 + Log Layer 4) ---

    def record_queue_wait(self, *, org: str, duration: float) -> None:
        """Record queue wait via MetricsContainer + structured log."""
        mc = self._mc()
        if mc:
            mc.code_execution_queue_wait_seconds.record(duration, {"org": org})
        _log_json(
            logging.INFO,
            "code_execution_queue_wait",
            org=org,
            wait_seconds=round(duration, 3),
        )

    def record_rejected(self, *, org: str) -> None:
        """Record rejected execution via MetricsContainer + structured log."""
        mc = self._mc()
        if mc:
            mc.code_execution_rejected_total.add(1, {"org": org})
        _log_json(logging.WARNING, "code_execution_rejected", org=org)

    def log_queued(self, *, org: str) -> None:
        """Log that an execution entered the queue."""
        _log_json(logging.DEBUG, "code_execution_queued", org=org)

    def log_dequeued(self, *, org: str, wait_seconds: float) -> None:
        """Log that an execution left the queue."""
        _log_json(
            logging.DEBUG,
            "code_execution_dequeued",
            org=org,
            wait_seconds=round(wait_seconds, 3),
        )

    # --- Cost tracking (OTEL Layer 1 + Log Layer 4) ---

    def record_resource_usage(
        self,
        *,
        org: str,
        language: str,
        cpu_seconds: float,
        memory_mb_seconds: float,
        duration: float,
    ) -> None:
        """Record resource usage via MetricsContainer + structured log."""
        mc = self._mc()
        attrs = {"org": org, "language": language}
        if mc:
            mc.code_execution_cpu_seconds.record(cpu_seconds, attrs)
            mc.code_execution_memory_mb_seconds.record(memory_mb_seconds, attrs)
        _log_json(
            logging.INFO,
            "code_execution_resource_usage",
            org=org,
            language=language,
            cpu_seconds=round(cpu_seconds, 4),
            memory_mb_seconds=round(memory_mb_seconds, 2),
            duration_seconds=round(duration, 3),
        )

    # --- OTEL Tracing (Layer 2) ---

    @contextmanager
    def trace_span(self, name: str, **attributes: Any) -> Any:
        """Context manager for OTEL tracing spans (sets as active span)."""
        if self._tracer is None:
            yield None
            return
        try:
            from opentelemetry import trace

            span = self._tracer.start_span(name, attributes=attributes)
            ctx = trace.set_span_in_context(span)
            token = trace.context_api.attach(ctx)
            try:
                yield span
            except Exception as exc:
                if span:
                    span.set_attribute("error", True)
                    span.set_attribute("error.message", str(exc))
                raise
            finally:
                if span:
                    span.end()
                trace.context_api.detach(token)
        except ImportError:
            yield None

    def start_span(self, name: str, **attributes: Any) -> Any:
        """Start an OTEL tracing span (manual end required)."""
        if self._tracer is None:
            return None
        return self._tracer.start_span(name, attributes=attributes)

    # --- Platform Audit (Layer 3) ---

    def emit_audit(
        self,
        *,
        language: str,
        status: str,
        exit_code: int,
        latency_ms: float,
        code_hash: str,
        namespace: str,
        image: str,
        job_name: str,
        timeout: int,
        stdout_bytes: int,
        stderr_bytes: int,
        scheduling_seconds: float = 0.0,
    ) -> None:
        """Emit audit event with user identity from context."""
        emit_audit_event(
            "code_execution",
            agent="orchestrator",
            language=language,
            status=status,
            exit_code=exit_code,
            latency_ms=latency_ms,
            code_hash=code_hash,
            namespace=namespace,
            image=image,
            job_name=job_name,
            timeout_seconds=timeout,
            stdout_bytes=stdout_bytes,
            stderr_bytes=stderr_bytes,
            scheduling_seconds=round(scheduling_seconds, 3),
        )

    # --- Structured Logging (Layer 4) ---

    def log_started(self, **fields: Any) -> None:
        """Log that a code execution has started."""
        _log_json(logging.INFO, "code_execution_started", **fields)

    def log_completed(self, **fields: Any) -> None:
        """Log that a code execution completed."""
        _log_json(logging.INFO, "code_execution_completed", **fields)

    def log_timeout(self, **fields: Any) -> None:
        """Log that a code execution timed out."""
        _log_json(logging.WARNING, "code_execution_timeout", **fields)

    def log_oom(self, **fields: Any) -> None:
        """Log that a code execution was OOM killed."""
        _log_json(logging.WARNING, "code_execution_oom_killed", **fields)

    def log_failed(self, **fields: Any) -> None:
        """Log that a code execution failed."""
        _log_json(logging.ERROR, "code_execution_failed", **fields)

    def log_cleanup(self, **fields: Any) -> None:
        """Log code execution cleanup."""
        _log_json(logging.DEBUG, "code_execution_cleanup", **fields)
