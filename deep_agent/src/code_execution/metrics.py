"""Observability for code execution — OTEL metrics, tracing, audit, logs.

Uses stdlib logging (not structlog) to ensure log output reaches
stdout/stderr inside the LangGraph graph-execution context, where
structlog's cached logger proxy may not be connected to the correct
handler chain.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
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


def _get_tracer() -> Any:
    """Return an OTEL tracer, or None if unavailable."""
    try:
        from deep_agent.aegra.otel import get_tracer

        return get_tracer("code_execution")
    except Exception:
        return None


def get_tracer() -> Any:
    """Return the code-execution OTEL tracer."""
    return _get_tracer()


def emit_audit_event(event_type: str, **details: Any) -> None:
    """Emit a platform audit event, falling back gracefully."""
    try:
        from deep_agent.src.audit.emitter import emit_audit_event as _emit

        _emit(event_type, **details)
    except ImportError:
        _log_json(logging.DEBUG, "audit_emitter_unavailable")
    except Exception as exc:
        _log_json(logging.WARNING, "audit_emit_failed", error=str(exc))


def compute_code_hash(code: str) -> str:
    """Return a SHA-256 hash of the code string."""
    return f"sha256:{hashlib.sha256(code.encode()).hexdigest()}"


class CodeExecutionMetrics:
    """Centralized observability for code execution."""

    def __init__(self) -> None:
        """Initialize metrics with an OTEL tracer."""
        self._tracer = get_tracer()

    def record_execution(
        self, *, language: str, org: str, exit_code: int, status: str, duration: float
    ) -> None:
        """Record an execution metric event."""
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
        """Record an execution error metric."""
        _log_json(
            logging.WARNING,
            "code_execution_error_metric",
            language=language,
            org=org,
            error_type=error_type,
        )

    def record_scheduling_latency(self, *, org: str, duration: float) -> None:
        """Record pod scheduling latency."""
        _log_json(
            logging.DEBUG,
            "code_execution_scheduling_metric",
            org=org,
            scheduling_seconds=round(duration, 3),
        )

    def increment_active(self, *, org: str) -> None:
        """Increment the active execution counter."""
        _log_json(logging.DEBUG, "code_execution_active_increment", org=org)

    def decrement_active(self, *, org: str) -> None:
        """Decrement the active execution counter."""
        _log_json(logging.DEBUG, "code_execution_active_decrement", org=org)

    def start_span(self, name: str, **attributes: Any) -> Any:
        """Start an OTEL tracing span."""
        if self._tracer is None:
            return None
        return self._tracer.start_span(name, attributes=attributes)

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
    ) -> None:
        """Emit a structured audit event for a code execution."""
        emit_audit_event(
            "code_execution",
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
        )

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
