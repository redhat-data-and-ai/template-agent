"""Observability for code execution — OTEL metrics, tracing, audit, logs."""

from __future__ import annotations

import hashlib
from typing import Any

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


def _get_tracer() -> Any:
    try:
        from deep_agent.aegra.otel import get_tracer

        return get_tracer("code_execution")
    except Exception:
        return None


def get_tracer() -> Any:
    """Return the OTEL tracer for code execution."""
    return _get_tracer()


def emit_audit_event(event_type: str, **details: Any) -> None:
    """Emit an audit event, silently skipping if emitter is unavailable."""
    try:
        from deep_agent.src.audit.emitter import emit_audit_event as _emit

        _emit(event_type, **details)
    except ImportError:
        logger.debug("Audit emitter not available, skipping audit event")
    except Exception as exc:
        logger.warning("Failed to emit audit event: %s", exc)


def compute_code_hash(code: str) -> str:
    """Return a SHA-256 hash of the given code string."""
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
        logger.info(
            "code_execution_metric",
            language=language,
            org=org,
            exit_code=exit_code,
            status=status,
            duration_seconds=round(duration, 3),
        )

    def record_error(self, *, language: str, org: str, error_type: str) -> None:
        """Record an execution error metric."""
        logger.warning(
            "code_execution_error_metric",
            language=language,
            org=org,
            error_type=error_type,
        )

    def record_scheduling_latency(self, *, org: str, duration: float) -> None:
        """Record pod scheduling latency."""
        logger.debug(
            "code_execution_scheduling_metric",
            org=org,
            scheduling_seconds=round(duration, 3),
        )

    def increment_active(self, *, org: str) -> None:
        """Increment the active execution counter."""
        logger.debug("code_execution_active_increment", org=org)

    def decrement_active(self, *, org: str) -> None:
        """Decrement the active execution counter."""
        logger.debug("code_execution_active_decrement", org=org)

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
        logger.info("code_execution_started", **fields)

    def log_completed(self, **fields: Any) -> None:
        """Log that a code execution completed."""
        logger.info("code_execution_completed", **fields)

    def log_timeout(self, **fields: Any) -> None:
        """Log that a code execution timed out."""
        logger.warning("code_execution_timeout", **fields)

    def log_oom(self, **fields: Any) -> None:
        """Log that a code execution was OOM-killed."""
        logger.warning("code_execution_oom_killed", **fields)

    def log_failed(self, **fields: Any) -> None:
        """Log that a code execution failed."""
        logger.error("code_execution_failed", **fields)

    def log_cleanup(self, **fields: Any) -> None:
        """Log a cleanup event for a code execution job."""
        logger.debug("code_execution_cleanup", **fields)
