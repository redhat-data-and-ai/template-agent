"""Audit context — user, org, trace_id for event envelopes."""

from __future__ import annotations

from contextvars import ContextVar

_trace_id_var: ContextVar[str | None] = ContextVar("audit_trace_id", default=None)
_user_var: ContextVar[str | None] = ContextVar("audit_user", default=None)
_org_var: ContextVar[str | None] = ContextVar("audit_org", default=None)


def bind_audit_context(
    *,
    trace_id: str | None = None,
    user: str | None = None,
    org: str | None = None,
) -> None:
    """Bind audit identifiers for the current async context."""
    MAX_LEN = 512  # Prevent memory exhaustion from extremely long strings

    if trace_id is not None:
        if not isinstance(trace_id, str):
            raise TypeError("trace_id must be a string")
        trace_id = trace_id.strip()
        if not trace_id:
            raise ValueError("trace_id cannot be empty or whitespace")
        if len(trace_id) > MAX_LEN:
            raise ValueError(f"trace_id exceeds maximum length of {MAX_LEN}")
        _trace_id_var.set(trace_id)

    if user is not None:
        if not isinstance(user, str):
            raise TypeError("user must be a string")
        user = user.strip()
        if not user:
            raise ValueError("user cannot be empty or whitespace")
        if len(user) > MAX_LEN:
            raise ValueError(f"user exceeds maximum length of {MAX_LEN}")
        _user_var.set(user)

    if org is not None:
        if not isinstance(org, str):
            raise TypeError("org must be a string")
        org = org.strip()
        if not org:
            raise ValueError("org cannot be empty or whitespace")
        if len(org) > MAX_LEN:
            raise ValueError(f"org exceeds maximum length of {MAX_LEN}")
        _org_var.set(org)


def clear_audit_context() -> None:
    """Reset audit context vars."""
    _trace_id_var.set(None)
    _user_var.set(None)
    _org_var.set(None)


def get_audit_context() -> dict[str, str | None]:
    """Return current audit context fields."""
    return {
        "trace_id": _trace_id_var.get(),
        "user": _user_var.get(),
        "org": _org_var.get(),
    }


def resolve_trace_id_from_config() -> str | None:
    """Read trace_id from LangGraph RunnableConfig metadata if available."""
    try:
        from langgraph.config import get_config

        config = get_config()
        metadata = config.get("metadata")
        if not isinstance(metadata, dict):
            return None
        trace_id = metadata.get("trace_id")
        if isinstance(trace_id, str) and trace_id.strip():
            return trace_id.strip()
    except Exception:  # Catch all: RuntimeError, AttributeError, TypeError, etc.
        pass
    return None


def resolve_trace_id_from_otel() -> str | None:
    """Read trace_id from the active OTEL span if present."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            trace_id = span.get_span_context().trace_id
            if isinstance(trace_id, int) and trace_id > 0:
                return format(trace_id, "032x")
    except Exception:  # Catch all: ImportError, AttributeError, TypeError, ValueError
        pass
    return None


def resolve_trace_id() -> str | None:
    """Best-effort trace_id from context, config metadata, or OTEL."""
    ctx = _trace_id_var.get()
    if ctx:
        return ctx
    from_config = resolve_trace_id_from_config()
    if from_config:
        return from_config
    return resolve_trace_id_from_otel()
