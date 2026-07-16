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
    if trace_id is not None:
        _trace_id_var.set(trace_id)
    if user is not None:
        _user_var.set(user)
    if org is not None:
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
