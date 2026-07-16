"""Audit event emitter — structured JSON to stdout.

Emits platform audit events as structured JSON lines to stdout.
Each event includes user, org, trace_id from the audit context,
plus event-specific details with sensitive key redaction.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from typing import Any

SENSITIVE_KEYS = frozenset(
    {
        "password",
        "token",
        "apikey",
        "api_key",
        "secret",
        "authorization",
        "cookie",
        "session",
        "credentials",
        "access_token",
        "refresh_token",
    }
)


def _is_sensitive_key(key: str) -> bool:
    """Return True if key names a sensitive field."""
    normalized = str(key).lower().replace("_", "").replace("-", "")
    return normalized in SENSITIVE_KEYS


def _scrub_details(details: dict[str, Any], depth: int = 0) -> dict[str, Any]:
    """Recursively redact sensitive keys from audit details."""
    if depth > 5:
        return {"error": "max_depth_exceeded"}
    scrubbed: dict[str, Any] = {}
    for key, value in details.items():
        if _is_sensitive_key(key):
            scrubbed[key] = "[REDACTED]"
        elif isinstance(value, dict):
            scrubbed[key] = _scrub_details(value, depth + 1)
        elif isinstance(value, (list, tuple)):
            items: list[Any] = [
                _scrub_details(v, depth + 1) if isinstance(v, dict) else v
                for v in list(value)[:100]
            ]
            scrubbed[key] = items
        else:
            scrubbed[key] = value
    return scrubbed


def _resolve_context() -> dict[str, str | None]:
    """Get audit context (user, org, trace_id) from available sources."""
    ctx: dict[str, str | None] = {
        "user": None,
        "org": None,
        "trace_id": None,
    }
    try:
        from deep_agent.src.audit.context import get_audit_context

        ctx.update(get_audit_context())
    except ImportError:
        pass

    if ctx["trace_id"] is None:
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            if span and span.get_span_context().is_valid:
                ctx["trace_id"] = format(span.get_span_context().trace_id, "032x")
        except Exception:
            pass

    return ctx


def emit_audit_event(audit_event_type: str, **details: Any) -> None:
    """Emit a platform audit event as structured JSON to stdout."""
    if not isinstance(audit_event_type, str) or not audit_event_type.strip():
        return

    ctx = _resolve_context()
    envelope = {
        "event": "platform.audit",
        "audit_event_type": audit_event_type.strip(),
        "user": ctx.get("user"),
        "org": ctx.get("org"),
        "trace_id": ctx.get("trace_id"),
        "timestamp": datetime.now(UTC).isoformat(),
        "details": _scrub_details(details) if details else {},
        "logger": "platform.audit",
        "level": "info",
        "service": "template-agent",
    }

    try:
        line = json.dumps(envelope, default=str, ensure_ascii=False)
        sys.stdout.write(f"{line}\n")
        sys.stdout.flush()
    except Exception:
        pass
