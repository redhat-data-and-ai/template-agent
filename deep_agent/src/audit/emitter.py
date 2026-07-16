"""Audit event emitter — structured JSON logging with local buffer fallback."""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from deep_agent.src.audit.buffer import drain, enqueue
from deep_agent.src.audit.config import is_audit_enabled
from deep_agent.src.audit.context import get_audit_context, resolve_trace_id
from deep_agent.utils.pylogger import SERVICE_NAME, get_python_logger

logger = get_python_logger()

# Sensitive keys to redact from audit details
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
        "auth",
        "credentials",
        "privatekey",
        "private_key",
        "accesstoken",
        "access_token",
        "refreshtoken",
        "refresh_token",
    }
)


def _is_sensitive_key(key: str) -> bool:
    """Return True if *key* names a sensitive field."""
    normalized = str(key).lower().replace("_", "").replace("-", "")
    if normalized in SENSITIVE_KEYS:
        return True
    parts = [p for p in re.split(r"[_\-.]", str(key).lower()) if p]
    return any(part in SENSITIVE_KEYS for part in parts)


def _scrub_details(details: dict[str, Any], depth: int = 0) -> dict[str, Any]:
    """Recursively redact sensitive keys from audit details."""
    MAX_DEPTH = 5
    MAX_ARRAY_LEN = 100

    if depth > MAX_DEPTH:
        return {"error": "max_depth_exceeded"}

    scrubbed: dict[str, Any] = {}
    for key, value in details.items():
        if _is_sensitive_key(key):
            scrubbed[key] = "[REDACTED]"
        elif isinstance(value, dict):
            scrubbed[key] = _scrub_details(value, depth + 1)
        elif isinstance(value, (list, tuple)):
            # Limit array length and recursively scrub dicts
            limited = list(value)[:MAX_ARRAY_LEN]
            scrubbed[key] = [
                _scrub_details(v, depth + 1) if isinstance(v, dict) else v
                for v in limited
            ]
        else:
            scrubbed[key] = value

    return scrubbed


def emit_audit_event(audit_event_type: str, **details: Any) -> None:
    """Emit a platform audit event. No-op when audit is disabled."""
    if not is_audit_enabled():
        return

    # Validate event type
    if not isinstance(audit_event_type, str) or not audit_event_type.strip():
        logger.error("invalid_audit_event_type", type=type(audit_event_type).__name__)
        return

    if len(audit_event_type) > 128:
        logger.error("audit_event_type_too_long", length=len(audit_event_type))
        return

    ctx = get_audit_context()
    envelope: dict[str, Any] = {
        "event": "platform.audit",
        "audit_event_type": audit_event_type.strip(),
        "user": ctx.get("user"),
        "org": ctx.get("org"),
        "trace_id": ctx.get("trace_id") or resolve_trace_id(),
        "timestamp": datetime.now(UTC).isoformat(),
        "details": _scrub_details(details) if details else {},
    }

    _emit_envelope(envelope)
    _flush_buffer()


def _format_record(envelope: dict[str, Any]) -> dict[str, Any]:
    """Shape audit JSON to match other template-agent stdout log lines."""
    return {
        **envelope,
        "logger": "platform.audit",
        "level": "info",
        "service": SERVICE_NAME,
    }


def _safe_json_default(obj: Any) -> str:
    """Safe JSON serializer - only converts known safe types."""
    # Allow datetime/date conversion
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "isoformat"):  # date, time, etc.
        return str(obj.isoformat())
    # Allow Path and UUID
    if isinstance(obj, (Path, UUID)):
        return str(obj)
    # Don't expose arbitrary objects - return type name only
    return f"<non-serializable:{type(obj).__name__}>"


def _emit_envelope(envelope: dict[str, Any]) -> None:
    MAX_SIZE = 1_000_000  # 1MB per event

    try:
        line = json.dumps(
            _format_record(envelope), default=_safe_json_default, ensure_ascii=False
        )

        if len(line) > MAX_SIZE:
            logger.warning(
                "audit_event_too_large",
                size=len(line),
                event_type=envelope.get("audit_event_type"),
            )
            # Emit a truncated error event instead
            error_envelope = {
                **envelope,
                "details": {"error": "event_too_large", "size": len(line)},
            }
            line = json.dumps(
                _format_record(error_envelope), default=_safe_json_default
            )

        sys.stdout.write(f"{line}\n")
        sys.stdout.flush()
    except Exception as exc:
        logger.warning(
            "audit_emit_failed",
            error=str(exc),
            error_type=type(exc).__name__,
            event_type=envelope.get("audit_event_type"),
        )
        enqueue(envelope)


def _flush_buffer() -> None:
    """Retry buffered events. Stops on first failure to preserve order."""
    pending = drain()
    for envelope in pending:
        try:
            line = json.dumps(
                _format_record(envelope), default=_safe_json_default, ensure_ascii=False
            )
            sys.stdout.write(f"{line}\n")
            sys.stdout.flush()
        except Exception as exc:
            logger.debug("audit_flush_failed", error=str(exc), remaining=len(pending))
            # Re-enqueue this event and stop (preserves order)
            enqueue(envelope)
            break
