"""Audit logging — structured events gated by PLATFORM_AUDIT_ENABLED."""

from deep_agent.src.audit.config import is_audit_enabled
from deep_agent.src.audit.emitter import emit_audit_event
from deep_agent.src.audit.events import AuditEventType

__all__ = [
    "AuditEventType",
    "emit_audit_event",
    "is_audit_enabled",
]
