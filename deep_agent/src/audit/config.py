"""Platform audit configuration.

Environment variables (loaded via ``settings.py``):
    PLATFORM_AUDIT_ENABLED:     Master switch (default: false)
    PLATFORM_AUDIT_BUFFER_MAX:  Max in-memory buffered events (default: 1000)

YAML reference (agent.yaml):
    platform.audit.enabled
    platform.audit.buffer_max
"""

from deep_agent.src.settings import settings


def is_audit_enabled() -> bool:
    """Return whether platform audit is enabled."""
    return settings.PLATFORM_AUDIT_ENABLED
