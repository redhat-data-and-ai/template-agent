"""Granite Guardian guardrails — content safety error hierarchy and public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from deep_agent.src.guardrails.config import GuardrailsConfig

# Sentinel embedded in every BLOCKED_RESULT ToolMessage and used by SafetyAwareRunnable
# to detect tool blocks in both ainvoke and astream_events.  Defined here so both
# tool_proxy.py and safety.py reference the same string without implicit coupling.
TOOL_SAFETY_REFUSAL = (
    "I wasn't able to complete this task due to a content safety policy issue."
)


class ContentSafetyError(ValueError):
    """Raised by GraniteGuardianCallbackHandler when content is flagged unsafe."""


class InputContentSafetyError(ContentSafetyError):
    """Raised when the user's input is flagged unsafe."""


class ToolContentSafetyError(ContentSafetyError):
    """Raised when a tool result is flagged unsafe."""


_config: Optional["GuardrailsConfig"] = None


def init_guardrails(config: "GuardrailsConfig") -> None:
    """Initialise the global guardrails config. Call once at process startup."""
    global _config  # noqa: PLW0603
    _config = config


def get_guardrails_config() -> Optional["GuardrailsConfig"]:
    """Return the global GuardrailsConfig, or None if not yet initialised."""
    return _config
