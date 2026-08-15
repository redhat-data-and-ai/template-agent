"""Granite Guardian guardrails -- content safety error hierarchy and public API."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from deep_agent.src.guardrails.config import GuardrailsConfig

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

_CIRCUIT_COOLDOWN_SECONDS: float = 300.0

_circuit_state: str = "closed"
_circuit_opened_at: float = 0.0
_circuit_reason: str = ""


def _time() -> float:
    return time.monotonic()


def init_guardrails(config: "GuardrailsConfig") -> None:
    """Initialise the global guardrails config. Call once at process startup."""
    global _config  # noqa: PLW0603
    _config = config


def get_guardrails_config() -> Optional["GuardrailsConfig"]:
    """Return the global GuardrailsConfig, or None if not yet initialised or circuit-open."""
    global _circuit_state  # noqa: PLW0603
    if _circuit_state == "open":
        elapsed = _time() - _circuit_opened_at
        if elapsed >= _CIRCUIT_COOLDOWN_SECONDS:
            _circuit_state = "half-open"
            from deep_agent.utils.pylogger import get_python_logger

            get_python_logger().warning(
                "guardian_circuit_half_open",
                message="Guardrails circuit breaker entering half-open state -- next call will probe.",
                cooldown_seconds=_CIRCUIT_COOLDOWN_SECONDS,
            )
            return _config
        return None
    return _config


def disable_guardrails_runtime(reason: str = "") -> None:
    """Open the circuit breaker after a configuration error.

    Guardrails are disabled for _CIRCUIT_COOLDOWN_SECONDS, then a probe
    request is allowed (half-open). A successful probe closes the circuit;
    another failure re-opens it.
    """
    global _circuit_state, _circuit_opened_at, _circuit_reason  # noqa: PLW0603
    if _circuit_state == "open":
        return
    _circuit_state = "open"
    _circuit_opened_at = _time()
    _circuit_reason = reason
    from deep_agent.utils.pylogger import get_python_logger

    get_python_logger().error(
        "guardian_circuit_open",
        reason=reason,
        cooldown_seconds=_CIRCUIT_COOLDOWN_SECONDS,
        message="Granite Guardian disabled due to a configuration error -- "
        "will probe again after cooldown.",
    )


def close_guardrails_circuit() -> None:
    """Close the circuit breaker after a successful probe."""
    global _circuit_state, _circuit_opened_at, _circuit_reason  # noqa: PLW0603
    if _circuit_state == "closed":
        return
    prev = _circuit_state
    _circuit_state = "closed"
    _circuit_opened_at = 0.0
    _circuit_reason = ""
    from deep_agent.utils.pylogger import get_python_logger

    get_python_logger().info(
        "guardian_circuit_closed",
        previous_state=prev,
        message="Granite Guardian circuit breaker closed -- guardrails re-enabled.",
    )


def guardrails_circuit_state() -> str:
    """Return the current circuit state: 'closed', 'open', or 'half-open'."""
    if _circuit_state == "open":
        elapsed = _time() - _circuit_opened_at
        if elapsed >= _CIRCUIT_COOLDOWN_SECONDS:
            return "half-open"
    return _circuit_state
