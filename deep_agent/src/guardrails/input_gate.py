"""Process-local Phase 1 gate.

The generic package lives in ``agent_guardrails``. This module is the
template-agent adapter: it builds the gate from agent.yaml and remembers the
decision for tool checks later in the same task.
"""

from __future__ import annotations

import contextvars
from typing import Any

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_gate: Any = None
_decision: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "input_guardrail_decision",
    default=None,
)


class _ClosedDecision:
    """Block every turn when the gate was requested but could not be built."""

    action = "block"
    blocked_by = ["configuration"]
    allowed_tools = None
    error = "Input guardrail is enabled but failed to initialize."
    redacted = False

    def __init__(self, text: str) -> None:
        self.text = text

    def tool_allowed(self, name: str) -> bool:
        del name
        return False


class _ClosedGate:
    """Fail-closed stand-in used when configuration or import fails."""

    def evaluate(self, text: str) -> _ClosedDecision:
        return _ClosedDecision(text)


def reset_input_gate() -> None:
    """Clear the cached gate. Tests call this between cases."""
    global _gate
    _gate = None
    _decision.set(None)


def init_input_gate(host_cfg: Any) -> bool:
    """Build the gate. Returns false when the process will fail closed."""
    global _gate
    raw = getattr(host_cfg, "gate", None)
    try:
        if not isinstance(raw, dict):
            raise TypeError("input_guardrail.gate must be a mapping")
        from agent_guardrails import InputGate, InputGateConfig

        _gate = InputGate(InputGateConfig.model_validate(raw))
        return True
    except Exception:
        logger.exception("input guardrail failed to initialize; failing closed")
        _gate = _ClosedGate()
        return False


def get_input_gate() -> Any:
    """Return the cached gate, or ``None`` when Phase 1 is inactive."""
    return _gate


def remember_decision(decision: Any) -> None:
    """Keep the input decision so a later tool call can apply its allowlist."""
    _decision.set(decision)


def current_decision() -> Any:
    """Return the decision for this task, if the input gate already ran."""
    return _decision.get()
