"""Decisions for the Phase 1 input gate.

The package never waits for a person. ``requires_approval`` stays false.
A block is not sent back through the agent, so ``retryable`` is false.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class Action(str, Enum):
    """What the host should do with the user message."""

    ALLOW = "allow"
    BLOCK = "block"
    REDACT = "redact"


class CheckResult(BaseModel):
    """One check inside the input gate."""

    name: str
    action: Action
    score: float | None = None
    detail: str = ""
    latency_ms: float = 0.0
    timed_out: bool = False


class IntentPrediction(BaseModel):
    """Label from the intent head. Only this head is trained."""

    label: str
    confidence: float
    read_only: bool = False


class InputDecision(BaseModel):
    """Handoff for one user message.

    ``text`` is safe to pass on: PII and secrets are already masked.
    ``allowed_tools`` is ``None`` when intent did not narrow the tool list.
    A list, including an empty list, is the permitted set for this turn.
    """

    action: Action
    text: str
    checks: list[CheckResult] = Field(default_factory=list)
    blocked_by: list[str] = Field(default_factory=list)
    intent: IntentPrediction | None = None
    allowed_tools: list[str] | None = None
    error: str | None = None
    retryable: bool = True
    redacted: bool = False
    latency_ms: float = 0.0
    model_calls: int = 0
    requires_approval: bool = False

    @model_validator(mode="after")
    def _decisions_are_automatic(self) -> InputDecision:
        self.requires_approval = False
        if self.action == Action.BLOCK:
            self.retryable = False
        return self

    def tool_allowed(self, name: str) -> bool:
        """Return whether this turn may call ``name``.

        A block rejects every tool. ``None`` means intent did not restrict tools.
        """
        if self.action == Action.BLOCK:
            return False
        if self.allowed_tools is None:
            return True
        return name in self.allowed_tools
