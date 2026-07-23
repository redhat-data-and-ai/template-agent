"""PII middleware configuration models."""

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class ActionType(str, Enum):
    """PII handling strategy."""

    scrub = "scrub"  # reversible: replace with [LABEL_N], restore in LLM output
    tokenize = "tokenize"  # alias for scrub (backwards compat)
    hash = "hash"  # one-way: HMAC-SHA256, deterministic for log correlation
    redact = "redact"  # one-way: replace with ***REDACTED***
    mask = "mask"  # one-way: partially mask (e.g. ****-****-****-1234 for credit card)
    block = "block"  # reject the entire request if this PII type appears in user input


class PIIRule(BaseModel):
    """Single PII detection rule with strategy and provider."""

    name: str
    regex: Optional[str] = None  # custom regex; absent = look up BUILTIN_PATTERNS[name]
    strategy: ActionType = ActionType.scrub  # scrub/mask/hash/redact/block
    provider: Literal["regex", "presidio", "custom", "default"] = (
        "regex"  # provider backend
    )
    label: Optional[str] = None  # token prefix; defaults to name.upper()

    @model_validator(mode="before")
    @classmethod
    def _normalise(cls, data: dict) -> dict:
        if "action" in data and "strategy" not in data:
            data["strategy"] = data.pop("action")
        if "detector" in data and "provider" not in data:
            data["provider"] = data.pop("detector")
        data.pop("pattern_type", None)  # no longer needed — regex presence is enough
        return data

    @property
    def action(self) -> ActionType:
        """Backwards-compatible alias."""
        return self.strategy

    @property
    def detector(self) -> str:
        """Backwards-compatible alias."""
        return self.provider

    @property
    def pattern_type(self) -> str:
        """Backwards-compatible alias — 'custom' if regex provided, else 'builtin'."""
        return "custom" if self.regex else "builtin"

    def effective_label(self) -> str:
        """Return the label used in token placeholders."""
        return (self.label or self.name).upper()


class PIIConfig(BaseModel):
    """Top-level PII middleware configuration."""

    enabled: bool = False
    trace_strategy: Literal["redact", "hash"] = (
        "hash"  # how PII appears in Langfuse traces
    )
    rules: list[PIIRule] = Field(default_factory=list)
