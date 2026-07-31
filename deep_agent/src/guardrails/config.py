"""Guardrails configuration model."""

from __future__ import annotations

from pydantic import BaseModel, model_validator


class GuardrailsConfig(BaseModel):
    """Guardian guardrail runtime configuration."""

    enabled: bool = False  # disabled when absent from agent.yaml or set to false
    model: str | None = None  # required in agent.yaml when enabled: true

    @model_validator(mode="after")
    def _require_model_when_enabled(self) -> GuardrailsConfig:
        if self.enabled and not self.model:
            raise ValueError("'model' is required when guardrails are enabled")
        return self
