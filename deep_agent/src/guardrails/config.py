"""Guardrails configuration model."""

from pydantic import BaseModel


class GuardrailsConfig(BaseModel):
    """Guardian guardrail runtime configuration."""

    enabled: bool = False  # disabled when absent from agent.yaml or set to false
    model: str = "ibm-granite/granite-guardian-3.2-5b"
