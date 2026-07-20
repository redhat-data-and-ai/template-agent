"""Guardrails configuration model."""

from pydantic import BaseModel


class GuardrailsConfig(BaseModel):
    """Guardian guardrail runtime configuration."""

    model: str = "ibm-granite/granite-guardian-3.2-5b"
