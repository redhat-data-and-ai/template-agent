"""OPA authorization configuration model.

Provides the validated Pydantic model for the ``opa:`` section of
config/agent/runtime/agent.yaml. Environment variables (OPA_ENABLED,
OPA_URL, OPA_TIMEOUT) override YAML values when explicitly set.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class OpaFileConfig(BaseModel):
    """OPA authorization config from agent.yaml ``opa:`` section."""

    enabled: bool = False
    url: str = "http://localhost:8181/v1/data/agent/authz"
    timeout: float = Field(default=2.0, gt=0)
    max_retries: int = Field(default=0, ge=0)
    fail_open: bool = Field(
        default=False,
        description="When True, OPA errors allow the request (fail open). When False, OPA errors deny the request (fail closed).",
    )
