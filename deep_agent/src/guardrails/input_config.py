"""Host switch for the Phase 1 package. The package does not import this module."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class InputGuardrailHostConfig(BaseModel):
    """Whether this process runs the Phase 1 gate, and the package config."""

    enabled: bool = False
    gate: dict[str, Any] = Field(default_factory=dict)
