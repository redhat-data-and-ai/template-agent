"""Pydantic models for A2A downstream agent configuration."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class A2ATargetAgent(BaseModel):
    """Represents a remote agent that this agent can delegate work to."""

    agent_id: str
    base_url: str
    description: str | None = None
    card: dict[str, Any] | None = None
    skills: list[str] = []
    capabilities: dict[str, Any] | None = None
