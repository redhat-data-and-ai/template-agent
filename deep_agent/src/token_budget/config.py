"""Token budget configuration models."""

from __future__ import annotations

from pydantic import BaseModel


class TokenBudgetConfig(BaseModel):
    """Per-thread token usage tracking from agent.yaml ``token_budget:`` section."""

    enabled: bool = False

    @property
    def is_active(self) -> bool:
        """Return True when tracking should run."""
        return self.enabled
