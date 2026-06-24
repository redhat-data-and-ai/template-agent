"""Pydantic models for policy settings."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PolicySettings(BaseModel):
    """User policy settings model."""

    user_id: str
    values: dict[str, Any]
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
