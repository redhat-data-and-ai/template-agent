"""Pydantic models for user personalization data."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Memory(BaseModel):
    """A single user memory — a fact the agent should recall across sessions."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    user_id: str
    content: str
    score: float = Field(default=1.0)
    cluster_id: uuid.UUID | None = Field(default=None)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class Rule(BaseModel):
    """A user-defined custom instruction that shapes agent behaviour."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    user_id: str
    content: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class UserPreferences(BaseModel):
    """Per-user feature preferences stored server-side."""

    user_id: str
    memory_enabled: bool = True
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
