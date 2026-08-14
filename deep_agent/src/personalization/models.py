"""Pydantic models for user personalization data."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class Memory(BaseModel):
    """A single user memory — a fact the agent should recall across sessions."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    user_id: str
    content: str
    score: float = Field(default=1.0)
    cluster_id: uuid.UUID | None = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Rule(BaseModel):
    """A user-defined custom instruction that shapes agent behaviour."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    user_id: str
    content: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
