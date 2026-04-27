"""Async-safe request context propagated through ContextVar."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class A2ARequestContext:
    """Values extracted by the auth middleware and consumed by the executor / delegation tool."""

    access_token: str | None = None
    calling_agent_id: str | None = None
    correlation_id: str | None = None


a2a_request_ctx: ContextVar[A2ARequestContext] = ContextVar(
    "a2a_request_ctx", default=A2ARequestContext()
)
