"""NodeEventEmitter — thin wrapper that reduces boilerplate in node files.

Instead of importing 10+ individual ``emit_*`` functions and calling
``ctx.emit_or_append(emit_agent_thinking(...), events)`` everywhere,
a node can do::

    from ._event_emitter import NodeEventEmitter

    emitter = NodeEventEmitter(ctx, events)
    emitter.thinking("Supervisor", "Reflecting on findings")
    emitter.decision("Supervisor", "Coverage sufficient", "90% coverage")
    emitter.raw(emit_supervisor_reflection(round_num, ...))
"""

from __future__ import annotations

from typing import Any

from template_agent.src.core.deep_research.events import (
    emit_agent_decision,
    emit_agent_message,
    emit_agent_thinking,
)
from template_agent.src.core.deep_research.state import ResearchContext


class NodeEventEmitter:
    """Convenience wrapper around ``ctx.emit_or_append`` and common emit helpers.

    Keeps a reference to the mutable *events* list and the *ctx* so callers
    don't have to thread both through every helper call.
    """

    __slots__ = ("_ctx", "_events")

    def __init__(self, ctx: ResearchContext, events: list[dict[str, Any]]) -> None:
        """Store context and events list for emitting events."""
        self._ctx = ctx
        self._events = events

    def thinking(self, agent: str, thought: str) -> None:
        """Emit an ``AGENT_THINKING`` event."""
        self._ctx.emit_or_append(emit_agent_thinking(agent, thought), self._events)

    def decision(self, agent: str, decision: str, reasoning: str = "") -> None:
        """Emit an ``AGENT_DECISION`` event."""
        self._ctx.emit_or_append(
            emit_agent_decision(agent, decision, reasoning), self._events
        )

    def message(
        self,
        from_agent: str,
        to_agent: str,
        msg: str,
        message_type: str = "request",
    ) -> None:
        """Emit an ``AGENT_MESSAGE`` event."""
        self._ctx.emit_or_append(
            emit_agent_message(from_agent, to_agent, msg, message_type),
            self._events,
        )

    def raw(self, event: dict[str, Any]) -> None:
        """Emit an arbitrary pre-built event dict."""
        self._ctx.emit_or_append(event, self._events)

    def append(self, event: dict[str, Any]) -> None:
        """Append an event directly to the events list (no streaming push)."""
        self._events.append(event)
