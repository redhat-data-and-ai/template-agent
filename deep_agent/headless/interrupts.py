"""Interrupt inspection shared by the threads API and the ``/invoke`` API."""

from __future__ import annotations

import json
from typing import Any


def _interrupt_value(interrupt: Any) -> Any:
    """Return the payload inside one interrupt."""
    if isinstance(interrupt, dict) and "value" in interrupt:
        value = interrupt["value"]
    else:
        value = interrupt
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def find_mcp_auth(interrupts: list[Any]) -> dict[str, Any] | None:
    """Return the MCP-auth interrupt when the agent is waiting for a browser login.

    The agent may put a JSON string in ``value``, or a gateway may already
    have parsed that string into a dict. Both shapes are accepted.
    """
    for interrupt in interrupts:
        value = _interrupt_value(interrupt)
        if isinstance(value, dict) and value.get("type") == "mcp_auth_required":
            return value
    return None


def count_actions(interrupts: list[Any]) -> int:
    """Count HITL decisions the caller must approve to resume."""
    count = 0
    for interrupt in interrupts:
        value = _interrupt_value(interrupt)
        if isinstance(value, dict):
            actions = value.get("action_requests") or []
            count += len(actions) if actions else 1
        else:
            count += 1
    return count


def local_start_body(message: str) -> dict[str, Any]:
    """Build the first ``/runs/stream`` body for the local Aegra API."""
    return {
        "assistant_id": "agent",
        "input": {"messages": [{"role": "human", "content": message}]},
        "stream_mode": "updates",
    }


def local_resume_body(action_count: int) -> dict[str, Any]:
    """Build a local resume body that approves every pending tool action."""
    return {
        "assistant_id": "agent",
        "command": {
            "resume": {"decisions": [{"type": "approve"}] * action_count},
        },
        "stream_mode": "updates",
    }


def invoke_resume_body(thread_id: str, action_count: int) -> dict[str, Any]:
    """Build the resume body for the agent's ``/invoke`` route."""
    return {
        "input": None,
        "thread_id": thread_id,
        "command": {
            "resume": {"decisions": [{"type": "approve"}] * action_count},
        },
        "assistant_id": "agent",
        "stream_mode": "updates",
    }


def aif_resume_body(thread_id: str, action_count: int) -> dict[str, Any]:
    """Alias for :func:`invoke_resume_body` (kept for older callers)."""
    return invoke_resume_body(thread_id, action_count)
