"""LangChain callback for Phase 1. A block raises the agent's safety errors."""

from __future__ import annotations

import hashlib
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import BaseMessage

from deep_agent.src.guardrails import InputContentSafetyError, ToolContentSafetyError
from deep_agent.src.guardrails.input_gate import (
    current_decision,
    get_input_gate,
    remember_decision,
)
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


class InputGuardrailHandler(AsyncCallbackHandler):
    """Screen the latest user message before the model runs.

    The handler is constructed with no arguments so LangChain can build it
    from the configure hook. Tool calls in the same task are checked against
    the allowlist from that decision. Argument bounds and secret scans of
    tool arguments belong to Phase 2 and are not done here.
    """

    raise_error = True

    def __init__(self) -> None:
        """Build a handler with an empty seen-content set."""
        super().__init__()
        self._seen: set[str] = set()

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Block, or replace the user text with the masked text."""
        del serialized, run_id, parent_run_id, tags, metadata, kwargs
        gate = get_input_gate()
        message = _last_human(messages)
        if gate is None or message is None:
            return
        content = _message_text(message)
        if not content:
            return
        key = hashlib.sha256(content.encode()).hexdigest()
        if key in self._seen:
            return
        decision = gate.evaluate(content)
        if _is_block(decision):
            logger.warning(
                "input_guardrail_blocked",
                blocked_by=list(getattr(decision, "blocked_by", [])),
            )
            raise InputContentSafetyError(
                "Request blocked by content safety policy. Please rephrase your message."
            )
        self._seen.add(key)
        remember_decision(decision)
        rewritten = getattr(decision, "text", content)
        if rewritten != content:
            message.content = rewritten
            self._seen.add(hashlib.sha256(rewritten.encode()).hexdigest())

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Reject a tool the Phase 1 allowlist does not permit."""
        del input_str, run_id, parent_run_id, tags, metadata, inputs, kwargs
        decision = current_decision()
        if decision is None:
            return
        name = str(serialized.get("name", "unknown"))
        if decision.tool_allowed(name):
            return
        logger.warning("input_guardrail_tool_blocked", tool=name)
        raise ToolContentSafetyError("That action is not available for this request.")


def _is_block(decision: Any) -> bool:
    action = getattr(decision, "action", "")
    value = getattr(action, "value", action)
    return str(value) == "block"


def _last_human(messages: list[list[BaseMessage]]) -> BaseMessage | None:
    if not messages:
        return None
    for message in reversed(messages[-1]):
        if getattr(message, "type", "") == "human":
            return message
    return None


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, list):
        return " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)
