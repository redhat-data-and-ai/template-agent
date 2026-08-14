"""LangChain callback handler for Granite Guardian input/output guardrails."""

from __future__ import annotations

import hashlib
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from deep_agent.src.guardrails import InputContentSafetyError, ToolContentSafetyError
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


def _extract_content(msg: BaseMessage) -> str:
    content = msg.content
    if isinstance(content, list):
        return " ".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in content
        )
    return str(content)


def _extract_messages_to_scan(
    messages: list[list[BaseMessage]],
) -> list[tuple[str, str]]:
    """Return (content, context_label) pairs to scan from this LLM round.

    Only checks the last human message. Tool results are scanned by GuardianToolProxy,
    which replaces unsafe content with a safe placeholder before it enters state.
    """
    to_scan: list[tuple[str, str]] = []
    if not messages:
        return to_scan
    for msg in reversed(messages[-1]):
        if getattr(msg, "type", "") == "human":
            content = _extract_content(msg)
            if content:
                to_scan.append((content, "input"))
            break
    return to_scan


def _extract_output_text(response: LLMResult) -> str:
    """Extract the first output text from an LLMResult."""
    for generation_list in response.generations:
        for generation in generation_list:
            message = getattr(generation, "message", None)
            if message is not None:
                content = getattr(message, "content", "")
                if isinstance(content, list):
                    return " ".join(
                        c.get("text", "") if isinstance(c, dict) else str(c)
                        for c in content
                    )
                return str(content)
            text = getattr(generation, "text", "")
            if text:
                return text
    return ""


class GraniteGuardianCallbackHandler(AsyncCallbackHandler):
    """Check user input and model output through Granite Guardian.

    Also emits structured audit logs for every tool invocation so that
    MCP tool abuse paths are visible even when Guardian cannot classify them.

    Guardian is called at most once per unique content string per handler
    instance to avoid rescanning the same human message on every LLM round
    in an agentic loop.
    """

    raise_error = True  # propagate ContentSafetyError so the LLM call is aborted

    def __init__(self) -> None:
        """Initialize with an empty set for deduplicating already-scanned content."""
        super().__init__()
        self._scanned: set[str] = set()

    def _already_scanned(self, content: str) -> bool:
        key = hashlib.sha256(content.encode()).hexdigest()
        if key in self._scanned:
            return True
        self._scanned.add(key)
        return False

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
        """Audit-log the tool invocation and pre-screen inputs through Granite Guardian."""
        tool_name = serialized.get("name", "unknown")
        logger.info(
            "tool_call_start",
            tool=tool_name,
            run_id=str(run_id),
            input_keys=sorted((inputs or {}).keys()),
        )
        # Arg safety is handled by GuardianToolProxy Phase 1, which returns a
        # BLOCKED_INPUT ToolMessage instead of raising — preserving parallel batch
        # isolation. Scanning here would duplicate that call and raise an exception
        # that cancels other in-flight tools in the same batch.

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Audit-log the tool result; result safety is handled by GuardianToolProxy."""
        logger.info("tool_call_end", run_id=str(run_id))
        # Result safety is handled by GuardianToolProxy, which replaces unsafe
        # content with a safe placeholder before it becomes a ToolMessage.
        # This callback is kept for audit logging only.

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Audit-log tool errors for observability."""
        logger.warning(
            "tool_call_error",
            run_id=str(run_id),
            error=str(error),
            error_type=type(error).__name__,
        )

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
        """Screen the latest human message through Guardian before each LLM call."""
        from deep_agent.src.guardrails import get_guardrails_config
        from deep_agent.src.guardrails.client import check_injection, check_safety

        if get_guardrails_config() is None:
            return

        for content, context in _extract_messages_to_scan(messages):
            if self._already_scanned(content):
                continue  # already scanned in a previous round — skip
            is_safe, verdict = await check_safety(content, context=context)
            if not is_safe:
                logger.warning("guardian_blocked", context=context, verdict=verdict)
                raise InputContentSafetyError(
                    "Request blocked by content safety policy. "
                    "Please rephrase your message."
                )
            is_safe, verdict = await check_injection(content, context=context)
            if not is_safe:
                logger.warning(
                    "guardian_injection_blocked", context=context, verdict=verdict
                )
                raise InputContentSafetyError(
                    "Request blocked by content safety policy. "
                    "Please rephrase your message."
                )

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Screen LLM output through Guardian, blocking unsafe responses."""
        from deep_agent.src.guardrails import get_guardrails_config
        from deep_agent.src.guardrails.client import check_safety

        if get_guardrails_config() is None:
            return

        content = _extract_output_text(response)
        if not content:
            return

        is_safe, verdict = await check_safety(content, context="output")
        if not is_safe:
            logger.warning("guardian_flagged_output", verdict=verdict)
            raise ToolContentSafetyError("Response blocked by content safety policy.")
