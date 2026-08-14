"""OPA authorization middleware.

Intercepts every tool call and enforces OPA policy. When OPA is enabled,
the tool call is blocked and a hardcoded denial message is returned.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    hook_config,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.constants import TAG_NOSTREAM
from langgraph.runtime import Runtime
from langgraph.types import Command

from deep_agent.src.opa.config import get_opa_max_retries
from deep_agent.src.opa.service import evaluate_message, evaluate_trajectory
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)

_BLOCKED_TOOL_MESSAGE = "Tool call blocked by OPA policy."
_BLOCKED_MODEL_MESSAGE = "I'm unable to respond to that request. Please try rephrasing."
_BLOCKED_TRAJECTORY_MESSAGE = "This thread has been terminated due to policy violation. Please start a new chat thread."


class _NoStreamModel:
    """Proxy that re-applies TAG_NOSTREAM after bind_tools/bind.

    The agent handler calls ``bind_tools()``, which drops ``with_config`` on the
    model. Re-tagging after bind keeps StreamMessagesHandler quiet without
    ``model_copy`` (which shares/closes the Gemini HTTP client on GC).
    """

    def __init__(self, model: Any) -> None:
        object.__setattr__(self, "_model", model)

    def bind_tools(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.bind_tools(*args, **kwargs).with_config(tags=[TAG_NOSTREAM])

    def bind(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.bind(*args, **kwargs).with_config(tags=[TAG_NOSTREAM])

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


class OPAMiddleware(AgentMiddleware):
    """Block model and tool calls based on OPA authorization policy."""

    def _extract_text(self, result: ModelResponse[Any]) -> str:
        """Return the text of the last AIMessage with non-empty text.

        Uses AIMessage.text which handles both str content and content-block
        lists, and returns '' for tool-call-only turns.
        """
        for message in reversed(result.result):
            if not isinstance(message, AIMessage):
                continue
            text = message.text
            if text:
                return str(text)
        return ""

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self,
        state: dict[str, Any],
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        """Evaluate conversation trajectory before each model call."""
        messages = state.get("messages") or []
        if not messages:
            return None

        trajectory = [
            m
            for m in messages
            if isinstance(m, BaseMessage) and not m.additional_kwargs.get("opa_retry")
        ]
        opa = await evaluate_trajectory(trajectory)
        logger.debug(
            "OPA abefore_model result: allowed=%s reason_count=%d",
            opa.allowed,
            len(opa.denial_reasons),
        )
        if opa.allowed:
            return None

        logger.debug(
            "OPA abefore_model blocking trajectory: reason_count=%d",
            len(opa.denial_reasons),
        )
        logger.info(
            "OPA abefore_model blocking trajectory: %s",
            opa.denial_reasons,
        )
        return {
            "messages": [AIMessage(content=_BLOCKED_TRAJECTORY_MESSAGE)],
            "jump_to": "end",
        }

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        """Async model hook: generate silently, then allow or block with retries."""
        max_retries = get_opa_max_retries()
        current_request = request

        for attempt in range(max_retries + 1):
            logger.debug(
                "OPA awrap_model_call attempt %d/%d",
                attempt + 1,
                max_retries + 1,
            )
            result = await handler(
                current_request.override(model=_NoStreamModel(current_request.model))
            )
            logger.debug(
                "OPA awrap_model_call result: message_count=%d", len(result.result)
            )
            text = self._extract_text(result)

            if not text.strip():
                return result

            opa = await evaluate_message("llm_response", agent_message=text)
            logger.debug(
                "OPA awrap_model_call result: allowed=%s reason_count=%d",
                opa.allowed,
                len(opa.denial_reasons),
            )
            if opa.allowed:
                return result

            logger.debug(
                "OPA awrap_model_call blocking model output: attempt=%d/%d reason_count=%d",
                attempt + 1,
                max_retries + 1,
                len(opa.denial_reasons),
            )

            if attempt < max_retries:
                denial_summary = (
                    "; ".join(opa.denial_reasons)
                    if opa.denial_reasons
                    else "policy violation"
                )
                feedback = HumanMessage(
                    content=(
                        f"Your previous response was blocked by security policy. "
                        f"Reasons: {denial_summary}. "
                        f"Please revise your response to comply with the policy."
                    ),
                    additional_kwargs={"opa_retry": True},
                )
                current_request = current_request.override(
                    messages=current_request.messages + [feedback]
                )

        orig = result.result[-1]
        return ModelResponse(
            result=[
                AIMessage(
                    content=_BLOCKED_MODEL_MESSAGE,
                    id=orig.id,
                    name=orig.name,
                )
            ]
        )

    def _extract_tool_content(self, result: ToolMessage | Command[Any]) -> str:
        """Return tool output text from a ToolMessage or task Command."""
        if isinstance(result, ToolMessage):
            if isinstance(result.content, str):
                return result.content
            return str(result.content) if result.content else ""

        if isinstance(result, Command):
            update = result.update or {}
            messages = update.get("messages") or []
            for message in reversed(messages):
                if not isinstance(message, ToolMessage):
                    continue
                if isinstance(message.content, str) and message.content.strip():
                    return message.content
                if message.content:
                    return str(message.content)
        return ""

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Async tool hook: run tool, then allow or block the result."""
        result = await handler(request)

        tool_call = request.tool_call
        is_dict = isinstance(tool_call, dict)
        tool_name = tool_call.get("name") if is_dict else tool_call.name
        tool_id = tool_call.get("id") if is_dict else tool_call.id

        tool_content = self._extract_tool_content(result)
        if not tool_content.strip():
            return result

        opa = await evaluate_message("tool_response", result=tool_content)
        logger.debug(
            "OPA awrap_tool_call result: allowed=%s reason_count=%d",
            opa.allowed,
            len(opa.denial_reasons),
        )
        if opa.allowed:
            return result

        logger.debug(
            "OPA awrap_tool_call blocking tool output: reason_count=%d",
            len(opa.denial_reasons),
        )

        denial_summary = (
            "; ".join(opa.denial_reasons) if opa.denial_reasons else "policy violation"
        )
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=(
                            f"{_BLOCKED_TOOL_MESSAGE} "
                            f"Reasons: {denial_summary}. "
                            f"Please revise the tool call arguments to comply with the policy and try again."
                        ),
                        tool_call_id=tool_id,
                        name=tool_name,
                        status="error",
                        additional_kwargs={"opa_retry": True},
                    ),
                ],
            },
        )
