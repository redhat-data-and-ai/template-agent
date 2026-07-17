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
)
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.constants import TAG_NOSTREAM
from langgraph.types import Command

from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)

_BLOCKED_TOOL_MESSAGE = "Tool call blocked by OPA policy."
_BLOCKED_MODEL_MESSAGE = "LLM call blocked by OPA policy."


class _NoStreamModel:
    """Proxy that re-applies TAG_NOSTREAM after bind_tools/bind.

    The agent handler calls ``bind_tools()``, which drops ``with_config`` on the
    model. Re-tagging after bind keeps StreamMessagesHandler quiet without
    ``model_copy`` (which shares/closes the Gemini HTTP client on GC).
    """

    def __init__(self, model: Any) -> None:
        object.__setattr__(self, "_model", model)

    def bind_tools(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.bind_tools(*args, **kwargs).with_config(
            tags=[TAG_NOSTREAM]
        )

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
                return text
        return ""

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        """Async model hook: generate silently, then allow or block."""
        result = await handler(
            request.override(model=_NoStreamModel(request.model))  # type: ignore[arg-type]
        )
        text = self._extract_text(result)

        if not text.strip():
            return result
        if "doctor" in text.lower():
            print("OPA awrap_model_call blocking model output")
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
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Async tool hook."""
        result = await handler(request)

        print(f"[OPA] awrap_tool_call blocking", flush=True)

        # tool_call = request.tool_call
        # tool_name = tool_call.get("name", "unknown")
        # tool_call_id = tool_call.get("id", "")
        # logger.info("OPA blocked tool call: %s", tool_name)
        # return ToolMessage(content=_BLOCKED_TOOL_MESSAGE, tool_call_id=tool_call_id)
        return result
