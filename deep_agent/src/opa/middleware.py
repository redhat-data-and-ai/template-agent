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
from langgraph.types import Command

from deep_agent.src.opa.config import is_opa_enabled
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_BLOCKED_TOOL_MESSAGE = "Tool call blocked by OPA policy."
_BLOCKED_MODEL_MESSAGE = "LLM call blocked by OPA policy."


class OPAMiddleware(AgentMiddleware):
    """Block model and tool calls based on OPA authorization policy."""

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Sync model hook."""
        handler(request)

        logger.info("OPA blocked model call")
        return ModelResponse(result=[AIMessage(content=_BLOCKED_MODEL_MESSAGE)])

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        """Async model hook."""
        await handler(request)

        logger.info("OPA blocked model call")
        return ModelResponse(result=[AIMessage(content=_BLOCKED_MODEL_MESSAGE)])

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Sync tool hook."""
        handler(request)

        tool_call = request.tool_call
        tool_name = tool_call.get("name", "unknown")
        tool_call_id = tool_call.get("id", "")
        logger.info("OPA blocked tool call: %s", tool_name)
        return ToolMessage(content=_BLOCKED_TOOL_MESSAGE, tool_call_id=tool_call_id)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Async tool hook."""
        await handler(request)

        tool_call = request.tool_call
        tool_name = tool_call.get("name", "unknown")
        tool_call_id = tool_call.get("id", "")
        logger.info("OPA blocked tool call: %s", tool_name)
        return ToolMessage(content=_BLOCKED_TOOL_MESSAGE, tool_call_id=tool_call_id)
