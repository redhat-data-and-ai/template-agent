"""Custom middleware template.

Replace MyCustomMiddleware with your middleware name and implement the
hooks you need. Delete any hooks you don't use.

Registration: add to config/agent/runtime/agent.yaml under middleware.extra:
  extra:
    - "your_module.path:MyCustomMiddleware"

Important: the constructor must accept zero arguments because
_import_middleware() calls the class with no args.
"""

from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import ToolMessage


class MyCustomMiddleware(AgentMiddleware):
    """Description of what this middleware does."""

    async def awrap_model_call(
        self, request: ModelRequest, handler: Any
    ) -> ModelResponse:
        """Wrap LLM calls with pre/post processing."""
        # Pre-processing: inspect/modify request
        # request.messages — the conversation messages
        # request.model — the LLM instance
        # request.system_message — the system prompt (if any)
        # request.override(**kwargs) — create a copy with modified fields

        response = await handler(request)

        # Post-processing: inspect/modify response
        # response.result — list of BaseMessage from the model

        return response

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Any
    ) -> ToolMessage:
        """Wrap tool executions with pre/post processing."""
        # Pre-processing: inspect tool call
        # request.tool_call — dict with 'name', 'args', 'id'
        # Example: filter by tool name
        # if request.tool_call.get("name") == "some_tool": ...

        result = await handler(request)

        # Post-processing: inspect/modify result
        # result is a ToolMessage or Command

        return result
