"""Wrap MCP tools to raise LangGraph interrupts when OAuth is required."""

from __future__ import annotations

import inspect
import json
from typing import Any

from langgraph.types import interrupt

from deep_agent.aegra.mcp_auth import NeedsAuthorization
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


def _mcp_auth_interrupt_payload(exc: NeedsAuthorization) -> str:
    return json.dumps(
        {
            "type": "mcp_auth_required",
            "mcp_name": exc.mcp_name,
            "connect_url": exc.connect_url,
            "message": f"Connect to {exc.mcp_name} to use these tools",
        }
    )


def wrap_mcp_tools_for_auth(tools: list[Any]) -> list[Any]:
    """Wrap MCP tools so ``NeedsAuthorization`` becomes a resumable interrupt."""
    wrapped: list[Any] = []
    for tool in tools:
        wrapped.append(_wrap_single_tool(tool))
    return wrapped


def _wrap_single_tool(tool: Any) -> Any:
    coroutine = getattr(tool, "coroutine", None)
    func = getattr(tool, "func", None)

    if inspect.iscoroutinefunction(coroutine):

        async def wrapped_coroutine(**kwargs: Any) -> Any:
            while True:
                try:
                    return await coroutine(**kwargs)
                except NeedsAuthorization as exc:
                    logger.info(
                        "MCP auth required for '%s' — interrupting run",
                        exc.mcp_name,
                    )
                    interrupt(_mcp_auth_interrupt_payload(exc))

        try:
            return tool.model_copy(update={"coroutine": wrapped_coroutine})
        except Exception:
            tool.coroutine = wrapped_coroutine
            return tool

    if func is not None and inspect.isfunction(func):

        def wrapped_func(**kwargs: Any) -> Any:
            try:
                return func(**kwargs)
            except NeedsAuthorization as exc:
                interrupt(_mcp_auth_interrupt_payload(exc))

        try:
            return tool.model_copy(update={"func": wrapped_func})
        except Exception:
            tool.func = wrapped_func
            return tool

    return tool
