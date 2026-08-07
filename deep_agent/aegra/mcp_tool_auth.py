"""Wrap MCP tools to raise LangGraph interrupts when OAuth is required.

When multiple MCP tools need auth simultaneously (parallel tool calls in a
single node), only the first fires an ``interrupt()``; the rest return an
error string so the LLM retries them on the next turn.  This prevents the
``RuntimeError: multiple pending interrupts`` that Aegra cannot resolve
because it does not pass ``resume_interrupt_id``.
"""

from __future__ import annotations

import contextvars
import inspect
import json
from typing import Any

from langgraph.types import interrupt

from deep_agent.aegra.mcp_auth import NeedsAuthorization
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_auth_interrupt_fired: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_auth_interrupt_fired", default=False
)


def reset_auth_interrupt_state() -> None:
    """Reset the per-invocation interrupt guard.

    Called at the start of each graph factory build so that a fresh run
    can fire an interrupt if needed.
    """
    _auth_interrupt_fired.set(False)


def _mcp_auth_interrupt_payload(exc: NeedsAuthorization) -> str:
    return json.dumps(
        {
            "type": "mcp_auth_required",
            "mcp_name": exc.mcp_name,
            "connect_url": exc.connect_url,
            "message": f"Connect to {exc.mcp_name} to use these tools",
        }
    )


def _handle_needs_auth(exc: NeedsAuthorization) -> str | None:
    """Fire an interrupt for the first auth failure; return error string for the rest.

    Returns ``None`` when this is the first failure (caller should call
    ``interrupt()``).  Returns an error-message string for subsequent
    failures in the same node execution.
    """
    if _auth_interrupt_fired.get(False):
        logger.info(
            "MCP auth also required for '%s' — skipping duplicate interrupt",
            exc.mcp_name,
        )
        return json.dumps(
            {
                "error": "mcp_auth_pending",
                "mcp_name": exc.mcp_name,
                "connect_url": exc.connect_url,
                "message": (
                    f"Authentication required for {exc.mcp_name}. "
                    "Another MCP is already awaiting authentication — "
                    "please connect and retry."
                ),
            }
        )
    _auth_interrupt_fired.set(True)
    return None


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
                    fallback = _handle_needs_auth(exc)
                    if fallback is not None:
                        return fallback
                    interrupt(_mcp_auth_interrupt_payload(exc))

        try:
            return tool.model_copy(update={"coroutine": wrapped_coroutine})
        except Exception:
            tool.coroutine = wrapped_coroutine
            return tool

    if func is not None and inspect.isfunction(func):

        def wrapped_func(**kwargs: Any) -> Any:
            while True:
                try:
                    return func(**kwargs)
                except NeedsAuthorization as exc:
                    fallback = _handle_needs_auth(exc)
                    if fallback is not None:
                        return fallback
                    interrupt(_mcp_auth_interrupt_payload(exc))

        try:
            return tool.model_copy(update={"func": wrapped_func})
        except Exception:
            tool.func = wrapped_func
            return tool

    return tool
