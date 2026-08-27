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


def _fix_stringified_json_args(tool: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Parse stringified JSON args when the tool schema expects object/array.

    Some models (notably Gemini) serialize nested objects as JSON strings
    instead of proper dicts when calling tools with complex input schemas.
    Parses args whose schema type is ``object``, ``array``, or untyped
    (``Any`` — no ``type`` key in the schema property). Args explicitly
    typed as ``string`` are never modified.
    """
    schema_props: dict[str, Any] = {}
    try:
        schema_props = getattr(tool, "args", {}) or {}
    except Exception:
        return kwargs

    if not schema_props:
        return kwargs

    fixed = dict(kwargs)
    for key, value in fixed.items():
        if not isinstance(value, str):
            continue
        prop = schema_props.get(key, {})
        expected_type = prop.get("type", "")
        if expected_type == "string" or (
            isinstance(expected_type, list) and "string" in expected_type
        ):
            continue
        union_schemas = prop.get("anyOf", []) + prop.get("oneOf", [])
        if any("string" in str(s.get("type", "")) for s in union_schemas):
            continue
        stripped = value.strip()
        if stripped and stripped[0] in ("{", "["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, (dict, list)):
                    fixed[key] = parsed
                    logger.debug(
                        "Fixed stringified JSON arg '%s' for tool '%s'",
                        key,
                        getattr(tool, "name", "?"),
                    )
            except (json.JSONDecodeError, ValueError, RecursionError):
                pass
    return fixed


def wrap_mcp_tools_for_auth(tools: list[Any]) -> list[Any]:
    """Wrap MCP tools so ``NeedsAuthorization`` becomes a resumable interrupt."""
    wrapped: list[Any] = []
    for tool in tools:
        wrapped.append(_wrap_single_tool(tool))
    return wrapped


def _make_safe_ainvoke(target_tool: Any) -> Any:
    """Build an ainvoke wrapper that catches MCP errors for *target_tool*."""
    original_ainvoke = target_tool.ainvoke

    async def safe_ainvoke(tool_input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Wrap ainvoke to catch auth interrupts and MCP errors."""
        from langchain_core.messages import ToolMessage
        from langgraph.errors import GraphBubbleUp

        try:
            return await original_ainvoke(tool_input, config, **kwargs)
        except NeedsAuthorization as exc:
            logger.info(
                "MCP auth required for '%s' — interrupting run",
                exc.mcp_name,
            )
            interrupt(_mcp_auth_interrupt_payload(exc))
            return await original_ainvoke(tool_input, config, **kwargs)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            tool_name = getattr(target_tool, "name", "unknown")
            tool_call_id = ""
            if isinstance(tool_input, dict):
                tool_call_id = str(tool_input.get("id", ""))
            logger.warning("MCP tool '%s' failed: %s", tool_name, exc)
            return ToolMessage(
                content=f"[TOOL_ERROR] {tool_name} failed: {exc}",
                name=tool_name,
                tool_call_id=tool_call_id,
                status="error",
            )

    return safe_ainvoke


def _wrap_single_tool(tool: Any) -> Any:
    coroutine = getattr(tool, "coroutine", None)
    func = getattr(tool, "func", None)

    if inspect.iscoroutinefunction(coroutine):

        async def wrapped_coroutine(**kwargs: Any) -> Any:
            while True:
                try:
                    kwargs = _fix_stringified_json_args(tool, kwargs)
                    return await coroutine(**kwargs)
                except NeedsAuthorization as exc:
                    logger.info(
                        "MCP auth required for '%s' — interrupting run",
                        exc.mcp_name,
                    )
                    interrupt(_mcp_auth_interrupt_payload(exc))

        try:
            wrapped = tool.model_copy(update={"coroutine": wrapped_coroutine})
        except Exception:
            tool.coroutine = wrapped_coroutine
            wrapped = tool
        object.__setattr__(wrapped, "ainvoke", _make_safe_ainvoke(wrapped))
        return wrapped

    if func is not None and inspect.isfunction(func):

        def wrapped_func(**kwargs: Any) -> Any:
            while True:
                try:
                    kwargs = _fix_stringified_json_args(tool, kwargs)
                    return func(**kwargs)
                except NeedsAuthorization as exc:
                    interrupt(_mcp_auth_interrupt_payload(exc))

        try:
            wrapped = tool.model_copy(update={"func": wrapped_func})
        except Exception:
            tool.func = wrapped_func
            wrapped = tool
        object.__setattr__(wrapped, "ainvoke", _make_safe_ainvoke(wrapped))
        return wrapped

    object.__setattr__(tool, "ainvoke", _make_safe_ainvoke(tool))
    return tool
