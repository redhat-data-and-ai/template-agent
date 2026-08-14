"""LangChain middleware for platform audit events.

Orchestrator and in-process subagents use the same ``AuditMiddleware`` with
identical classification rules:

- ``llm_call`` — every model invocation (sync + async paths)
- ``mcp_tool_call`` — tools in the subagent/orchestrator MCP tool name set
- ``memory_write`` — ``edit_file`` / ``write_file`` under ``/memories/`` (log only; no memory setup)
- ``subagent_delegation`` — ``task`` tool (orchestrator delegating to subagent)

Events include ``agent`` (``orchestrator`` or subagent name).
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deep_agent.src.audit.config import is_audit_enabled
from deep_agent.src.audit.emitter import emit_audit_event
from deep_agent.src.audit.events import AuditEventType

_MEMORY_TOOLS = frozenset({"edit_file", "write_file"})
_SUBAGENT_TOOL = "task"
_ORCHESTRATOR_AGENT = "orchestrator"


def _tool_path(args: dict[str, Any]) -> str:
    for key in ("path", "file_path", "filename", "file"):
        value = args.get(key)
        if isinstance(value, str):
            return value
    return ""


def _is_memory_write(tool_name: str, args: dict[str, Any]) -> bool:
    if tool_name not in _MEMORY_TOOLS:
        return False
    path = _tool_path(args)
    return "memories" in path.replace("\\", "/")


def _model_name(request: ModelRequest[Any]) -> str:
    model = request.model
    if isinstance(model, str):
        return model
    return (
        getattr(model, "model_name", None) or getattr(model, "model", None) or "unknown"
    )


def classify_tool_call(
    tool_name: str,
    args: dict[str, Any],
    *,
    mcp_tool_names: frozenset[str],
) -> str:
    """Classify a tool call using orchestrator/subagent parity rules."""
    if tool_name == _SUBAGENT_TOOL:
        return AuditEventType.SUBAGENT_DELEGATION
    if tool_name in mcp_tool_names:
        return AuditEventType.MCP_TOOL_CALL
    if _is_memory_write(tool_name, args):
        return AuditEventType.MEMORY_WRITE
    return ""


class AuditMiddleware(AgentMiddleware):
    """Emit platform audit events for LLM and tool operations."""

    def __init__(
        self,
        *,
        mcp_tool_names: frozenset[str] | None = None,
        subagent: str | None = None,
        agent: str | None = None,
    ) -> None:
        """Initialize with optional MCP tool filter and agent identity."""
        self._mcp_tool_names = mcp_tool_names or frozenset()
        self._agent = agent or subagent or _ORCHESTRATOR_AGENT

    def _base_details(self) -> dict[str, Any]:
        return {"agent": self._agent}

    def _emit_llm_phase(
        self,
        *,
        phase: str,
        model: str,
        message_count: int,
        status: str | None = None,
        latency_ms: float | None = None,
        error: str | None = None,
    ) -> None:
        details: dict[str, Any] = {
            "phase": phase,
            "model": model,
            **self._base_details(),
        }
        if phase == "start":
            details["message_count"] = message_count
        if status is not None:
            details["status"] = status
        if latency_ms is not None:
            details["latency_ms"] = latency_ms
        if error:
            details["error"] = error
        emit_audit_event(AuditEventType.LLM_CALL, **details)

    def _audit_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        if not is_audit_enabled():
            return handler(request)

        model = _model_name(request)
        started = time.monotonic()
        self._emit_llm_phase(
            phase="start",
            model=model,
            message_count=len(request.messages),
        )
        try:
            response = handler(request)
        except Exception as exc:
            elapsed_ms = round((time.monotonic() - started) * 1000, 2)
            self._emit_llm_phase(
                phase="complete",
                model=model,
                message_count=len(request.messages),
                status="error",
                latency_ms=elapsed_ms,
                error=str(exc) or type(exc).__name__,
            )
            raise

        elapsed_ms = round((time.monotonic() - started) * 1000, 2)
        self._emit_llm_phase(
            phase="complete",
            model=model,
            message_count=len(request.messages),
            status="success",
            latency_ms=elapsed_ms,
        )
        return response

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Sync model hook — subagents use ``Runnable.invoke()``."""
        return self._audit_model_call(request, handler)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        """Async wrapper that audits LLM model invocations."""
        if not is_audit_enabled():
            return await handler(request)

        model = _model_name(request)
        started = time.monotonic()
        self._emit_llm_phase(
            phase="start",
            model=model,
            message_count=len(request.messages),
        )
        try:
            response = await handler(request)
        except Exception as exc:
            elapsed_ms = round((time.monotonic() - started) * 1000, 2)
            self._emit_llm_phase(
                phase="complete",
                model=model,
                message_count=len(request.messages),
                status="error",
                latency_ms=elapsed_ms,
                error=str(exc) or type(exc).__name__,
            )
            raise

        elapsed_ms = round((time.monotonic() - started) * 1000, 2)
        self._emit_llm_phase(
            phase="complete",
            model=model,
            message_count=len(request.messages),
            status="success",
            latency_ms=elapsed_ms,
        )
        return response

    def _classify_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        return classify_tool_call(tool_name, args, mcp_tool_names=self._mcp_tool_names)

    def _emit_tool_event(
        self,
        audit_type: str,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        status: str,
        latency_ms: float,
        error: str | None = None,
    ) -> None:
        if not audit_type:
            return

        details: dict[str, Any] = {
            "tool": tool_name,
            "status": status,
            "latency_ms": latency_ms,
            **self._base_details(),
        }
        if error:
            details["error"] = error

        if audit_type == AuditEventType.SUBAGENT_DELEGATION:
            details["delegated_subagent"] = tool_args.get("subagent") or tool_args.get(
                "name"
            )
        elif audit_type == AuditEventType.MEMORY_WRITE:
            details["path"] = _tool_path(tool_args)
        elif audit_type == AuditEventType.MCP_TOOL_CALL:
            details["args_keys"] = sorted(tool_args.keys())

        emit_audit_event(audit_type, **details)

    def _audit_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        if not is_audit_enabled():
            return handler(request)

        tool_call = request.tool_call
        tool_name = tool_call.get("name", "unknown")
        tool_args = tool_call.get("args")
        if not isinstance(tool_args, dict):
            tool_args = {}
        audit_type = self._classify_tool(tool_name, tool_args)

        started = time.monotonic()
        try:
            result = handler(request)
            status = "success"
            error: str | None = None
        except Exception as exc:
            elapsed_ms = round((time.monotonic() - started) * 1000, 2)
            self._emit_tool_event(
                audit_type,
                tool_name=tool_name,
                tool_args=tool_args,
                status="error",
                latency_ms=elapsed_ms,
                error=str(exc) or type(exc).__name__,
            )
            raise

        elapsed_ms = round((time.monotonic() - started) * 1000, 2)
        self._emit_tool_event(
            audit_type,
            tool_name=tool_name,
            tool_args=tool_args,
            status=status,
            latency_ms=elapsed_ms,
            error=error,
        )
        return result

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Sync tool hook — subagents use ``Runnable.invoke()``."""
        return self._audit_tool_call(request, handler)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Async wrapper that audits tool invocations."""
        if not is_audit_enabled():
            return await handler(request)

        tool_call = request.tool_call
        tool_name = tool_call.get("name", "unknown")
        tool_args = tool_call.get("args")
        if not isinstance(tool_args, dict):
            tool_args = {}
        audit_type = self._classify_tool(tool_name, tool_args)

        started = time.monotonic()
        try:
            result = await handler(request)
            status = "success"
            error: str | None = None
        except Exception as exc:
            elapsed_ms = round((time.monotonic() - started) * 1000, 2)
            self._emit_tool_event(
                audit_type,
                tool_name=tool_name,
                tool_args=tool_args,
                status="error",
                latency_ms=elapsed_ms,
                error=str(exc) or type(exc).__name__,
            )
            raise

        elapsed_ms = round((time.monotonic() - started) * 1000, 2)
        self._emit_tool_event(
            audit_type,
            tool_name=tool_name,
            tool_args=tool_args,
            status=status,
            latency_ms=elapsed_ms,
            error=error,
        )
        return result
