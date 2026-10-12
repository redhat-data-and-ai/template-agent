"""CapabilityToolProxy — runtime enforcement point for the capability manifest.

Wraps each tool so manifest membership is checked before every dispatch,
independent of the LLM's context. Mirrors the ``GuardianToolProxy`` pattern
so LangGraph/deepagents see an ordinary ``BaseTool`` and wrappers compose.
"""

from __future__ import annotations

from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, PrivateAttr

from deep_agent.src.capability.manifest import CapabilityManifest
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

CAPABILITY_DENIED_RESULT = (
    "[CAPABILITY_DENIED] This tool is not in the agent's authorized capability "
    "manifest and was blocked by platform policy. Do not retry this or any "
    "other unlisted tool. Tell the user this action is not available."
)


def _get_tool_call_id(input: Any) -> str:
    """Extract tool_call_id from a LangGraph ToolCall dict."""
    if isinstance(input, dict):
        return str(input.get("id", ""))
    return ""


def _make_denied_result(tool_name: str, input: Any) -> Any:
    """Return CAPABILITY_DENIED_RESULT as a ToolMessage for the graph state."""
    from langchain_core.messages import ToolMessage

    return ToolMessage(
        content=CAPABILITY_DENIED_RESULT,
        name=tool_name,
        tool_call_id=_get_tool_call_id(input),
        status="error",
    )


def _emit_denied_audit(tool_name: str, manifest: CapabilityManifest) -> None:
    """Best-effort platform audit event for a capability denial; never raises."""
    try:
        from deep_agent.src.audit.emitter import emit_audit_event
        from deep_agent.src.audit.events import AuditEventType

        emit_audit_event(
            AuditEventType.CAPABILITY_DENIED,
            agent=manifest.agent_name,
            tool=tool_name,
            manifest_source=manifest.source,
        )
    except Exception:
        logger.debug("capability_audit_emit_failed", tool=tool_name, exc_info=True)


class CapabilityToolProxy(BaseTool):
    """Transparent BaseTool wrapper that enforces manifest membership per call.

    All attributes (name, description, args_schema) are copied from the inner
    tool so LangGraph and deepagents see the same interface. Both ``ainvoke``
    (the LangGraph hot path) and ``_run`` (sync fallback) check the manifest
    before delegating to the inner tool.
    """

    name: str = ""
    description: str = ""
    _inner: Any = PrivateAttr()
    _manifest: CapabilityManifest = PrivateAttr()

    def __init__(self, inner_tool: Any, manifest: CapabilityManifest) -> None:
        """Copy name/description/args_schema from inner_tool; bind the manifest."""
        schema: Optional[Type[BaseModel]] = getattr(inner_tool, "args_schema", None)
        super().__init__(
            name=getattr(inner_tool, "name", ""),
            description=getattr(inner_tool, "description", ""),
            args_schema=schema,
        )
        self._inner = inner_tool
        self._manifest = manifest

    def _denied(self) -> bool:
        """Return True and emit audit if this tool is not in the manifest."""
        if self._manifest.allows(self.name):
            return False
        logger.warning(
            "capability_denied",
            agent=self._manifest.agent_name,
            tool=self.name,
            manifest_source=self._manifest.source,
        )
        _emit_denied_audit(self.name, self._manifest)
        return True

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Deny immediately if the tool is outside the manifest; else delegate."""
        if self._denied():
            return _make_denied_result(self.name, input)
        return await self._inner.ainvoke(input, config, **kwargs)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Sync fallback; same manifest check as ainvoke.

        Reassembles input for ``invoke()`` since ``BaseTool.run()`` delivers
        structured-tool fields as kwargs, but ``invoke()`` expects a single
        ``input`` argument.
        """
        if self._denied():
            return CAPABILITY_DENIED_RESULT
        tool_input: Any = args[0] if args else kwargs
        return self._inner.invoke(tool_input)


def enforce_capability(tools: list[Any], manifest: CapabilityManifest) -> list[Any]:
    """Wrap every tool with a CapabilityToolProxy bound to *manifest*.

    Defense-in-depth: tools already match the manifest under normal
    operation, but a regression or dynamic injection that bypasses
    resolution is caught at dispatch time.
    """
    if not tools:
        return tools
    return [CapabilityToolProxy(t, manifest) for t in tools]
