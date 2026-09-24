"""CapabilityToolProxy -- the runtime enforcement point for the capability manifest.

Wraps a tool so its manifest membership is checked immediately before the
inner tool executes, on every call, no matter which code path assembled the
graph's tool list. This is deliberately independent of the LLM: the check
reads a ``CapabilityManifest`` computed once at graph-build time, so nothing
in the running conversation -- including a prompt injection surfaced through
a tool result -- can add a tool to the allow-list or bypass the check.

This mirrors the existing ``GuardianToolProxy`` pattern (same ``BaseTool``
wrapper shape) so LangGraph's ``ToolNode`` and deepagents see an ordinary
tool, and other wrappers (Guardian content-safety, MCP auth injection) can be
composed with it in any order.
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
        """Check the manifest and log/audit a denial if this tool isn't authorized."""
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

    # ainvoke is the hot path -- LangGraph's ToolNode calls this.
    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Deny immediately if the tool is outside the manifest; else delegate."""
        if self._denied():
            return _make_denied_result(self.name, input)
        return await self._inner.ainvoke(input, config, **kwargs)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Sync fallback path; same enforcement as ainvoke.

        ``BaseTool.run()`` parses the caller's input against the tool's
        schema and, for a generic ``_run(*args, **kwargs)`` signature like
        this one, always delivers it as either a single positional value
        (unstructured tools) or as keyword arguments matching the schema
        fields (structured tools) -- never both. ``BaseTool.invoke()``
        expects that same input back as a single ``input`` argument, so it
        must be reassembled here rather than re-spread with ``*args,
        **kwargs``, which would try to satisfy ``invoke``'s ``input``
        parameter from field-named kwargs and raise
        ``TypeError: missing 1 required positional argument: 'input'`` for
        every structured (multi-field) tool.
        """
        if self._denied():
            return CAPABILITY_DENIED_RESULT
        tool_input: Any = args[0] if args else kwargs
        return self._inner.invoke(tool_input)


def enforce_capability(tools: list[Any], manifest: CapabilityManifest) -> list[Any]:
    """Wrap every tool in *tools* with a CapabilityToolProxy bound to *manifest*.

    This is the always-on defense-in-depth gate: under normal operation every
    tool passed in is already a member of *manifest* (both are derived from
    the same resolution step), so no legitimate call is affected. The value is
    that any future code path that adds a tool to the graph without going
    through capability resolution -- a regression, a bug in a merge, or later
    a dynamically-injected tool -- is denied at dispatch time instead of
    silently executing.
    """
    if not tools:
        return tools
    return [CapabilityToolProxy(t, manifest) for t in tools]
