"""Capability manifest resolution.

Resolves the enforced tool allow-list for an agent or subagent from
deploy-time frontmatter. A running conversation cannot alter it.

Key behaviours:

- Explicit ``tools:`` list → authoritative manifest; never widened with
  additional MCP server tools.
- ``mcps:`` declared without ``tools:`` → implicit grant of every tool
  those servers expose, logged and audited as ``implicit_all_mcp``.
- ``tools: []`` (explicit empty) → empty manifest; most restrictive.

An explicit ``tools:`` list is the recommended way to pin capabilities
to what was reviewed at publish time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

EXPLICIT = "explicit"
IMPLICIT_ALL_MCP = "implicit_all_mcp"
NO_MANIFEST = "no_manifest"


@dataclass(frozen=True)
class CapabilityManifest:
    """Frozen set of tool names one agent/subagent is authorized to invoke.

    Built once per request/graph-build from deploy-time config; never mutated
    afterwards, and never derived from anything inside the model's context.
    """

    agent_name: str
    allowed_tool_names: frozenset[str]
    source: str  # EXPLICIT | IMPLICIT_ALL_MCP | NO_MANIFEST

    def allows(self, tool_name: str) -> bool:
        """Return True if *tool_name* is authorized under this manifest."""
        return tool_name in self.allowed_tool_names

    def merged_with(self, extra_tool_names: Any) -> "CapabilityManifest":
        """Return a new manifest that additionally authorizes *extra_tool_names*.

        Intended for tool families that carry their own scoping mechanism
        internally (e.g. MCP resource-read tools, which enforce a per-call
        URI allowlist independent of the tool name) and are therefore safe
        to add to the dispatch-time gate without going through name-based
        resolution against ``available_tools``.
        """
        return CapabilityManifest(
            agent_name=self.agent_name,
            allowed_tool_names=self.allowed_tool_names | frozenset(extra_tool_names),
            source=self.source,
        )


def resolve_capability_manifest(
    tool_names: list[str] | None,
    available_tools: list[Any],
    mcp_server_names: list[str] | None,
    agent_name: str = "agent",
) -> tuple[list[Any], CapabilityManifest]:
    """Resolve the enforced tool list and capability manifest.

    Args:
        tool_names: Explicit ``tools:`` list, or ``None`` when omitted.
            Pass ``None`` (not ``[]``) for absent keys so the omitted-vs-
            explicit-empty distinction is preserved.
        available_tools: Tools reachable from the agent's declared MCP servers.
        mcp_server_names: Declared ``mcps:`` list, if any.
        agent_name: Agent/subagent name for logging.

    Returns:
        ``(tools, manifest)`` tuple.
    """
    # Lazy import so tests can patch agent_config before this runs.
    from deep_agent.src.agent.config import agent_config

    tools = (
        agent_config.resolve_tools(tool_names, available_tools, agent_name=agent_name)
        if tool_names
        else []
    )

    if tools:
        return list(tools), CapabilityManifest(
            agent_name=agent_name,
            allowed_tool_names=frozenset(t.name for t in tools),
            source=EXPLICIT,
        )

    # Explicit tools: list was provided but nothing resolved — builder
    # intended a restriction; honour it as an explicit (empty) manifest
    # so downstream code knows this is deliberate, not absent.
    if tool_names and not tools:
        logger.warning(
            "capability_explicit_empty",
            agent=agent_name,
            declared_tools=sorted(tool_names),
            message=(
                "Agent declared explicit 'tools:' but none could be resolved "
                "against available MCP server tools. The agent will have no "
                "MCP server tools. Verify tool names match MCP tool names."
            ),
        )
        return [], CapabilityManifest(
            agent_name=agent_name,
            allowed_tool_names=frozenset(),
            source=EXPLICIT,
        )

    # Explicit empty list (tools: []) → no tools; only omitted field gets
    # the implicit-all-mcp fallback.
    if tool_names is None and mcp_server_names and available_tools:
        logger.info(
            "capability_implicit_grant",
            agent=agent_name,
            mcp_servers=sorted(mcp_server_names),
            tool_count=len(available_tools),
            message=(
                "Agent declared MCP server(s) without an explicit 'tools:' "
                "list -- granting every tool those servers currently expose. "
                "This tracks the server's live tool set, not a reviewed "
                "manifest; add 'tools:' to pin capabilities to what was "
                "actually reviewed."
            ),
        )
        _emit_manifest_audit_event(
            agent_name=agent_name,
            mcp_server_names=mcp_server_names,
            tool_names=[t.name for t in available_tools],
        )
        return list(available_tools), CapabilityManifest(
            agent_name=agent_name,
            allowed_tool_names=frozenset(t.name for t in available_tools),
            source=IMPLICIT_ALL_MCP,
        )

    return [], CapabilityManifest(
        agent_name=agent_name, allowed_tool_names=frozenset(), source=NO_MANIFEST
    )


def _emit_manifest_audit_event(
    *,
    agent_name: str,
    mcp_server_names: list[str],
    tool_names: list[str],
) -> None:
    """Best-effort platform audit event; never raises into the caller."""
    try:
        from deep_agent.src.audit.emitter import emit_audit_event
        from deep_agent.src.audit.events import AuditEventType

        emit_audit_event(
            AuditEventType.CAPABILITY_IMPLICIT_GRANT,
            agent=agent_name,
            mcp_servers=sorted(mcp_server_names),
            tools=sorted(tool_names),
        )
    except Exception:
        logger.debug("capability_audit_emit_failed", agent=agent_name, exc_info=True)
