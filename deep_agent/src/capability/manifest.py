"""Capability manifest resolution.

The manifest is the enforced answer to "which tools may this agent invoke at
runtime" -- as distinct from "which MCP servers did the agent declare", which
is all the platform previously enforced (see OFFSEC-384). It is derived once
per request from the frontmatter agent-engine materializes into this pod at
deploy time from the reviewed ``AGENTS.md``. A running conversation cannot
alter it, so it stays intact even if a tool result carries a prompt injection.

``resolve_capability_manifest`` replaces two separate inline fallbacks that
existed before this package, one in ``graph.py`` and (nearly identically)
twice in ``subagents.py``:

- An agent that declared ``mcps:`` without an explicit ``tools:`` list
  silently got every tool those servers happened to expose. Behaviour for
  these agents is unchanged here -- they still get every tool the server
  exposes -- but the grant is now a named, logged, and enforced
  ``CapabilityManifest`` instead of an implicit side effect of list-building.
- On ``graph.py`` specifically, an agent *with* an explicit ``tools:`` list
  still had every other tool from its declared MCP server(s) unioned back
  in (any MCP tool not already in the resolved set was appended
  unconditionally). That silently widened a reviewed allow-list to the
  server's entire live tool set and is the exact gap OFFSEC-384 calls out --
  this resolver does not reproduce it: an explicit ``tools:`` list is now
  authoritative and is never unioned with the rest of the server's tools.

An explicit ``tools:`` list remains the recommended way to pin an agent's
capabilities to what was actually reviewed at publish time, since an MCP
server's live tool set can otherwise drift after review without the agent
ever being re-reviewed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

EXPLICIT = "explicit"
IMPLICIT_ALL_MCP = "implicit_all_mcp"
NONE = "none"


@dataclass(frozen=True)
class CapabilityManifest:
    """Frozen set of tool names one agent/subagent is authorized to invoke.

    Built once per request/graph-build from deploy-time config; never mutated
    afterwards, and never derived from anything inside the model's context.
    """

    agent_name: str
    allowed_tool_names: frozenset[str]
    source: str  # EXPLICIT | IMPLICIT_ALL_MCP | NONE

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
    """Resolve the enforced tool list and the manifest that will police it.

    Args:
        tool_names: Explicit ``tools:`` frontmatter list, if any. Callers
            must pass ``None`` when the ``tools:`` key is absent from
            frontmatter entirely (e.g. ``agent_cfg.get("tools")``, not
            ``agent_cfg.get("tools", [])``) so an *omitted* field can be
            told apart from an author writing ``tools: []`` on purpose --
            the two must not collapse to the same fallback behaviour below.
        available_tools: All tools currently reachable from the agent's
            declared MCP servers.
        mcp_server_names: Declared ``mcps:`` frontmatter list, if any.
        agent_name: Orchestrator or subagent name, for logging/audit.

    Returns:
        ``(tools, manifest)`` -- ``tools`` is the resolved tool list (the same
        result the old inline fallbacks produced); ``manifest`` is the frozen
        allow-list a ``CapabilityToolProxy`` enforces before every dispatch.
    """
    # Imported lazily (not at module load time) so tests -- and any future
    # caller -- that patch `deep_agent.src.agent.config.agent_config` before
    # invoking this function see the patched singleton, matching the
    # existing lazy-import pattern used by graph.py/subagents.py for the
    # same reason.
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

    # `tool_names == []` (author wrote an explicit empty allow-list) must NOT
    # fall through to the implicit-all-mcp grant below -- that would make the
    # most restrictive possible declaration behave identically to omitting
    # `tools:` altogether, i.e. the opposite of what the author asked for.
    # Only a genuinely *omitted* field (`tool_names is None`) gets the
    # implicit fallback.
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
        agent_name=agent_name, allowed_tool_names=frozenset(), source=NONE
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
