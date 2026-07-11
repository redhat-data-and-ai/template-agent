"""Audit event type constants.

Orchestrator and subagents emit the same event types:
  llm_call, mcp_tool_call, memory_write, subagent_delegation
"""

from typing import Final

LLM_CALL: Final = "llm_call"
MCP_TOOL_CALL: Final = "mcp_tool_call"
MEMORY_WRITE: Final = "memory_write"
SUBAGENT_DELEGATION: Final = "subagent_delegation"

# Event types audited via AuditMiddleware (orchestrator + in-process subagents).
AUDITED_MIDDLEWARE_EVENTS: frozenset[str] = frozenset(
    {LLM_CALL, MCP_TOOL_CALL, MEMORY_WRITE, SUBAGENT_DELEGATION}
)


class AuditEventType:
    """Namespace for platform audit event type strings."""

    LLM_CALL = LLM_CALL
    MCP_TOOL_CALL = MCP_TOOL_CALL
    MEMORY_WRITE = MEMORY_WRITE
    SUBAGENT_DELEGATION = SUBAGENT_DELEGATION
