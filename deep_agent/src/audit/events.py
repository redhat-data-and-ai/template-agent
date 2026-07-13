"""Audit event type constants."""

from __future__ import annotations

from typing import Final

LLM_CALL: Final = "llm_call"
MCP_TOOL_CALL: Final = "mcp_tool_call"
MEMORY_WRITE: Final = "memory_write"
SUBAGENT_DELEGATION: Final = "subagent_delegation"
CODE_EXECUTION: Final = "code_execution"

AUDITED_MIDDLEWARE_EVENTS: frozenset[str] = frozenset(
    {LLM_CALL, MCP_TOOL_CALL, MEMORY_WRITE, SUBAGENT_DELEGATION, CODE_EXECUTION}
)


class AuditEventType:
    """Namespace for audit event type constants."""

    LLM_CALL = LLM_CALL
    MCP_TOOL_CALL = MCP_TOOL_CALL
    MEMORY_WRITE = MEMORY_WRITE
    SUBAGENT_DELEGATION = SUBAGENT_DELEGATION
    CODE_EXECUTION = CODE_EXECUTION
