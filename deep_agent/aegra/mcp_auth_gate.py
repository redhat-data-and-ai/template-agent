"""Auth-first gate for MCP servers requiring OAuth/DCR.

Intercepts the graph BEFORE the LLM runs. If any MCP server needs
authentication, fires a LangGraph interrupt so the user can connect
before the model ever sees the tool list. On resume (after auth),
the graph is rebuilt with real tools and this middleware is a no-op.
"""

from __future__ import annotations

import json
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langgraph.types import interrupt

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


class McpAuthGateMiddleware(AgentMiddleware):
    """Interrupt before the LLM runs if MCP servers need OAuth."""

    name = "mcp_auth_gate"

    def __init__(self, pending_auth: list[dict[str, str]]) -> None:
        """Initialize with the list of MCP servers awaiting authentication."""
        self.pending_auth = pending_auth

    async def abefore_agent(self, state: Any, config: Any = None) -> Any:
        """Interrupt with mcp_auth_required if any server needs OAuth."""
        if not self.pending_auth:
            return state

        server = self.pending_auth[0]
        logger.info(
            "MCP auth gate: interrupting for '%s' before LLM runs",
            server["mcp_name"],
        )
        interrupt(
            json.dumps(
                {
                    "type": "mcp_auth_required",
                    "mcp_name": server["mcp_name"],
                    "connect_url": server["connect_url"],
                    "message": f"Connect to {server['mcp_name']} to use these tools",
                }
            )
        )
        return state
