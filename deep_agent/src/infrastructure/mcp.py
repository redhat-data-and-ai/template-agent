"""MCP client — re-export from aegra runtime layer.

This module moved to deep_agent.aegra.mcp as part of the runtime
consolidation. This shim preserves backward compatibility.
"""

from deep_agent.aegra.mcp import (  # noqa: F401
    get_mcp_tools,
    refresh_access_token,
)
