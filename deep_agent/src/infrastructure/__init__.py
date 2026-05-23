"""Infrastructure layer for external system integrations.

This package contains modules that interface with external systems and services:
- MCP servers for tools
- Backend execution environments
- Subagent configuration loading

These modules form the boundary between our application and external dependencies.
"""

from .backend import get_backend, get_configured_backend
from .mcp import get_mcp_tools
from .subagents import load_subagents

__all__ = [
    "get_mcp_tools",
    "get_backend",
    "get_configured_backend",
    "load_subagents",
]
