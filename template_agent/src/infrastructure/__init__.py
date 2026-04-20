"""Infrastructure layer for external system integrations.

This package contains modules that interface with external systems and services:
- MCP servers for tools
- PostgreSQL for state persistence
- Backend execution environments
- Subagent configuration loading

These modules form the boundary between our application and external dependencies.
"""

from .backend import get_backend, initialize_backend
from .checkpointer import get_checkpointer, initialize_checkpointer
from .mcp import get_mcp_tools
from .subagents import load_subagents

__all__ = [
    "get_mcp_tools",
    "get_backend",
    "initialize_backend",
    "get_checkpointer",
    "initialize_checkpointer",
    "load_subagents",
]
