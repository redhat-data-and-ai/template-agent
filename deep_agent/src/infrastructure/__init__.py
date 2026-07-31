"""Infrastructure layer for external system integrations.

This package contains modules that interface with external systems and services:
- MCP servers for tools
- Backend execution environments
- Subagent configuration loading

These modules form the boundary between our application and external dependencies.
"""

from .backend import get_backend, get_configured_backend

__all__ = [
    "get_mcp_tools",
    "get_backend",
    "get_configured_backend",
    "load_subagents",
]


def __getattr__(name: str) -> object:
    if name == "get_mcp_tools":
        from .mcp import get_mcp_tools

        return get_mcp_tools
    if name == "load_subagents":
        from .subagents import load_subagents

        return load_subagents
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
