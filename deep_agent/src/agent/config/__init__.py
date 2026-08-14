"""Agent configuration management.

This package handles loading and processing agent configurations from the
config/ directory at the repository root. It provides a singleton AgentConfig
class that loads orchestrator, subagent, skill, and MCP configurations.

Modules:
    loader: Main AgentConfig singleton class
    parser: Frontmatter parsing and runtime value injection
    resolver: Skill and tool name resolution

Main exports:
    AgentConfig: Singleton configuration manager
    agent_config: Pre-initialized singleton instance
"""

from .loader import AgentConfig, agent_config

__all__ = ["AgentConfig", "agent_config"]
