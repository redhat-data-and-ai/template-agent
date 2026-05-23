"""Skill and tool resolution utilities.

This module resolves skill names to directory paths and tool names to tool objects.
It handles validation, logging of missing dependencies, and returns only the
successfully resolved items.

Why this exists:
    Agent configs reference skills and tools by name (strings). This module
    looks up those names in the available skills directory and MCP tools list,
    returning the actual paths/objects needed for agent initialization.

Functions:
    resolve_skill_paths: Convert skill names to skill directory paths
    resolve_tools: Convert tool names to tool objects
"""

from pathlib import Path
from typing import Any

from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)


def resolve_skill_paths(
    skill_names: list[str],
    available_skills: dict[str, Path],
    agent_name: str = "agent",
) -> list[str]:
    """Resolve skill names to skill directory paths using cached skill index.

    Args:
        skill_names: List of skill names from frontmatter.
        available_skills: Dict mapping skill name to skill directory path.
        agent_name: Name of the agent (for logging).

    Returns:
        List of skill directory paths as strings.
    """
    skill_paths: list[str] = []
    missing: list[str] = []

    for skill_name in skill_names:
        if skill_name in available_skills:
            skill_path = available_skills[skill_name]
            skill_paths.append(str(skill_path))
            logger.debug(f"Agent '{agent_name}' resolved skill: {skill_name}")
        else:
            missing.append(skill_name)

    if missing:
        logger.warning(f"Agent '{agent_name}' references unknown skills: {missing}")

    return skill_paths


def resolve_tools(
    tool_names: list[str],
    available_tools: list[Any],
    agent_name: str = "agent",
) -> list[Any]:
    """Resolve tool names to actual tool objects.

    Args:
        tool_names: List of tool names from frontmatter.
        available_tools: List of available tool objects.
        agent_name: Name of the agent (for logging).

    Returns:
        List of resolved tool objects.
    """
    tool_by_name = {t.name: t for t in available_tools}
    resolved = [tool_by_name[n] for n in tool_names if n in tool_by_name]
    missing = [n for n in tool_names if n not in tool_by_name]

    if missing:
        logger.warning(f"Agent '{agent_name}' references unknown tools: {missing}")

    return resolved
