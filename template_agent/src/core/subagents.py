"""Subagent initialization and configuration module.

This module handles loading and configuring subagents from markdown files
with YAML frontmatter, including tool and skill resolution.
"""

from pathlib import Path
from typing import Any

import yaml
from deepagents import SubAgent

from template_agent.src.core.llm import create_model
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)


def _resolve_tools(
    agent_name: str,
    tool_names: list[str],
    tool_by_name: dict[str, Any],
) -> list[Any]:
    """Resolve tool names to actual tool objects.

    Args:
        agent_name: Name of the agent (for logging)
        tool_names: List of tool names from frontmatter
        tool_by_name: Dictionary mapping tool names to tool objects

    Returns:
        List of resolved tool objects
    """
    resolved = [tool_by_name[n] for n in tool_names if n in tool_by_name]
    missing = [n for n in tool_names if n not in tool_by_name]

    if missing:
        logger.warning(f"Subagent '{agent_name}' references unknown tools: {missing}")

    return resolved


def _resolve_skills(
    agent_name: str,
    skill_names: list[str],
    skills_base: Path,
) -> list[str]:
    """Resolve skill names to skill directory paths.

    Args:
        agent_name: Name of the agent (for logging)
        skill_names: List of skill names from frontmatter
        skills_base: Base directory containing skills

    Returns:
        List of skill directory paths as strings
    """
    skill_paths: list[str] = []

    for skill_name in skill_names:
        skill_dir = skills_base / skill_name
        if skill_dir.exists():
            skill_paths.append(str(skill_dir))
            logger.info(f"Subagent '{agent_name}' skill loaded: {skill_dir}")
        else:
            logger.warning(f"Subagent '{agent_name}' skill not found: {skill_dir}")

    return skill_paths


def parse_agent_frontmatter(path: Path) -> dict[str, Any]:
    r"""Parse a markdown agent file with YAML frontmatter.

    Expects the format: ``--- \\n <yaml> \\n --- \\n <markdown body>``.
    The markdown body is returned under the ``"body"`` key as the
    subagent's system prompt.

    Args:
        path: Path to the ``.md`` agent definition file.

    Returns:
        A dict of frontmatter fields plus ``body`` (the markdown body).
    """
    content = path.read_text()
    if not content.startswith("---"):
        return {"body": content.strip()}

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {"body": content.strip()}

    frontmatter: dict[str, Any] = yaml.safe_load(parts[1]) or {}
    frontmatter["body"] = parts[2].strip()
    return frontmatter


def load_subagents(
    agents_dir: Path,
    tools: list,
    skills_base: Path,
) -> list[SubAgent] | None:
    """Load and configure subagents from markdown files.

    Args:
        agents_dir: Directory containing agent definition files (*.md)
        tools: List of available MCP tools
        skills_base: Base directory for skills

    Returns:
        List of configured SubAgent instances, or None if agents_dir doesn't exist
    """
    if not agents_dir.is_dir():
        logger.warning(f"Agents directory not found at {agents_dir}")
        return None

    logger.info(f"Loading subagents from {agents_dir}")

    # Create tool lookup map
    tool_by_name = {t.name: t for t in tools}

    subagents_config: list[SubAgent] = []

    for agent_file in sorted(agents_dir.glob("*.md")):
        config = parse_agent_frontmatter(agent_file)
        name = config.get("name", agent_file.stem)

        # Model is required for subagents
        model_name = config.get("model")
        if not model_name:
            raise ValueError(
                f"Subagent '{name}' in {agent_file} is missing required 'model' field in frontmatter"
            )

        logger.info(f"Subagent '{name}' using model: {model_name}")

        # Build SubAgent with all required fields
        sa: SubAgent = SubAgent(
            name=name,
            model=create_model(model_name=model_name),
            description=config.get("description", ""),
            system_prompt=config.get("body", ""),
        )

        # Resolve tool names to loaded MCP tools
        tool_names = config.get("tools", [])
        if tool_names:
            sa["tools"] = _resolve_tools(name, tool_names, tool_by_name)

        # Resolve skill names to paths under skills/
        skill_names = config.get("skills", [])
        if skill_names:
            skill_paths = _resolve_skills(name, skill_names, skills_base)
            if skill_paths:
                sa["skills"] = skill_paths

        subagents_config.append(sa)

    logger.info(f"Loaded {len(subagents_config)} subagents")
    return subagents_config
