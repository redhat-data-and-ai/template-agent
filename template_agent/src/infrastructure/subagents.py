"""Subagent loading from configuration files.

This module builds SubAgent instances from the markdown configuration files in
agent_config/subagents/. It reads each subagent's config, resolves their tools
and skills, creates appropriate LLM instances, and returns ready-to-use SubAgent
objects for the orchestrator.

Why this exists:
    Subagents are specialized agents that handle specific tasks (e.g., analyst,
    publisher). This module transforms their declarative configs into executable
    SubAgent instances that the orchestrator can delegate work to.

Functions:
    load_subagents: Build all subagents from agent_config/subagents/*.md
"""

from typing import Any

from deepagents import SubAgent

from template_agent.src.agent.config import agent_config
from template_agent.src.agent.llm import create_model
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)


def load_subagents(
    tools: list,
) -> list[SubAgent] | None:
    """Build subagents from pre-loaded configurations.

    Args:
        tools: List of available MCP tools

    Returns:
        List of configured SubAgent instances, or None if no subagents configured
    """
    all_subagent_configs = agent_config.get_all_subagent_configs()

    if not all_subagent_configs:
        logger.warning("No subagent configurations found")
        return None

    logger.info(f"Building {len(all_subagent_configs)} subagent(s)")

    subagents_list: list[SubAgent] = []

    for name, agent_cfg in all_subagent_configs.items():
        # Model is required for subagents
        model_name = agent_cfg.get("model")
        if not model_name:
            raise ValueError(
                f"Subagent '{name}' is missing required 'model' field in frontmatter"
            )

        logger.info(f"Subagent '{name}' using model: {model_name}")

        # Resolve tools and get pre-resolved skill paths
        tool_names = agent_cfg.get("tools", [])
        resolved_tools = (
            agent_config.resolve_tools(tool_names, tools, agent_name=name)
            if tool_names
            else []
        )

        # Skills are already resolved during config loading
        skill_paths = agent_cfg.get("skill_paths", [])

        # Build subagent params dict
        subagent_params: dict[str, Any] = {
            "name": name,
            "model": create_model(model_name=model_name),
            "description": agent_cfg.get("description", ""),
            "system_prompt": agent_cfg.get("body", ""),
        }

        # Add optional parameters only if they have values
        if resolved_tools:
            subagent_params["tools"] = resolved_tools
        if skill_paths:
            subagent_params["skills"] = skill_paths

        # Build SubAgent with all parameters at once
        sa = SubAgent(**subagent_params)

        subagents_list.append(sa)

    logger.info(f"Built {len(subagents_list)} subagent(s) successfully")
    return subagents_list
