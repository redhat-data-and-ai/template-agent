"""Agent implementation for the template agent system.

This module provides the core agent functionality using the deepagents library,
including initialization, configuration, and agent creation with MCP tools,
skills, subagents, and memory.
"""

from contextlib import asynccontextmanager

from deepagents import create_deep_agent

from template_agent.src.core.agent_config import agent_config
from template_agent.src.core.backend import get_backend
from template_agent.src.core.checkpointer import get_checkpointer
from template_agent.src.core.llm import create_model
from template_agent.src.core.mcp import get_mcp_tools
from template_agent.src.core.subagents import load_subagents
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)


@asynccontextmanager
async def get_template_agent(sso_token: str | None = None):
    """Get a fully initialized deep agent with MCP tools, skills, subagents, and memory.

    This function creates and configures a deep agent using the deepagents library
    with the necessary tools from MCP, skills, subagents, and memory. It uses an
    async context manager to ensure proper resource cleanup.

    Args:
        sso_token: Optional access token for authentication. If provided,
            it will be used for authorization headers in MCP client requests.

    Yields:
        The initialized deep agent instance.

    Raises:
        Exception: If there are issues with database connections or agent setup.
    """
    # Get pre-loaded orchestrator configuration
    orchestrator_cfg = agent_config.get_orchestrator_config()

    # Extract configuration from frontmatter
    agent_name = orchestrator_cfg.get("name", "orchestrator")
    model_name = orchestrator_cfg.get("model", "gemini-3.1-pro-preview")
    system_prompt = orchestrator_cfg.get("body", "")
    skill_paths = orchestrator_cfg.get("skill_paths", [])
    tool_names = orchestrator_cfg.get("tools", [])

    logger.info(
        f"Initializing orchestrator agent '{agent_name}' with model: {model_name}"
    )

    # Initialize the language model
    model = create_model(model_name=model_name)

    # Initialize MCP client and get tools
    mcp_tools = await get_mcp_tools(sso_token=sso_token)

    # Resolve tools from tool names in frontmatter
    tools = agent_config.resolve_tools(tool_names, mcp_tools, agent_name=agent_name)

    # Build subagents from pre-loaded configs
    subagents = load_subagents(tools=mcp_tools)

    # Load and configure backend
    backend = get_backend()

    async with get_checkpointer() as checkpointer:
        agent = create_deep_agent(
            name=agent_name,
            model=model,
            system_prompt=system_prompt,
            skills=skill_paths,
            tools=tools,
            subagents=subagents,
            backend=backend,
            checkpointer=checkpointer,
            store=None,  # TODO: Add store support
        )
        logger.info(f"Orchestrator agent '{agent_name}' initialized successfully")
        yield agent
