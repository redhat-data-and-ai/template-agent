"""Agent implementation for the template agent system.

This module provides the core agent functionality using the deepagents library,
including initialization, configuration, and agent creation with MCP tools,
skills, subagents, and memory.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from deepagents import create_deep_agent
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from template_agent.src.core.backend import get_backend
from template_agent.src.core.llm import create_model
from template_agent.src.core.mcp import get_mcp_tools
from template_agent.src.core.prompt import get_system_prompt
from template_agent.src.core.storage import get_global_checkpoint, get_global_store
from template_agent.src.core.subagents import load_subagents
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)

CONFIG_DIR = Path(__file__).parent.parent.parent / "agent_config"


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
    # Initialize MCP client and get tools
    tools = await get_mcp_tools(sso_token=sso_token)

    # Initialize the language model (credentials handled in llm.py)
    model = create_model(model_name="gemini-3.1-pro-preview")

    # Load subagent definitions from agents/ directory (markdown + frontmatter)
    agents_dir = CONFIG_DIR / "agents"
    skills_base = CONFIG_DIR / "skills"

    # Main agent skills — flat directory under skills/
    main_skills_dir = skills_base / "client-intake"
    main_skills_path = [str(main_skills_dir)] if main_skills_dir.exists() else []

    # Load and configure subagents
    subagents_config = load_subagents(
        agents_dir=agents_dir,
        tools=tools,
        skills_base=skills_base,
    )

    # Load system prompt (identity + routing + behavior from system-prompt.md)
    system_prompt = get_system_prompt()
    logger.info("Loaded system prompt from agent_config/system-prompt.md")

    if main_skills_path:
        logger.info(f"Main agent skills: {main_skills_dir}")
    else:
        logger.warning(f"Main agent skills directory not found: {main_skills_dir}")

    backend = get_backend()

    # Resolve checkpointer and store
    checkpointer = None
    store = None
    pg_ctx = None

    if settings.USE_INMEMORY_SAVER:
        checkpointer = get_global_checkpoint()
        store = get_global_store()
        logger.info(
            f"Using in-memory checkpoint={type(checkpointer).__name__} "
            f"store={type(store).__name__}"
        )
    else:
        logger.info("Using PostgreSQL checkpoint")
        pg_ctx = AsyncPostgresSaver.from_conn_string(settings.database_uri)
        checkpointer = await pg_ctx.__aenter__()
        logger.info(f"PostgreSQL checkpointer ready: {type(checkpointer).__name__}")
        if hasattr(checkpointer, "setup"):
            await checkpointer.setup()

    logger.info(
        f"Creating deep agent with checkpointer={type(checkpointer).__name__ if checkpointer else None} "
        f"store={type(store).__name__ if store else None}"
    )

    try:
        agent = create_deep_agent(
            model=model,
            system_prompt=system_prompt,
            skills=main_skills_path,
            tools=[],
            subagents=subagents_config,
            backend=backend,
            checkpointer=checkpointer,
            store=store,
        )
        logger.info("Deep agent initialized successfully")
        yield agent
    finally:
        if pg_ctx is not None:
            await pg_ctx.__aexit__(None, None, None)
