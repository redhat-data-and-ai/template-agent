"""Agent implementation for the template agent system.

This module provides the core agent functionality for the template agent,
including initialization, configuration, and agent creation utilities.
"""

from contextlib import asynccontextmanager
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.prebuilt import create_react_agent

from template_agent.src.core.prompt import get_system_prompt
from template_agent.src.core.storage import get_global_checkpoint
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)


@asynccontextmanager
async def get_template_agent(
    sso_token: Optional[str] = None, enable_checkpointing: bool = True
):
    """Get a fully initialized template agent.

    This function creates and configures a template agent with the necessary
    tools, model, and database connections. It uses an async context manager
    to ensure proper resource cleanup.

    Args:
        sso_token: Optional access token for authentication. If provided,
            it will be used for authorization headers in MCP client requests.
        enable_checkpointing: Whether to enable checkpointing/persistence.
            Set to False for streaming-only operations that shouldn't save to DB.

    Yields:
        The initialized template agent instance.

    Raises:
        Exception: If there are issues with database connections or agent setup.
    """
    # Initialize MCP client and get tools (optional for local development)
    tools = []
    try:
        client = MultiServerMCPClient(
            {
                "template-mcp-server": {
                    "url": "http://localhost:5001/mcp/",
                    "transport": "streamable_http",
                    "headers": {"Authorization": f"Bearer {sso_token}"}
                    if sso_token
                    else {},
                },
            }
        )
        tools = await client.get_tools()
        logger.info(
            f"Successfully connected to MCP server and loaded {len(tools)} tools"
        )
    except Exception as e:
        if settings.USE_INMEMORY_SAVER:
            logger.warning(f"Could not connect to MCP server: {e}")
            logger.info("Running in local development mode without MCP tools")
            tools = []  # No tools for local development
        else:
            logger.error(f"Failed to connect to MCP server in production mode: {e}")
            raise

    # Initialize the language model
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    if not enable_checkpointing:
        # Create agent without checkpointing for streaming-only operations
        logger.info(
            "Creating agent without checkpointing for streaming-only operations"
        )
        agent_redhat = create_react_agent(
            model=model,
            prompt=get_system_prompt(),
            tools=tools,
            # No checkpointer or store - streaming only, no persistence
        )
        logger.info("Template agent initialized successfully without checkpointing")
        yield agent_redhat
    elif settings.USE_INMEMORY_SAVER:
        # Use single global checkpoint for local development
        logger.info("Using single global checkpoint for local development")
        # Use single checkpoint instance for both checkpointer and store
        checkpoint = get_global_checkpoint()
        agent_redhat = create_react_agent(
            model=model,
            prompt=get_system_prompt(),
            tools=tools,
            checkpointer=checkpoint,
            store=checkpoint,
        )
        logger.info(
            "Template agent initialized successfully with single global checkpoint"
        )
        yield agent_redhat
    else:
        # Use PostgreSQL storage for production
        logger.info("Using PostgreSQL checkpoint for production")
        async with AsyncPostgresSaver.from_conn_string(
            settings.database_uri
        ) as checkpoint:
            # Setup database connection once
            if hasattr(checkpoint, "setup"):
                await checkpoint.setup()

            # Create the agent with single checkpoint instance for both checkpointer and store
            agent_redhat = create_react_agent(
                model=model,
                prompt=get_system_prompt(),
                tools=tools,
                checkpointer=checkpoint,
                store=checkpoint,
            )

            logger.info(
                "Template agent initialized successfully with PostgreSQL checkpoint"
            )
            yield agent_redhat
