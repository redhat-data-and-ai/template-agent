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

from template_agent.src.core.exceptions.exceptions import AppException, AppExceptionCode
from template_agent.src.core.prompt import get_system_prompt
from template_agent.src.core.storage import get_global_checkpoint
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)


async def initialize_database() -> None:
    """Initialize PostgreSQL database schema on application startup.

    This function ensures the checkpoints table and related schema are created
    before any requests are processed. Only runs when using PostgreSQL storage
    (USE_INMEMORY_SAVER=False).

    Raises:
        AppException: If database connection or schema creation fails.
    """
    if settings.USE_INMEMORY_SAVER:
        logger.info("Using in-memory storage - skipping database initialization")
        return

    try:
        logger.info("Initializing PostgreSQL database schema")
        async with AsyncPostgresSaver.from_conn_string(
            settings.database_uri
        ) as checkpoint:
            # Setup database schema - creates checkpoints table and indexes
            if hasattr(checkpoint, "setup"):
                await checkpoint.setup()
                logger.info("Database schema initialized successfully")
            else:
                logger.warning(
                    "AsyncPostgresSaver does not have setup method - schema may need manual creation"
                )
    except Exception as e:
        logger.error(f"Failed to initialize database schema: {e}", exc_info=True)
        raise AppException(
            f"Database initialization failed: {str(e)}",
            AppExceptionCode.CONFIGURATION_INITIALIZATION_ERROR,
        )


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
    # Initialize MCP client and get tools
    tools = []

    # Log MCP connection details for debugging
    logger.info(f"Attempting to connect to MCP server at {settings.MCP_SERVER_URL}")
    logger.info(f"MCP server name: {settings.MCP_SERVER_NAME}")
    logger.info(f"MCP transport protocol: {settings.MCP_TRANSPORT_PROTOCOL}")
    logger.info(f"MCP connection timeout: {settings.MCP_CONNECTION_TIMEOUT}s")
    logger.info(f"SSO authentication: {'Yes' if sso_token else 'No'}")

    try:
        import asyncio

        # Add timeout wrapper for MCP connection
        async def connect_with_timeout():
            # Configure MCP client with SSL verification setting
            server_config = {
                "url": settings.MCP_SERVER_URL,
                "transport": settings.MCP_TRANSPORT_PROTOCOL,
                "headers": {"Authorization": f"Bearer {sso_token}"}
                if sso_token
                else {},
            }

            # Add SSL verification setting (verify=False disables cert verification)
            if not settings.MCP_SSL_VERIFY:
                server_config["verify"] = False
                logger.warning(
                    "SSL certificate verification disabled for MCP connection"
                )

            client = MultiServerMCPClient({settings.MCP_SERVER_NAME: server_config})
            return await client.get_tools()

        tools = await asyncio.wait_for(
            connect_with_timeout(), timeout=settings.MCP_CONNECTION_TIMEOUT
        )
        logger.info(
            f"Successfully connected to MCP server and loaded {len(tools)} tools"
        )
    except asyncio.TimeoutError:
        # Handle timeout specifically
        error_msg = (
            f"Timeout connecting to MCP server at {settings.MCP_SERVER_URL} "
            f"after {settings.MCP_CONNECTION_TIMEOUT}s. "
            f"Server may be down or unreachable."
        )
        logger.error(error_msg)

        if settings.USE_INMEMORY_SAVER:
            logger.warning("Running in local development mode without MCP tools")
            tools = []
        else:
            logger.critical(error_msg)
            raise AppException(
                error_msg,
                AppExceptionCode.PRODUCTION_MCP_CONNECTION_ERROR,
            )
    except Exception as e:
        # Log detailed error information for other exceptions
        logger.error(
            f"Failed to connect to MCP server at {settings.MCP_SERVER_URL}",
            exc_info=True,
        )
        logger.error(f"MCP connection error type: {type(e).__name__}")
        logger.error(f"MCP connection error details: {str(e)}")

        if settings.USE_INMEMORY_SAVER:
            logger.warning("Running in local development mode without MCP tools")
            tools = []  # No tools for local development
        else:
            # In production, MCP is required
            error_msg = (
                f"Failed to connect to required MCP server at {settings.MCP_SERVER_URL}. "
                f"Error: {type(e).__name__}: {str(e)}"
            )
            logger.critical(error_msg)
            raise AppException(
                error_msg,
                AppExceptionCode.PRODUCTION_MCP_CONNECTION_ERROR,
            )

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
