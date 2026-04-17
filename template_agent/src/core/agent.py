"""Agent implementation for the template agent system.

This module provides the core agent functionality for the template agent,
including initialization, configuration, and agent creation utilities.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

import httpx
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
    tools: list = []

    mcp_defs = settings.mcp_servers
    logger.info(
        f"MCP connection timeout: {settings.MCP_CONNECTION_TIMEOUT}s | "
        f"SSO authentication: {'Yes' if sso_token else 'No'} | "
        f"Enabled servers: {list(mcp_defs.keys()) or '(none)'}"
    )

    def _build_server_config(
        url: str,
        transport: str,
        ssl_verify: bool,
        token: str | None,
    ) -> dict:
        config: dict = {
            "url": url,
            "transport": transport,
            "headers": {"Authorization": f"Bearer {token}"} if token else {},
        }
        if not ssl_verify:
            logger.warning(f"SSL verification disabled for MCP server at {url}")
            config["httpx_client_factory"] = (
                lambda **kwargs: httpx.AsyncClient(verify=False, **kwargs)  # nosec B501
            )
        return config

    server_configs: dict[str, dict] = {}
    for name, defn in mcp_defs.items():
        server_configs[name] = _build_server_config(
            url=defn["url"],
            transport=defn.get("transport", "streamable_http"),
            ssl_verify=defn.get("ssl_verify", True),
            token=sso_token,
        )
        logger.info(
            f"MCP server '{name}' configured: {defn['url']} "
            f"(transport={defn.get('transport', 'streamable_http')}, "
            f"ssl_verify={defn.get('ssl_verify', True)})"
        )

    if server_configs:
        async def _try_single_server(
            srv_name: str, srv_cfg: dict
        ) -> list:
            """Connect to one MCP server; return tools or empty on failure."""
            try:
                client = MultiServerMCPClient({srv_name: srv_cfg})
                srv_tools = await asyncio.wait_for(
                    client.get_tools(),
                    timeout=settings.MCP_CONNECTION_TIMEOUT,
                )
                logger.info(
                    f"MCP server '{srv_name}': loaded {len(srv_tools)} tools"
                )
                return srv_tools
            except asyncio.TimeoutError:
                logger.error(
                    f"MCP server '{srv_name}': timeout after "
                    f"{settings.MCP_CONNECTION_TIMEOUT}s"
                )
            except Exception as exc:
                logger.error(
                    f"MCP server '{srv_name}': {type(exc).__name__}: {exc}",
                    exc_info=True,
                )
            return []

        results = await asyncio.gather(
            *[
                _try_single_server(name, cfg)
                for name, cfg in server_configs.items()
            ]
        )
        for server_tools in results:
            tools.extend(server_tools)

        logger.info(f"Total MCP tools loaded across all servers: {len(tools)}")
    else:
        logger.warning("No MCP servers enabled — agent will run without MCP tools")

    if not tools and not settings.USE_INMEMORY_SAVER:
        error_msg = (
            "No MCP tools loaded and in-memory saver is disabled. "
            "At least one MCP server must be reachable in production."
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
