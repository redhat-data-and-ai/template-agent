"""Agent implementation for the template agent system.

This module provides the core agent functionality using the deepagents library,
including initialization, configuration, and agent creation with MCP tools,
skills, subagents, and memory.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import yaml
from deepagents import SubAgent, create_deep_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from template_agent.src.core.backend import get_backend
from template_agent.src.core.exceptions.exceptions import AppException, AppExceptionCode
from template_agent.src.core.prompt import get_system_prompt
from template_agent.src.core.storage import get_global_checkpoint, get_global_store
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)

CONFIG_DIR = Path(__file__).parent.parent.parent / "agent_config"


def _parse_agent_frontmatter(path: Path) -> dict[str, Any]:
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
    tools: list = []

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
            server_config: dict = {
                "url": settings.MCP_SERVER_URL,
                "transport": settings.MCP_TRANSPORT_PROTOCOL,
                "headers": {"Authorization": f"Bearer {sso_token}"}
                if sso_token
                else {},
            }

            if not settings.MCP_SSL_VERIFY:
                logger.warning(
                    "SSL certificate verification disabled for MCP connection"
                )
                server_config["httpx_client_factory"] = (
                    lambda **kwargs: httpx.AsyncClient(verify=False, **kwargs)  # nosec B501
                )

            client = MultiServerMCPClient({settings.MCP_SERVER_NAME: server_config})
            return await client.get_tools()

        tools = await asyncio.wait_for(
            connect_with_timeout(), timeout=settings.MCP_CONNECTION_TIMEOUT
        )
        logger.info(
            f"Successfully connected to MCP server and loaded {len(tools)} tools"
        )
        logger.info(f"Available tools: {[tool.name for tool in tools]}")
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

    # Initialize the language model with service account credentials
    import google.auth

    credentials, project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    # Disable HTTP keepalive to prevent stale TLS connections after tool-call
    # pauses.  Long-lived pooled connections to Vertex/Gemini can trigger
    # ssl.SSLError("passed invalid argument") when reused; fresh connections
    # on every request eliminate the issue.
    _no_keepalive = httpx.Limits(max_keepalive_connections=0)

    model = ChatGoogleGenerativeAI(
        model="gemini-3.1-pro-preview",
        temperature=0,
        credentials=credentials,
        project=project,
        client_args={"limits": _no_keepalive},
    )

    # Load subagent definitions from agents/ directory (markdown + frontmatter)
    agents_dir = CONFIG_DIR / "agents"
    logger.info(f"Loading subagents from {agents_dir}")

    tool_by_name = {t.name: t for t in tools}
    skills_base = CONFIG_DIR / "skills"

    # Main agent skills — flat directory under skills/
    main_skills_dir = skills_base / "client-intake"
    main_skills_path = [str(main_skills_dir)] if main_skills_dir.exists() else []

    subagents_config: list[SubAgent] | None = None
    if agents_dir.is_dir():
        subagents_config = []
        for agent_file in sorted(agents_dir.glob("*.md")):
            config = _parse_agent_frontmatter(agent_file)
            name = config.get("name", agent_file.stem)

            # Add model if specified
            model_name = config.get("model")
            model = None
            if model_name:
                logger.info(f"Subagent '{name}' using model: {model_name}")
                model = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=0,
                    credentials=credentials,
                    project=project,
                    client_args={"limits": _no_keepalive},
                )

            sa: SubAgent = SubAgent(
                name=name,
                model=model,
                description=config.get("description", ""),
                system_prompt=config.get("body", ""),
            )

            # Resolve tool names to loaded MCP tools
            yaml_tool_names = config.get("tools", [])
            if yaml_tool_names:
                resolved = [
                    tool_by_name[n] for n in yaml_tool_names if n in tool_by_name
                ]
                missing = [n for n in yaml_tool_names if n not in tool_by_name]
                if missing:
                    logger.warning(
                        f"Subagent '{name}' references unknown tools: {missing}"
                    )
                sa["tools"] = resolved

            # Resolve skill names to paths under skills/
            skill_names = config.get("skills", [])
            if skill_names:
                skill_paths: list[str] = []
                for skill_name in skill_names:
                    skill_dir = skills_base / skill_name
                    if skill_dir.exists():
                        skill_paths.append(str(skill_dir))
                        logger.info(f"Subagent '{name}' skill loaded: {skill_dir}")
                    else:
                        logger.warning(
                            f"Subagent '{name}' skill not found: {skill_dir}"
                        )
                if skill_paths:
                    sa["skills"] = skill_paths

            subagents_config.append(sa)
        logger.info(f"Loaded {len(subagents_config)} subagents")
    else:
        logger.warning(f"Agents directory not found at {agents_dir}")

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
