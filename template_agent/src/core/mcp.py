"""MCP client initialization and connection utilities.

This module handles connecting to MCP servers and retrieving tools,
with proper timeout and error handling.
"""

import asyncio

import httpx
from langchain_mcp_adapters.client import MultiServerMCPClient

from template_agent.src.core.exceptions.exceptions import AppException, AppExceptionCode
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)


def _build_server_config(sso_token: str | None) -> dict:
    """Build MCP server configuration.

    Args:
        sso_token: Optional SSO token for authentication

    Returns:
        Server configuration dictionary
    """
    server_config: dict = {
        "url": settings.MCP_SERVER_URL,
        "transport": settings.MCP_TRANSPORT_PROTOCOL,
        "headers": {"Authorization": f"Bearer {sso_token}"} if sso_token else {},
    }

    if not settings.MCP_SSL_VERIFY:
        logger.warning("SSL certificate verification disabled for MCP connection")
        server_config["httpx_client_factory"] = (
            lambda **kwargs: httpx.AsyncClient(verify=False, **kwargs)  # nosec B501
        )

    return server_config


def _handle_connection_error(error: Exception) -> list:
    """Handle MCP connection errors based on environment.

    Args:
        error: The exception that occurred

    Returns:
        Empty list in development mode

    Raises:
        AppException: In production mode
    """
    if isinstance(error, asyncio.TimeoutError):
        error_msg = (
            f"Timeout connecting to MCP server at {settings.MCP_SERVER_URL} "
            f"after {settings.MCP_CONNECTION_TIMEOUT}s. "
            f"Server may be down or unreachable."
        )
        logger.error(error_msg)
    else:
        error_msg = (
            f"Failed to connect to required MCP server at {settings.MCP_SERVER_URL}. "
            f"Error: {type(error).__name__}: {str(error)}"
        )
        logger.error(
            error_msg,
            error_type=type(error).__name__,
            exc_info=True,
        )

    if settings.USE_INMEMORY_SAVER:
        logger.warning("Running in local development mode without MCP tools")
        return []

    logger.critical(error_msg)
    raise AppException(
        error_msg,
        AppExceptionCode.PRODUCTION_MCP_CONNECTION_ERROR,
    )


async def get_mcp_tools(sso_token: str | None = None) -> list:
    """Connect to MCP server and retrieve available tools.

    Args:
        sso_token: Optional SSO token for authentication

    Returns:
        List of available MCP tools

    Raises:
        AppException: If connection fails in production mode
    """
    # Log MCP connection details for debugging
    logger.info(f"Attempting to connect to MCP server at {settings.MCP_SERVER_URL}")
    logger.info(f"MCP server name: {settings.MCP_SERVER_NAME}")
    logger.info(f"MCP transport protocol: {settings.MCP_TRANSPORT_PROTOCOL}")
    logger.info(f"MCP connection timeout: {settings.MCP_CONNECTION_TIMEOUT}s")
    logger.info(f"SSO authentication: {'Yes' if sso_token else 'No'}")

    try:
        server_config = _build_server_config(sso_token)
        client = MultiServerMCPClient({settings.MCP_SERVER_NAME: server_config})

        tools = await asyncio.wait_for(
            client.get_tools(), timeout=settings.MCP_CONNECTION_TIMEOUT
        )

        logger.info(
            f"Successfully connected to MCP server and loaded {len(tools)} tools"
        )
        logger.info(f"Available tools: {[tool.name for tool in tools]}")
        return tools

    except (asyncio.TimeoutError, Exception) as e:
        return _handle_connection_error(e)
