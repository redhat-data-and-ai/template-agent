"""MCP (Model Context Protocol) client for external tool integration.

This module manages connections to MCP servers that provide tools for agents.
It reads server configurations from agent_config/mcp.json, establishes parallel
connections with fault isolation, and retrieves all available tools.

Why this exists:
    MCP servers provide external capabilities (APIs, databases, etc.) as tools
    that agents can use. This module bridges the gap between our agent system
    and external MCP-compatible services.

Functions:
    get_mcp_tools: Connect to all MCP servers and retrieve their tools
"""

import asyncio

import httpx
from langchain_mcp_adapters.client import MultiServerMCPClient

from template_agent.src.agent.config import agent_config
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)


def _get_server_configs() -> dict[str, dict]:
    """Get pre-loaded MCP server configurations.

    Returns:
        ``{server_name: {url, transport, enabled, auth, ssl_verify, timeout}}``
    """
    return agent_config.get_mcp_servers()


def _build_server_config(entry: dict, sso_token: str | None) -> dict:
    """Build MultiServerMCPClient config from server definition.

    Args:
        entry: Server definition with url, auth, ssl_verify, transport.
        sso_token: Optional bearer token for authentication.

    Returns:
        Config dict for MultiServerMCPClient.
    """
    headers = (
        {"Authorization": f"Bearer {sso_token}"}
        if entry.get("auth", True) and sso_token
        else {}
    )

    config = {
        "url": entry["url"],
        "transport": entry.get("transport", "streamable_http"),
        "headers": headers,
    }

    if not entry.get("ssl_verify", True):
        config["httpx_client_factory"] = lambda **kw: httpx.AsyncClient(
            verify=False, **kw
        )  # nosec B501

    return config


async def _connect_single_server(name: str, config: dict, timeout: int) -> list:
    """Connect to one MCP server and return its tools.

    Failures are logged and return empty list for fault isolation.
    """
    try:
        async with asyncio.timeout(timeout):
            client = MultiServerMCPClient({name: config})
            tools = await client.get_tools()
        logger.info(f"[{name}] loaded {len(tools)} tool(s)")
        return tools
    except TimeoutError:
        logger.error(f"[{name}] timeout after {timeout}s ({config.get('url')})")
    except Exception:
        logger.error(f"[{name}] connection failed ({config.get('url')})", exc_info=True)
    return []


async def get_mcp_tools(sso_token: str | None = None) -> list:
    """Connect to MCP server(s) and retrieve available tools.

    Loads server definitions from ``agent_config/mcp.json``, connects to
    each enabled server in parallel, and returns a deduplicated flat list.

    Connection failures are logged but do not raise exceptions, ensuring
    the application continues with an empty tool list.

    Args:
        sso_token: Optional SSO token for authentication.

    Returns:
        List of available MCP tools (empty list if all connections fail).
    """
    servers = _get_server_configs()
    enabled = {k: v for k, v in servers.items() if v.get("enabled", False)}

    if not enabled:
        logger.warning("No MCP servers enabled")
        return []

    logger.info(f"Connecting to {len(enabled)} MCP server(s): {', '.join(enabled)}")

    results = await asyncio.gather(
        *[
            _connect_single_server(
                name=name,
                config=_build_server_config(entry, sso_token),
                timeout=entry.get("timeout", 30),
            )
            for name, entry in enabled.items()
        ]
    )

    # Deduplicate tools by name (first occurrence wins)
    seen = set()
    tools = []
    for tool_list in results:
        for tool in tool_list:
            if tool.name not in seen:
                seen.add(tool.name)
                tools.append(tool)
            else:
                logger.warning(f"Duplicate tool '{tool.name}' skipped")

    if not tools:
        logger.warning("All MCP servers failed to load tools")
        return []

    logger.info(f"Loaded {len(tools)} MCP tool(s): {', '.join(seen)}")
    return tools
