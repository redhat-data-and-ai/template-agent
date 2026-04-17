"""MCP client initialization and connection utilities.

This module handles connecting to one or more MCP servers defined in
``agent_config/mcp.json`` and retrieving their tools, with
parallel connections, per-server auth/SSL/timeout, fault isolation,
and tool-name deduplication.

If the JSON config file is absent the module falls back to the single-
server settings exposed via environment variables (backward compatible).
"""

import asyncio
import json
from pathlib import Path

import httpx
from langchain_mcp_adapters.client import MultiServerMCPClient

from template_agent.src.core.exceptions import AppException, ErrorCodes
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)

_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "agent_config" / "mcp.json"
)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_server_configs() -> dict[str, dict]:
    """Load MCP server definitions from JSON or fall back to env vars.

    Returns:
        ``{server_name: {url, transport, enabled, auth, ssl_verify, timeout}}``

    Raises:
        AppException: On JSON parse errors or missing required fields.
    """
    if _CONFIG_PATH.is_file():
        return _load_from_json()
    return _fallback_from_settings()


def _load_from_json() -> dict[str, dict]:
    """Parse and validate ``mcp.json``."""
    try:
        data = json.loads(_CONFIG_PATH.read_bytes())
    except json.JSONDecodeError as exc:
        raise AppException(
            f"Invalid JSON in {_CONFIG_PATH}: {exc}",
            ErrorCodes.CONFIGURATION_VALIDATION_ERROR,
        ) from exc

    all_servers = data.get("mcpServers") or {}
    if not isinstance(all_servers, dict):
        raise AppException(
            "mcpServers must be a JSON object",
            ErrorCodes.CONFIGURATION_VALIDATION_ERROR,
        )

    for name, entry in all_servers.items():
        if "url" not in entry:
            raise AppException(
                f"MCP server '{name}' missing required field 'url'",
                ErrorCodes.CONFIGURATION_VALIDATION_ERROR,
            )

    logger.info(
        f"Loaded {len(all_servers)} MCP server definition(s) from {_CONFIG_PATH.name}"
    )
    return all_servers


def _fallback_from_settings() -> dict[str, dict]:
    """Build a single-server config dict from Settings env vars."""
    logger.info("No mcp.json found; falling back to env-var config")
    return {
        settings.MCP_SERVER_NAME: {
            "url": settings.MCP_SERVER_URL,
            "transport": settings.MCP_TRANSPORT_PROTOCOL,
            "enabled": True,
            "auth": True,
            "ssl_verify": settings.MCP_SSL_VERIFY,
            "timeout": settings.MCP_CONNECTION_TIMEOUT,
        }
    }


# ---------------------------------------------------------------------------
# Per-server config builder
# ---------------------------------------------------------------------------


def _build_server_config(entry: dict, sso_token: str | None) -> dict:
    """Translate a JSON entry into a ``MultiServerMCPClient``-compatible dict.

    Args:
        entry: Single server definition from the config.
        sso_token: Optional bearer token for authentication.

    Returns:
        Config dict ready for ``MultiServerMCPClient``.
    """
    wants_auth = entry.get("auth", True)
    headers: dict[str, str] = {}
    if wants_auth and sso_token:
        headers["Authorization"] = f"Bearer {sso_token}"

    config: dict = {
        "url": entry["url"],
        "transport": entry.get("transport", "streamable_http"),
        "headers": headers,
    }

    if not entry.get("ssl_verify", True):
        config["httpx_client_factory"] = (
            lambda **kwargs: httpx.AsyncClient(verify=False, **kwargs)  # nosec B501 # NOSONAR
        )

    return config


# ---------------------------------------------------------------------------
# Single-server connection (fault-isolated)
# ---------------------------------------------------------------------------


async def _connect_single_server(
    name: str,
    config: dict,
    timeout: int,
) -> list:
    """Connect to one MCP server and return its tools.

    On any failure the error is logged and an empty list is returned so
    that other servers are not affected.
    """
    try:
        client = MultiServerMCPClient({name: config})
        async with asyncio.timeout(timeout):
            tools = await client.get_tools()
        logger.info(f"[{name}] loaded {len(tools)} tool(s)")
        return tools
    except TimeoutError:
        logger.error(
            f"[{name}] timeout after {timeout}s connecting to {config.get('url')}"
        )
        return []
    except Exception:
        logger.error(
            f"[{name}] failed to connect to {config.get('url')}",
            exc_info=True,
        )
        return []


# ---------------------------------------------------------------------------
# Public API (signature unchanged for backward compatibility)
# ---------------------------------------------------------------------------


async def get_mcp_tools(sso_token: str | None = None) -> list:
    """Connect to MCP server(s) and retrieve available tools.

    Loads server definitions from ``agent_config/mcp.json`` (or
    falls back to env-var settings), connects to each enabled server in
    parallel, and returns a deduplicated flat list of tools.

    Args:
        sso_token: Optional SSO token for authentication.

    Returns:
        List of available MCP tools.

    Raises:
        AppException: If connection fails in production mode.
    """
    server_defs = _load_server_configs()

    enabled = {
        name: entry
        for name, entry in server_defs.items()
        if entry.get("enabled", False)
    }

    if not enabled:
        logger.warning("No MCP servers are enabled in configuration")
        return _handle_no_tools()

    logger.info(f"Connecting to {len(enabled)} MCP server(s): {list(enabled.keys())}")

    tasks = [
        _connect_single_server(
            name=name,
            config=_build_server_config(entry, sso_token),
            timeout=entry.get("timeout", settings.MCP_CONNECTION_TIMEOUT),
        )
        for name, entry in enabled.items()
    ]

    results = await asyncio.gather(*tasks)

    all_tools: list = []
    seen_names: set[str] = set()
    for tool_list in results:
        for tool in tool_list:
            if tool.name in seen_names:
                logger.warning(f"Duplicate tool '{tool.name}' skipped")
                continue
            seen_names.add(tool.name)
            all_tools.append(tool)

    if not all_tools:
        return _handle_no_tools()

    logger.info(f"Total MCP tools loaded: {len(all_tools)} ({', '.join(seen_names)})")
    return all_tools


def _handle_no_tools() -> list:
    """Apply the dev/prod error policy when no tools were retrieved."""
    if settings.USE_INMEMORY_SAVER:
        logger.warning("Running in local development mode without MCP tools")
        return []

    raise AppException(
        "No MCP tools available — all servers failed or none are enabled",
        ErrorCodes.PRODUCTION_MCP_CONNECTION_ERROR,
    )
