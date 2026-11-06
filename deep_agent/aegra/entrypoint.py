"""Container entrypoint — validates config mount and starts the agent.

Starts the HTTP API server via uvicorn.

Expected directory structure at ``CONFIG_PATH`` (default ``/app/config/agent``):

  - PROMPT.md
  - mcp.json
  - runtime/agent.yaml
  - skills/ (optional)
  - subagents/ (optional)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(os.getenv("CONFIG_PATH", "/app/config/agent"))
REQUIRED_FILES = ["PROMPT.md", "mcp.json"]

_LOGGER: Any = None


def _logger() -> Any:
    """Return the structured logger, configuring logging on first use.

    Boot output happens before ``aegra_api.main`` is imported (that import is
    what normally calls ``setup_logging``), so without configuring it here this
    module's lines would reach the log pipeline as plain text while every other
    line is JSON. ``setup_logging()`` is safe to run twice — importing
    ``aegra_api.main`` in :func:`start_server` calls it again at module level.
    """
    global _LOGGER  # noqa: PLW0603
    if _LOGGER is None:
        import structlog
        from aegra_api.utils.setup_logging import setup_logging

        setup_logging()
        _LOGGER = structlog.stdlib.get_logger("deep_agent.aegra.entrypoint")
    return _LOGGER


def validate_config_mount() -> None:
    """Ensure config volume is mounted and contains required files."""
    log = _logger()

    if not CONFIG_PATH.exists():
        log.error(
            "Config path not found — expected the config volume to be mounted; "
            "check the deployment spec",
            config_path=str(CONFIG_PATH),
        )
        sys.exit(1)

    missing_files = [
        name for name in REQUIRED_FILES if not (CONFIG_PATH / name).exists()
    ]

    if missing_files:
        found_files = [
            str(path.relative_to(CONFIG_PATH))
            for path in CONFIG_PATH.rglob("*")
            if path.is_file()
        ]
        log.error(
            "Missing required config files",
            missing_files=missing_files,
            config_path=str(CONFIG_PATH),
            found_files=found_files,
        )
        sys.exit(1)

    log.info(
        "Config validation passed",
        config_path=str(CONFIG_PATH),
        prompt_md_bytes=(CONFIG_PATH / "PROMPT.md").stat().st_size,
        mcp_json_bytes=(CONFIG_PATH / "mcp.json").stat().st_size,
    )

    try:
        from deep_agent.src.agent.config.loader import _strip_jsonc_comments

        with open(CONFIG_PATH / "mcp.json") as f:
            mcp_config = json.loads(_strip_jsonc_comments(f.read()))
            servers = list(mcp_config.get("mcpServers", {}).keys())
            log.info("MCP servers configured", mcp_servers=servers)
    except json.JSONDecodeError as e:
        log.warning("Invalid mcp.json", error=str(e))
    except OSError as e:
        log.warning("Could not read mcp.json", error=str(e))


def start_server() -> None:
    """Start the HTTP API server."""
    import uvicorn
    from aegra_api.main import app

    host = os.getenv("AGENT_HOST", "0.0.0.0")
    port = int(os.getenv("AGENT_PORT", "5002"))

    _logger().info(
        "Starting agent runtime",
        host=host,
        port=port,
        config_path=str(CONFIG_PATH),
    )

    # Importing aegra_api.main above already ran setup_logging(), which routes
    # every logger through structlog. Both kwargs below stop uvicorn undoing it:
    #
    # log_config=None  — uvicorn otherwise dictConfig's its own LOGGING_CONFIG
    #   here, re-attaching plain-text handlers to uvicorn.error. That is what
    #   made "Invalid HTTP request received." and the startup banner emit as
    #   unparseable text while everything else was JSON. With None, uvicorn.error
    #   propagates to aegra's root handler and renders as JSON like the rest.
    #
    # access_log=False — aegra's StructLogMiddleware already emits a structured
    #   access line per request (honouring LOG_EXCLUDE_PATHS), so uvicorn's own
    #   access logger would duplicate every one of them. Clearing log_config is
    #   not enough on its own: uvicorn still calls setLevel on uvicorn.access
    #   for log_level, so the handler must be removed explicitly.
    #
    # log_level still governs uvicorn.error, so startup, shutdown and
    # malformed-request reports are kept — they are now JSON rather than text.
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=False,
        log_config=None,
    )


def main() -> None:
    """Validate config mount and start the server."""
    validate_config_mount()
    start_server()


if __name__ == "__main__":
    main()
