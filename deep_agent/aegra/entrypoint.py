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

CONFIG_PATH = Path(os.getenv("CONFIG_PATH", "/app/config/agent"))

# Logical names, resolved through config.paths rather than hardcoded
# filenames. `mcp` lives at `integrations/mcp.json` with the old
# root-level `mcp.json` still accepted; a literal name here would have
# aborted startup the moment the file moved — the entrypoint refusing
# to boot over a rename the rest of the code handles.
REQUIRED_CONFIGS = ["prompt", "mcp"]


def validate_config_mount() -> None:
    """Ensure config volume is mounted and contains required files."""
    if not CONFIG_PATH.exists():
        print(f"ERROR: Config path not found: {CONFIG_PATH}", file=sys.stderr)
        print("   Expected config to be mounted as a volume.", file=sys.stderr)
        print("   Check deployment spec for volume mount.", file=sys.stderr)
        sys.exit(1)

    from deep_agent.src.agent.config.paths import candidates_for

    def _find(name: str) -> Path | None:
        for relative in candidates_for(name):
            path = CONFIG_PATH / relative
            if path.is_file():
                return path
        return None

    resolved = {name: _find(name) for name in REQUIRED_CONFIGS}
    missing_files = [name for name, path in resolved.items() if path is None]

    if missing_files:
        print(
            f"ERROR: Missing required config files: {missing_files}",
            file=sys.stderr,
        )
        print(f"   Config path: {CONFIG_PATH}", file=sys.stderr)
        found_files = [
            str(path.relative_to(CONFIG_PATH))
            for path in CONFIG_PATH.rglob("*")
            if path.is_file()
        ]
        print(f"   Found files: {found_files}", file=sys.stderr)
        sys.exit(1)

    print(f"Config validation passed: {CONFIG_PATH}")
    for path in resolved.values():
        assert path is not None  # missing names exited above
        print(f"   {path.relative_to(CONFIG_PATH)}: {path.stat().st_size} bytes")

    try:
        from deep_agent.src.agent.config.loader import _strip_jsonc_comments

        mcp_path = resolved["mcp"]
        assert mcp_path is not None  # asserted for every name above
        with open(mcp_path) as f:
            mcp_config = json.loads(_strip_jsonc_comments(f.read()))
            servers = list(mcp_config.get("mcpServers", {}).keys())
            print(f"   MCP servers: {servers if servers else 'none'}")
    except json.JSONDecodeError as e:
        print(f"WARNING: Invalid mcp.json: {e}", file=sys.stderr)
    except OSError as e:
        print(f"WARNING: Could not read mcp.json: {e}", file=sys.stderr)


def start_server() -> None:
    """Start the HTTP API server."""
    import uvicorn
    from aegra_api.main import app

    host = os.getenv("AGENT_HOST", "0.0.0.0")
    port = int(os.getenv("AGENT_PORT", "5002"))

    print("Starting agent runtime...")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Config: {CONFIG_PATH}")

    uvicorn.run(app, host=host, port=port, log_level="info")


def main() -> None:
    """Validate config mount and start the server."""
    validate_config_mount()
    start_server()


if __name__ == "__main__":
    main()
