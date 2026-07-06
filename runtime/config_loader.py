#!/usr/bin/env python3
"""Config Loader - Validates config volume mount before starting agent runtime.

Expected directory structure at $CONFIG_PATH (/app/config/agent):
  - PROMPT.md
  - mcp.json
  - skills/
  - subagents/
  - runtime/ (optional)
"""

import json
import os
import sys
from pathlib import Path

CONFIG_PATH = Path(os.getenv("CONFIG_PATH", "/app/config/agent"))
REQUIRED_FILES = ["PROMPT.md", "mcp.json"]


def validate_config_mount() -> None:
    """Ensure config volume is mounted and contains required files."""
    if not CONFIG_PATH.exists():
        print(f"❌ ERROR: Config path not found: {CONFIG_PATH}", file=sys.stderr)
        print("   Expected config to be mounted as a volume.", file=sys.stderr)
        print("   Check deployment spec for volume mount.", file=sys.stderr)
        sys.exit(1)

    missing_files = []
    for file in REQUIRED_FILES:
        if not (CONFIG_PATH / file).exists():
            missing_files.append(file)

    if missing_files:
        print(
            f"❌ ERROR: Missing required config files: {missing_files}", file=sys.stderr
        )
        print(f"   Config path: {CONFIG_PATH}", file=sys.stderr)
        found_files = [
            str(p.relative_to(CONFIG_PATH))
            for p in CONFIG_PATH.rglob("*")
            if p.is_file()
        ]
        print(f"   Found files: {found_files}", file=sys.stderr)
        sys.exit(1)

    print(f"✅ Config validation passed: {CONFIG_PATH}")
    print(f"   PROMPT.md: {(CONFIG_PATH / 'PROMPT.md').stat().st_size} bytes")
    print(f"   mcp.json: {(CONFIG_PATH / 'mcp.json').stat().st_size} bytes")

    # Validate mcp.json structure
    try:
        with open(CONFIG_PATH / "mcp.json") as f:
            mcp_config = json.load(f)
            servers = list(mcp_config.get("servers", {}).keys())
            print(f"   MCP servers: {servers if servers else 'none'}")
    except json.JSONDecodeError as e:
        print(f"⚠️  WARNING: Invalid mcp.json: {e}", file=sys.stderr)
    except Exception as e:
        print(f"⚠️  WARNING: Could not read mcp.json: {e}", file=sys.stderr)


def start_agent() -> None:
    """Start the agent runtime after config validation."""
    import uvicorn
    from aegra_api.main import app

    host = os.getenv("AGENT_HOST", "0.0.0.0")
    port = int(os.getenv("AGENT_PORT", "5002"))

    print("🚀 Starting agent runtime...")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Config: {CONFIG_PATH}")

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    validate_config_mount()
    start_agent()
