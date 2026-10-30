"""Config data reader - lets the agent read files from config/agent/data/.

Provides sandboxed read-only access to static data files bundled with the
agent (product catalog, table metadata, etc.) without depending on the
deepagents filesystem middleware or MCP resource tools.
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.tools import tool

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_CONFIG_DIR = Path(os.environ.get("CONFIG_PATH", "config/agent"))
_DATA_DIR = _CONFIG_DIR / "data"

_MAX_SIZE_BYTES = 512_000  # 500 KB safety limit


@tool
def read_config_data(filename: str) -> str:
    """Read a data file bundled with the agent configuration.

    Use this tool to access static reference data such as the product catalog
    or table metadata. Files live in the agent's config/agent/data/ directory.

    Available files:
    - product_catalog.yaml: Full catalog of data products with descriptions,
      domains, schemas, and usage notes.
    - product_tables.json: Detailed table metadata (columns, types, sample
      values) for all data products.

    Args:
        filename: Name of the file to read (e.g. "product_catalog.yaml").
            Only the filename is needed, not the full path.

    Returns:
        The file contents as a string, or an error message if the file
        is not found or access is denied.
    """
    safe_name = Path(filename).name
    if safe_name != filename:
        logger.warning("Path traversal blocked: %r -> %r", filename, safe_name)
        return f"Error: Only filenames are accepted, not paths. Use '{safe_name}' instead."

    target = _DATA_DIR / safe_name

    if not target.exists():
        available = sorted(f.name for f in _DATA_DIR.iterdir() if f.is_file()) if _DATA_DIR.is_dir() else []
        return (
            f"Error: File '{safe_name}' not found in config data directory.\n"
            f"Available files: {', '.join(available) if available else '(none)'}"
        )

    if not target.is_file():
        return f"Error: '{safe_name}' is not a regular file."

    size = target.stat().st_size
    if size > _MAX_SIZE_BYTES:
        return f"Error: File '{safe_name}' is {size:,} bytes, exceeding the {_MAX_SIZE_BYTES:,} byte limit."

    try:
        content = target.read_text(encoding="utf-8")
        logger.info("read_config_data: served %s (%d bytes)", safe_name, len(content))
        return content
    except Exception as exc:
        logger.error("read_config_data: failed to read %s: %s", safe_name, exc)
        return f"Error reading '{safe_name}': {exc}"
