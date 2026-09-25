"""Frontmatter parsing and runtime value injection.

This module handles parsing markdown files with YAML frontmatter (used for agent
configurations) and injecting runtime values like {{current_date}} into the content.

Why this exists:
    Agent configs are written in markdown with YAML frontmatter. This module
    extracts the frontmatter metadata and body content, and replaces template
    variables with runtime values.

Functions:
    parse_frontmatter: Parse markdown file with YAML frontmatter
"""

from pathlib import Path
from typing import Any

import yaml


def parse_frontmatter(path: Path) -> dict[str, Any]:
    r"""Parse a markdown file with YAML frontmatter.

    Expects the format: ``--- \n <yaml> \n --- \n <markdown body>``.
    The markdown body is returned under the ``"body"`` key.

    Args:
        path: Path to the ``.md`` file.

    Returns:
        A dict of frontmatter fields plus ``body`` (the markdown content).
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
