"""Parse PROMPT.md front matter for LDAP group-to-role mappings.

Reads the ``groups`` and ``accessibility`` fields from the YAML front
matter of PROMPT.md.  The result is cached in-memory and invalidated
when the file's mtime changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

ROLE_HIERARCHY: dict[str, int] = {
    "owners": 4,
    "admins": 3,
    "builders": 2,
    "users": 1,
}

PRIVILEGED_ROLES: set[str] = {"owners", "admins", "builders"}


@dataclass
class GroupRoleMapping:
    """Maps an LDAP group CN to a role name (e.g. owners, admins)."""

    role: str
    group: str


@dataclass
class PromptAccessConfig:
    """Parsed PROMPT.md access control configuration."""

    groups: list[GroupRoleMapping] | None = None
    accessibility: str = "private"


_cached_config: PromptAccessConfig | None = None
_cached_mtime: float = 0.0


def _prompt_md_path() -> Path:
    """Return the path to PROMPT.md based on CONFIG_PATH env var."""
    config_dir = os.environ.get("CONFIG_PATH", "config/agent")
    return Path(config_dir) / "PROMPT.md"


def _parse_groups_from_frontmatter(path: Path) -> PromptAccessConfig:
    """Parse groups and accessibility from PROMPT.md YAML front matter."""
    try:
        from deep_agent.src.agent.config.parser import parse_frontmatter

        fm = parse_frontmatter(path)
    except Exception:
        logger.warning("Could not read %s — failing closed (denied)", path)
        return PromptAccessConfig(groups=[], accessibility="private")

    accessibility = "public" if fm.get("accessibility") == "public" else "private"

    raw_groups = fm.get("groups")
    if not isinstance(raw_groups, list) or len(raw_groups) == 0:
        return PromptAccessConfig(groups=None, accessibility=accessibility)

    groups: list[GroupRoleMapping] = []
    for entry in raw_groups:
        if (
            isinstance(entry, dict)
            and isinstance(entry.get("role"), str)
            and isinstance(entry.get("group"), str)
        ):
            groups.append(
                GroupRoleMapping(
                    role=entry["role"].lower(),
                    group=entry["group"],
                )
            )

    return PromptAccessConfig(
        groups=groups if groups else None,
        accessibility=accessibility,
    )


def get_prompt_access_config() -> PromptAccessConfig:
    """Return the cached PROMPT.md access config, reloading when the file changes."""
    global _cached_config, _cached_mtime

    path = _prompt_md_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        if _cached_config is not None:
            return _cached_config
        logger.warning("PROMPT.md not found at %s — failing closed (denied)", path)
        return PromptAccessConfig(groups=[], accessibility="private")

    if _cached_config is not None and mtime == _cached_mtime:
        return _cached_config

    config = _parse_groups_from_frontmatter(path)
    _cached_config = config
    _cached_mtime = mtime

    group_count = len(config.groups) if config.groups else 0
    logger.info(
        "PROMPT.md access config loaded: groups=%d accessibility=%s",
        group_count,
        config.accessibility,
    )
    return config


def reset_prompt_config() -> None:
    """Clear the cached config so the next call re-reads the file."""
    global _cached_config, _cached_mtime
    _cached_config = None
    _cached_mtime = 0.0
