"""Filesystem permissions builder.

Converts declarative permission rules from filesystem.yaml into
deepagents FilesystemPermission instances that are passed to
create_deep_agent(permissions=...).

Template-agent users only edit YAML. This module handles the conversion.
"""

from __future__ import annotations

from typing import Any

from deep_agent.src.agent.config.filesystem import (
    FilesystemFileConfig,
)
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


def build_permissions(
    config: FilesystemFileConfig,
) -> list[Any] | None:
    """Build FilesystemPermission list from config.

    Args:
        config: Parsed filesystem.yaml config.

    Returns:
        List of FilesystemPermission instances, or None if no rules defined.
        None means deepagents uses its default (all operations allowed).
    """
    if not config.permissions:
        return None

    try:
        from deepagents.middleware.filesystem import FilesystemPermission
    except ImportError:
        logger.warning(
            "FilesystemPermission not available — permissions config ignored"
        )
        return None

    permissions: list[Any] = []

    for rule in config.permissions:
        try:
            perm = FilesystemPermission(
                operations=rule.operations,
                paths=rule.paths,
                mode=rule.mode,
            )
            permissions.append(perm)
        except Exception as e:
            logger.warning("Skipping invalid permission rule %r: %s", rule, e)

    if permissions:
        logger.info("Built %d filesystem permission rule(s)", len(permissions))

    return permissions or None
