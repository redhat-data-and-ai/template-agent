"""OAuth scope parsing and validation for MCP token flows."""

from __future__ import annotations

from typing import Any

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


def requested_scopes(oauth_cfg: dict[str, Any]) -> list[str]:
    """Return normalized scope list from MCP OAuth config."""
    scopes = oauth_cfg.get("scopes") or []
    if isinstance(scopes, list):
        return [str(s) for s in scopes if s]
    if isinstance(scopes, str) and scopes:
        return scopes.split()
    return []


def parse_token_scopes(body: dict[str, Any]) -> list[str] | None:
    """Parse granted scopes from an OAuth token response body."""
    scope_raw = body.get("scope")
    if isinstance(scope_raw, str) and scope_raw:
        return scope_raw.split()
    if isinstance(scope_raw, list):
        return [str(s) for s in scope_raw if s]
    return None


def validate_granted_scopes(
    granted: list[str] | None,
    requested: list[str],
    mcp_name: str,
) -> list[str] | None:
    """Return granted scopes when they include all requested scopes, else None."""
    if not requested:
        return granted

    if not granted:
        logger.error(
            "OAuth token for '%s' returned no scopes; requested %s",
            mcp_name,
            requested,
        )
        return None

    missing = [scope for scope in requested if scope not in set(granted)]
    if missing:
        logger.error(
            "OAuth token for '%s' missing requested scopes %s (granted: %s)",
            mcp_name,
            missing,
            granted,
        )
        return None

    return granted
