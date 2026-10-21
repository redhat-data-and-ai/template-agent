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
    """Parse granted scopes from an OAuth token response body.

    Handles standard OAuth (top-level ``scope``) and Slack-style responses
    where scopes live under ``authed_user.scope`` as a comma-separated string.
    """
    scope_raw = body.get("scope")
    if scope_raw is None:
        authed_user = body.get("authed_user")
        if isinstance(authed_user, dict):
            scope_raw = authed_user.get("scope")
    if isinstance(scope_raw, str) and scope_raw:
        sep = "," if "," in scope_raw else " "
        return [s.strip() for s in scope_raw.split(sep) if s.strip()]
    if isinstance(scope_raw, list):
        scopes: list[str] = []
        for s in scope_raw:
            raw = str(s).strip()
            if not raw:
                continue
            sep = "," if "," in raw else " "
            scopes.extend(part.strip() for part in raw.split(sep) if part.strip())
        return scopes or None
    return None


def validate_granted_scopes(
    granted: list[str] | None,
    requested: list[str],
    mcp_name: str,
) -> list[str] | None:
    """Return the scopes this token actually carries, or None if it carries none.

    **A narrower grant than requested is not a failure.** RFC 6749 §3.3
    lets the authorization server issue a token whose scope differs
    from the request, and real providers do it constantly for reasons
    that have nothing to do with us:

    - Atlassian advertises `read:bitbucket:…` and `read:jira-align:…`
      in `scopes_supported`, then grants neither to a site that has
      no Bitbucket and no Jira Align. It granted 32 other scopes —
      Jira, Confluence, projects, teams, search — which is every scope
      the agent needs.
    - `delete:` and `manage:` variants require admin consent an
      ordinary user cannot give.

    `scopes_supported` is an advertisement of everything the server can
    *ever* issue, across every product and every user. It was never a
    list any one person can obtain, so requiring it rejects tokens that
    work perfectly.

    Rejecting also does not fail safe, which is the part worth being
    clear about. It converts a partly-capable integration into a dead
    one, and the retry produces the same narrower grant — so the user
    loops through consent forever. A missing scope surfaces anyway, as
    a 403 on the one tool that needs it, which is both accurate and
    survivable.

    The one real failure is a token with *no* scopes when scopes were
    requested: nothing was authorised and nothing will work.
    """
    if not requested:
        return granted

    # RFC 6749 §5.1: scope is optional when identical to what was requested.
    # Only assume requested when scope is truly absent (None), not explicitly empty.
    if granted is None:
        logger.debug(
            "OAuth token for '%s' omitted scopes; assuming requested %s",
            mcp_name,
            requested,
        )
        return list(requested)

    if not granted:
        logger.error(
            "OAuth token for '%s' granted no scopes (requested: %s)",
            mcp_name,
            requested,
        )
        return None

    withheld = [scope for scope in requested if scope not in set(granted)]
    if withheld:
        logger.warning(
            "OAuth token for '%s' was granted %d of %d requested scopes; "
            "withheld: %s. Tools needing these will fail individually.",
            mcp_name,
            len(granted),
            len(requested),
            withheld,
        )

    return granted
