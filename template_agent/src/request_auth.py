"""Resolve API / SSO tokens from request headers."""

from fastapi import Request

MISSING_AUTH_DETAIL = "Missing credentials: send X-Token or Authorization: Bearer"


def access_token_from_request(request: Request) -> str | None:
    """Return token from X-Token or Authorization: Bearer (case-insensitive scheme)."""
    raw = request.headers.get("X-Token")
    if raw:
        return raw.strip()
    auth = request.headers.get("authorization")
    if not auth:
        return None
    parts = auth.strip().split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None
