"""Shared authentication helpers for Aegra route handlers."""

from __future__ import annotations

import asyncio

from fastapi import HTTPException, Request

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


async def authenticated_user_id(
    request: Request, *, reject_anonymous: bool = False
) -> str:
    """Extract and return the authenticated user's identity from the JWT.

    Args:
        request: The incoming FastAPI request.
        reject_anonymous: If True, raise 401 when credentials are missing
            instead of returning ``"anonymous"``.

    Returns:
        The ``sub`` claim from the JWT, ``DEV_USER_ID`` when auth is
        disabled, or ``"anonymous"`` if credentials are absent and
        *reject_anonymous* is False.
    """
    from deep_agent.aegra.auth import DEV_USER_ID, ENABLE_AUTH, _decode_token

    if not ENABLE_AUTH:
        return DEV_USER_ID

    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        if reject_anonymous:
            raise HTTPException(
                status_code=401, detail="Missing or invalid Authorization header"
            )
        return "anonymous"

    payload = await asyncio.to_thread(_decode_token, auth_header[7:])
    return str(payload["sub"])
