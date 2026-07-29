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


async def memory_user_id(request: Request) -> str:
    """User id for LangGraph Store memories (preferred_username / X-User-ID).

    Custom Rules and memories both use this id — the same value the BFF
    puts on ``X-User-ID`` and run ``metadata.user_id``:

    - Auth off (local): ``X-User-ID``, then ``DEV_USER_ID``.
    - Auth on (prod): verified JWT ``preferred_username`` or ``sub``. A
      present ``X-User-ID`` must match that claim (403 if it does not).
    """
    from deep_agent.aegra.auth import DEV_USER_ID, ENABLE_AUTH, _decode_token

    header = (request.headers.get("x-user-id") or "").strip()

    if not ENABLE_AUTH:
        return header or DEV_USER_ID

    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Missing or invalid Authorization header"
        )

    payload = await asyncio.to_thread(_decode_token, auth_header[7:])
    token_uid = str(
        payload.get("preferred_username") or payload.get("sub") or ""
    ).strip()
    if not token_uid:
        raise HTTPException(
            status_code=401, detail="Token missing preferred_username and sub"
        )

    if header and header != token_uid:
        raise HTTPException(
            status_code=403, detail="X-User-ID does not match authenticated user"
        )

    return token_uid


async def rule_user_ids(request: Request) -> list[str]:
    """Owner key for Custom Rules — same id as memories and the chat graph."""
    return [await memory_user_id(request)]
