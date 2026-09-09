"""Shared authentication helpers for Aegra route handlers."""

from __future__ import annotations

import asyncio
from typing import cast

from fastapi import HTTPException, Request

from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


def _normalize_roles(payload: dict) -> list[str]:
    """Extract realm_access.roles as a flat list of strings."""
    realm = payload.get("realm_access") or {}
    roles = realm.get("roles") if isinstance(realm, dict) else None
    if isinstance(roles, list):
        return [r for r in roles if isinstance(r, str)]
    return []


def check_group_access(permissions: list[str], *, developer_only: bool = False) -> None:
    """Enforce group-based access from DEVELOPER_GROUP / USER_GROUP.

    Both empty: unrestricted (caller still enforces ENABLE_AUTH).
    Only one group set: that group is an allow-list.
    Both set: DEVELOPER_GROUP has full access; USER_GROUP has non-eval access.
    developer_only=True requires DEVELOPER_GROUP (eval endpoints).
    """
    # Settings validator normalizes to list[str] at runtime
    dev_groups = cast(list[str], settings.DEVELOPER_GROUP)
    user_groups = cast(list[str], settings.USER_GROUP)
    if not dev_groups and not user_groups:
        return

    if developer_only:
        if dev_groups and any(g in permissions for g in dev_groups):
            return
        detail = (
            "Access denied: developer group membership required."
            if dev_groups
            else "Access denied: DEVELOPER_GROUP is not configured."
        )
        raise HTTPException(status_code=403, detail=detail)

    if dev_groups and any(g in permissions for g in dev_groups):
        return
    if user_groups and any(g in permissions for g in user_groups):
        return
    all_groups = dev_groups + user_groups
    allowed = " or ".join(f"'{g}'" for g in all_groups)
    raise HTTPException(
        status_code=403,
        detail=f"Access denied: {allowed} group membership required.",
    )


async def authenticated_user_id(
    request: Request, *, reject_anonymous: bool = False, developer_only: bool = False
) -> str:
    """Extract and return the authenticated user's identity from the JWT.

    Args:
        request: The incoming FastAPI request.
        reject_anonymous: If True, raise 401 when credentials are missing
            instead of returning ``"anonymous"``.
        developer_only: If True, only DEVELOPER_GROUP members pass group check.

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
    permissions = _normalize_roles(payload)
    check_group_access(permissions, developer_only=developer_only)
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
