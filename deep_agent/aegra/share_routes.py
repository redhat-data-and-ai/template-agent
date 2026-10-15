"""Share a conversation by link.

See `deep_agent/src/sharing/store.py` for the four defaults this is
built on — snapshot not live, authenticated not public, expiring not
permanent, attachments named not embedded.
"""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from deep_agent.aegra.auth_helpers import authenticated_user_id
from deep_agent.src.settings import settings
from deep_agent.src.sharing import store
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

router = APIRouter(tags=["sharing"])

_TIMEOUT = 30


class ShareRequest(BaseModel):
    """Who may open a share link, and for how long."""

    scope: str = Field(default="authenticated")
    expiry_days: int = Field(default=store.DEFAULT_EXPIRY_DAYS, ge=1)


@router.post("/threads/{thread_id}/share")
async def share_thread(
    thread_id: str, body: ShareRequest, request: Request
) -> dict[str, Any]:
    """Publish a snapshot of this conversation behind a link."""
    user_id = await authenticated_user_id(request, reject_anonymous=True)

    headers = {}
    auth = request.headers.get("Authorization")
    if auth:
        headers["Authorization"] = auth

    # State is read through the Agent Protocol rather than out of the
    # checkpoint tables — same reasoning as forking: the storage layout
    # moves, the protocol does not.
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{settings.AGENT_PORT}",
        headers=headers,
        timeout=_TIMEOUT,
    ) as client:
        state = await client.get(f"/threads/{thread_id}/state")
        if state.status_code != 200:
            raise HTTPException(status_code=404, detail="Thread not found")
        meta = await client.get(f"/threads/{thread_id}")

    values = (state.json() or {}).get("values") or {}
    title = ""
    if meta.status_code == 200:
        title = str(((meta.json() or {}).get("metadata") or {}).get("title") or "")

    snapshot = store.build_snapshot(values, title or "Shared conversation")
    if not snapshot["messages"]:
        # Sharing an empty conversation produces a link to nothing,
        # which the recipient reads as a broken link rather than as an
        # empty conversation.
        raise HTTPException(
            status_code=422, detail="There is nothing in this conversation to share"
        )

    try:
        share = await store.create_share(
            thread_id=thread_id,
            owner_id=user_id,
            title=snapshot["title"],
            snapshot=snapshot,
            scope=body.scope,
            expiry_days=body.expiry_days,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    result = share.as_dict()
    result["path"] = f"/shared/{share.share_id}"
    result["messages"] = len(snapshot["messages"])
    result["attachment_names"] = snapshot["attachment_names"]
    return result


@router.get("/threads/{thread_id}/share")
async def get_thread_share(thread_id: str, request: Request) -> dict[str, Any]:
    """Whether this conversation is shared, and how it has been used."""
    user_id = await authenticated_user_id(request, reject_anonymous=True)
    share = await store.get_share_for_thread(thread_id, user_id)
    if share is None:
        return {"shared": False}
    return {"shared": True, **share.as_dict(), "path": f"/shared/{share.share_id}"}


@router.delete("/threads/{thread_id}/share")
async def unshare_thread(thread_id: str, request: Request) -> dict[str, Any]:
    """Revoke the link."""
    user_id = await authenticated_user_id(request, reject_anonymous=True)
    revoked = await store.revoke_share(thread_id, user_id)
    return {"revoked": revoked}


@router.get("/shares")
async def my_shares(request: Request) -> dict[str, Any]:
    """Everything this user has shared.

    The feature nobody builds and everybody needs: without it, the
    only way to find out what you have published is to remember.
    """
    user_id = await authenticated_user_id(request, reject_anonymous=True)
    shares = await store.list_shares(user_id)
    return {
        "shares": [
            {**share.as_dict(), "path": f"/shared/{share.share_id}"} for share in shares
        ]
    }


@router.get("/shared/{share_id}")
async def read_shared(share_id: str, request: Request) -> dict[str, Any]:
    """Read a shared conversation."""
    found = await store.read_share(share_id)
    if found is None:
        raise HTTPException(status_code=404, detail="This link is not valid")

    share, snapshot = found

    if share.scope != "public":
        # An `authenticated` share is still a link, not a grant: any
        # signed-in user who has it may read it. What it rules out is
        # the link working for the whole internet, which is the
        # difference that matters for an internal tool.
        await authenticated_user_id(request, reject_anonymous=True)

    if not share.active:
        # Expired and revoked are reported apart, because "ask them to
        # share it again" and "they took it down" are different things
        # for the reader to do.
        detail = (
            "This link was revoked" if share.revoked_at else "This link has expired"
        )
        raise HTTPException(status_code=410, detail=detail)

    return {
        "share_id": share.share_id,
        "title": share.title,
        "scope": share.scope,
        "captured_at": snapshot.get("captured_at"),
        "messages": snapshot.get("messages", []),
        # Named only — the files themselves are not part of a share.
        "attachment_names": snapshot.get("attachment_names", []),
    }
