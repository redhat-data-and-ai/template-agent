"""Fork a conversation at an earlier turn.

Search finds the conversation; this is what you do once you are in it
and want to try a different question from the middle without losing the
original.

**Forks copy into a new thread rather than branching in place.**
LangGraph can genuinely branch — run from an older checkpoint and leave
a divergent history behind in the same thread — and that was the first
design. It was rejected for two reasons. The original conversation stops
being a stable thing you can link to or search, since its history now
depends on which branch you ask for. And the UI would have to render a
tree, which is a large change to pay for a feature whose usual shape is
"ask that again, differently".

Copying is also version-proof: it uses the Agent Protocol's state
endpoints rather than reaching into checkpoint tables, so it does not
break when the storage layout moves — which, per §19 and §22, it does.
"""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from deep_agent.aegra.auth_helpers import authenticated_user_id
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

router = APIRouter(tags=["threads"])

# A fork copies state; it never runs the graph, so this only has to
# cover a few local calls.
_TIMEOUT = settings.FORK_REQUEST_TIMEOUT_SECONDS


class ForkRequest(BaseModel):
    """Where to cut.

    One of the two must be given. `checkpoint_id` is what the history
    endpoint hands back and is exact. `message_index` is what a UI has
    when the user clicks a message, and means "keep this many messages".
    """

    checkpoint_id: str | None = None
    message_index: int | None = Field(default=None, ge=0)
    title: str | None = Field(default=None, max_length=200)


def _client(request: Request) -> httpx.AsyncClient:
    """Loopback client carrying the caller's own credentials.

    The fork must be subject to the same authorisation as the caller —
    borrowing a service identity here would let anyone fork any thread.
    """
    headers = {}
    auth = request.headers.get("Authorization")
    if auth:
        headers["Authorization"] = auth
    return httpx.AsyncClient(
        base_url=settings.loopback_base_url,
        headers=headers,
        timeout=_TIMEOUT,
    )


def _messages_at(history: list[dict[str, Any]], checkpoint_id: str) -> list[dict]:
    for entry in history:
        if entry.get("checkpoint_id") == checkpoint_id:
            return list((entry.get("values") or {}).get("messages") or [])
    raise HTTPException(status_code=404, detail="Checkpoint not found in this thread")


def _latest_messages(history: list[dict[str, Any]]) -> list[dict]:
    """History is newest-first; the longest message list is the current one."""
    best: list[dict] = []
    for entry in history:
        messages = (entry.get("values") or {}).get("messages") or []
        if len(messages) > len(best):
            best = list(messages)
    return best


def _strip_for_replay(messages: list[Any]) -> list[dict]:
    """Drop fields that must not be copied into a new thread.

    Message ids are the big one: re-using them makes LangGraph treat the
    seeded messages as updates to messages that do not exist in the new
    thread, and they silently vanish.
    """
    cleaned = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        copy = {k: v for k, v in message.items() if k not in ("id", "run_id")}
        if copy.get("type") in ("human", "ai", "system", "tool"):
            cleaned.append(copy)
    return cleaned


@router.post("/threads/{thread_id}/fork")
async def fork_thread(
    thread_id: str, body: ForkRequest, request: Request
) -> dict[str, Any]:
    """Copy a conversation up to a point into a new thread."""
    user_id = await authenticated_user_id(request, reject_anonymous=True)

    if body.checkpoint_id is None and body.message_index is None:
        raise HTTPException(
            status_code=400, detail="Provide checkpoint_id or message_index"
        )

    async with _client(request) as client:
        # Reading history through the API rather than SQL is also the
        # ownership check: a thread the caller cannot see 404s here.
        history_response = await client.get(f"/threads/{thread_id}/history")
        if history_response.status_code == 404:
            raise HTTPException(status_code=404, detail="Thread not found")
        if history_response.status_code != 200:
            raise HTTPException(status_code=502, detail="Could not read thread history")

        history = history_response.json() or []
        if not isinstance(history, list) or not history:
            raise HTTPException(status_code=400, detail="Thread has no history to fork")

        if body.checkpoint_id:
            messages = _messages_at(history, body.checkpoint_id)
        else:
            messages = _latest_messages(history)[: body.message_index]

        messages = _strip_for_replay(messages)
        if not messages:
            raise HTTPException(status_code=400, detail="Nothing to fork at that point")

        source = await client.get(f"/threads/{thread_id}")
        source_metadata = (
            (source.json() or {}).get("metadata") or {}
            if source.status_code == 200
            else {}
        )

        # graph_id is required up front: a thread with no graph
        # association rejects a state update, so seeding a bare thread
        # fails with "has no associated graph".
        created = await client.post(
            "/threads",
            json={
                "metadata": {
                    "graph_id": source_metadata.get("graph_id") or "agent",
                    "assistant_id": source_metadata.get("assistant_id") or "agent",
                    "thread_name": body.title or _forked_title(source_metadata),
                    # Provenance, so a fork is traceable back to what it
                    # came from rather than looking like a fresh chat.
                    "forked_from": thread_id,
                    "forked_at_checkpoint": body.checkpoint_id,
                    "forked_message_count": len(messages),
                    # Carried over so a fork lands in the same project.
                    "project_id": source_metadata.get("project_id"),
                },
            },
        )
        if created.status_code not in (200, 201):
            raise HTTPException(status_code=502, detail="Could not create the fork")
        new_thread_id = created.json()["thread_id"]

        seeded = await client.post(
            f"/threads/{new_thread_id}/state",
            json={
                "values": {"messages": messages},
                "assistant_id": source_metadata.get("assistant_id") or "agent",
            },
        )
        if seeded.status_code not in (200, 201):
            # Leaving an empty thread behind would look like a chat the
            # user started and abandoned.
            await client.delete(f"/threads/{new_thread_id}")
            logger.error(
                "Fork seeding failed (%s): %s", seeded.status_code, seeded.text[:200]
            )
            raise HTTPException(status_code=502, detail="Could not seed the fork")

    logger.info(
        "Forked thread %s -> %s (%d messages) for %s",
        thread_id,
        new_thread_id,
        len(messages),
        user_id,
    )
    return {
        "thread_id": new_thread_id,
        "forked_from": thread_id,
        "message_count": len(messages),
    }


def _forked_title(metadata: dict[str, Any]) -> str:
    original = str(metadata.get("thread_name") or "").strip()
    return f"{original} (fork)"[:200] if original else "Forked conversation"
