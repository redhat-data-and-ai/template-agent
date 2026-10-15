"""Tags on conversations.

Projects already give a conversation one home; tags give it as many
labels as it needs. The two answer different questions — "which piece
of work is this part of" versus "what is this about" — and collapsing
them would force a choice nobody wants to make.

**Stored in `thread.metadata_json`, not in a new table.** The column
already carries a GIN index on `jsonb_path_ops`, which is exactly the
index a containment query over a tag array wants, so filtering is an
index scan with no migration and nothing new to keep consistent with
the thread it describes. Deleting a thread deletes its tags, which is
the correct behaviour and free.
"""

from __future__ import annotations

import re
from typing import Any

import psycopg
from fastapi import APIRouter, HTTPException, Query, Request
from psycopg.rows import dict_row
from pydantic import BaseModel, Field

from deep_agent.aegra.auth_helpers import authenticated_user_id
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

router = APIRouter(tags=["threads"])

# Long enough for "q3-planning", short enough that a tag stays a label
# rather than becoming a note.
MAX_TAG_LENGTH = 32

# Past this, tags stop being a way to find things.
MAX_TAGS_PER_THREAD = 20

# Lower-case, because `Billing` and `billing` are the same tag and a
# UI that shows both is a UI that lost half the conversations.
_ALLOWED = re.compile(r"^[a-z0-9][a-z0-9 _-]*$")


class TagsRequest(BaseModel):
    """The full tag set for a thread — a replace, not an append."""

    tags: list[str] = Field(default_factory=list)


def normalise_tag(raw: str) -> str:
    """Fold a tag to its canonical form, or raise.

    Normalisation happens on write rather than on read: a tag written
    once and compared a thousand times should be cheap to compare.
    """
    tag = " ".join(str(raw or "").strip().lower().split())
    if not tag:
        raise ValueError("a tag cannot be empty")
    if len(tag) > MAX_TAG_LENGTH:
        raise ValueError(f"'{tag[:20]}…' is longer than {MAX_TAG_LENGTH} characters")
    if not _ALLOWED.match(tag):
        raise ValueError(
            f"'{tag}' may only contain letters, digits, spaces, hyphens and underscores"
        )
    return tag


def normalise_tags(raw: list[str]) -> list[str]:
    """Normalise, de-duplicate and order a tag list."""
    seen: list[str] = []
    for item in raw:
        tag = normalise_tag(item)
        if tag not in seen:
            seen.append(tag)
    if len(seen) > MAX_TAGS_PER_THREAD:
        raise ValueError(f"a conversation may have at most {MAX_TAGS_PER_THREAD} tags")
    return sorted(seen)


async def _connect() -> psycopg.AsyncConnection:
    return await psycopg.AsyncConnection.connect(
        settings.database_uri, row_factory=dict_row
    )


@router.put("/threads/{thread_id}/tags")
async def set_tags(
    thread_id: str, body: TagsRequest, request: Request
) -> dict[str, Any]:
    """Replace a conversation's tags.

    Replace rather than add: the UI edits a set, and an add/remove
    pair would need the client to diff against state it may not have.
    """
    user_id = await authenticated_user_id(request, reject_anonymous=True)

    try:
        tags = normalise_tags(body.tags)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    async with await _connect() as conn:
        # Ownership is checked in the UPDATE itself rather than with a
        # SELECT first — two statements leave a window where the thread
        # changes hands between the check and the write.
        result = await conn.execute(
            """
            UPDATE thread
               SET metadata_json = jsonb_set(
                       metadata_json, '{tags}', %(tags)s::jsonb, true
                   ),
                   updated_at = now()
             WHERE thread_id = %(thread_id)s AND user_id = %(user_id)s
            RETURNING thread_id
            """,
            {
                "tags": psycopg.types.json.Json(tags),
                "thread_id": thread_id,
                "user_id": user_id,
            },
        )
        if await result.fetchone() is None:
            # 404 for someone else's thread as well as a missing one.
            # A 403 confirms the thread exists, which is more than a
            # stranger should learn from a URL.
            raise HTTPException(status_code=404, detail="Thread not found")
        await conn.commit()

    return {"thread_id": thread_id, "tags": tags}


@router.get("/threads/{thread_id}/tags")
async def get_tags(thread_id: str, request: Request) -> dict[str, Any]:
    """A conversation's tags."""
    user_id = await authenticated_user_id(request, reject_anonymous=True)

    async with await _connect() as conn:
        row = await (
            await conn.execute(
                """
                SELECT metadata_json -> 'tags' AS tags
                  FROM thread
                 WHERE thread_id = %(thread_id)s AND user_id = %(user_id)s
                """,
                {"thread_id": thread_id, "user_id": user_id},
            )
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"thread_id": thread_id, "tags": row.get("tags") or []}


@router.get("/tags")
async def list_tags(request: Request) -> dict[str, Any]:
    """Every tag this user has used, with counts.

    Counts are what make a tag list usable — a tag applied once is
    noise, and without the number there is no way to tell it from the
    one applied forty times.
    """
    user_id = await authenticated_user_id(request, reject_anonymous=True)

    async with await _connect() as conn:
        rows = await (
            await conn.execute(
                """
                SELECT tag, COUNT(*) AS threads
                  FROM thread,
                       LATERAL jsonb_array_elements_text(
                           CASE WHEN jsonb_typeof(metadata_json -> 'tags') = 'array'
                                THEN metadata_json -> 'tags' ELSE '[]'::jsonb END
                       ) AS tag
                 WHERE user_id = %(user_id)s
                 GROUP BY tag
                 ORDER BY threads DESC, tag
                """,
                {"user_id": user_id},
            )
        ).fetchall()

    return {"tags": [{"tag": r["tag"], "threads": int(r["threads"])} for r in rows]}


# Under /tags rather than /threads/by-tag: Aegra already owns
# /threads/{thread_id}, and a sibling literal path would be matched as
# a thread called "by-tag" depending on which router registered first.
@router.get("/tags/{tag}/threads")
async def threads_by_tag(
    tag: str,
    request: Request,
    limit: int = Query(50, ge=1, le=200),
) -> dict[str, Any]:
    """Conversations carrying a tag, newest first."""
    user_id = await authenticated_user_id(request, reject_anonymous=True)

    try:
        wanted = normalise_tag(tag)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    async with await _connect() as conn:
        rows = await (
            await conn.execute(
                # Containment, so the GIN index on metadata_json is
                # usable. Unnesting the array would work and would scan
                # every thread the user owns.
                """
                SELECT thread_id,
                       metadata_json ->> 'title' AS title,
                       metadata_json -> 'tags'   AS tags,
                       updated_at
                  FROM thread
                 WHERE user_id = %(user_id)s
                   AND metadata_json @> %(probe)s::jsonb
                 ORDER BY updated_at DESC
                 LIMIT %(limit)s
                """,
                {
                    "user_id": user_id,
                    "probe": psycopg.types.json.Json({"tags": [wanted]}),
                    "limit": limit,
                },
            )
        ).fetchall()

    return {
        "tag": wanted,
        "threads": [
            {
                "thread_id": r["thread_id"],
                "title": r["title"],
                "tags": r["tags"] or [],
                "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
            }
            for r in rows
        ],
    }
