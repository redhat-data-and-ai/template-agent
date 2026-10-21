"""Shared conversations: the table, the snapshot, and what it leaves out.

Sharing a conversation is an authorisation decision, so the defaults
here matter more than the feature does. Four of them, each chosen
against the obvious alternative:

**A snapshot, not a live view.** A link that keeps updating as the
owner keeps talking publishes messages they had not written when they
shared it. Nobody reasons about a link that way. Re-sharing takes a
new snapshot, explicitly.

**Authenticated by default, public only on request.** "Anyone with the
link" on an internal tool carrying customer data is a decision somebody
should have to make, not a default they inherit.

**Links expire.** open-webui's shares do not, and a link that outlives
the reason it was created is the actual risk — not the moment of
sharing, the year afterwards.

**Attachments are excluded.** A snapshot that embedded uploaded files
would re-publish documents through a URL the document's owner never
saw. The snapshot records that files were attached and names them; it
does not carry their contents.

The table is created with IF NOT EXISTS rather than an Alembic
revision, following `thread_search`: it belongs to us, not to Aegra's
migration chain.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

# 32 hex characters. Long enough that guessing is not a strategy, which
# matters most for `public` shares, where the token is the only control.
_TOKEN_BYTES = settings.SHARE_TOKEN_BYTES

DEFAULT_EXPIRY_DAYS = settings.SHARE_DEFAULT_EXPIRY_DAYS
MAX_EXPIRY_DAYS = settings.SHARE_MAX_EXPIRY_DAYS

# Beyond this a snapshot stops being a conversation anyone will read
# and starts being a payload.
MAX_SNAPSHOT_MESSAGES = settings.SHARE_MAX_SNAPSHOT_MESSAGES
MAX_MESSAGE_CHARS = settings.SHARE_MAX_MESSAGE_CHARS

_SCHEMA = """
CREATE TABLE IF NOT EXISTS thread_share (
    share_id       TEXT PRIMARY KEY,
    thread_id      TEXT NOT NULL,
    owner_id       TEXT NOT NULL,
    title          TEXT NOT NULL DEFAULT '',
    -- The conversation as it stood when shared. Deliberately a copy:
    -- revoking a share must not depend on the thread still existing,
    -- and deleting a thread must not silently change what a link
    -- already published.
    snapshot       JSONB NOT NULL DEFAULT '{}'::jsonb,
    scope          TEXT NOT NULL DEFAULT 'authenticated',
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at     TIMESTAMPTZ,
    revoked_at     TIMESTAMPTZ,
    views          INTEGER NOT NULL DEFAULT 0,
    last_viewed_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_thread_share_owner ON thread_share (owner_id);
CREATE INDEX IF NOT EXISTS idx_thread_share_thread ON thread_share (thread_id);
"""

SCOPES = ("authenticated", "public")


@dataclass
class Share:
    """A share link: who made it, what it points at, and when it lapses."""

    share_id: str
    thread_id: str
    owner_id: str
    title: str
    scope: str
    created_at: datetime
    expires_at: datetime | None
    revoked_at: datetime | None
    views: int
    last_viewed_at: datetime | None

    @property
    def active(self) -> bool:
        """Is this share still usable — neither revoked nor expired?"""
        if self.revoked_at is not None:
            return False
        return not (self.expires_at and self.expires_at <= datetime.now(UTC))

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, as the API returns it."""
        return {
            "share_id": self.share_id,
            "thread_id": self.thread_id,
            "title": self.title,
            "scope": self.scope,
            "active": self.active,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "revoked_at": self.revoked_at.isoformat() if self.revoked_at else None,
            # Surfaced so an owner can tell a link that was used from
            # one that was sent and ignored. A share nobody opened is
            # a share worth revoking.
            "views": self.views,
            "last_viewed_at": (
                self.last_viewed_at.isoformat() if self.last_viewed_at else None
            ),
        }


async def _connect() -> psycopg.AsyncConnection:
    return await psycopg.AsyncConnection.connect(
        settings.database_uri, row_factory=dict_row
    )


async def ensure_schema(conn: psycopg.AsyncConnection) -> None:
    """Create the shares table if it is not there yet."""
    await conn.execute(_SCHEMA)


def build_snapshot(values: dict[str, Any], title: str) -> dict[str, Any]:
    """Reduce thread state to the part that is safe to publish.

    Everything not explicitly kept is dropped. An allowlist is the only
    workable rule here: thread state grows new keys as middleware is
    added, and a denylist would publish each new one by default.
    """
    messages: list[dict[str, Any]] = []

    for raw in (values or {}).get("messages", [])[:MAX_SNAPSHOT_MESSAGES]:
        if not isinstance(raw, dict):
            continue
        role = str(raw.get("type") or raw.get("role") or "").lower()
        if role not in ("human", "ai", "user", "assistant"):
            # Tool and system messages carry retrieved documents, file
            # contents and prompt internals. None of that is what
            # somebody meant to share.
            continue

        content = raw.get("content")
        if isinstance(content, list):
            # Multimodal: keep the text parts, drop image and audio
            # blocks — those are the attachment bytes by another name.
            text = "\n".join(
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            text = str(content or "")

        if not text.strip():
            continue

        messages.append(
            {
                "role": "user" if role in ("human", "user") else "assistant",
                "content": text[:MAX_MESSAGE_CHARS],
            }
        )

    # Named, not included. A reader can see that the conversation was
    # grounded in a document without the document travelling with it.
    # `files` is a mapping of path to content, but a thread written
    # by another writer can hold a list of entries instead. Iterating
    # that yields dicts, and `.rsplit` on a dict is a 500 in place of
    # a snapshot.
    raw_files = (values or {}).get("files") or {}
    if isinstance(raw_files, dict):
        paths = list(raw_files)
    elif isinstance(raw_files, list):
        paths = [
            str(entry.get("path") or entry.get("name") or "")
            if isinstance(entry, dict)
            else str(entry)
            for entry in raw_files
        ]
    else:
        paths = []
    attachments = sorted(
        path.rsplit("/", 1)[-1] for path in paths if isinstance(path, str) and path
    )

    return {
        "title": title,
        "messages": messages,
        "attachment_names": attachments,
        "captured_at": datetime.now(UTC).isoformat(),
    }


async def create_share(
    *,
    thread_id: str,
    owner_id: str,
    title: str,
    snapshot: dict[str, Any],
    scope: str = "authenticated",
    expiry_days: int = DEFAULT_EXPIRY_DAYS,
) -> Share:
    """Share a conversation, replacing any previous share of it.

    Replacing rather than accumulating: a thread with four live links
    is four things to revoke, and an owner who clicks Share twice did
    not mean to create a second one.
    """
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")

    days = max(1, min(int(expiry_days), MAX_EXPIRY_DAYS))
    share_id = secrets.token_hex(_TOKEN_BYTES)
    expires_at = datetime.now(UTC) + timedelta(days=days)

    async with await _connect() as conn:
        await ensure_schema(conn)
        # The old link stops working the moment a new one is made.
        await conn.execute(
            "DELETE FROM thread_share WHERE thread_id = %s AND owner_id = %s",
            (thread_id, owner_id),
        )
        row = await (
            await conn.execute(
                """
                INSERT INTO thread_share
                    (share_id, thread_id, owner_id, title, snapshot, scope, expires_at)
                VALUES
                    (%(share_id)s, %(thread_id)s, %(owner_id)s, %(title)s,
                     %(snapshot)s, %(scope)s, %(expires_at)s)
                RETURNING *
                """,
                {
                    "share_id": share_id,
                    "thread_id": thread_id,
                    "owner_id": owner_id,
                    "title": title[:200],
                    "snapshot": Json(snapshot),
                    "scope": scope,
                    "expires_at": expires_at,
                },
            )
        ).fetchone()
        await conn.commit()

    logger.info(
        "Shared thread %s as %s (scope=%s, expires in %dd)",
        thread_id,
        share_id,
        scope,
        days,
    )
    return _to_share(row or {})


async def revoke_share(thread_id: str, owner_id: str) -> bool:
    """Stop a link working. Returns False when there was nothing to revoke."""
    async with await _connect() as conn:
        await ensure_schema(conn)
        result = await conn.execute(
            """
            UPDATE thread_share
               SET revoked_at = now()
             WHERE thread_id = %s AND owner_id = %s AND revoked_at IS NULL
            RETURNING share_id
            """,
            (thread_id, owner_id),
        )
        found = await result.fetchone()
        await conn.commit()
    return found is not None


async def get_share_for_thread(thread_id: str, owner_id: str) -> Share | None:
    """This owner's share for a thread, or None if they have not made one."""
    async with await _connect() as conn:
        await ensure_schema(conn)
        row = await (
            await conn.execute(
                "SELECT * FROM thread_share WHERE thread_id = %s AND owner_id = %s",
                (thread_id, owner_id),
            )
        ).fetchone()
    return _to_share(row) if row else None


async def list_shares(owner_id: str) -> list[Share]:
    """Every share this owner has created, newest first."""
    async with await _connect() as conn:
        await ensure_schema(conn)
        rows = await (
            await conn.execute(
                "SELECT * FROM thread_share WHERE owner_id = %s ORDER BY created_at DESC",
                (owner_id,),
            )
        ).fetchall()
    return [_to_share(row) for row in rows]


async def read_share(
    share_id: str, *, record_view: bool = True
) -> tuple[Share, dict[str, Any]] | None:
    """Fetch a share and its snapshot, recording the view by default.

    The view counter is incremented here rather than by the caller so
    that every path that reads a share is counted — an audit trail
    that depends on remembering to call it is not an audit trail.

    `record_view=False` is for a caller that has still to decide
    whether this reader may see it at all. An `authenticated` share
    rejected with 401 counted as a view, which inflated the number
    owners use to decide what to revoke.
    """
    async with await _connect() as conn:
        await ensure_schema(conn)
        row = await (
            await conn.execute(
                "SELECT * FROM thread_share WHERE share_id = %s", (share_id,)
            )
        ).fetchone()
        if row is None:
            return None

        share = _to_share(row)
        if share.active and record_view:
            await _count_view(conn, share_id)

    return share, dict(row.get("snapshot") or {})


async def _count_view(conn: Any, share_id: str) -> None:
    await conn.execute(
        """
        UPDATE thread_share
           SET views = views + 1, last_viewed_at = now()
         WHERE share_id = %s
        """,
        (share_id,),
    )
    await conn.commit()


async def record_share_view(share_id: str) -> None:
    """Count one view, after the reader has been allowed through."""
    async with await _connect() as conn:
        await ensure_schema(conn)
        await _count_view(conn, share_id)


def _to_share(row: dict[str, Any]) -> Share:
    return Share(
        share_id=row["share_id"],
        thread_id=row["thread_id"],
        owner_id=row["owner_id"],
        title=row.get("title") or "",
        scope=row.get("scope") or "authenticated",
        created_at=row["created_at"],
        expires_at=row.get("expires_at"),
        revoked_at=row.get("revoked_at"),
        views=int(row.get("views") or 0),
        last_viewed_at=row.get("last_viewed_at"),
    )
