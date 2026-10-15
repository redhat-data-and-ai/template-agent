"""The `thread_search` table: build it, keep it current, query it.

Why a derived table at all. Message text is msgpack inside
`checkpoint_blobs`, so Postgres cannot search it — `LIKE` over a bytea
column matches nothing useful. Decoding every blob at query time was the
alternative, and it is O(threads) msgpack parses per keystroke.

Why it refreshes lazily rather than on write. Writing an index entry when
a run finishes would mean hooking `finalize_run`, which lives in aegra
and moves between versions; §19 is a long story about what that costs.
The read path instead reindexes only the threads that changed since they
were last indexed, for the one user doing the search. That is a handful
of threads in practice, and it is self-healing: an index that is dropped,
corrupted or never built converges on first use.

The table is created with IF NOT EXISTS rather than an Alembic revision,
deliberately — it is a cache, it is derived from data we still have, and
it must not join a migration chain owned by a dependency.
"""

from __future__ import annotations

from typing import Any

import psycopg

from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

# Bounds a single refresh so one user with a huge history cannot turn a
# search into a minutes-long request.
MAX_THREADS_PER_REFRESH = 200
# Keeps one pathological thread from dominating the index.
MAX_BODY_CHARS = 100_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS thread_search (
    thread_id         TEXT PRIMARY KEY,
    user_id           TEXT NOT NULL,
    title             TEXT NOT NULL DEFAULT '',
    body              TEXT NOT NULL DEFAULT '',
    -- Generated, so the vector can never drift from the text it indexes.
    search_vector     tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(body, '')), 'B')
    ) STORED,
    source_updated_at TIMESTAMPTZ,
    indexed_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_thread_search_vector
    ON thread_search USING GIN (search_vector);
CREATE INDEX IF NOT EXISTS idx_thread_search_user
    ON thread_search (user_id);
"""


def _ext_hook(code: int, data: bytes) -> Any:
    """Tolerate msgpack ext types we do not understand.

    Gemini thought signatures are binary ext payloads; raising on one
    would lose the whole conversation from the index over a field we do
    not want to search anyway.
    """
    import msgpack

    try:
        return msgpack.unpackb(data, raw=False, ext_hook=_ext_hook)
    except Exception:
        return data


def extract_text(blob: bytes) -> str:
    """Pull searchable text out of one checkpoint blob.

    Blob format is LangGraph's serde: a list of
    ``[module_path, class_name, message_dict]``. Only human and AI text
    is indexed — tool payloads are mostly JSON and searching them
    returns noise for every query that happens to share a key name.
    """
    import msgpack

    try:
        items = msgpack.unpackb(bytes(blob), raw=False, ext_hook=_ext_hook)
    except Exception:
        return ""
    if not isinstance(items, (list, tuple)):
        return ""

    parts: list[str] = []
    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        message = item[2]
        if not isinstance(message, dict):
            continue
        if message.get("type") not in ("human", "ai"):
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content)
        elif isinstance(content, list):
            # Multimodal content: a list of blocks, only some textual.
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])

    return "\n".join(parts)[:MAX_BODY_CHARS]


async def ensure_schema(conn: psycopg.AsyncConnection) -> None:
    """Create the search tables if they are not there yet."""
    await conn.execute(_SCHEMA)


async def _stale_threads(
    conn: psycopg.AsyncConnection, user_id: str
) -> list[tuple[str, Any]]:
    """Threads of this user that the index has not caught up with."""
    rows = await conn.execute(
        """
        SELECT t.thread_id, t.updated_at
        FROM thread t
        LEFT JOIN thread_search s ON s.thread_id = t.thread_id
        WHERE t.user_id = %s
          AND (s.thread_id IS NULL OR s.source_updated_at IS DISTINCT FROM t.updated_at)
        ORDER BY t.updated_at DESC
        LIMIT %s
        """,
        (user_id, MAX_THREADS_PER_REFRESH),
    )
    return [(row[0], row[1]) for row in await rows.fetchall()]


async def _thread_body(conn: psycopg.AsyncConnection, thread_id: str) -> str:
    """Current conversation text for one thread.

    Reads both checkpoint tables, because which one holds the messages
    depends on the LangGraph version and on how big the channel value
    got. On this deployment every message is in `checkpoint_writes` and
    `checkpoint_blobs` has no `messages` channel at all — a layout that
    only turned up by querying a live database, and one that would have
    made a blobs-only index silently return nothing.

    `checkpoint_writes` holds per-step deltas rather than the whole
    list, so this unions every step. For ranking that is right: the
    index needs the words that were said, not their order, and a word
    repeated across steps is not double-counted by tsvector.

    Root namespace only — subagent namespaces repeat the orchestrator's
    text and would multiply every match.
    """
    chunks: list[str] = []

    blobs = await conn.execute(
        """
        SELECT DISTINCT ON (channel) blob
        FROM checkpoint_blobs
        WHERE thread_id = %s AND checkpoint_ns = '' AND channel = 'messages'
        ORDER BY channel, version DESC
        """,
        (thread_id,),
    )
    chunks += [extract_text(blob) for (blob,) in await blobs.fetchall() if blob]

    writes = await conn.execute(
        """
        SELECT blob
        FROM checkpoint_writes
        WHERE thread_id = %s AND checkpoint_ns = '' AND channel = 'messages'
        """,
        (thread_id,),
    )
    chunks += [extract_text(blob) for (blob,) in await writes.fetchall() if blob]

    seen: set[str] = set()
    unique: list[str] = []
    for chunk in chunks:
        if chunk and chunk not in seen:
            seen.add(chunk)
            unique.append(chunk)
    return "\n".join(unique)[:MAX_BODY_CHARS]


def derive_title(body: str) -> str:
    """First line of the first message, near enough to a subject line."""
    for line in body.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:200]
    return ""


async def refresh_user(user_id: str) -> int:
    """Bring this user's index entries up to date. Returns rows written."""
    written = 0
    async with await psycopg.AsyncConnection.connect(settings.database_uri) as conn:
        await ensure_schema(conn)
        stale = await _stale_threads(conn, user_id)

        for thread_id, updated_at in stale:
            body = await _thread_body(conn, thread_id)
            await conn.execute(
                """
                INSERT INTO thread_search
                    (thread_id, user_id, title, body, source_updated_at, indexed_at)
                VALUES (%s, %s, %s, %s, %s, now())
                ON CONFLICT (thread_id) DO UPDATE SET
                    user_id = EXCLUDED.user_id,
                    title = EXCLUDED.title,
                    body = EXCLUDED.body,
                    source_updated_at = EXCLUDED.source_updated_at,
                    indexed_at = now()
                """,
                (thread_id, user_id, derive_title(body), body, updated_at),
            )
            written += 1

        await conn.commit()

    if written:
        logger.info("Reindexed %d thread(s) for search", written)
    return written


async def search(
    user_id: str, query: str, *, limit: int = 20, offset: int = 0
) -> list[dict[str, Any]]:
    """Ranked matches for one user.

    `websearch_to_tsquery` rather than `plainto_tsquery`: users type
    quoted phrases and `-exclusions` into a search box whatever the
    parser supports, and websearch understands both instead of erroring
    or matching them literally.
    """
    if not query.strip():
        return []

    async with await psycopg.AsyncConnection.connect(settings.database_uri) as conn:
        await ensure_schema(conn)
        rows = await conn.execute(
            """
            SELECT
                s.thread_id,
                s.title,
                ts_rank(s.search_vector, websearch_to_tsquery('english', %(q)s)) AS rank,
                ts_headline(
                    'english', s.body, websearch_to_tsquery('english', %(q)s),
                    'MaxFragments=2, MinWords=5, MaxWords=18, StartSel=<<, StopSel=>>'
                ) AS snippet,
                s.source_updated_at
            FROM thread_search s
            WHERE s.user_id = %(user)s
              AND s.search_vector @@ websearch_to_tsquery('english', %(q)s)
            ORDER BY rank DESC, s.source_updated_at DESC NULLS LAST
            LIMIT %(limit)s OFFSET %(offset)s
            """,
            {
                "q": query,
                "user": user_id,
                "limit": max(1, min(limit, 100)),
                "offset": max(0, offset),
            },
        )
        return [
            {
                "thread_id": row[0],
                "title": row[1],
                "rank": float(row[2]),
                "snippet": row[3],
                "updated_at": row[4].isoformat() if row[4] else None,
            }
            for row in await rows.fetchall()
        ]
