"""Async Postgres repository for message feedback."""

from __future__ import annotations

from typing import Any, Literal

import psycopg
from psycopg.rows import dict_row

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_TABLE_ENSURED = False

CREATE_FEEDBACK_TABLE = """
CREATE TABLE IF NOT EXISTS message_feedback (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id   TEXT NOT NULL,
    message_id  TEXT NOT NULL,
    user_id     TEXT NOT NULL DEFAULT 'anonymous',
    feedback    TEXT NOT NULL CHECK (feedback IN ('up', 'down')),
    trace_id    TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (thread_id, message_id, user_id)
);
CREATE INDEX IF NOT EXISTS idx_feedback_thread ON message_feedback (thread_id);
"""


class FeedbackRepository:
    """Thin async wrapper around the message_feedback table."""

    def __init__(self, database_uri: str) -> None:
        """Initialize with a Postgres connection URI."""
        self._uri = database_uri

    async def ensure_table(self) -> None:
        """Create message_feedback table if it does not already exist (lazy, once)."""
        global _TABLE_ENSURED  # noqa: PLW0603
        if _TABLE_ENSURED:
            return
        async with await psycopg.AsyncConnection.connect(self._uri) as conn:
            await conn.execute(CREATE_FEEDBACK_TABLE)
            await conn.commit()
        _TABLE_ENSURED = True
        logger.info("message_feedback table ensured")

    async def upsert_feedback(
        self,
        thread_id: str,
        message_id: str,
        user_id: str,
        feedback: Literal["up", "down"],
        trace_id: str | None = None,
    ) -> None:
        """Insert or update feedback for a message (per thread and user)."""
        await self.ensure_table()
        uid = user_id if user_id else "anonymous"
        async with await psycopg.AsyncConnection.connect(self._uri) as conn:
            await conn.execute(
                """
                INSERT INTO message_feedback (
                    thread_id, message_id, user_id, feedback, trace_id, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, now())
                ON CONFLICT (thread_id, message_id, user_id)
                DO UPDATE SET
                    feedback = EXCLUDED.feedback,
                    trace_id = EXCLUDED.trace_id,
                    updated_at = now()
                """,
                (thread_id, message_id, uid, feedback, trace_id),
            )
            await conn.commit()

    async def delete_feedback(
        self,
        thread_id: str,
        message_id: str,
        user_id: str,
    ) -> bool:
        """Remove feedback row (un-vote). Returns True if a row was deleted."""
        await self.ensure_table()
        uid = user_id if user_id else "anonymous"
        async with await psycopg.AsyncConnection.connect(self._uri) as conn:
            cur = await conn.execute(
                """
                DELETE FROM message_feedback
                WHERE thread_id = %s AND message_id = %s AND user_id = %s
                """,
                (thread_id, message_id, uid),
            )
            await conn.commit()
            return bool(cur.rowcount > 0)

    async def list_feedback(
        self,
        thread_id: str,
        user_id: str,
    ) -> list[dict[str, Any]]:
        """Return feedback entries for the thread and user as ``{message_id, feedback}``."""
        await self.ensure_table()
        uid = user_id if user_id else "anonymous"
        async with await psycopg.AsyncConnection.connect(
            self._uri, row_factory=dict_row
        ) as conn:
            cur = await conn.execute(
                """
                SELECT message_id, feedback
                FROM message_feedback
                WHERE thread_id = %s AND user_id = %s
                ORDER BY updated_at ASC
                """,
                (thread_id, uid),
            )
            rows = await cur.fetchall()
            return [
                {"message_id": str(r["message_id"]), "feedback": r["feedback"]}
                for r in rows
            ]
