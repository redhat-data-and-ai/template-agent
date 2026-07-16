"""Async Postgres repository for task audit trail.

Persists every task status change to PostgreSQL for audit purposes.
Redis remains the primary store for speed; Postgres is the durable
record that survives Redis TTL expiry.
"""

from __future__ import annotations

import json
from typing import Any

import psycopg
from psycopg.rows import dict_row

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_TABLES_ENSURED = False

CREATE_TASKS_TABLE = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id     TEXT PRIMARY KEY,
    task_name   TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'queued',
    payload     JSONB NOT NULL DEFAULT '{}',
    result      TEXT,
    error       TEXT,
    thread_id   TEXT,
    user_id     TEXT,
    delivered   BOOLEAN NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_tasks_user_id ON tasks (user_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks (status);
CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks (created_at DESC);
"""


class TaskRepository:
    """Async Postgres repository for task audit records."""

    def __init__(self, database_uri: str) -> None:
        """Initialize with a PostgreSQL connection URI."""
        self._database_uri = database_uri

    async def ensure_table(self) -> None:
        """Create the tasks table if it doesn't exist."""
        global _TABLES_ENSURED  # noqa: PLW0603
        if _TABLES_ENSURED:
            return
        try:
            async with await psycopg.AsyncConnection.connect(
                self._database_uri
            ) as conn:
                await conn.execute(CREATE_TASKS_TABLE)
                await conn.commit()
            _TABLES_ENSURED = True
            logger.info("tasks audit table ensured")
        except Exception:
            logger.warning("failed to create tasks table", exc_info=True)

    async def insert_task(
        self,
        task_id: str,
        task_name: str,
        payload: dict[str, Any],
        thread_id: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """Insert a new task record."""
        try:
            async with await psycopg.AsyncConnection.connect(
                self._database_uri
            ) as conn:
                await conn.execute(
                    """INSERT INTO tasks (task_id, task_name, status, payload, thread_id, user_id)
                       VALUES (%s, %s, 'queued', %s, %s, %s)
                       ON CONFLICT (task_id) DO NOTHING""",
                    (task_id, task_name, json.dumps(payload, default=str), thread_id, user_id),
                )
                await conn.commit()
        except Exception:
            logger.warning("failed to insert task audit record", task_id=task_id, exc_info=True)

    async def update_status(
        self,
        task_id: str,
        status: str,
        result: str | None = None,
        error: str | None = None,
    ) -> None:
        """Update task status in the audit table."""
        try:
            completed_at = "now()" if status in ("completed", "failed") else None
            async with await psycopg.AsyncConnection.connect(
                self._database_uri
            ) as conn:
                if completed_at:
                    await conn.execute(
                        """UPDATE tasks SET status = %s, result = %s, error = %s,
                           updated_at = now(), completed_at = now()
                           WHERE task_id = %s""",
                        (status, result, error, task_id),
                    )
                else:
                    await conn.execute(
                        """UPDATE tasks SET status = %s, result = %s, error = %s,
                           updated_at = now() WHERE task_id = %s""",
                        (status, result, error, task_id),
                    )
                await conn.commit()
        except Exception:
            logger.warning("failed to update task audit record", task_id=task_id, exc_info=True)

    async def mark_delivered(self, task_id: str) -> None:
        """Mark task as delivered in the audit table."""
        try:
            async with await psycopg.AsyncConnection.connect(
                self._database_uri
            ) as conn:
                await conn.execute(
                    "UPDATE tasks SET delivered = TRUE, updated_at = now() WHERE task_id = %s",
                    (task_id,),
                )
                await conn.commit()
        except Exception:
            logger.warning("failed to mark task delivered", task_id=task_id, exc_info=True)

    async def get_task_history(
        self,
        user_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Query task history for audit purposes."""
        try:
            async with await psycopg.AsyncConnection.connect(
                self._database_uri, row_factory=dict_row
            ) as conn:
                conditions = []
                params: list[Any] = []
                if user_id:
                    conditions.append("user_id = %s")
                    params.append(user_id)
                if status:
                    conditions.append("status = %s")
                    params.append(status)

                where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
                params.append(limit)

                rows = await conn.execute(
                    f"SELECT * FROM tasks {where} ORDER BY created_at DESC LIMIT %s",
                    params,
                )
                return [dict(r) for r in await rows.fetchall()]
        except Exception:
            logger.warning("failed to query task history", exc_info=True)
            return []
