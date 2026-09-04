"""Async Postgres repository for projects.

Projects allow users to organize conversations (threads) into named
groups. Each project belongs to a single user. Thread membership is
stored on Aegra ``thread.metadata_json`` (so ``/threads/search`` can list it)
and mirrored onto checkpoint metadata when those rows exist.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import psycopg
from psycopg.errors import UndefinedTable
from psycopg.rows import dict_row

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_TABLES_ENSURED = False
_OPTIONAL_INDEXES_ENSURED = False

T = TypeVar("T")

CREATE_PROJECTS_TABLE = """
CREATE TABLE IF NOT EXISTS projects (
    project_id          TEXT PRIMARY KEY,
    project_name        TEXT NOT NULL,
    project_description TEXT,
    username            TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_projects_username_name
    ON projects (username, project_name);
"""

CREATE_THREAD_PROJECT_INDEX = """
CREATE INDEX IF NOT EXISTS idx_thread_project_id
    ON thread ((metadata_json->>'project_id'))
    WHERE metadata_json->>'project_id' IS NOT NULL;
"""


class ProjectsStorageError(Exception):
    """Raised when a project operation fails."""


class ProjectNotFoundError(ProjectsStorageError):
    """Raised when a project row is missing for the given user."""


class ThreadNotFoundError(ProjectsStorageError):
    """Raised when assign matches no thread row."""


class ProjectsRepository:
    """Thin async wrapper around the projects table."""

    def __init__(self, database_uri: str) -> None:
        """Initialise with a Postgres connection URI."""
        self._uri = database_uri

    @classmethod
    async def ensure_tables(cls, database_uri: str | None = None) -> None:
        """Create tables/indexes if they don't exist yet."""
        global _TABLES_ENSURED, _OPTIONAL_INDEXES_ENSURED  # noqa: PLW0603
        if _TABLES_ENSURED and _OPTIONAL_INDEXES_ENSURED:
            return
        if not database_uri:
            return
        async with await psycopg.AsyncConnection.connect(database_uri) as conn:
            # Commit projects DDL in its own transaction. Optional indexes on
            # other tables must not abort this work — Postgres would otherwise
            # roll back CREATE TABLE while _TABLES_ENSURED stayed True.
            if not _TABLES_ENSURED:
                await conn.execute(CREATE_PROJECTS_TABLE)
                await conn.commit()
                _TABLES_ENSURED = True
                logger.info("Projects tables ensured")
            if _OPTIONAL_INDEXES_ENSURED:
                return
            try:
                await conn.execute(CREATE_THREAD_PROJECT_INDEX)
                await conn.commit()
                _OPTIONAL_INDEXES_ENSURED = True
            except UndefinedTable:
                await conn.rollback()
                logger.debug("Thread project index skipped (table may not exist yet)")

    async def _ensure(self) -> None:
        await self.ensure_tables(self._uri)

    # ── CRUD ──────────────────────────────────────────────────

    async def create_project(
        self, username: str, project_name: str, project_description: str | None = None
    ) -> dict[str, Any]:
        """Insert a project row and return it."""
        await self._ensure()
        project_id = str(uuid.uuid4())
        async with await psycopg.AsyncConnection.connect(
            self._uri, row_factory=dict_row
        ) as conn:
            cur = await conn.execute(
                "INSERT INTO projects (project_id, project_name, project_description, username) "
                "VALUES (%s, %s, %s, %s) RETURNING *",
                (project_id, project_name, project_description, username),
            )
            row = await cur.fetchone()
            await conn.commit()
        if row is None:
            raise ProjectsStorageError("Failed to create project")
        return dict(row)

    async def list_projects_with_thread_counts(
        self, username: str
    ) -> list[dict[str, Any]]:
        """Return the user's projects with conversation counts."""
        await self._ensure()
        async with await psycopg.AsyncConnection.connect(
            self._uri, row_factory=dict_row
        ) as conn:

            async def from_thread() -> list[dict[str, Any]]:
                cur = await conn.execute(
                    """
                    SELECT p.*,
                           COALESCE(tc.cnt, 0) AS thread_count
                    FROM projects p
                    LEFT JOIN (
                        SELECT metadata_json->>'project_id' AS pid,
                               COUNT(*) AS cnt
                        FROM thread
                        WHERE user_id = %s
                          AND metadata_json->>'project_id' IS NOT NULL
                        GROUP BY metadata_json->>'project_id'
                    ) tc ON tc.pid = p.project_id
                    WHERE p.username = %s
                    ORDER BY p.updated_at DESC
                    """,
                    (username, username),
                )
                return [dict(r) for r in await cur.fetchall()]

            rows = await _optional_table_op(conn, from_thread)
            if rows is not None:
                return rows
            cur = await conn.execute(
                """
                SELECT p.*, 0 AS thread_count
                FROM projects p
                WHERE p.username = %s
                ORDER BY p.updated_at DESC
                """,
                (username,),
            )
            return [dict(r) for r in await cur.fetchall()]

    async def update_project(
        self,
        project_id: str,
        username: str,
        project_name: str | None = None,
        project_description: str | None = None,
    ) -> dict[str, Any] | None:
        """Update name and/or description; return the row or None."""
        await self._ensure()
        sets: list[str] = ["updated_at = now()"]
        params: list[Any] = []
        if project_name is not None:
            sets.append("project_name = %s")
            params.append(project_name)
        if project_description is not None:
            sets.append("project_description = %s")
            params.append(project_description)
        params.extend([project_id, username])
        async with await psycopg.AsyncConnection.connect(
            self._uri, row_factory=dict_row
        ) as conn:
            cur = await conn.execute(
                f"UPDATE projects SET {', '.join(sets)} "  # noqa: S608
                "WHERE project_id = %s AND username = %s RETURNING *",
                tuple(params),
            )
            row = await cur.fetchone()
            await conn.commit()
        return dict(row) if row else None

    async def delete_project_with_threads(
        self,
        project_id: str,
        username: str,
        *,
        keep_threads: bool = False,
    ) -> list[str]:
        """Delete a project and either unassign or purge its conversations.

        All PostgreSQL writes run in one connection/transaction. Conversation
        rows are removed via the same cleanup as ``DELETE /threads/{id}``.
        MongoDB token-usage cleanup runs after commit as best-effort.
        """
        await self._ensure()
        from deep_agent.aegra.thread_cleanup import _delete_token_usage

        async with await psycopg.AsyncConnection.connect(self._uri) as conn:
            async with conn.transaction():
                await _lock_owned_project(conn, project_id, username)
                if keep_threads:
                    await _unassign_on_conn(conn, project_id, username)
                    deleted_ids: list[str] = []
                else:
                    deleted_ids = await _purge_member_threads(
                        conn, project_id, username
                    )

                result = await conn.execute(
                    "DELETE FROM projects WHERE project_id = %s AND username = %s",
                    (project_id, username),
                )
                if result.rowcount == 0:
                    raise ProjectNotFoundError(f"Project {project_id} not found")

        if not keep_threads:
            for thread_id in deleted_ids:
                try:
                    await _delete_token_usage(thread_id)
                except Exception:
                    logger.warning(
                        "token_usage_cleanup_failed_after_project_delete",
                        thread_id=thread_id,
                        exc_info=True,
                    )

        logger.info(
            "Deleted project %s (keep_threads=%s, threads=%d)",
            project_id,
            keep_threads,
            len(deleted_ids),
        )
        return deleted_ids

    async def verify_project_ownership(self, project_id: str, username: str) -> bool:
        """Return True if the user owns the given project."""
        await self._ensure()
        async with await psycopg.AsyncConnection.connect(self._uri) as conn:
            cur = await conn.execute(
                "SELECT 1 FROM projects WHERE project_id = %s AND username = %s",
                (project_id, username),
            )
            return (await cur.fetchone()) is not None

    async def count_project_threads(self, project_id: str, username: str) -> int:
        """Return how many conversations belong to the project."""
        await self._ensure()
        async with await psycopg.AsyncConnection.connect(self._uri) as conn:

            async def from_thread() -> int:
                cur = await conn.execute(
                    "SELECT COUNT(*) AS cnt FROM thread "
                    "WHERE metadata_json->>'project_id' = %s AND user_id = %s",
                    (project_id, username),
                )
                row = await cur.fetchone()
                return int(row[0]) if row else 0

            count = await _optional_table_op(conn, from_thread)
            if count is not None:
                return count
            return 0

    async def get_project_threads(
        self, project_id: str, username: str
    ) -> list[dict[str, Any]]:
        """Return threads belonging to a project, most recently updated first."""
        await self._ensure()
        async with await psycopg.AsyncConnection.connect(
            self._uri, row_factory=dict_row
        ) as conn:

            async def from_thread() -> list[dict[str, Any]]:
                cur = await conn.execute(
                    """
                    SELECT thread_id,
                           metadata_json->>'thread_name' AS thread_title,
                           metadata_json->>'project_id' AS project_id,
                           updated_at
                    FROM thread
                    WHERE metadata_json->>'project_id' = %s
                      AND user_id = %s
                    ORDER BY updated_at DESC
                    """,
                    (project_id, username),
                )
                return [dict(r) for r in await cur.fetchall()]

            rows = await _optional_table_op(conn, from_thread)
            if rows is not None:
                return rows
            return []

    async def assign_thread_to_project(
        self, thread_id: str, project_id: str | None, username: str
    ) -> None:
        """Update thread (and checkpoint) metadata to assign/unassign a project."""
        await self._ensure()
        async with await psycopg.AsyncConnection.connect(self._uri) as conn:
            if project_id:
                thread_sql = (
                    "UPDATE thread "
                    "SET metadata_json = jsonb_set("
                    "  COALESCE(metadata_json, '{}'::jsonb), '{project_id}', to_jsonb(%s::text)"
                    "), updated_at = now() "
                    "WHERE thread_id = %s AND user_id = %s"
                )
                thread_params: tuple[Any, ...] = (project_id, thread_id, username)
                owned_checkpoint_sql = (
                    "UPDATE checkpoints "
                    "SET metadata = jsonb_set("
                    "  COALESCE(metadata, '{}'::jsonb), '{project_id}', to_jsonb(%s::text)"
                    ") "
                    "WHERE thread_id = %s"
                )
                owned_checkpoint_params: tuple[Any, ...] = (project_id, thread_id)
            else:
                thread_sql = (
                    "UPDATE thread "
                    "SET metadata_json = COALESCE(metadata_json, '{}'::jsonb) - 'project_id', "
                    "    updated_at = now() "
                    "WHERE thread_id = %s AND user_id = %s"
                )
                thread_params = (thread_id, username)
                owned_checkpoint_sql = (
                    "UPDATE checkpoints "
                    "SET metadata = metadata - 'project_id' "
                    "WHERE thread_id = %s"
                )
                owned_checkpoint_params = (thread_id,)

            async def upd_thread() -> Any:
                return await conn.execute(thread_sql, thread_params)

            async def upd_owned_checkpoint() -> Any:
                return await conn.execute(owned_checkpoint_sql, owned_checkpoint_params)

            async with conn.transaction():
                if project_id:
                    await _lock_owned_project(conn, project_id, username)
                else:
                    current = await _peek_thread_project_id(conn, thread_id, username)
                    if current:
                        try:
                            await _lock_owned_project(conn, current, username)
                        except ProjectNotFoundError:
                            pass
                thread_cur = await _optional_table_op(conn, upd_thread)
                if thread_cur is None or _rowcount(thread_cur) == 0:
                    raise ThreadNotFoundError(f"Thread {thread_id} not found")
                await _optional_table_op(conn, upd_owned_checkpoint)

    async def unassign_all_threads(self, project_id: str, username: str) -> list[str]:
        """Remove project membership from every thread in the project.

        Conversations themselves are kept. Returns the unassigned thread IDs.
        """
        await self._ensure()
        async with await psycopg.AsyncConnection.connect(self._uri) as conn:
            async with conn.transaction():
                await _lock_owned_project(conn, project_id, username)
                thread_ids = await _unassign_on_conn(conn, project_id, username)

        logger.info(
            "Unassigned %d thread(s) from project %s", len(thread_ids), project_id
        )
        return thread_ids


async def _optional_table_op(
    conn: Any, operation: Callable[[], Awaitable[T]]
) -> T | None:
    """Run *operation* in a savepoint; return None if the relation is missing.

    Other errors propagate so a successful sibling write is not committed
    as success when a real dual-write failure occurs.
    """
    try:
        async with conn.transaction():
            return await operation()
    except UndefinedTable:
        return None


async def _lock_owned_project(conn: Any, project_id: str, username: str) -> None:
    """Lock the project row for this user, or raise if it is missing."""
    cur = await conn.execute(
        "SELECT 1 FROM projects WHERE project_id = %s AND username = %s FOR UPDATE",
        (project_id, username),
    )
    if await cur.fetchone() is None:
        raise ProjectNotFoundError(f"Project {project_id} not found")


async def _peek_thread_project_id(
    conn: Any, thread_id: str, username: str
) -> str | None:
    """Return the thread's current project_id, or None if unknown."""

    async def select() -> str | None:
        cur = await conn.execute(
            "SELECT metadata_json->>'project_id' FROM thread "
            "WHERE thread_id = %s AND user_id = %s",
            (thread_id, username),
        )
        row = await cur.fetchone()
        if not row or row[0] is None or row[0] == "":
            return None
        return str(row[0])

    return await _optional_table_op(conn, select)


def _rowcount(cursor: Any) -> int:
    """Return a non-negative UPDATE/DELETE rowcount, or 0 if unknown."""
    if cursor is None:
        return 0
    raw = getattr(cursor, "rowcount", 0) or 0
    try:
        count = int(raw)
    except (TypeError, ValueError):
        return 0
    return count if count > 0 else 0


async def _purge_member_threads(conn: Any, project_id: str, username: str) -> list[str]:
    """Delete conversations currently in the project; return their thread IDs.

    Membership is decided at DELETE time (``WHERE project_id``), not from a
    list taken earlier, so a concurrent move-out is not purged.
    """
    from deep_agent.aegra.thread_cleanup import _delete_pg_data

    async def delete_thread_rows() -> list[str]:
        cur = await conn.execute(
            "DELETE FROM thread "
            "WHERE metadata_json->>'project_id' = %s AND user_id = %s "
            "RETURNING thread_id",
            (project_id, username),
        )
        return [_thread_id_from_row(r) for r in await cur.fetchall()]

    thread_ids = await _optional_table_op(conn, delete_thread_rows) or []
    for thread_id in thread_ids:
        await _delete_pg_data(thread_id, username, conn)
    return list(thread_ids)


async def _unassign_on_conn(conn: Any, project_id: str, username: str) -> list[str]:
    """Strip ``project_id`` from thread and checkpoint metadata on *conn*."""

    async def update_thread() -> list[str]:
        cur = await conn.execute(
            "UPDATE thread "
            "SET metadata_json = COALESCE(metadata_json, '{}'::jsonb) - 'project_id', "
            "    updated_at = now() "
            "WHERE metadata_json->>'project_id' = %s AND user_id = %s "
            "RETURNING thread_id",
            (project_id, username),
        )
        return [_thread_id_from_row(r) for r in await cur.fetchall()]

    thread_ids = await _optional_table_op(conn, update_thread) or []

    async def update_checkpoints() -> None:
        await conn.execute(
            "UPDATE checkpoints "
            "SET metadata = metadata - 'project_id' "
            "WHERE thread_id = ANY(%s)",
            (thread_ids,),
        )

    if thread_ids:
        await _optional_table_op(conn, update_checkpoints)
    return list(thread_ids)


def _thread_id_from_row(row: Any) -> str:
    """Return thread_id from a tuple or mapping row."""
    if isinstance(row, dict):
        return str(row["thread_id"])
    return str(row[0])
