"""Unit tests for ProjectsRepository (mocked DB)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_agent.src.projects import (
    ProjectNotFoundError,
    ProjectsRepository,
    ThreadNotFoundError,
)


class _TxnCM:
    """Async context manager standing in for ``conn.transaction()``."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.fixture(autouse=True)
def _reset_tables_flag():
    import deep_agent.src.projects as repo_mod

    repo_mod._TABLES_ENSURED = False
    repo_mod._OPTIONAL_INDEXES_ENSURED = False
    yield
    repo_mod._TABLES_ENSURED = False
    repo_mod._OPTIONAL_INDEXES_ENSURED = False


@pytest.fixture
def mock_conn():
    conn = AsyncMock()
    cursor = AsyncMock()
    cursor.fetchall = AsyncMock(return_value=[])
    cursor.fetchone = AsyncMock(return_value=None)
    cursor.rowcount = 0
    conn.execute = AsyncMock(return_value=cursor)
    conn.commit = AsyncMock()
    conn.rollback = AsyncMock()
    conn.transaction = MagicMock(return_value=_TxnCM())
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=False)
    conn._cursor = cursor
    return conn


@pytest.fixture
def repo():
    return ProjectsRepository("postgresql://test:test@localhost/testdb")


def _own_project(mock_conn):
    mock_conn._cursor.fetchone = AsyncMock(return_value=(1,))


class TestEnsureTables:
    @pytest.mark.asyncio
    async def test_creates_tables_once(self, repo, mock_conn):
        with patch(
            "deep_agent.src.projects.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            await repo.ensure_tables(repo._uri)
            assert mock_conn.execute.call_count >= 1
            assert mock_conn.commit.await_count >= 1
            thread_index_sql = next(
                str(call.args[0])
                for call in mock_conn.execute.await_args_list
                if "idx_thread_project_id" in str(call.args[0])
            )
            assert "metadata_json" in thread_index_sql

    @pytest.mark.asyncio
    async def test_skips_if_already_ensured(self, repo, mock_conn):
        import deep_agent.src.projects as repo_mod

        repo_mod._TABLES_ENSURED = True
        repo_mod._OPTIONAL_INDEXES_ENSURED = True
        with patch(
            "deep_agent.src.projects.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            await repo.ensure_tables(repo._uri)
            mock_conn.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_without_uri(self, repo, mock_conn):
        with patch(
            "deep_agent.src.projects.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            await ProjectsRepository.ensure_tables(None)
            mock_conn.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_optional_index_failure_commits_projects_table_first(
        self, repo, mock_conn
    ):
        """Optional thread index must not roll back projects DDL.

        Postgres aborts the whole transaction if CREATE INDEX on a missing
        table fails. Committing the projects table first keeps GET /projects
        from hitting UndefinedTable after a swallowed index error.
        """
        import deep_agent.src.projects as repo_mod

        from psycopg.errors import UndefinedTable

        events: list[str] = []

        async def execute(query, *_args, **_kwargs):
            sql = str(query)
            events.append(f"execute:{sql}")
            if "idx_thread_project_id" in sql:
                raise UndefinedTable("thread")
            return mock_conn._cursor

        async def commit():
            events.append("commit")

        async def rollback():
            events.append("rollback")

        mock_conn.execute = AsyncMock(side_effect=execute)
        mock_conn.commit = AsyncMock(side_effect=commit)
        mock_conn.rollback = AsyncMock(side_effect=rollback)

        with patch(
            "deep_agent.src.projects.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            await repo.ensure_tables(repo._uri)

        create_i = next(
            i
            for i, event in enumerate(events)
            if event.startswith("execute:")
            and "CREATE TABLE IF NOT EXISTS projects" in event
        )
        first_commit_i = events.index("commit")
        thread_i = next(
            i for i, event in enumerate(events) if "idx_thread_project_id" in event
        )

        assert create_i < first_commit_i < thread_i
        assert "rollback" in events
        assert repo_mod._TABLES_ENSURED is True
        assert repo_mod._OPTIONAL_INDEXES_ENSURED is False

    @pytest.mark.asyncio
    async def test_retries_optional_indexes_after_tables_exist(self, repo, mock_conn):
        import deep_agent.src.projects as repo_mod

        repo_mod._TABLES_ENSURED = True
        repo_mod._OPTIONAL_INDEXES_ENSURED = False
        with patch(
            "deep_agent.src.projects.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            await repo.ensure_tables(repo._uri)
        sqls = [str(call.args[0]) for call in mock_conn.execute.await_args_list]
        assert any("idx_thread_project_id" in s for s in sqls)
        assert repo_mod._OPTIONAL_INDEXES_ENSURED is True


class TestCreateProject:
    @pytest.mark.asyncio
    async def test_inserts_and_returns_row(self, repo, mock_conn):
        import deep_agent.src.projects as repo_mod

        repo_mod._TABLES_ENSURED = True
        repo_mod._OPTIONAL_INDEXES_ENSURED = True
        mock_conn._cursor.fetchone = AsyncMock(
            return_value={
                "project_id": "p1",
                "project_name": "Alpha",
                "project_description": None,
                "username": "u1",
                "created_at": "2026-01-01",
                "updated_at": "2026-01-01",
            }
        )
        with patch(
            "deep_agent.src.projects.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            row = await repo.create_project("u1", "Alpha")
        assert row["project_name"] == "Alpha"
        mock_conn.commit.assert_awaited()


class TestGetAndList:
    @pytest.mark.asyncio
    async def test_list_projects_with_counts(self, repo, mock_conn):
        import deep_agent.src.projects as repo_mod

        repo_mod._TABLES_ENSURED = True
        repo_mod._OPTIONAL_INDEXES_ENSURED = True
        mock_conn._cursor.fetchall = AsyncMock(
            return_value=[
                {
                    "project_id": "p1",
                    "project_name": "Alpha",
                    "username": "u1",
                    "thread_count": 2,
                    "project_description": None,
                    "created_at": "t",
                    "updated_at": "t",
                }
            ]
        )
        with patch(
            "deep_agent.src.projects.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            rows = await repo.list_projects_with_thread_counts("u1")
        assert len(rows) == 1
        assert rows[0]["thread_count"] == 2


class TestUpdateAndDelete:
    @pytest.mark.asyncio
    async def test_update_project(self, repo, mock_conn):
        import deep_agent.src.projects as repo_mod

        repo_mod._TABLES_ENSURED = True
        repo_mod._OPTIONAL_INDEXES_ENSURED = True
        mock_conn._cursor.fetchone = AsyncMock(
            return_value={
                "project_id": "p1",
                "project_name": "Beta",
                "username": "u1",
                "project_description": "d",
                "created_at": "t",
                "updated_at": "t",
            }
        )
        with patch(
            "deep_agent.src.projects.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            row = await repo.update_project("p1", "u1", project_name="Beta")
        assert row["project_name"] == "Beta"

    @pytest.mark.asyncio
    async def test_delete_missing_raises(self, repo, mock_conn):
        import deep_agent.src.projects as repo_mod

        repo_mod._TABLES_ENSURED = True
        repo_mod._OPTIONAL_INDEXES_ENSURED = True
        mock_conn._cursor.rowcount = 0
        with patch(
            "deep_agent.src.projects.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            with pytest.raises(ProjectNotFoundError):
                await repo.delete_project_with_threads("p1", "u1")

    @pytest.mark.asyncio
    async def test_delete_returns_thread_ids(self, repo, mock_conn):
        import deep_agent.src.projects as repo_mod

        repo_mod._TABLES_ENSURED = True
        repo_mod._OPTIONAL_INDEXES_ENSURED = True
        _own_project(mock_conn)
        mock_conn._cursor.fetchall = AsyncMock(return_value=[("t1",), ("t2",)])
        mock_conn._cursor.rowcount = 1
        with patch(
            "deep_agent.src.projects.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            ids = await repo.delete_project_with_threads("p1", "u1")
        assert ids == ["t1", "t2"]


class TestAssignAndOwnership:
    @pytest.mark.asyncio
    async def test_verify_ownership_true(self, repo, mock_conn):
        import deep_agent.src.projects as repo_mod

        repo_mod._TABLES_ENSURED = True
        repo_mod._OPTIONAL_INDEXES_ENSURED = True
        mock_conn._cursor.fetchone = AsyncMock(return_value=(1,))
        with patch(
            "deep_agent.src.projects.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            assert await repo.verify_project_ownership("p1", "u1") is True

    @pytest.mark.asyncio
    async def test_assign_and_unassign(self, repo, mock_conn):
        import deep_agent.src.projects as repo_mod

        repo_mod._TABLES_ENSURED = True
        repo_mod._OPTIONAL_INDEXES_ENSURED = True
        _own_project(mock_conn)
        mock_conn._cursor.rowcount = 1
        with patch(
            "deep_agent.src.projects.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            await repo.assign_thread_to_project("th1", "p1", "u1")
            await repo.assign_thread_to_project("th1", None, "u1")
        assert mock_conn.transaction.call_count >= 2
        mock_conn.commit.assert_not_awaited()
        sqls = [str(call.args[0]) for call in mock_conn.execute.await_args_list]
        assert any("FOR UPDATE" in s for s in sqls)

    @pytest.mark.asyncio
    async def test_assign_checkpoint_error_does_not_commit(self, repo, mock_conn):
        import deep_agent.src.projects as repo_mod

        repo_mod._TABLES_ENSURED = True
        repo_mod._OPTIONAL_INDEXES_ENSURED = True
        _own_project(mock_conn)
        mock_conn._cursor.rowcount = 1

        async def execute(query, *_args, **_kwargs):
            if "UPDATE checkpoints" in str(query):
                raise RuntimeError("deadlock")
            return mock_conn._cursor

        mock_conn.execute = AsyncMock(side_effect=execute)
        with patch(
            "deep_agent.src.projects.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            with pytest.raises(RuntimeError, match="deadlock"):
                await repo.assign_thread_to_project("th1", "p1", "u1")
        mock_conn.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_assign_zero_rows_raises_thread_not_found(self, repo, mock_conn):
        import deep_agent.src.projects as repo_mod

        repo_mod._TABLES_ENSURED = True
        repo_mod._OPTIONAL_INDEXES_ENSURED = True
        _own_project(mock_conn)
        mock_conn._cursor.rowcount = 0
        with patch(
            "deep_agent.src.projects.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            with pytest.raises(ThreadNotFoundError):
                await repo.assign_thread_to_project("th1", "p1", "u1")

    @pytest.mark.asyncio
    async def test_assign_raises_when_thread_row_missing_even_if_checkpoint_matches(
        self, repo, mock_conn
    ):
        import deep_agent.src.projects as repo_mod

        repo_mod._TABLES_ENSURED = True
        repo_mod._OPTIONAL_INDEXES_ENSURED = True
        _own_project(mock_conn)

        async def execute(query, *_args, **_kwargs):
            sql = str(query)
            cur = AsyncMock()
            cur.fetchone = AsyncMock(return_value=(1,))
            cur.fetchall = AsyncMock(return_value=[])
            cur.rowcount = 0 if "UPDATE thread" in sql else 1
            return cur

        mock_conn.execute = AsyncMock(side_effect=execute)
        with patch(
            "deep_agent.src.projects.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            with pytest.raises(ThreadNotFoundError):
                await repo.assign_thread_to_project("th1", "p1", "u1")
        sqls = [str(call.args[0]) for call in mock_conn.execute.await_args_list]
        assert not any("user_identity" in s for s in sqls)

    @pytest.mark.asyncio
    async def test_assign_uses_aegra_metadata_json_column(self, repo, mock_conn):
        """Aegra's thread table stores JSON in metadata_json, not metadata."""
        import deep_agent.src.projects as repo_mod

        repo_mod._TABLES_ENSURED = True
        repo_mod._OPTIONAL_INDEXES_ENSURED = True
        _own_project(mock_conn)
        mock_conn._cursor.rowcount = 1
        with patch(
            "deep_agent.src.projects.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            await repo.assign_thread_to_project("th1", "p1", "u1")

        sqls = [str(call.args[0]) for call in mock_conn.execute.await_args_list]
        thread_sql = next(s for s in sqls if "UPDATE thread" in s)
        checkpoint_sql = next(s for s in sqls if "UPDATE checkpoints" in s)
        assert "SET metadata_json" in thread_sql
        assert "COALESCE(metadata_json" in thread_sql
        assert "SET metadata =" not in thread_sql
        assert "SET metadata =" in checkpoint_sql or "jsonb_set(" in checkpoint_sql
        assert "user_identity" not in checkpoint_sql
        assert "WHERE thread_id = %s" in checkpoint_sql

    @pytest.mark.asyncio
    async def test_unassign_uses_aegra_metadata_json_column(self, repo, mock_conn):
        import deep_agent.src.projects as repo_mod

        repo_mod._TABLES_ENSURED = True
        repo_mod._OPTIONAL_INDEXES_ENSURED = True
        mock_conn._cursor.rowcount = 1
        with patch(
            "deep_agent.src.projects.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            await repo.assign_thread_to_project("th1", None, "u1")

        sqls = [str(call.args[0]) for call in mock_conn.execute.await_args_list]
        thread_sql = next(s for s in sqls if "UPDATE thread" in s)
        assert "metadata_json" in thread_sql
        assert "SET metadata =" not in thread_sql

    @pytest.mark.asyncio
    async def test_unassign_all_strips_project_id_from_thread_metadata_json(
        self, repo, mock_conn
    ):
        import deep_agent.src.projects as repo_mod

        repo_mod._TABLES_ENSURED = True
        repo_mod._OPTIONAL_INDEXES_ENSURED = True
        _own_project(mock_conn)
        mock_conn._cursor.fetchall = AsyncMock(return_value=[("t1",), ("t2",)])
        with patch(
            "deep_agent.src.projects.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            ids = await repo.unassign_all_threads("p1", "u1")

        assert ids == ["t1", "t2"]
        sqls = [str(call.args[0]) for call in mock_conn.execute.await_args_list]
        thread_sql = next(s for s in sqls if "UPDATE thread" in s)
        assert "metadata_json" in thread_sql
        assert "project_id" in thread_sql
        assert mock_conn.transaction.called
        mock_conn.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_thread_queries_use_metadata_json(self, repo, mock_conn):
        import deep_agent.src.projects as repo_mod

        repo_mod._TABLES_ENSURED = True
        repo_mod._OPTIONAL_INDEXES_ENSURED = True
        _own_project(mock_conn)
        mock_conn._cursor.rowcount = 1
        with patch(
            "deep_agent.src.projects.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            await repo.list_projects_with_thread_counts("u1")
            await repo.count_project_threads("p1", "u1")
            await repo.get_project_threads("p1", "u1")
            await repo.delete_project_with_threads("p1", "u1")
            await repo.unassign_all_threads("p1", "u1")

        thread_sqls = [
            str(call.args[0])
            for call in mock_conn.execute.await_args_list
            if "FROM thread" in str(call.args[0])
            or "UPDATE thread" in str(call.args[0])
        ]
        assert thread_sqls
        for sql in thread_sqls:
            if sql.strip().startswith("DELETE FROM thread"):
                continue
            assert "metadata_json" in sql
            assert "metadata->>" not in sql.replace("metadata_json", "COL")

    @pytest.mark.asyncio
    async def test_list_counts_filter_by_user_id(self, repo, mock_conn):
        import deep_agent.src.projects as repo_mod

        repo_mod._TABLES_ENSURED = True
        repo_mod._OPTIONAL_INDEXES_ENSURED = True
        with patch(
            "deep_agent.src.projects.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            await repo.list_projects_with_thread_counts("u1")
        sql, params = mock_conn.execute.await_args_list[0].args[:2]
        assert "user_id" in str(sql)
        assert params[0] == "u1"

    @pytest.mark.asyncio
    async def test_unassign_all_propagates_non_missing_table_errors(
        self, repo, mock_conn
    ):
        import deep_agent.src.projects as repo_mod

        repo_mod._TABLES_ENSURED = True
        repo_mod._OPTIONAL_INDEXES_ENSURED = True
        _own_project(mock_conn)
        mock_conn._cursor.fetchall = AsyncMock(return_value=[("t1",)])

        async def execute(query, *_args, **_kwargs):
            if "UPDATE checkpoints" in str(query):
                raise RuntimeError("deadlock")
            return mock_conn._cursor

        mock_conn.execute = AsyncMock(side_effect=execute)
        with patch(
            "deep_agent.src.projects.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            with pytest.raises(RuntimeError, match="deadlock"):
                await repo.unassign_all_threads("p1", "u1")

    @pytest.mark.asyncio
    async def test_delete_purges_thread_and_checkpoint_rows(self, repo, mock_conn):
        import deep_agent.src.projects as repo_mod

        repo_mod._TABLES_ENSURED = True
        repo_mod._OPTIONAL_INDEXES_ENSURED = True
        _own_project(mock_conn)
        mock_conn._cursor.fetchall = AsyncMock(return_value=[("t1",)])
        mock_conn._cursor.rowcount = 1
        with (
            patch(
                "deep_agent.src.projects.psycopg.AsyncConnection.connect",
                return_value=mock_conn,
            ),
            patch(
                "deep_agent.aegra.thread_cleanup._delete_token_usage",
                new_callable=AsyncMock,
            ),
        ):
            ids = await repo.delete_project_with_threads("p1", "u1")
        assert ids == ["t1"]
        sqls = [str(call.args[0]) for call in mock_conn.execute.await_args_list]
        assert any("DELETE FROM thread" in s for s in sqls)
        assert any("DELETE FROM projects" in s for s in sqls)

    @pytest.mark.asyncio
    async def test_assign_missing_project_raises(self, repo, mock_conn):
        import deep_agent.src.projects as repo_mod

        repo_mod._TABLES_ENSURED = True
        repo_mod._OPTIONAL_INDEXES_ENSURED = True
        mock_conn._cursor.fetchone = AsyncMock(return_value=None)
        mock_conn._cursor.rowcount = 1
        with patch(
            "deep_agent.src.projects.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            with pytest.raises(ProjectNotFoundError):
                await repo.assign_thread_to_project("th1", "p1", "u1")
        sqls = [str(call.args[0]) for call in mock_conn.execute.await_args_list]
        assert not any("UPDATE thread" in s for s in sqls)

    @pytest.mark.asyncio
    async def test_hard_delete_purges_only_current_project_members(
        self, repo, mock_conn
    ):
        import deep_agent.src.projects as repo_mod

        repo_mod._TABLES_ENSURED = True
        repo_mod._OPTIONAL_INDEXES_ENSURED = True

        async def execute(query, *_args, **_kwargs):
            sql = str(query)
            cur = AsyncMock()
            cur.rowcount = 1
            if "FOR UPDATE" in sql:
                cur.fetchone = AsyncMock(return_value=(1,))
                return cur
            if "DELETE FROM thread" in sql and "project_id" in sql:
                cur.fetchall = AsyncMock(return_value=[("t1",)])
                return cur
            cur.fetchall = AsyncMock(return_value=[])
            cur.fetchone = AsyncMock(return_value=None)
            return cur

        mock_conn.execute = AsyncMock(side_effect=execute)
        with (
            patch(
                "deep_agent.src.projects.psycopg.AsyncConnection.connect",
                return_value=mock_conn,
            ),
            patch(
                "deep_agent.aegra.thread_cleanup._delete_token_usage",
                new_callable=AsyncMock,
            ),
        ):
            ids = await repo.delete_project_with_threads("p1", "u1")
        assert ids == ["t1"]
        pred = next(
            c
            for c in mock_conn.execute.await_args_list
            if "DELETE FROM thread" in str(c.args[0]) and "project_id" in str(c.args[0])
        )
        assert pred.args[1] == ("p1", "u1")
        per_id = [
            c.args[1]
            for c in mock_conn.execute.await_args_list
            if "DELETE FROM thread" in str(c.args[0])
            and "project_id" not in str(c.args[0])
        ]
        assert all(args[0] == "t1" for args in per_id)

    @pytest.mark.asyncio
    async def test_hard_delete_does_not_purge_thread_already_moved_out(
        self, repo, mock_conn
    ):
        """Purge membership by current project_id, not a stale SELECT list.

        A concurrent assign can move a thread out after members are listed.
        DELETE ... WHERE project_id misses that row; delete-by-id would not.
        """
        import deep_agent.src.projects as repo_mod

        repo_mod._TABLES_ENSURED = True
        repo_mod._OPTIONAL_INDEXES_ENSURED = True

        async def execute(query, *_args, **_kwargs):
            sql = str(query)
            cur = AsyncMock()
            cur.rowcount = 1
            if "FOR UPDATE" in sql:
                cur.fetchone = AsyncMock(return_value=(1,))
                return cur
            if "DELETE FROM thread" in sql and "project_id" in sql:
                cur.fetchall = AsyncMock(return_value=[("t-kept",)])
                return cur
            if "DELETE FROM projects" in sql:
                cur.rowcount = 1
                return cur
            cur.fetchall = AsyncMock(return_value=[])
            cur.fetchone = AsyncMock(return_value=None)
            return cur

        mock_conn.execute = AsyncMock(side_effect=execute)
        with (
            patch(
                "deep_agent.src.projects.psycopg.AsyncConnection.connect",
                return_value=mock_conn,
            ),
            patch(
                "deep_agent.aegra.thread_cleanup._delete_token_usage",
                new_callable=AsyncMock,
            ),
        ):
            ids = await repo.delete_project_with_threads("p1", "u1")
        assert ids == ["t-kept"]
        pred = next(
            c
            for c in mock_conn.execute.await_args_list
            if "DELETE FROM thread" in str(c.args[0]) and "project_id" in str(c.args[0])
        )
        assert pred.args[1] == ("p1", "u1")
        per_id = [
            c.args[1]
            for c in mock_conn.execute.await_args_list
            if "DELETE FROM thread" in str(c.args[0])
            and "project_id" not in str(c.args[0])
        ]
        assert all(args[0] == "t-kept" for args in per_id)

    @pytest.mark.asyncio
    async def test_delete_keep_threads_unassigns_without_purging_threads(
        self, repo, mock_conn
    ):
        import deep_agent.src.projects as repo_mod

        repo_mod._TABLES_ENSURED = True
        repo_mod._OPTIONAL_INDEXES_ENSURED = True
        _own_project(mock_conn)
        mock_conn._cursor.fetchall = AsyncMock(return_value=[("t1",)])
        mock_conn._cursor.rowcount = 1
        with patch(
            "deep_agent.src.projects.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            ids = await repo.delete_project_with_threads("p1", "u1", keep_threads=True)
        assert ids == []
        sqls = [str(call.args[0]) for call in mock_conn.execute.await_args_list]
        assert any("UPDATE thread" in s for s in sqls)
        assert any("DELETE FROM projects" in s for s in sqls)
        assert not any("DELETE FROM thread" in s for s in sqls)

    @pytest.mark.asyncio
    async def test_unassign_locks_current_project(self, repo, mock_conn):
        import deep_agent.src.projects as repo_mod

        repo_mod._TABLES_ENSURED = True
        repo_mod._OPTIONAL_INDEXES_ENSURED = True
        mock_conn._cursor.rowcount = 1

        async def execute(query, *_args, **_kwargs):
            sql = str(query)
            cur = AsyncMock()
            cur.rowcount = 1
            if "SELECT metadata_json->>'project_id' FROM thread" in sql:
                cur.fetchone = AsyncMock(return_value=("p1",))
                return cur
            if "FOR UPDATE" in sql:
                cur.fetchone = AsyncMock(return_value=(1,))
                return cur
            return mock_conn._cursor

        mock_conn.execute = AsyncMock(side_effect=execute)
        with patch(
            "deep_agent.src.projects.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            await repo.assign_thread_to_project("th1", None, "u1")

        sqls = [str(call.args[0]) for call in mock_conn.execute.await_args_list]
        assert any("FOR UPDATE" in s for s in sqls)
        lock = next(
            call
            for call in mock_conn.execute.await_args_list
            if "FOR UPDATE" in str(call.args[0])
        )
        assert lock.args[1] == ("p1", "u1")

    @pytest.mark.asyncio
    async def test_unassign_continues_when_current_project_is_gone(
        self, repo, mock_conn
    ):
        import deep_agent.src.projects as repo_mod

        repo_mod._TABLES_ENSURED = True
        repo_mod._OPTIONAL_INDEXES_ENSURED = True
        mock_conn._cursor.rowcount = 1

        async def execute(query, *_args, **_kwargs):
            sql = str(query)
            cur = AsyncMock()
            cur.rowcount = 1
            if "SELECT metadata_json->>'project_id' FROM thread" in sql:
                cur.fetchone = AsyncMock(return_value=("p1",))
                return cur
            if "FOR UPDATE" in sql:
                cur.fetchone = AsyncMock(return_value=None)
                return cur
            return mock_conn._cursor

        mock_conn.execute = AsyncMock(side_effect=execute)
        with patch(
            "deep_agent.src.projects.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            await repo.assign_thread_to_project("th1", None, "u1")

        sqls = [str(call.args[0]) for call in mock_conn.execute.await_args_list]
        assert any("UPDATE thread" in s for s in sqls)

    @pytest.mark.asyncio
    async def test_unassign_all_strips_checkpoints_by_returned_thread_ids(
        self, repo, mock_conn
    ):
        import deep_agent.src.projects as repo_mod

        repo_mod._TABLES_ENSURED = True
        repo_mod._OPTIONAL_INDEXES_ENSURED = True

        async def execute(query, *_args, **_kwargs):
            sql = str(query)
            cur = AsyncMock()
            cur.rowcount = 1
            if "FOR UPDATE" in sql:
                cur.fetchone = AsyncMock(return_value=(1,))
                return cur
            if "UPDATE thread" in sql:
                cur.fetchall = AsyncMock(return_value=[("t1",)])
                return cur
            cur.fetchall = AsyncMock(return_value=[])
            cur.fetchone = AsyncMock(return_value=None)
            return cur

        mock_conn.execute = AsyncMock(side_effect=execute)
        with patch(
            "deep_agent.src.projects.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            ids = await repo.unassign_all_threads("p1", "u1")
        assert ids == ["t1"]
        checkpoint_sql = next(
            str(call.args[0])
            for call in mock_conn.execute.await_args_list
            if "UPDATE checkpoints" in str(call.args[0])
        )
        assert "ANY(%s)" in checkpoint_sql
        assert "user_identity" not in checkpoint_sql
