"""Unit tests for thread deletion with full data cleanup."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from deep_agent.aegra.http_app import app


class TestDeletePgData:
    @pytest.mark.asyncio
    async def test_deletes_all_tables_in_one_transaction(self):
        from deep_agent.aegra.thread_cleanup import _delete_pg_data

        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.rowcount = 5
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        counts = await _delete_pg_data("thread-123", "user-1", mock_conn)

        assert counts["checkpoints"] == 15  # 5 per table x 3 tables
        assert counts["feedback"] == 5
        # 3 checkpoint tables + 1 feedback + 1 runs + 1 thread = 6 queries
        assert mock_conn.execute.call_count == 6
        queries = [c[0][0] for c in mock_conn.execute.call_args_list]
        assert any("checkpoint_writes" in q for q in queries)
        assert any("checkpoint_blobs" in q for q in queries)
        assert any("checkpoints" in q for q in queries)
        assert any("message_feedback" in q for q in queries)
        assert any("runs" in q for q in queries)
        assert any("DELETE FROM thread" in q for q in queries)

    @pytest.mark.asyncio
    async def test_propagates_exception_without_commit(self):
        """A failure mid-transaction prevents any data from being deleted."""
        from deep_agent.aegra.thread_cleanup import _delete_pg_data

        call_count = 0
        original_return = AsyncMock(rowcount=2)

        async def fail_on_third(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                raise RuntimeError("pg connection lost")
            return original_return

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(side_effect=fail_on_third)

        with pytest.raises(RuntimeError, match="pg connection lost"):
            await _delete_pg_data("thread-123", "user-1", mock_conn)


class TestDeleteThreadWithCleanup:
    def test_endpoint_deletes_all_data_atomically(self):
        """DELETE /threads/{id} runs ownership check + cleanup in one transaction."""
        thread_id = "00000000-0000-0000-0000-000000000001"

        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value={"thread_id": thread_id})
        mock_cursor.rowcount = 1
        mock_conn.execute = AsyncMock(return_value=mock_cursor)
        mock_conn.commit = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        pg_counts = {"checkpoints": 10, "feedback": 3}

        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", False),
            patch("deep_agent.aegra.thread_cleanup.settings") as mock_settings,
            patch(
                "psycopg.AsyncConnection.connect",
                new_callable=AsyncMock,
                return_value=mock_conn,
            ),
            patch(
                "deep_agent.aegra.thread_cleanup._delete_pg_data",
                new_callable=AsyncMock,
                return_value=pg_counts,
            ) as mock_pg,
            patch(
                "deep_agent.aegra.thread_cleanup._delete_token_usage",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_tu,
        ):
            mock_settings.database_uri = "postgresql://test"

            client = TestClient(app)
            res = client.delete(f"/threads/{thread_id}")

            assert res.status_code == 200
            assert res.json() == {"status": "deleted"}
            mock_pg.assert_awaited_once_with(thread_id, None, mock_conn)
            mock_conn.commit.assert_awaited()
            mock_tu.assert_awaited_once_with(thread_id)

            first_query = mock_conn.execute.call_args_list[0][0][0]
            assert "FOR UPDATE" in first_query

    def test_pg_failure_rolls_back_no_partial_delete(self):
        """If PG cleanup fails, nothing is committed (single transaction)."""
        thread_id = "00000000-0000-0000-0000-000000000001"

        mock_conn = AsyncMock()
        call_count = 0

        async def ownership_then_fail(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                cursor = AsyncMock()
                cursor.fetchone = AsyncMock(return_value={"thread_id": thread_id})
                return cursor
            raise RuntimeError("disk full")

        mock_conn.execute = AsyncMock(side_effect=ownership_then_fail)
        mock_conn.commit = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", False),
            patch("deep_agent.aegra.thread_cleanup.settings") as mock_settings,
            patch(
                "psycopg.AsyncConnection.connect",
                new_callable=AsyncMock,
                return_value=mock_conn,
            ),
        ):
            mock_settings.database_uri = "postgresql://test"

            client = TestClient(app, raise_server_exceptions=False)
            res = client.delete(f"/threads/{thread_id}")

            assert res.status_code == 500
            mock_conn.commit.assert_not_awaited()

    def test_mongodb_failure_still_returns_success(self):
        """MongoDB token-usage cleanup is best-effort after PG commit."""
        thread_id = "00000000-0000-0000-0000-000000000001"

        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value={"thread_id": thread_id})
        mock_cursor.rowcount = 1
        mock_conn.execute = AsyncMock(return_value=mock_cursor)
        mock_conn.commit = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", False),
            patch("deep_agent.aegra.thread_cleanup.settings") as mock_settings,
            patch(
                "psycopg.AsyncConnection.connect",
                new_callable=AsyncMock,
                return_value=mock_conn,
            ),
            patch(
                "deep_agent.aegra.thread_cleanup._delete_token_usage",
                new_callable=AsyncMock,
                side_effect=RuntimeError("MongoDB down"),
            ),
        ):
            mock_settings.database_uri = "postgresql://test"

            client = TestClient(app)
            res = client.delete(f"/threads/{thread_id}")

            assert res.status_code == 200
            assert res.json() == {"status": "deleted"}


class TestCleanupUserScoping:
    def test_thread_ownership_checked_before_delete(self):
        """DELETE returns 404 when thread doesn't belong to the authenticated user."""
        thread_id = "00000000-0000-0000-0000-000000000002"
        test_user_id = "auth-user-abc"

        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock(return_value=mock_cursor)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("deep_agent.aegra.auth.ENABLE_AUTH", True),
            patch(
                "deep_agent.aegra.auth._decode_token",
                return_value={"sub": test_user_id},
            ),
            patch("deep_agent.aegra.thread_cleanup.settings") as mock_settings,
            patch(
                "psycopg.AsyncConnection.connect",
                new_callable=AsyncMock,
                return_value=mock_conn,
            ),
            patch(
                "deep_agent.aegra.thread_cleanup._delete_pg_data",
                new_callable=AsyncMock,
            ) as mock_pg,
            patch(
                "deep_agent.aegra.thread_cleanup._delete_token_usage",
                new_callable=AsyncMock,
            ) as mock_tu,
        ):
            mock_settings.database_uri = "postgresql://test"

            client = TestClient(app)
            res = client.delete(
                f"/threads/{thread_id}",
                headers={"Authorization": "Bearer fake-jwt-token"},
            )

            assert res.status_code == 404
            query_args = mock_conn.execute.call_args_list[0]
            assert thread_id in query_args[0][1]
            assert test_user_id in query_args[0][1]
            mock_pg.assert_not_awaited()
            mock_tu.assert_not_awaited()
