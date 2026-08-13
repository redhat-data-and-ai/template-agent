"""Integration tests for lifecycle state persistence.

These tests exercise the full lifecycle flow using mock Postgres.
They verify cross-component interactions rather than individual
functions.

Marked as integration tests — skip in fast CI with:
    pytest -m "not integration"
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_agent.aegra.lifecycle import (
    POD_ID,
    persist_inflight_runs,
    resume_interrupted_runs,
)


@pytest.mark.integration
class TestConversationSurvivesPodRestart:
    """End-to-end: shutdown-persist expires leases, startup-resume picks them up."""

    def test_persist_expires_leases_via_postgres(self):
        """persist_inflight_runs reads active_runs and expires leases in Postgres."""
        with (
            patch(
                "deep_agent.aegra.lifecycle._get_active_runs",
                return_value={"run-A": MagicMock()},
            ),
            patch(
                "deep_agent.aegra.lifecycle._expire_run_leases_batch",
                return_value=1,
            ) as mock_expire,
        ):
            count = persist_inflight_runs()

        assert count == 1
        mock_expire.assert_called_once_with(["run-A"])

    def test_persist_multiple_runs(self):
        """Multiple active runs all get their leases expired."""
        with (
            patch(
                "deep_agent.aegra.lifecycle._get_active_runs",
                return_value={
                    "run-A": MagicMock(),
                    "run-B": MagicMock(),
                    "run-C": MagicMock(),
                },
            ),
            patch(
                "deep_agent.aegra.lifecycle._expire_run_leases_batch",
                return_value=3,
            ) as mock_expire,
        ):
            count = persist_inflight_runs()

        assert count == 3
        call_args = mock_expire.call_args[0][0]
        assert sorted(call_args) == ["run-A", "run-B", "run-C"]

    async def test_resume_after_persist(self):
        """Verify resume picks up what persist left behind."""
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(
            return_value=[("run-A", "thread-A", "asst_1", None)]
        )

        cursor_ctx = MagicMock()
        cursor_ctx.__aenter__ = AsyncMock(return_value=mock_cursor)
        cursor_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = cursor_ctx
        mock_conn.commit = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        resume_fn = AsyncMock()

        with patch("psycopg.AsyncConnection.connect", return_value=mock_conn):
            results = await resume_interrupted_runs("postgresql://test", resume_fn)

        assert results["run-A"] == "resumed"
        resume_fn.assert_awaited_once_with("run-A", "thread-A")

        execute_calls = mock_cursor.execute.call_args_list
        claim_sql = execute_calls[1][0][0]
        assert "claimed_by" in claim_sql
        assert "lease_expires_at" in claim_sql


@pytest.mark.integration
class TestStaleLeaseRecovery:
    """Verify that runs with expired leases are reclaimed."""

    async def test_stale_lease_reclaimed(self):
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(
            return_value=[("run-stale", "thread-stale", "asst_1", None)]
        )

        cursor_ctx = MagicMock()
        cursor_ctx.__aenter__ = AsyncMock(return_value=mock_cursor)
        cursor_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = cursor_ctx
        mock_conn.commit = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        resume_fn = AsyncMock()

        with patch("psycopg.AsyncConnection.connect", return_value=mock_conn):
            results = await resume_interrupted_runs("postgresql://test", resume_fn)

        assert results["run-stale"] == "resumed"
        resume_fn.assert_awaited_once_with("run-stale", "thread-stale")

        execute_calls = mock_cursor.execute.call_args_list
        claim_sql = execute_calls[1][0][0]
        claim_params = execute_calls[1][0][1]
        assert "claimed_by" in claim_sql
        assert POD_ID in claim_params
