"""Unit tests for FeedbackRepository (mocked DB)."""

from unittest.mock import AsyncMock, patch

import pytest

from deep_agent.src.feedback import repository as feedback_repo_mod
from deep_agent.src.feedback.repository import FeedbackRepository


@pytest.fixture(autouse=True)
def _reset_feedback_table_flag():
    """Reset module-level _TABLE_ENSURED before each test."""
    feedback_repo_mod._TABLE_ENSURED = False
    yield
    feedback_repo_mod._TABLE_ENSURED = False


@pytest.fixture
def mock_conn():
    """Create a mock async connection context manager."""
    conn = AsyncMock()
    cursor = AsyncMock()
    cursor.fetchall = AsyncMock(return_value=[])
    cursor.rowcount = 0
    conn.execute = AsyncMock(return_value=cursor)
    conn.commit = AsyncMock()
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=False)
    conn._cursor = cursor
    return conn


@pytest.fixture
def repo():
    return FeedbackRepository("postgresql://test:test@localhost/testdb")


class TestEnsureTable:
    @pytest.mark.asyncio
    async def test_creates_table_once(self, repo, mock_conn):
        with patch(
            "deep_agent.src.feedback.repository.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            await repo.ensure_table()
            mock_conn.execute.assert_awaited_once()
            mock_conn.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_idempotent_second_call(self, repo, mock_conn):
        with patch(
            "deep_agent.src.feedback.repository.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            await repo.ensure_table()
            mock_conn.execute.reset_mock()
            mock_conn.commit.reset_mock()
            await repo.ensure_table()
            mock_conn.execute.assert_not_called()
            mock_conn.commit.assert_not_called()


class TestUpsertFeedback:
    @pytest.mark.asyncio
    async def test_insert_calls_execute_and_commit(self, repo, mock_conn):
        feedback_repo_mod._TABLE_ENSURED = True
        with patch(
            "deep_agent.src.feedback.repository.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            await repo.upsert_feedback(
                "t1",
                "m1",
                "u1",
                "up",
                "trace-1",
            )
            mock_conn.execute.assert_awaited_once()
            mock_conn.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_update_second_upsert(self, repo, mock_conn):
        """Second upsert with same keys runs ON CONFLICT UPDATE (still one execute)."""
        feedback_repo_mod._TABLE_ENSURED = True
        with patch(
            "deep_agent.src.feedback.repository.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            await repo.upsert_feedback("t1", "m1", "u1", "up", None)
            await repo.upsert_feedback("t1", "m1", "u1", "down", None)
            assert mock_conn.execute.await_count == 2
            assert mock_conn.commit.await_count == 2


class TestDeleteFeedback:
    @pytest.mark.asyncio
    async def test_delete_returns_true_when_row_removed(self, repo, mock_conn):
        feedback_repo_mod._TABLE_ENSURED = True
        mock_conn._cursor.rowcount = 1
        mock_conn.execute.return_value = mock_conn._cursor
        with patch(
            "deep_agent.src.feedback.repository.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            result = await repo.delete_feedback("t1", "m1", "u1")
            assert result is True

    @pytest.mark.asyncio
    async def test_delete_returns_false_when_missing(self, repo, mock_conn):
        feedback_repo_mod._TABLE_ENSURED = True
        mock_conn._cursor.rowcount = 0
        mock_conn.execute.return_value = mock_conn._cursor
        with patch(
            "deep_agent.src.feedback.repository.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            result = await repo.delete_feedback("t1", "m1", "u1")
            assert result is False


class TestListFeedback:
    @pytest.mark.asyncio
    async def test_returns_message_id_and_feedback(self, repo, mock_conn):
        feedback_repo_mod._TABLE_ENSURED = True
        mock_conn._cursor.fetchall = AsyncMock(
            return_value=[
                {"message_id": "m1", "feedback": "up"},
                {"message_id": "m2", "feedback": "down"},
            ]
        )
        with patch(
            "deep_agent.src.feedback.repository.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            rows = await repo.list_feedback("t1", "u1")
            assert rows == [
                {"message_id": "m1", "feedback": "up"},
                {"message_id": "m2", "feedback": "down"},
            ]

    @pytest.mark.asyncio
    async def test_empty_list(self, repo, mock_conn):
        feedback_repo_mod._TABLE_ENSURED = True
        with patch(
            "deep_agent.src.feedback.repository.psycopg.AsyncConnection.connect",
            return_value=mock_conn,
        ):
            rows = await repo.list_feedback("t1", "u1")
            assert rows == []
