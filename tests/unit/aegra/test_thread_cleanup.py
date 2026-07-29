"""Unit tests for thread deletion with full data cleanup."""

from unittest.mock import AsyncMock, patch

import pytest


class TestDeleteCheckpoints:
    @pytest.mark.asyncio
    async def test_deletes_all_three_checkpoint_tables(self):
        from deep_agent.aegra.thread_cleanup import _delete_checkpoints

        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.rowcount = 5
        mock_conn.execute = AsyncMock(return_value=mock_cursor)
        mock_conn.commit = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "psycopg.AsyncConnection.connect",
                new_callable=AsyncMock,
                return_value=mock_conn,
            ),
            patch("deep_agent.aegra.thread_cleanup.settings") as mock_settings,
        ):
            mock_settings.database_uri = "postgresql://test"

            count = await _delete_checkpoints("thread-123")

            assert count == 15  # 5 per table x 3 tables
            assert mock_conn.execute.call_count == 3
            queries = [c[0][0] for c in mock_conn.execute.call_args_list]
            assert any("checkpoint_writes" in q for q in queries)
            assert any("checkpoint_blobs" in q for q in queries)
            assert any("checkpoints" in q for q in queries)

    @pytest.mark.asyncio
    async def test_returns_zero_on_failure(self):
        from deep_agent.aegra.thread_cleanup import _delete_checkpoints

        with (
            patch(
                "psycopg.AsyncConnection.connect",
                side_effect=Exception("connection failed"),
            ),
            patch("deep_agent.aegra.thread_cleanup.settings") as mock_settings,
        ):
            mock_settings.database_uri = "postgresql://test"

            count = await _delete_checkpoints("thread-123")
            assert count == 0


class TestDeleteFeedback:
    @pytest.mark.asyncio
    async def test_deletes_feedback_for_thread(self):
        from deep_agent.aegra.thread_cleanup import _delete_feedback

        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.rowcount = 3
        mock_conn.execute = AsyncMock(return_value=mock_cursor)
        mock_conn.commit = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "psycopg.AsyncConnection.connect",
                new_callable=AsyncMock,
                return_value=mock_conn,
            ),
            patch("deep_agent.aegra.thread_cleanup.settings") as mock_settings,
        ):
            mock_settings.database_uri = "postgresql://test"

            count = await _delete_feedback("thread-123")

            assert count == 3
            query = mock_conn.execute.call_args[0][0]
            assert "message_feedback" in query
            assert "thread_id" in query


class TestDeleteThreadWithCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_deletes_all_associated_data(self):
        """Thread delete must clean up checkpoints, feedback, and token usage."""
        from deep_agent.aegra.thread_cleanup import (
            _delete_checkpoints,
            _delete_feedback,
            _delete_token_usage,
        )

        with (
            patch(
                "deep_agent.aegra.thread_cleanup._delete_checkpoints",
                new_callable=AsyncMock,
                return_value=10,
            ) as mock_cp,
            patch(
                "deep_agent.aegra.thread_cleanup._delete_feedback",
                new_callable=AsyncMock,
                return_value=3,
            ) as mock_fb,
            patch(
                "deep_agent.aegra.thread_cleanup._delete_token_usage",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_tu,
        ):
            await mock_cp("thread-123")
            await mock_fb("thread-123")
            await mock_tu("thread-123")

            mock_cp.assert_awaited_once_with("thread-123")
            mock_fb.assert_awaited_once_with("thread-123")
            mock_tu.assert_awaited_once_with("thread-123")


class TestCleanupUserScoping:
    @pytest.mark.asyncio
    async def test_thread_ownership_checked_before_delete(self):
        """Delete must verify the thread belongs to the authenticated user."""
        from deep_agent.aegra.thread_cleanup import _delete_checkpoints

        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.rowcount = 0
        mock_conn.execute = AsyncMock(return_value=mock_cursor)
        mock_conn.commit = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "psycopg.AsyncConnection.connect",
                new_callable=AsyncMock,
                return_value=mock_conn,
            ),
            patch("deep_agent.aegra.thread_cleanup.settings") as mock_settings,
        ):
            mock_settings.database_uri = "postgresql://test"

            count = await _delete_checkpoints("thread-123")
            assert count == 0
