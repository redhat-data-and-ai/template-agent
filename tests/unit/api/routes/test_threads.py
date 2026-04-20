"""Unit tests for threads route."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from template_agent.src.api.routes.memory.threads import list_threads


class TestListThreads:
    """Tests for list_threads endpoint."""

    @pytest.mark.asyncio
    async def test_successful_thread_retrieval(self):
        """Test successful retrieval of user threads."""
        mock_checkpointer = MagicMock()
        mock_cursor = MagicMock()

        # Mock cursor behavior
        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchall = AsyncMock(
            return_value=[
                {"thread_id": "thread1"},
                {"thread_id": "thread2"},
                {"thread_id": "thread3"},
            ]
        )
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=None)

        # Mock conn.cursor
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_checkpointer.conn = mock_conn

        # Mock checkpointer context manager
        mock_checkpointer.__aenter__ = AsyncMock(return_value=mock_checkpointer)
        mock_checkpointer.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "template_agent.src.api.routes.memory.threads.get_checkpointer"
        ) as mock_get_checkpointer:
            mock_get_checkpointer.return_value = mock_checkpointer

            result = await list_threads("user123")

            assert result == ["thread1", "thread2", "thread3"]
            mock_cursor.execute.assert_called_once()
            # Verify the SQL query includes user_id filter
            call_args = mock_cursor.execute.call_args
            assert "metadata->>'user_id'" in call_args[0][0]
            assert call_args[0][1] == ("user123",)

    @pytest.mark.asyncio
    async def test_no_threads_found(self):
        """Test when user has no threads."""
        mock_checkpointer = MagicMock()
        mock_cursor = MagicMock()

        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[])
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=None)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_checkpointer.conn = mock_conn

        mock_checkpointer.__aenter__ = AsyncMock(return_value=mock_checkpointer)
        mock_checkpointer.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "template_agent.src.api.routes.memory.threads.get_checkpointer"
        ) as mock_get_checkpointer:
            mock_get_checkpointer.return_value = mock_checkpointer

            result = await list_threads("usernothreads")

            assert result == []

    @pytest.mark.asyncio
    async def test_database_error_raises_http_exception(self):
        """Test that database errors raise HTTPException."""
        mock_checkpointer = MagicMock()
        mock_cursor = MagicMock()

        mock_cursor.execute = AsyncMock(
            side_effect=Exception("Database connection failed")
        )
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=None)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_checkpointer.conn = mock_conn

        mock_checkpointer.__aenter__ = AsyncMock(return_value=mock_checkpointer)
        mock_checkpointer.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "template_agent.src.api.routes.memory.threads.get_checkpointer"
        ) as mock_get_checkpointer:
            mock_get_checkpointer.return_value = mock_checkpointer

            with pytest.raises(HTTPException) as exc_info:
                await list_threads("user123")

            assert exc_info.value.status_code == 500
            assert "Failed to retrieve threads" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_threads_sorted_by_checkpoint_id_desc(self):
        """Test that threads are sorted by checkpoint_id descending (newest first)."""
        mock_checkpointer = MagicMock()
        mock_cursor = MagicMock()

        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchall = AsyncMock(
            return_value=[
                {"thread_id": "newestthread"},
                {"thread_id": "middlethread"},
                {"thread_id": "oldestthread"},
            ]
        )
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=None)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_checkpointer.conn = mock_conn

        mock_checkpointer.__aenter__ = AsyncMock(return_value=mock_checkpointer)
        mock_checkpointer.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "template_agent.src.api.routes.memory.threads.get_checkpointer"
        ) as mock_get_checkpointer:
            mock_get_checkpointer.return_value = mock_checkpointer

            result = await list_threads("user123")

            # Verify SQL uses checkpoint_id not step
            call_args = mock_cursor.execute.call_args
            sql_query = call_args[0][0]
            assert "ORDER BY" in sql_query
            assert "MAX(checkpoint_id)" in sql_query
            assert "DESC" in sql_query
            # Should NOT use step anymore
            assert "step" not in sql_query.lower()
            assert result == ["newestthread", "middlethread", "oldestthread"]
