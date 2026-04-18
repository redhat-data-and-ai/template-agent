"""Unit tests for history route."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage

from template_agent.src.routes.history import history
from template_agent.src.schema import ChatHistoryResponse


class TestHistory:
    """Tests for history endpoint."""

    @pytest.mark.asyncio
    async def test_successful_history_retrieval(self):
        """Test successful retrieval of chat history."""
        mock_checkpointer = MagicMock()
        mock_cursor = MagicMock()

        # Mock ownership check
        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value={"exists": 1})
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=None)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_checkpointer.conn = mock_conn

        # Mock checkpoint data with messages
        checkpoint_data = {
            "channel_values": {
                "messages": [
                    HumanMessage(content="Hello", id="msg1"),
                    AIMessage(content="Hi there!", id="msg2"),
                ]
            }
        }
        mock_checkpointer.aget = AsyncMock(return_value=checkpoint_data)

        # Mock metadata
        mock_metadata_tuple = MagicMock()
        mock_metadata_tuple.metadata = {"run_id": "run123", "session_id": "session456"}
        mock_checkpointer.aget_tuple = AsyncMock(return_value=mock_metadata_tuple)

        mock_checkpointer.__aenter__ = AsyncMock(return_value=mock_checkpointer)
        mock_checkpointer.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "template_agent.src.routes.history.get_checkpointer"
        ) as mock_get_checkpointer:
            mock_get_checkpointer.return_value = mock_checkpointer

            result = await history("user123", "thread456")

            assert isinstance(result, ChatHistoryResponse)
            assert len(result.messages) == 2
            assert result.messages[0].type == "human"
            assert result.messages[0].content == "Hello"
            assert result.messages[1].type == "ai"
            assert result.messages[1].content == "Hi there!"

    @pytest.mark.asyncio
    async def test_ownership_check_fails(self):
        """Test when user doesn't own the thread."""
        mock_checkpointer = MagicMock()
        mock_cursor = MagicMock()

        # Mock ownership check returning None (no access)
        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=None)
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=None)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_checkpointer.conn = mock_conn

        mock_checkpointer.__aenter__ = AsyncMock(return_value=mock_checkpointer)
        mock_checkpointer.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "template_agent.src.routes.history.get_checkpointer"
        ) as mock_get_checkpointer:
            mock_get_checkpointer.return_value = mock_checkpointer

            result = await history("user123", "thread456")

            assert isinstance(result, ChatHistoryResponse)
            assert result.messages == []

    @pytest.mark.asyncio
    async def test_no_messages_in_checkpoint(self):
        """Test when checkpoint exists but has no messages."""
        mock_checkpointer = MagicMock()
        mock_cursor = MagicMock()

        # Mock ownership check passes
        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value={"exists": 1})
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=None)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_checkpointer.conn = mock_conn

        # Mock checkpoint with no messages
        checkpoint_data = {"channel_values": {}}
        mock_checkpointer.aget = AsyncMock(return_value=checkpoint_data)

        mock_checkpointer.__aenter__ = AsyncMock(return_value=mock_checkpointer)
        mock_checkpointer.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "template_agent.src.routes.history.get_checkpointer"
        ) as mock_get_checkpointer:
            mock_get_checkpointer.return_value = mock_checkpointer

            result = await history("user123", "thread456")

            assert isinstance(result, ChatHistoryResponse)
            assert result.messages == []

    @pytest.mark.asyncio
    async def test_no_checkpoint_data(self):
        """Test when no checkpoint data exists."""
        mock_checkpointer = MagicMock()
        mock_cursor = MagicMock()

        # Mock ownership check passes
        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value={"exists": 1})
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=None)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_checkpointer.conn = mock_conn

        # Mock no checkpoint data
        mock_checkpointer.aget = AsyncMock(return_value=None)

        mock_checkpointer.__aenter__ = AsyncMock(return_value=mock_checkpointer)
        mock_checkpointer.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "template_agent.src.routes.history.get_checkpointer"
        ) as mock_get_checkpointer:
            mock_get_checkpointer.return_value = mock_checkpointer

            result = await history("user123", "thread456")

            assert isinstance(result, ChatHistoryResponse)
            assert result.messages == []

    @pytest.mark.asyncio
    async def test_message_conversion_error_continues(self):
        """Test that message conversion errors don't stop processing."""
        mock_checkpointer = MagicMock()
        mock_cursor = MagicMock()

        # Mock ownership check
        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value={"exists": 1})
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=None)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_checkpointer.conn = mock_conn

        # Mock checkpoint with mixed valid/invalid messages
        checkpoint_data = {
            "channel_values": {
                "messages": [
                    HumanMessage(content="Valid message", id="msg1"),
                    MagicMock(),  # Invalid message that will fail conversion
                    AIMessage(content="Another valid", id="msg3"),
                ]
            }
        }
        mock_checkpointer.aget = AsyncMock(return_value=checkpoint_data)

        mock_metadata_tuple = MagicMock()
        mock_metadata_tuple.metadata = {}
        mock_checkpointer.aget_tuple = AsyncMock(return_value=mock_metadata_tuple)

        mock_checkpointer.__aenter__ = AsyncMock(return_value=mock_checkpointer)
        mock_checkpointer.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "template_agent.src.routes.history.get_checkpointer"
        ) as mock_get_checkpointer:
            mock_get_checkpointer.return_value = mock_checkpointer

            result = await history("user123", "thread456")

            # Should have 2 valid messages, skipping the invalid one
            assert isinstance(result, ChatHistoryResponse)
            assert len(result.messages) == 2

    @pytest.mark.asyncio
    async def test_metadata_enrichment(self):
        """Test that messages are enriched with metadata."""
        mock_checkpointer = MagicMock()
        mock_cursor = MagicMock()

        # Mock ownership check
        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value={"exists": 1})
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=None)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_checkpointer.conn = mock_conn

        checkpoint_data = {
            "channel_values": {"messages": [HumanMessage(content="Test", id="msg1")]}
        }
        mock_checkpointer.aget = AsyncMock(return_value=checkpoint_data)

        # Mock metadata with run_id and session_id
        mock_metadata_tuple = MagicMock()
        mock_metadata_tuple.metadata = {
            "run_id": "run-xyz",
            "session_id": "session-abc",
        }
        mock_checkpointer.aget_tuple = AsyncMock(return_value=mock_metadata_tuple)

        mock_checkpointer.__aenter__ = AsyncMock(return_value=mock_checkpointer)
        mock_checkpointer.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "template_agent.src.routes.history.get_checkpointer"
        ) as mock_get_checkpointer:
            mock_get_checkpointer.return_value = mock_checkpointer

            result = await history("user123", "thread456")

            assert len(result.messages) == 1
            assert result.messages[0].run_id == "run-xyz"
            assert result.messages[0].session_id == "session-abc"
            assert result.messages[0].thread_id == "thread456"

    @pytest.mark.asyncio
    async def test_database_error_raises_http_exception(self):
        """Test that database errors raise HTTPException."""
        mock_checkpointer = MagicMock()
        mock_cursor = MagicMock()

        # Mock database error
        mock_cursor.execute = AsyncMock(side_effect=Exception("DB connection lost"))
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=None)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_checkpointer.conn = mock_conn

        mock_checkpointer.__aenter__ = AsyncMock(return_value=mock_checkpointer)
        mock_checkpointer.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "template_agent.src.routes.history.get_checkpointer"
        ) as mock_get_checkpointer:
            mock_get_checkpointer.return_value = mock_checkpointer

            with pytest.raises(HTTPException) as exc_info:
                await history("user123", "thread456")

            assert exc_info.value.status_code == 500
            assert "Failed to retrieve chat history" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_ownership_check_sql_injection_protection(self):
        """Test that ownership check uses parameterized query."""
        mock_checkpointer = MagicMock()
        mock_cursor = MagicMock()

        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=None)
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=None)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_checkpointer.conn = mock_conn

        mock_checkpointer.__aenter__ = AsyncMock(return_value=mock_checkpointer)
        mock_checkpointer.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "template_agent.src.routes.history.get_checkpointer"
        ) as mock_get_checkpointer:
            mock_get_checkpointer.return_value = mock_checkpointer

            # Try with malicious input
            await history("user'; DROP TABLE checkpoints; --", "thread123")

            # Verify parameterized query was used
            call_args = mock_cursor.execute.call_args
            assert call_args[0][1] == ("thread123", "user'; DROP TABLE checkpoints; --")

    @pytest.mark.asyncio
    async def test_checkpointer_context_manager_cleanup(self):
        """Test that checkpointer context manager is properly cleaned up."""
        mock_checkpointer = MagicMock()
        mock_cursor = MagicMock()

        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=None)
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=None)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_checkpointer.conn = mock_conn

        mock_checkpointer.__aenter__ = AsyncMock(return_value=mock_checkpointer)
        mock_checkpointer.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "template_agent.src.routes.history.get_checkpointer"
        ) as mock_get_checkpointer:
            mock_get_checkpointer.return_value = mock_checkpointer

            await history("user123", "thread456")

            mock_checkpointer.__aenter__.assert_called_once()
            mock_checkpointer.__aexit__.assert_called_once()
