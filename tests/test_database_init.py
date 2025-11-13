"""Tests for database initialization functionality.

This module tests the database schema initialization to ensure the checkpoints
table is created properly on application startup when using PostgreSQL storage.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from template_agent.src.core.agent import initialize_database
from template_agent.src.core.exceptions.exceptions import AppException


class TestDatabaseInitialization:
    """Test cases for database initialization."""

    @pytest.mark.asyncio
    async def test_initialize_database_skips_when_inmemory(self):
        """Test that database initialization is skipped when using in-memory storage."""
        with patch("template_agent.src.core.agent.settings") as mock_settings:
            mock_settings.USE_INMEMORY_SAVER = True

            # Should not raise any exceptions
            await initialize_database()

    @pytest.mark.asyncio
    async def test_initialize_database_calls_setup(self):
        """Test that database initialization calls setup on the checkpoint."""
        with patch("template_agent.src.core.agent.settings") as mock_settings:
            mock_settings.USE_INMEMORY_SAVER = False
            mock_settings.database_uri = "postgresql://user:pass@localhost:5432/db"

            # Create mock checkpoint with setup method
            mock_checkpoint = AsyncMock()
            mock_checkpoint.setup = AsyncMock()
            mock_checkpoint.__aenter__ = AsyncMock(return_value=mock_checkpoint)
            mock_checkpoint.__aexit__ = AsyncMock(return_value=None)

            with patch(
                "template_agent.src.core.agent.AsyncPostgresSaver.from_conn_string",
                return_value=mock_checkpoint,
            ):
                await initialize_database()

                # Verify setup was called
                mock_checkpoint.setup.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_database_handles_no_setup_method(self):
        """Test that database initialization handles checkpoints without setup method."""
        with patch("template_agent.src.core.agent.settings") as mock_settings:
            mock_settings.USE_INMEMORY_SAVER = False
            mock_settings.database_uri = "postgresql://user:pass@localhost:5432/db"

            # Create mock checkpoint without setup method
            mock_checkpoint = AsyncMock()
            mock_checkpoint.__aenter__ = AsyncMock(return_value=mock_checkpoint)
            mock_checkpoint.__aexit__ = AsyncMock(return_value=None)

            with patch(
                "template_agent.src.core.agent.AsyncPostgresSaver.from_conn_string",
                return_value=mock_checkpoint,
            ):
                # Should not raise exception, just log warning
                await initialize_database()

    @pytest.mark.asyncio
    async def test_initialize_database_raises_on_connection_error(self):
        """Test that database initialization raises AppException on connection failure."""
        with patch("template_agent.src.core.agent.settings") as mock_settings:
            mock_settings.USE_INMEMORY_SAVER = False
            mock_settings.database_uri = "postgresql://user:pass@localhost:5432/db"

            with patch(
                "template_agent.src.core.agent.AsyncPostgresSaver.from_conn_string",
                side_effect=Exception("Connection failed"),
            ):
                with pytest.raises(AppException) as exc_info:
                    await initialize_database()

                assert "Database initialization failed" in str(exc_info.value)
                assert "Connection failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_initialize_database_raises_on_setup_error(self):
        """Test that database initialization raises AppException on setup failure."""
        with patch("template_agent.src.core.agent.settings") as mock_settings:
            mock_settings.USE_INMEMORY_SAVER = False
            mock_settings.database_uri = "postgresql://user:pass@localhost:5432/db"

            # Create mock checkpoint that fails on setup
            mock_checkpoint = AsyncMock()
            mock_checkpoint.setup = AsyncMock(side_effect=Exception("Setup failed"))
            mock_checkpoint.__aenter__ = AsyncMock(return_value=mock_checkpoint)
            mock_checkpoint.__aexit__ = AsyncMock(return_value=None)

            with patch(
                "template_agent.src.core.agent.AsyncPostgresSaver.from_conn_string",
                return_value=mock_checkpoint,
            ):
                with pytest.raises(AppException) as exc_info:
                    await initialize_database()

                assert "Database initialization failed" in str(exc_info.value)
                assert "Setup failed" in str(exc_info.value)
