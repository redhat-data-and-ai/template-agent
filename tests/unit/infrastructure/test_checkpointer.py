"""Unit tests for checkpointer module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from template_agent.src.exceptions import AppException
from template_agent.src.infrastructure.checkpointer import (
    get_checkpointer,
    initialize_checkpointer,
)


class TestInitializeCheckpointer:
    """Tests for initialize_checkpointer function."""

    @pytest.mark.asyncio
    async def test_successful_initialization(self):
        """Test successful database schema initialization."""
        mock_checkpointer = MagicMock()
        mock_checkpointer.setup = AsyncMock()
        mock_checkpointer.__aenter__ = AsyncMock(return_value=mock_checkpointer)
        mock_checkpointer.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "template_agent.src.infrastructure.checkpointer.AsyncPostgresSaver.from_conn_string"
        ) as mock_from_conn:
            mock_from_conn.return_value = mock_checkpointer

            await initialize_checkpointer()

            mock_from_conn.assert_called_once()
            mock_checkpointer.setup.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialization_failure_raises_app_exception(self):
        """Test that database initialization failures raise AppException."""
        mock_checkpointer = MagicMock()
        mock_checkpointer.setup = AsyncMock(side_effect=Exception("Connection failed"))
        mock_checkpointer.__aenter__ = AsyncMock(return_value=mock_checkpointer)
        mock_checkpointer.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "template_agent.src.infrastructure.checkpointer.AsyncPostgresSaver.from_conn_string"
        ) as mock_from_conn:
            mock_from_conn.return_value = mock_checkpointer

            with pytest.raises(AppException) as exc_info:
                await initialize_checkpointer()

            assert "Database initialization failed" in str(exc_info.value)
            assert exc_info.value.code == "E_008"

    @pytest.mark.asyncio
    async def test_connection_failure_raises_app_exception(self):
        """Test that connection failures raise AppException."""
        with patch(
            "template_agent.src.infrastructure.checkpointer.AsyncPostgresSaver.from_conn_string"
        ) as mock_from_conn:
            mock_from_conn.side_effect = Exception("Connection refused")

            with pytest.raises(AppException) as exc_info:
                await initialize_checkpointer()

            assert "Database initialization failed" in str(exc_info.value)


class TestGetCheckpointer:
    """Tests for get_checkpointer context manager."""

    @pytest.mark.asyncio
    async def test_yields_checkpointer_instance(self):
        """Test that get_checkpointer yields AsyncPostgresSaver instance."""
        mock_checkpointer = MagicMock()
        mock_checkpointer.__aenter__ = AsyncMock(return_value=mock_checkpointer)
        mock_checkpointer.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "template_agent.src.infrastructure.checkpointer.AsyncPostgresSaver.from_conn_string"
        ) as mock_from_conn:
            mock_from_conn.return_value = mock_checkpointer

            async with get_checkpointer() as checkpointer:
                assert checkpointer is mock_checkpointer

            mock_from_conn.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_database_uri_from_settings(self):
        """Test that get_checkpointer uses database_uri from settings."""
        mock_checkpointer = MagicMock()
        mock_checkpointer.__aenter__ = AsyncMock(return_value=mock_checkpointer)
        mock_checkpointer.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "template_agent.src.infrastructure.checkpointer.AsyncPostgresSaver.from_conn_string"
            ) as mock_from_conn,
            patch(
                "template_agent.src.infrastructure.checkpointer.settings"
            ) as mock_settings,
        ):
            mock_settings.database_uri = "postgresql://test:test@localhost/testdb"
            mock_from_conn.return_value = mock_checkpointer

            async with get_checkpointer():
                pass

            mock_from_conn.assert_called_once_with(
                "postgresql://test:test@localhost/testdb"
            )

    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self):
        """Test that context manager properly cleans up resources."""
        mock_checkpointer = MagicMock()
        mock_checkpointer.__aenter__ = AsyncMock(return_value=mock_checkpointer)
        mock_checkpointer.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "template_agent.src.infrastructure.checkpointer.AsyncPostgresSaver.from_conn_string"
        ) as mock_from_conn:
            mock_from_conn.return_value = mock_checkpointer

            async with get_checkpointer():
                pass

            mock_checkpointer.__aenter__.assert_called_once()
            mock_checkpointer.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_error_propagates(self):
        """Test that connection errors propagate properly."""
        with patch(
            "template_agent.src.infrastructure.checkpointer.AsyncPostgresSaver.from_conn_string"
        ) as mock_from_conn:
            mock_from_conn.side_effect = Exception("Connection failed")

            with pytest.raises(Exception) as exc_info:
                async with get_checkpointer():
                    pass

            assert "Connection failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_multiple_concurrent_checkpointers(self):
        """Test that multiple concurrent get_checkpointer calls work independently."""
        mock_checkpointer1 = MagicMock()
        mock_checkpointer1.__aenter__ = AsyncMock(return_value=mock_checkpointer1)
        mock_checkpointer1.__aexit__ = AsyncMock(return_value=None)

        mock_checkpointer2 = MagicMock()
        mock_checkpointer2.__aenter__ = AsyncMock(return_value=mock_checkpointer2)
        mock_checkpointer2.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "template_agent.src.infrastructure.checkpointer.AsyncPostgresSaver.from_conn_string"
        ) as mock_from_conn:
            mock_from_conn.side_effect = [mock_checkpointer1, mock_checkpointer2]

            async with get_checkpointer() as cp1:
                async with get_checkpointer() as cp2:
                    assert cp1 is not cp2
                    assert cp1 is mock_checkpointer1
                    assert cp2 is mock_checkpointer2

            assert mock_from_conn.call_count == 2
