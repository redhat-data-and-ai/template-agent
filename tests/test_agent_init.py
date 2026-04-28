"""Tests for agent.py - database initialization and agent creation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from template_agent.src.core.exceptions.exceptions import AppException


class TestInitializeDatabase:
    """Tests for initialize_database function."""

    async def test_skips_when_inmemory_saver(self, monkeypatch):
        """Skips database initialization when USE_INMEMORY_SAVER is True."""
        monkeypatch.setenv("USE_INMEMORY_SAVER", "true")
        with patch("template_agent.src.core.agent.settings") as mock_settings:
            mock_settings.USE_INMEMORY_SAVER = True
            from template_agent.src.core.agent import initialize_database

            await initialize_database()

    async def test_initializes_postgres_schema(self, monkeypatch):
        """Initializes PostgreSQL schema when not using in-memory saver."""
        monkeypatch.setenv("USE_INMEMORY_SAVER", "false")
        monkeypatch.setenv("AGENT_PG_HOST", "localhost")
        monkeypatch.setenv("AGENT_PG_PORT", "5432")
        monkeypatch.setenv("AGENT_PG_DATABASE", "test")
        monkeypatch.setenv("AGENT_PG_USER", "test")
        monkeypatch.setenv("AGENT_PG_PASSWORD", "test")

        mock_checkpoint = MagicMock()
        mock_checkpoint.setup = AsyncMock()
        mock_checkpoint.__aenter__ = AsyncMock(return_value=mock_checkpoint)
        mock_checkpoint.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "template_agent.src.core.agent.AsyncPostgresSaver.from_conn_string",
            return_value=mock_checkpoint,
        ):
            from template_agent.src.core.agent import initialize_database

            await initialize_database()

        mock_checkpoint.setup.assert_awaited_once()

    async def test_handles_missing_setup_method(self, monkeypatch):
        """Handles checkpoint without setup method gracefully."""
        monkeypatch.setenv("USE_INMEMORY_SAVER", "false")
        monkeypatch.setenv("AGENT_PG_HOST", "localhost")
        monkeypatch.setenv("AGENT_PG_PORT", "5432")
        monkeypatch.setenv("AGENT_PG_DATABASE", "test")
        monkeypatch.setenv("AGENT_PG_USER", "test")
        monkeypatch.setenv("AGENT_PG_PASSWORD", "test")

        mock_checkpoint = MagicMock()
        del mock_checkpoint.setup
        mock_checkpoint.__aenter__ = AsyncMock(return_value=mock_checkpoint)
        mock_checkpoint.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "template_agent.src.core.agent.AsyncPostgresSaver.from_conn_string",
            return_value=mock_checkpoint,
        ):
            from template_agent.src.core.agent import initialize_database

            await initialize_database()

    async def test_raises_on_connection_failure(self, monkeypatch):
        """Raises AppException on database connection failure."""
        monkeypatch.setenv("USE_INMEMORY_SAVER", "false")
        monkeypatch.setenv("AGENT_PG_HOST", "localhost")
        monkeypatch.setenv("AGENT_PG_PORT", "5432")
        monkeypatch.setenv("AGENT_PG_DATABASE", "test")
        monkeypatch.setenv("AGENT_PG_USER", "test")
        monkeypatch.setenv("AGENT_PG_PASSWORD", "test")

        mock_checkpoint = MagicMock()
        mock_checkpoint.__aenter__ = AsyncMock(
            side_effect=Exception("Connection failed")
        )
        mock_checkpoint.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "template_agent.src.core.agent.AsyncPostgresSaver.from_conn_string",
            return_value=mock_checkpoint,
        ):
            from template_agent.src.core.agent import initialize_database

            with pytest.raises(AppException) as exc_info:
                await initialize_database()

            assert "Database initialization failed" in str(exc_info.value)
