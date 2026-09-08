"""Unit tests for the shared async PostgreSQL connection pool module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_agent.aegra import db


@pytest.fixture(autouse=True)
def _reset_pool():
    """Ensure the global pool is None before and after each test."""
    db._pool = None
    yield
    db._pool = None


class TestInitPool:
    async def test_creates_pool_and_opens_it(self):
        mock_pool = AsyncMock()
        with patch("deep_agent.aegra.db.AsyncConnectionPool", return_value=mock_pool):
            await db.init_pool("postgresql://test")

        assert db._pool is mock_pool
        mock_pool.open.assert_awaited_once_with(wait=True)

    async def test_appends_connect_timeout_to_conninfo(self):
        mock_pool = AsyncMock()
        with patch(
            "deep_agent.aegra.db.AsyncConnectionPool", return_value=mock_pool
        ) as mock_cls:
            await db.init_pool("postgresql://test")

        call_kwargs = mock_cls.call_args[1]
        assert "connect_timeout=5" in call_kwargs["conninfo"]

    async def test_pool_not_assigned_on_open_failure(self):
        mock_pool = AsyncMock()
        mock_pool.open.side_effect = OSError("connection refused")
        with (
            patch("deep_agent.aegra.db.AsyncConnectionPool", return_value=mock_pool),
            pytest.raises(OSError, match="connection refused"),
        ):
            await db.init_pool("postgresql://test")

        assert db._pool is None

    async def test_idempotent_when_pool_exists(self):
        db._pool = MagicMock()
        existing = db._pool
        with patch("deep_agent.aegra.db.AsyncConnectionPool") as mock_cls:
            await db.init_pool("postgresql://test")

        mock_cls.assert_not_called()
        assert db._pool is existing


class TestClosePool:
    async def test_closes_and_clears_pool(self):
        mock_pool = AsyncMock()
        db._pool = mock_pool

        await db.close_pool()

        mock_pool.close.assert_awaited_once()
        assert db._pool is None

    async def test_noop_when_no_pool(self):
        await db.close_pool()
        assert db._pool is None


class TestGetPool:
    def test_returns_none_when_not_initialised(self):
        assert db.get_pool() is None

    def test_returns_pool_when_initialised(self):
        sentinel = MagicMock()
        db._pool = sentinel
        assert db.get_pool() is sentinel


class TestAsyncConnection:
    async def test_uses_pool_when_available(self):
        mock_conn = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.connection.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.connection.return_value.__aexit__ = AsyncMock(return_value=False)
        db._pool = mock_pool

        async with db.async_connection() as conn:
            assert conn is mock_conn

    async def test_sets_row_factory_on_pool_connection(self):
        from psycopg.rows import dict_row

        mock_conn = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.connection.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.connection.return_value.__aexit__ = AsyncMock(return_value=False)
        db._pool = mock_pool

        async with db.async_connection(row_factory=dict_row) as conn:
            assert conn.row_factory is dict_row

    async def test_restores_row_factory_after_pool_connection(self):
        from psycopg.rows import dict_row, tuple_row

        mock_conn = MagicMock()
        mock_conn.row_factory = tuple_row
        mock_pool = MagicMock()
        mock_pool.connection.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.connection.return_value.__aexit__ = AsyncMock(return_value=False)
        db._pool = mock_pool

        async with db.async_connection(row_factory=dict_row) as conn:
            assert conn.row_factory is dict_row

        assert mock_conn.row_factory is tuple_row

    async def test_falls_back_to_direct_connection(self):
        mock_conn = AsyncMock()
        mock_connect = AsyncMock(return_value=mock_conn)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        mock_settings = MagicMock()
        mock_settings.database_uri = "postgresql://fallback"

        with (
            patch("psycopg.AsyncConnection.connect", mock_connect),
            patch("deep_agent.src.settings.settings", mock_settings),
        ):
            async with db.async_connection() as conn:
                assert conn is mock_conn

        call_kwargs = mock_connect.call_args
        assert call_kwargs[1]["connect_timeout"] == db._CONNECT_TIMEOUT
