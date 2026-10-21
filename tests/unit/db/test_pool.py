"""Tests for the shared Postgres pool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from psycopg.rows import dict_row, tuple_row
from psycopg_pool import AsyncConnectionPool

import deep_agent.src.db.pool as dbp


@pytest.fixture(autouse=True)
def _clean_pool():
    dbp._pool = None
    yield
    dbp._pool = None


class TestGetPool:
    async def test_checks_liveness_on_checkout(self):
        """A connection the server already closed must not be handed out.

        psycopg's pool defaults to ``check=None``, which makes the
        next borrower absorb a failure it had nothing to do with —
        and after a failover that is one per pooled connection at
        once.
        """
        with patch.object(dbp, "AsyncConnectionPool", autospec=True) as pool_cls:
            pool_cls.check_connection = AsyncConnectionPool.check_connection
            pool_cls.return_value.open = AsyncMock()
            await dbp.get_pool()

        kwargs = pool_cls.call_args.kwargs
        assert kwargs["check"] is AsyncConnectionPool.check_connection

    async def test_does_not_use_server_side_prepared_statements(self):
        """They pin a session on PgBouncer and RDS Proxy."""
        with patch.object(dbp, "AsyncConnectionPool", autospec=True) as pool_cls:
            pool_cls.return_value.open = AsyncMock()
            await dbp.get_pool()

        assert pool_cls.call_args.kwargs["kwargs"]["prepare_threshold"] is None

    async def test_is_created_once(self):
        with patch.object(dbp, "AsyncConnectionPool", autospec=True) as pool_cls:
            pool_cls.return_value.open = AsyncMock()
            first = await dbp.get_pool()
            second = await dbp.get_pool()

        assert first is second
        assert pool_cls.call_count == 1

    async def test_bounds_come_from_settings(self):
        with (
            patch.object(dbp.settings, "DB_POOL_MAX_SIZE", 3),
            patch.object(dbp.settings, "DB_POOL_TIMEOUT", 7.0),
            patch.object(dbp, "AsyncConnectionPool", autospec=True) as pool_cls,
        ):
            pool_cls.return_value.open = AsyncMock()
            await dbp.get_pool()

        kwargs = pool_cls.call_args.kwargs
        assert kwargs["max_size"] == 3
        assert kwargs["timeout"] == 7.0


class TestConfigure:
    async def test_sets_a_statement_timeout_once_per_connection(self):
        """On the connection, not per request.

        A bare ``SET`` inside a handler would follow the pooled
        connection to its next borrower, and pin the session on a
        transaction pooler.
        """
        conn = AsyncMock()
        with patch.object(dbp.settings, "DB_STATEMENT_TIMEOUT_MS", 5000):
            await dbp._configure(conn)
        conn.execute.assert_awaited_once_with("SET statement_timeout = 5000")

    async def test_zero_disables_it(self):
        conn = AsyncMock()
        with patch.object(dbp.settings, "DB_STATEMENT_TIMEOUT_MS", 0):
            await dbp._configure(conn)
        conn.execute.assert_not_awaited()


class TestConnection:
    async def _pool_yielding(self, conn):
        pool = MagicMock()
        pool.connection.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.connection.return_value.__aexit__ = AsyncMock(return_value=False)
        return pool

    async def test_restores_the_row_factory(self):
        """The connection outlives the request that borrowed it.

        Leaving a caller's row factory behind would silently change
        the shape of rows for whatever borrows it next.
        """
        conn = MagicMock()
        conn.row_factory = dict_row
        pool = await self._pool_yielding(conn)

        with patch.object(dbp, "get_pool", AsyncMock(return_value=pool)):
            async with dbp.connection(row_factory=tuple_row) as got:
                assert got is conn
                assert conn.row_factory is tuple_row

        assert conn.row_factory is dict_row

    async def test_restores_the_row_factory_on_exception(self):
        conn = MagicMock()
        conn.row_factory = dict_row
        pool = await self._pool_yielding(conn)

        with patch.object(dbp, "get_pool", AsyncMock(return_value=pool)):
            with pytest.raises(RuntimeError, match="boom"):
                async with dbp.connection(row_factory=tuple_row):
                    raise RuntimeError("boom")

        assert conn.row_factory is dict_row


class TestPoolStats:
    def test_empty_before_the_pool_opens(self):
        assert dbp.pool_stats() == {}
