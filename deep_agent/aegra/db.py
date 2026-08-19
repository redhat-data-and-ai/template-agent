"""Shared async PostgreSQL connection pool.

Centralises all database access behind a single ``psycopg_pool.AsyncConnectionPool``
so that:
- Connection overhead is amortised across requests (no per-query TCP handshake).
- ``connect_timeout`` is enforced uniformly (prevents 2-min OS-level TCP hangs).
- Pool size is bounded, preventing connection exhaustion under load.

Usage::

    from deep_agent.aegra.db import async_connection

    async with async_connection() as conn:
        await conn.execute("SELECT 1")
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)

_pool: AsyncConnectionPool | None = None

_CONNECT_TIMEOUT = 5
_POOL_MIN_SIZE = 2
_POOL_MAX_SIZE = 10


async def init_pool(uri: str) -> None:
    """Create and open the shared connection pool."""
    global _pool  # noqa: PLW0603
    if _pool is not None:
        return
    conninfo = f"{uri} connect_timeout={_CONNECT_TIMEOUT}"
    pool = AsyncConnectionPool(
        conninfo=conninfo,
        min_size=_POOL_MIN_SIZE,
        max_size=_POOL_MAX_SIZE,
        open=False,
    )
    await pool.open(wait=True)
    _pool = pool
    logger.info(
        "Async connection pool opened (min=%d, max=%d)", _POOL_MIN_SIZE, _POOL_MAX_SIZE
    )


async def close_pool() -> None:
    """Close the shared connection pool."""
    global _pool  # noqa: PLW0603
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("Async connection pool closed")


def get_pool() -> AsyncConnectionPool | None:
    """Return the shared pool, or None if not initialised."""
    return _pool


@asynccontextmanager
async def async_connection(**kwargs: Any) -> AsyncIterator[Any]:
    """Yield a connection from the shared pool.

    Falls back to a direct connection when the pool is not initialised
    (e.g. during early startup or in tests).
    """
    if _pool is not None:
        async with _pool.connection() as conn:
            original_row_factory = conn.row_factory
            if kwargs.get("row_factory") is dict_row:
                conn.row_factory = dict_row
            try:
                yield conn
            finally:
                conn.row_factory = original_row_factory
    else:
        import psycopg

        from deep_agent.src.settings import settings

        uri = settings.database_uri
        async with await psycopg.AsyncConnection.connect(
            uri, **kwargs, connect_timeout=_CONNECT_TIMEOUT
        ) as conn:
            yield conn
