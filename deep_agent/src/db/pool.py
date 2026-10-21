"""One Postgres pool for everything in deep_agent that is not LangGraph.

Before this module there were four pools and two unpooled call sites,
each sized by whoever wrote it. Per pod that came to 64 connections
from the pools alone, plus an unbounded number from
``eval_routes._pg_conn`` opening a raw connection per request and
``personalization_routes`` building an entire asyncpg pool per call.
Against a server ``max_connections`` of 100 and an HPA that scales to
ten pods, the budget was both over the limit and impossible to
compute. ``docs/scalability.md`` has the arithmetic.

Two properties matter beyond the count:

**Liveness on checkout.** ``check`` is set, which psycopg's pool
leaves at ``None`` by default. A connection can die server-side
without the client noticing — a proxy's idle timeout, a database
failover, a network partition — and a pool with no check hands that
corpse to the next borrower, failing a request that had nothing to do
with the cause. After a failover that is not one bad request but one
per pooled connection, arriving together and looking like an
application fault. The SQLAlchemy engine already does this via
``pool_pre_ping``; this is the psycopg equivalent.

**A statement timeout, applied once.** Set in ``configure``, which
runs when a connection is created, rather than per request. ``SET``
inside a handler would leak the setting to the next borrower and, on
a transaction pooler, pin the session.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_pool: AsyncConnectionPool | None = None
_lock = asyncio.Lock()


async def _configure(conn: Any) -> None:
    """Run once per new connection, not per checkout."""
    if settings.DB_STATEMENT_TIMEOUT_MS > 0:
        await conn.execute(
            f"SET statement_timeout = {int(settings.DB_STATEMENT_TIMEOUT_MS)}"
        )


async def get_pool() -> AsyncConnectionPool:
    """Return the shared pool, opening it on first use."""
    global _pool  # noqa: PLW0603
    if _pool is not None:
        return _pool
    async with _lock:
        if _pool is None:
            pool = AsyncConnectionPool(
                settings.database_uri,
                min_size=settings.DB_POOL_MIN_SIZE,
                max_size=settings.DB_POOL_MAX_SIZE,
                timeout=settings.DB_POOL_TIMEOUT,
                max_idle=settings.DB_POOL_MAX_IDLE,
                max_lifetime=settings.DB_POOL_MAX_LIFETIME,
                # Hand out connections the server still has open.
                check=AsyncConnectionPool.check_connection,
                configure=_configure,
                kwargs={
                    "autocommit": True,
                    # No server-side prepared statements: they pin a
                    # session on PgBouncer and RDS Proxy, which defeats
                    # the multiplexing those exist to provide.
                    "prepare_threshold": None,
                    "row_factory": dict_row,
                    "keepalives": 1,
                    "keepalives_idle": 300,
                    "keepalives_interval": 30,
                    "keepalives_count": 3,
                },
                open=False,
            )
            await pool.open()
            _pool = pool
            logger.info(
                "Postgres pool open: min=%d max=%d timeout=%.1fs "
                "statement_timeout=%dms",
                settings.DB_POOL_MIN_SIZE,
                settings.DB_POOL_MAX_SIZE,
                settings.DB_POOL_TIMEOUT,
                settings.DB_STATEMENT_TIMEOUT_MS,
            )
    return _pool


@asynccontextmanager
async def connection(row_factory: Any = None) -> AsyncIterator[Any]:
    """Borrow a connection, returning it on success or on exception.

    ``row_factory`` overrides the pool default for this checkout only;
    it is restored before the connection goes back. Callers that read
    rows positionally need ``psycopg.rows.tuple_row``.
    """
    pool = await get_pool()
    async with pool.connection() as conn:
        if row_factory is None:
            yield conn
            return
        previous = conn.row_factory
        conn.row_factory = row_factory
        try:
            yield conn
        finally:
            # The connection outlives this request. Leaving a caller's
            # row factory on it would silently change the shape of
            # whatever borrows it next.
            conn.row_factory = previous


async def close_pool() -> None:
    """Release the pool on shutdown."""
    global _pool  # noqa: PLW0603
    if _pool is not None:
        await _pool.close()
        _pool = None


def pool_stats() -> dict[str, Any]:
    """Pool counters for health and metrics, or empty if never opened."""
    if _pool is None:
        return {}
    stats = _pool.get_stats()
    return {
        "size": stats.get("pool_size", 0),
        "available": stats.get("pool_available", 0),
        "waiting": stats.get("requests_waiting", 0),
        "max_size": settings.DB_POOL_MAX_SIZE,
    }
