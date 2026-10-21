"""Shared Postgres access for deep_agent."""

from deep_agent.src.db.pool import close_pool, connection, get_pool

__all__ = ["close_pool", "connection", "get_pool"]
