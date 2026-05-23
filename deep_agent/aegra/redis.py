"""Redis connection configuration for aegra deployment (MR-20).

Provides a Redis client factory for caching, rate limiting, and
pub/sub in the LangGraph Platform deployment. Falls back gracefully
if Redis is unavailable — the agent operates without caching.

Environment variables:
    REDIS_URL: Full Redis URL (default: redis://localhost:6379/0)
    REDIS_MAX_CONNECTIONS: Pool size (default: 10)
    REDIS_SOCKET_TIMEOUT: Seconds (default: 5)
    REDIS_RETRY_ON_TIMEOUT: Enable retry (default: true)
"""

import os
from typing import Any

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
REDIS_MAX_CONNECTIONS = int(os.environ.get("REDIS_MAX_CONNECTIONS", "10"))
REDIS_SOCKET_TIMEOUT = int(os.environ.get("REDIS_SOCKET_TIMEOUT", "5"))
REDIS_RETRY_ON_TIMEOUT = (
    os.environ.get("REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true"
)
REDIS_KEY_PREFIX = os.environ.get("REDIS_KEY_PREFIX", "aegra:")

_client: Any = None


def get_redis_config() -> dict[str, Any]:
    """Return the Redis configuration dict for documentation/debugging."""
    return {
        "url": REDIS_URL,
        "max_connections": REDIS_MAX_CONNECTIONS,
        "socket_timeout": REDIS_SOCKET_TIMEOUT,
        "retry_on_timeout": REDIS_RETRY_ON_TIMEOUT,
        "key_prefix": REDIS_KEY_PREFIX,
    }


def get_redis_client() -> Any:
    """Get or create a Redis client with connection pooling.

    Returns:
        Redis client instance, or None if Redis is unavailable.
    """
    global _client  # noqa: PLW0603
    if _client is not None:
        return _client

    try:
        import redis

        _client = redis.from_url(
            REDIS_URL,
            max_connections=REDIS_MAX_CONNECTIONS,
            socket_timeout=REDIS_SOCKET_TIMEOUT,
            retry_on_timeout=REDIS_RETRY_ON_TIMEOUT,
            decode_responses=True,
        )
        _client.ping()
        logger.info("Redis connected: %s", REDIS_URL)
        return _client
    except ImportError:
        logger.warning("redis package not installed — caching disabled")
        return None
    except Exception:
        logger.warning(
            "Redis unavailable at %s — caching disabled", REDIS_URL, exc_info=True
        )
        _client = None
        return None


def cache_get(key: str) -> str | None:
    """Read a value from Redis cache. Returns None on miss or error."""
    client = get_redis_client()
    if client is None:
        return None
    try:
        val = client.get(f"{REDIS_KEY_PREFIX}{key}")
        return str(val) if val is not None else None
    except Exception:
        logger.debug("Cache read failed for key '%s'", key, exc_info=True)
        return None


def cache_set(key: str, value: str, ttl_seconds: int = 300) -> bool:
    """Write a value to Redis cache with TTL. Returns False on error."""
    client = get_redis_client()
    if client is None:
        return False
    try:
        client.setex(f"{REDIS_KEY_PREFIX}{key}", ttl_seconds, value)
        return True
    except Exception:
        logger.debug("Cache write failed for key '%s'", key, exc_info=True)
        return False


def cache_delete(key: str) -> bool:
    """Delete a key from Redis cache. Returns False on error."""
    client = get_redis_client()
    if client is None:
        return False
    try:
        client.delete(f"{REDIS_KEY_PREFIX}{key}")
        return True
    except Exception:
        return False
