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

import asyncio
import os
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

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


def close_redis_client() -> None:
    """Close the Redis client connection if open. Idempotent."""
    global _client  # noqa: PLW0603
    if _client is None:
        return
    try:
        _client.close()
        logger.info("Redis client closed")
    except Exception:
        logger.debug("Redis close error", exc_info=True)
    finally:
        _client = None


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


def cache_set_persistent(key: str, value: str) -> bool:
    """Write a value to Redis without expiry. Returns False on error."""
    client = get_redis_client()
    if client is None:
        return False
    try:
        client.set(f"{REDIS_KEY_PREFIX}{key}", value)
        return True
    except Exception:
        logger.debug("Persistent cache write failed for key '%s'", key, exc_info=True)
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


_RELEASE_LOCK_LUA = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""


def _lock_key(name: str) -> str:
    return f"{REDIS_KEY_PREFIX}lock:{name}"


def acquire_distributed_lock(
    name: str,
    *,
    ttl_seconds: int = 30,
    wait_seconds: float = 10.0,
    poll_interval: float = 0.05,
) -> str | None:
    """Acquire a Redis lock. Returns a token, or None if unavailable or timed out."""
    client = get_redis_client()
    if client is None:
        return None

    token = secrets.token_urlsafe(16)
    key = _lock_key(name)
    deadline = time.monotonic() + wait_seconds

    while True:
        try:
            if client.set(key, token, nx=True, ex=ttl_seconds):
                return token
        except Exception:
            logger.debug("Lock acquire failed for '%s'", name, exc_info=True)
            return None

        if time.monotonic() >= deadline:
            return None
        time.sleep(poll_interval)


def release_distributed_lock(name: str, token: str) -> bool:
    """Release a Redis lock when the token still matches."""
    client = get_redis_client()
    if client is None:
        return False
    try:
        return bool(client.eval(_RELEASE_LOCK_LUA, 1, _lock_key(name), token))
    except Exception:
        logger.debug("Lock release failed for '%s'", name, exc_info=True)
        return False


LockState = Literal["held", "no_redis", "timeout"]


@asynccontextmanager
async def distributed_lock(
    name: str,
    *,
    ttl_seconds: int = 30,
    wait_seconds: float = 10.0,
) -> AsyncIterator[LockState]:
    """Yield lock state for a Redis-backed distributed lock."""
    if get_redis_client() is None:
        yield "no_redis"
        return

    token = await asyncio.to_thread(
        acquire_distributed_lock,
        name,
        ttl_seconds=ttl_seconds,
        wait_seconds=wait_seconds,
    )
    if token is None:
        yield "timeout"
        return

    try:
        yield "held"
    finally:
        await asyncio.to_thread(release_distributed_lock, name, token)
