"""Cache backend implementations.

Provides a ``CacheBackend`` protocol and three implementations:

- ``NullCache``: No-op (returned when caching is disabled)
- ``InMemoryCache``: Process-local TTLCache via ``cachetools``
- ``RedisCache``: Shared cache via the existing ``aegra.redis`` client
"""

import threading
from typing import Any, Protocol, runtime_checkable

from cachetools import TTLCache  # type: ignore[import-untyped]

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


@runtime_checkable
class CacheBackend(Protocol):
    """Minimal cache interface — get/set/delete/clear with string values."""

    def get(self, key: str) -> str | None:
        """Retrieve a cached value by key, or None on miss."""
        ...

    def set(self, key: str, value: str, ttl: int | None = None) -> bool:
        """Store a value; return True on success."""
        ...

    def delete(self, key: str) -> bool:
        """Remove a key; return True if it existed."""
        ...

    def clear(self) -> None:
        """Remove all entries."""
        ...

    @property
    def name(self) -> str:
        """Human-readable backend name."""
        ...


class NullCache:
    """No-op cache — every operation is a silent miss."""

    @property
    def name(self) -> str:
        """Return backend name."""
        return "null"

    def get(self, key: str) -> str | None:
        """Always return None."""
        return None

    def set(self, key: str, value: str, ttl: int | None = None) -> bool:
        """Always return False (nothing stored)."""
        return False

    def delete(self, key: str) -> bool:
        """Always return False (nothing to delete)."""
        return False

    def clear(self) -> None:
        """No-op."""


class InMemoryCache:
    """Process-local TTL cache backed by ``cachetools.TTLCache``.

    Thread-safe via an internal lock.

    Args:
        max_size: Maximum number of entries.
        default_ttl: Default time-to-live in seconds.
    """

    def __init__(self, max_size: int = 256, default_ttl: int = 300) -> None:
        """Initialise with capacity and TTL."""
        self._cache: TTLCache[str, str] = TTLCache(maxsize=max_size, ttl=default_ttl)
        self._default_ttl = default_ttl
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        """Return backend name."""
        return "memory"

    def get(self, key: str) -> str | None:
        """Look up *key* in the TTL cache."""
        with self._lock:
            result: str | None = self._cache.get(key)
            return result

    def set(self, key: str, value: str, ttl: int | None = None) -> bool:
        """Insert or overwrite *key*."""
        with self._lock:
            self._cache[key] = value
            return True

    def delete(self, key: str) -> bool:
        """Remove *key* if present."""
        with self._lock:
            try:
                del self._cache[key]
                return True
            except KeyError:
                return False

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        """Current number of entries."""
        with self._lock:
            return len(self._cache)


class RedisCache:
    """Shared cache via the existing ``aegra.redis`` client.

    Falls back to no-op if Redis is unavailable — never raises.

    Args:
        default_ttl: Default TTL in seconds.
        key_prefix: Prefix prepended to all keys (namespacing).
    """

    def __init__(self, default_ttl: int = 300, key_prefix: str = "cache:") -> None:
        """Initialise with TTL and key prefix."""
        self._default_ttl = default_ttl
        self._prefix = key_prefix
        self._client: Any = None
        self._checked = False

    def _get_client(self) -> Any:
        if not self._checked:
            try:
                from deep_agent.aegra.redis import get_redis_client

                self._client = get_redis_client()
            except Exception:
                logger.debug("Redis unavailable for cache layer", exc_info=True)
                self._client = None
            self._checked = True
        return self._client

    @property
    def name(self) -> str:
        """Return backend name."""
        return "redis"

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def get(self, key: str) -> str | None:
        """Read from Redis; return None on miss or error."""
        client = self._get_client()
        if client is None:
            return None
        try:
            result: str | None = client.get(self._key(key))
            return result
        except Exception:
            logger.debug("Redis cache GET failed for '%s'", key, exc_info=True)
            return None

    def set(self, key: str, value: str, ttl: int | None = None) -> bool:
        """Write to Redis with TTL."""
        client = self._get_client()
        if client is None:
            return False
        try:
            client.setex(self._key(key), ttl or self._default_ttl, value)
            return True
        except Exception:
            logger.debug("Redis cache SET failed for '%s'", key, exc_info=True)
            return False

    def delete(self, key: str) -> bool:
        """Delete from Redis."""
        client = self._get_client()
        if client is None:
            return False
        try:
            client.delete(self._key(key))
            return True
        except Exception:
            return False

    def clear(self) -> None:
        """Clear is not supported for Redis (too dangerous). No-op."""
