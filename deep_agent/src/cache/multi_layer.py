"""Two-layer cache: L1 in-process memory + L2 shared Redis.

On ``get``:
    L1 hit → return immediately
    L1 miss → check L2 → backfill L1 on hit

On ``set``:
    Write to both L1 and L2

On ``delete``:
    Delete from both L1 and L2
"""

from deep_agent.src.cache import metrics
from deep_agent.src.cache.backend import CacheBackend, NullCache
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


class MultiLayerCache:
    """Composite cache with L1 (fast/local) and optional L2 (shared/Redis).

    Args:
        name: Human-readable name used in metrics and logging.
        l1: Primary (fast) cache backend.
        l2: Secondary (shared) cache backend, or None to skip.
    """

    def __init__(
        self,
        name: str,
        l1: CacheBackend,
        l2: CacheBackend | None = None,
    ) -> None:
        """Initialise with a name and one or two backend layers."""
        self._name = name
        self._l1 = l1
        self._l2 = l2

    @property
    def name(self) -> str:
        """Human-readable cache name used in metrics."""
        return self._name

    def get(self, key: str) -> str | None:
        """Look up *key* in L1, then L2. Backfills L1 on L2 hit."""
        value = self._l1.get(key)
        if value is not None:
            metrics.record_hit(self._name)
            return value

        if self._l2 is not None:
            value = self._l2.get(key)
            if value is not None:
                self._l1.set(key, value)
                metrics.record_hit(self._name)
                return value

        metrics.record_miss(self._name)
        return None

    def set(self, key: str, value: str, ttl: int | None = None) -> bool:
        """Write *value* to L1 and L2."""
        metrics.record_set(self._name)
        ok = self._l1.set(key, value, ttl)
        if self._l2 is not None:
            self._l2.set(key, value, ttl)
        return ok

    def delete(self, key: str) -> bool:
        """Remove *key* from both layers."""
        metrics.record_delete(self._name)
        ok = self._l1.delete(key)
        if self._l2 is not None:
            self._l2.delete(key)
        return ok

    def clear(self) -> None:
        """Clear L1. L2 clear is intentionally a no-op (safety)."""
        self._l1.clear()


def create_null_layer(name: str) -> MultiLayerCache:
    """Return a no-op MultiLayerCache (used when caching is disabled)."""
    return MultiLayerCache(name=name, l1=NullCache())
