"""Cache metrics — hit/miss/eviction counters per cache name.

Counters are in-memory per process. When ``CACHE_METRICS_ENABLED``
is true, periodic summaries are logged at INFO level.
"""

import threading
from typing import Any

from deep_agent.src.cache.config import cache_settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_lock = threading.Lock()
_counters: dict[str, dict[str, int]] = {}


def _ensure(name: str) -> dict[str, int]:
    if name not in _counters:
        _counters[name] = {"hits": 0, "misses": 0, "sets": 0, "deletes": 0}
    return _counters[name]


def record_hit(cache_name: str) -> None:
    """Increment hit counter for *cache_name*."""
    if not cache_settings.is_enabled("metrics"):
        return
    with _lock:
        _ensure(cache_name)["hits"] += 1


def record_miss(cache_name: str) -> None:
    """Increment miss counter for *cache_name*."""
    if not cache_settings.is_enabled("metrics"):
        return
    with _lock:
        _ensure(cache_name)["misses"] += 1


def record_set(cache_name: str) -> None:
    """Increment set counter for *cache_name*."""
    if not cache_settings.is_enabled("metrics"):
        return
    with _lock:
        _ensure(cache_name)["sets"] += 1


def record_delete(cache_name: str) -> None:
    """Increment delete counter for *cache_name*."""
    if not cache_settings.is_enabled("metrics"):
        return
    with _lock:
        _ensure(cache_name)["deletes"] += 1


def snapshot() -> dict[str, dict[str, int]]:
    """Return a copy of all counters."""
    with _lock:
        return {k: dict(v) for k, v in _counters.items()}


def reset() -> None:
    """Clear all counters."""
    with _lock:
        _counters.clear()


def log_summary() -> None:
    """Log current counters at INFO level."""
    if not cache_settings.is_enabled("metrics"):
        return
    stats = snapshot()
    if not stats:
        return
    for name, counts in stats.items():
        total = counts["hits"] + counts["misses"]
        rate = (counts["hits"] / total * 100) if total > 0 else 0.0
        logger.info(
            "Cache '%s': %d hits, %d misses (%.1f%% hit rate), %d sets, %d deletes",
            name,
            counts["hits"],
            counts["misses"],
            rate,
            counts["sets"],
            counts["deletes"],
        )


def get_stats() -> dict[str, Any]:
    """Return metrics as a JSON-serialisable dict (for /health or /metrics)."""
    stats = snapshot()
    result: dict[str, Any] = {}
    for name, counts in stats.items():
        total = counts["hits"] + counts["misses"]
        result[name] = {
            **counts,
            "total": total,
            "hit_rate": round(counts["hits"] / total * 100, 1) if total > 0 else 0.0,
        }
    return result
