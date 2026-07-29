"""Personalization cache — Redis L2 for user memories and rules.

Avoids hitting Postgres on every request for the same user's
personalization data. Stores serialised JSON in Redis, keyed by
``user_id``.

Feature flag: ``CACHE_PERSONALIZATION_ENABLED`` (+ ``CACHE_ENABLED``).
"""

import json
from typing import Any

from deep_agent.src.cache import metrics
from deep_agent.src.cache.backend import RedisCache
from deep_agent.src.cache.config import cache_settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_KEY_PREFIX = "personalization:"

_redis: RedisCache | None = None


def _get_redis() -> RedisCache:
    global _redis  # noqa: PLW0603
    if _redis is None:
        _redis = RedisCache(
            default_ttl=cache_settings.CACHE_PERSONALIZATION_TTL,
            key_prefix=_KEY_PREFIX,
        )
    return _redis


def _cache_key(user_id: str) -> str:
    return f"user:{user_id}"


async def get_personalization(
    user_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Return cached ``(memories, rules)`` dicts or None on miss.

    When disabled, always returns None (caller falls through to DB).
    """
    if not cache_settings.is_enabled("personalization"):
        return None

    raw = _get_redis().get(_cache_key(user_id))
    if raw is None:
        metrics.record_miss("personalization")
        return None

    try:
        data = json.loads(raw)
        metrics.record_hit("personalization")
        logger.debug("Personalization cache HIT for user %s", user_id[:8])
        return data["memories"], data["rules"]
    except (json.JSONDecodeError, KeyError):
        logger.debug(
            "Personalization cache corrupt for user %s — evicting", user_id[:8]
        )
        _get_redis().delete(_cache_key(user_id))
        metrics.record_miss("personalization")
        return None


async def set_personalization(
    user_id: str,
    memories: list[dict[str, Any]],
    rules: list[dict[str, Any]],
) -> None:
    """Store personalization data in Redis cache."""
    if not cache_settings.is_enabled("personalization"):
        return

    payload = json.dumps({"memories": memories, "rules": rules})
    _get_redis().set(_cache_key(user_id), payload)
    metrics.record_set("personalization")
    logger.debug(
        "Personalization cached for user %s (%d memories, %d rules)",
        user_id[:8],
        len(memories),
        len(rules),
    )


async def invalidate(user_id: str | None = None, *, _retries: int = 2) -> None:
    """Evict cached personalization for a user.

    Retries up to ``_retries`` times on failure so a transient Redis
    hiccup doesn't leave stale data for the full TTL window.

    Args:
        user_id: Specific user to evict. ``None`` is a no-op
            (clearing all Redis keys is too dangerous).
    """
    if user_id is None:
        return
    key = _cache_key(user_id)
    for attempt in range(_retries + 1):
        try:
            _get_redis().delete(key)
            metrics.record_delete("personalization")
            return
        except Exception:
            if attempt < _retries:
                logger.debug(
                    "Cache invalidation retry %d/%d for user %s",
                    attempt + 1,
                    _retries,
                    user_id[:8],
                )
            else:
                logger.warning(
                    "Cache invalidation failed after %d retries for user %s",
                    _retries + 1,
                    user_id[:8],
                    exc_info=True,
                )
                raise
