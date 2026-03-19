"""Store-first plan storage for deep research with local LRU cache.

Plans are persisted to the LangGraph memory store (Postgres-backed) as the
source of truth.  A local in-memory LRU cache provides fast reads within the
same pod.  All writes go to the store *first*; the local cache is updated
only after persistence succeeds.  This guarantees multi-pod consistency:
any pod can resume a plan that was created on a different pod.

# TODO: The module-level OrderedDict caches (_PLAN_STORE, _PLAN_METADATA,
# _PLAN_ENRICHMENT, _PLAN_CONTEXT, _THREAD_OWNERS) are local to each pod.
# In a horizontally-scaled deployment the LangGraph store (Postgres) is the
# authoritative source.  If you observe stale reads across pods, consider
# adding a TTL to the local cache or switching to Redis as a shared L1 cache.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from template_agent.src.core.deep_research.utils import get_setting as _get_setting
from template_agent.utils.pylogger import get_python_logger

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore

logger = get_python_logger()

_MAX_PLANS = 500
_MAX_THREAD_OWNERS = 10_000

_PLAN_STORE: OrderedDict[str, list[str]] = OrderedDict()
_PLAN_METADATA: OrderedDict[str, dict[str, Any]] = OrderedDict()
_PLAN_ENRICHMENT: OrderedDict[str, dict[str, Any]] = OrderedDict()
_PLAN_CONTEXT: OrderedDict[str, dict[str, Any]] = OrderedDict()
_THREAD_OWNERS: OrderedDict[str, str] = OrderedDict()
_LOCKS: dict[int, asyncio.Lock] = {}


def _get_lock() -> asyncio.Lock:
    """Return an asyncio.Lock scoped to the current event loop."""
    loop_id = id(asyncio.get_running_loop())
    lock = _LOCKS.get(loop_id)
    if lock is None:
        lock = asyncio.Lock()
        _LOCKS[loop_id] = lock
    return lock


def _cache_key(thread_id: str, user_id: str | None = None) -> str:
    """Build a user-scoped cache key to prevent cross-user cache poisoning."""
    if user_id:
        return f"{user_id}:{thread_id}"
    return thread_id


def _evict_oldest() -> None:
    """Remove oldest entries when store exceeds max capacity. Must be called under lock."""
    max_plans = _get_setting("DEEP_RESEARCH_MAX_CACHED_PLANS", _MAX_PLANS)
    while len(_PLAN_STORE) > max_plans:
        evicted_key, _ = _PLAN_STORE.popitem(last=False)
        _PLAN_METADATA.pop(evicted_key, None)
        _PLAN_ENRICHMENT.pop(evicted_key, None)
        _PLAN_CONTEXT.pop(evicted_key, None)
        logger.debug(
            "Evicted plan for thread %s (store size exceeded %d)",
            evicted_key,
            max_plans,
        )


PLAN_NAMESPACE_SUFFIX = ("deep_research", "plans")


async def _cache_plan_locally(
    thread_id: str,
    plan: list[str],
    metadata: dict[str, Any] | None = None,
    enrichment: dict[str, Any] | None = None,
    user_id: str | None = None,
) -> None:
    """Update the local in-memory LRU cache (async-safe)."""
    key = _cache_key(thread_id, user_id)
    async with _get_lock():
        _PLAN_STORE[key] = list(plan)
        _PLAN_STORE.move_to_end(key)
        if metadata:
            _PLAN_METADATA[key] = dict(metadata)
            _PLAN_METADATA.move_to_end(key)
        if enrichment:
            _PLAN_ENRICHMENT[key] = dict(enrichment)
            _PLAN_ENRICHMENT.move_to_end(key)
        _evict_oldest()


async def set_plan(
    thread_id: str,
    plan: list[str],
    metadata: dict[str, Any] | None = None,
    enrichment: dict[str, Any] | None = None,
    store: BaseStore | None = None,
    user_id: str | None = None,
) -> None:
    """Persist a plan to the external store first, then cache locally.

    Store-first semantics: when *store* and *user_id* are provided the plan
    is written to the store **before** the local cache is updated.

    When store/user_id are unavailable (e.g. unit tests) the plan is cached
    locally only -- callers should treat this as a degraded path.
    """
    if store and user_id and thread_id:
        persisted = await persist_plan_to_store(
            store,
            user_id,
            thread_id,
            plan,
            query=(metadata or {}).get("query", ""),
            metadata=metadata,
        )
        if not persisted:
            logger.error(
                "Plan persistence to store failed for thread %s — "
                "caching locally only (DATA LOSS RISK in multi-pod)",
                thread_id,
            )
    elif thread_id:
        logger.debug(
            "No store/user_id for set_plan on thread %s — local cache only",
            thread_id,
        )

    await _cache_plan_locally(thread_id, plan, metadata, enrichment, user_id=user_id)


async def get_plan(thread_id: str, user_id: str | None = None) -> list[str] | None:
    """Retrieve a plan for the given thread (async-safe, in-memory only).

    For cross-session loading when in-memory misses, use get_plan_with_load().
    """
    key = _cache_key(thread_id, user_id)
    async with _get_lock():
        stored = _PLAN_STORE.get(key)
        if stored is not None:
            _PLAN_STORE.move_to_end(key)
            return list(stored)
        return None


async def get_plan_enrichment(
    thread_id: str, user_id: str | None = None
) -> dict[str, Any] | None:
    """Retrieve cached enrichment data for the given thread (async-safe)."""
    key = _cache_key(thread_id, user_id)
    async with _get_lock():
        stored = _PLAN_ENRICHMENT.get(key)
        if stored is not None:
            _PLAN_ENRICHMENT.move_to_end(key)
            return dict(stored)
        return None


async def _fetch_plan_value_from_store(
    store: BaseStore,
    namespace: tuple[str, ...],
    thread_id: str,
) -> Any:
    """Fetch plan value from store via aget or asearch fallback."""
    value = None
    if hasattr(store, "aget"):
        value = await store.aget(namespace=namespace, key=thread_id)
    if value is None and hasattr(store, "asearch"):
        results = await store.asearch(namespace, query=thread_id, limit=10)
        for item in results:
            v = getattr(item, "value", item) if hasattr(item, "value") else item
            if isinstance(v, dict) and v.get("thread_id") == thread_id:
                return v
    return value


def _update_cache_from_plan_value(thread_id: str, value: dict[str, Any]) -> None:
    """Update local LRU cache from plan value. Must be called under lock."""
    plan = value.get("plan")
    if not plan:
        return
    _PLAN_STORE[thread_id] = list(plan)
    _PLAN_STORE.move_to_end(thread_id)
    meta = {
        k: v
        for k, v in value.items()
        if k not in ("plan", "thread_id", "subquery_count")
    }
    if meta:
        _PLAN_METADATA[thread_id] = meta
        _PLAN_METADATA.move_to_end(thread_id)


async def load_plan_from_store(
    store: BaseStore,
    user_id: str,
    thread_id: str,
) -> list[str] | None:
    """Load a plan from the LangGraph store by thread_id."""
    if not user_id or not thread_id:
        return None
    try:
        namespace = (user_id, *PLAN_NAMESPACE_SUFFIX)
        value = await _fetch_plan_value_from_store(store, namespace, thread_id)
        if not value or not isinstance(value, dict):
            return None
        plan = value.get("plan")
        if not plan:
            return None
        key = _cache_key(thread_id, user_id)
        async with _get_lock():
            _update_cache_from_plan_value(key, value)
        return list(plan)
    except Exception as e:
        logger.warning("Failed to load plan from store: %s", e)
    return None


async def get_plan_with_load(
    thread_id: str,
    store: BaseStore | None = None,
    user_id: str | None = None,
) -> list[str] | None:
    """Get plan from memory, or load from store if cache miss."""
    plan = await get_plan(thread_id, user_id=user_id)
    if plan is not None:
        return plan
    if store and user_id and thread_id:
        return await load_plan_from_store(store, user_id, thread_id)
    return None


async def get_plan_metadata(
    thread_id: str, user_id: str | None = None
) -> dict[str, Any] | None:
    """Retrieve plan metadata for the given thread (async-safe)."""
    key = _cache_key(thread_id, user_id)
    async with _get_lock():
        stored = _PLAN_METADATA.get(key)
        return dict(stored) if stored else None


_CONTEXT_NAMESPACE_SUFFIX = ("deep_research", "plan_context")


async def set_plan_context(
    thread_id: str,
    context: dict[str, Any],
    user_id: str | None = None,
    store: Any | None = None,
) -> None:
    """Persist pre-plan state for resume.

    Writes to the remote store first (when available) so the context
    survives pod restarts and is accessible from any pod.
    """
    key = _cache_key(thread_id, user_id)
    ctx_copy = dict(context)

    if store and user_id:
        try:
            namespace = (user_id, *_CONTEXT_NAMESPACE_SUFFIX)
            await store.aput(namespace=namespace, key=thread_id, value=ctx_copy)
        except Exception as e:
            logger.warning(
                "Failed to persist plan context to store for thread %s: %s",
                thread_id,
                e,
            )

    async with _get_lock():
        _PLAN_CONTEXT[key] = ctx_copy
        _PLAN_CONTEXT.move_to_end(key)


async def get_plan_context(
    thread_id: str,
    user_id: str | None = None,
    store: Any | None = None,
) -> dict[str, Any] | None:
    """Retrieve persisted pre-plan state for the given thread (async-safe)."""
    key = _cache_key(thread_id, user_id)
    async with _get_lock():
        stored = _PLAN_CONTEXT.get(key)
        if stored is not None:
            _PLAN_CONTEXT.move_to_end(key)
            return dict(stored)

    if store and user_id:
        try:
            namespace = (user_id, *_CONTEXT_NAMESPACE_SUFFIX)
            item = await store.aget(namespace=namespace, key=thread_id)
            if item and hasattr(item, "value") and item.value:
                ctx = dict(item.value)
                async with _get_lock():
                    _PLAN_CONTEXT[key] = ctx
                    _PLAN_CONTEXT.move_to_end(key)
                return ctx
        except Exception as e:
            logger.warning(
                "Failed to load plan context from store for thread %s: %s", thread_id, e
            )
    return None


async def clear_plan(thread_id: str, user_id: str | None = None) -> None:
    """Clear a stored plan for the given thread (async-safe)."""
    key = _cache_key(thread_id, user_id)
    async with _get_lock():
        _PLAN_STORE.pop(key, None)
        _PLAN_METADATA.pop(key, None)
        _PLAN_ENRICHMENT.pop(key, None)
        _PLAN_CONTEXT.pop(key, None)


async def register_thread_owner(thread_id: str, user_id: str) -> None:
    """Bind a thread to its owner (first-write-wins, async-safe)."""
    async with _get_lock():
        if thread_id in _THREAD_OWNERS:
            return
        _THREAD_OWNERS[thread_id] = user_id
        _THREAD_OWNERS.move_to_end(thread_id)
        while len(_THREAD_OWNERS) > _MAX_THREAD_OWNERS:
            _THREAD_OWNERS.popitem(last=False)


async def get_thread_owner(thread_id: str) -> str | None:
    """Return the registered owner of a thread, or None if unregistered."""
    async with _get_lock():
        return _THREAD_OWNERS.get(thread_id)


_PERSIST_MAX_RETRIES = 2
_PERSIST_BASE_DELAY_S = 0.5


async def persist_plan_to_store(
    store: BaseStore,
    user_id: str,
    thread_id: str,
    plan: list[str],
    query: str,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Persist a research plan to the LangGraph memory store.

    Retries up to ``_PERSIST_MAX_RETRIES`` times with exponential backoff
    before giving up.
    """
    if not user_id:
        logger.debug("Skipping plan persistence: no user_id")
        return False

    namespace = (user_id, *PLAN_NAMESPACE_SUFFIX)
    value = {
        "thread_id": thread_id,
        "query": query,
        "plan": plan,
        "subquery_count": len(plan),
        **(metadata or {}),
    }

    last_error: Exception | None = None
    for attempt in range(_PERSIST_MAX_RETRIES + 1):
        try:
            await store.aput(namespace=namespace, key=thread_id, value=value)
            logger.info(
                "Persisted research plan to store: user=%s, thread=%s, subqueries=%d",
                user_id,
                thread_id,
                len(plan),
            )
            return True
        except Exception as e:
            last_error = e
            if attempt < _PERSIST_MAX_RETRIES:
                delay = _PERSIST_BASE_DELAY_S * (2**attempt)
                logger.warning(
                    "Plan persistence attempt %d/%d failed for thread %s: %s — retrying in %.1fs",
                    attempt + 1,
                    _PERSIST_MAX_RETRIES + 1,
                    thread_id,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)

    logger.error(
        "Plan persistence FAILED after %d attempts for thread %s: %s — "
        "plan cached locally only (DATA LOSS RISK on pod restart)",
        _PERSIST_MAX_RETRIES + 1,
        thread_id,
        last_error,
    )
    return False


async def delete_plan_for_thread(
    store: BaseStore,
    user_id: str,
    thread_id: str,
) -> bool:
    """Delete a persisted plan from the LangGraph memory store."""
    if not user_id or not thread_id:
        await clear_plan(thread_id, user_id=user_id)
        return True
    if not hasattr(store, "adelete"):
        logger.debug("Store lacks adelete – skipping plan store cleanup")
        await clear_plan(thread_id, user_id=user_id)
        return True

    tid_hash = hashlib.sha256(thread_id.encode()).hexdigest()[:12]
    try:
        namespace = (user_id, *PLAN_NAMESPACE_SUFFIX)
        await store.adelete(namespace, thread_id)
        logger.info(
            "Deleted plan from store: user=%s, thread_hash=%s", user_id, tid_hash
        )
    except Exception as e:
        logger.warning(
            "Failed to delete plan from store (thread_hash=%s): %s — "
            "keeping local cache intact to avoid inconsistency",
            tid_hash,
            e,
        )
        return False

    await clear_plan(thread_id, user_id=user_id)
    return True


async def load_similar_plans(
    store: BaseStore,
    user_id: str,
    query: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Load similar past research plans using semantic search.

    Retrieves plans from previous sessions that are semantically similar
    to the current query, enabling strategy reuse and consistency.
    """
    if not user_id:
        return []

    try:
        namespace = (user_id, *PLAN_NAMESPACE_SUFFIX)
        similar_plans: list[dict[str, Any]] = []

        if not hasattr(store, "asearch"):
            logger.debug("Store does not support asearch, skipping semantic search")
            return similar_plans

        results = await store.asearch(namespace, query=query, limit=limit)

        for item in results:
            if hasattr(item, "value") and item.value:
                similar_plans.append(item.value)

        if similar_plans:
            logger.info(
                "Found %d similar past plans for query: %s",
                len(similar_plans),
                query[:50],
            )
        return similar_plans
    except Exception as e:
        logger.warning("Failed to load similar plans: %s", e)
        return []


_CLEANUP_INTERVAL_SECONDS = 3600
_cleanup_task: asyncio.Task[None] | None = None


async def _cleanup_stale_data_loop() -> None:
    """Background loop that prunes old plans."""
    while True:
        try:
            await asyncio.sleep(_CLEANUP_INTERVAL_SECONDS)
            await _run_stale_cleanup()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("Stale data cleanup error: %s", e)


async def _run_stale_cleanup() -> None:
    """Remove plans older than threshold. Override for store-specific cleanup."""
    use_postgres = _get_setting("USE_POSTGRES_STORAGE", False)
    if not use_postgres:
        await asyncio.sleep(0)  # Yield to event loop
        return


def start_cleanup_task() -> None:
    """Start the background cleanup loop (called once during app lifespan)."""
    global _cleanup_task
    if _cleanup_task is None or _cleanup_task.done():
        _cleanup_task = asyncio.create_task(
            _cleanup_stale_data_loop(), name="stale-data-cleanup"
        )


async def stop_cleanup_task() -> None:
    """Cancel the background cleanup loop (called during shutdown)."""
    global _cleanup_task
    if _cleanup_task and not _cleanup_task.done():
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            logger.debug("Cleanup task cancelled during shutdown")
            raise
    _cleanup_task = None
