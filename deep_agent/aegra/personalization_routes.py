"""REST API for user rules (PersonalizationRepository) and memory files (LangGraph Store)."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, field_validator

from deep_agent.src.personalization.repository import PersonalizationRepository
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

router = APIRouter(prefix="/personalization", tags=["personalization"])

_AEGRA_JSON = Path(__file__).resolve().parent.parent.parent / "aegra.json"


def _require_memory() -> None:
    """Raise 404 if the memory feature is disabled."""
    if not settings.MEMORY_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Memory feature is disabled"
        )


def _require_rules() -> None:
    """Raise 404 if the user rules feature is disabled."""
    if not settings.USER_RULES_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User rules feature is disabled",
        )


MAX_RULES_PER_USER = 50


# ── Models ────────────────────────────────────────────────────────────


class MemoryItemOut(BaseModel):
    """An individual memory fact parsed from the LangGraph Store."""

    id: str
    content: str
    created_at: str


class RuleCreate(BaseModel):
    """Request body for creating or upserting a rule."""

    content: str
    is_active: bool = True

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate rule content is non-empty and within word limit."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Rule content cannot be empty")
        if len(stripped.split()) > 100:
            raise ValueError("Rule must be 100 words or fewer")
        return stripped


class RuleOut(BaseModel):
    """Serialised rule returned to the client."""

    id: UUID
    user_id: str
    content: str
    is_active: bool
    created_at: str
    updated_at: str


# ── Helpers ───────────────────────────────────────────────────────────


def _get_user_id(request: Request) -> str:
    """Extract user identity from Aegra auth, UI header, or local fallback.

    Resolution order:
    1. request.state.user.identity — set by Aegra's auth middleware (SSO token)
    2. X-User-ID header — sent by the UI's BFF proxy (trusted internal header)
    3. OS username — local dev fallback
    """
    import os

    user = getattr(request.state, "user", None)
    if user is not None:
        identity: str | None = getattr(user, "identity", None)
        if identity:
            return identity

    header_id: str | None = request.headers.get("x-user-id")
    if header_id:
        return header_id

    return os.getenv("USER", "default")


def _get_default_graph_name() -> str:
    """Read the first graph name from aegra.json."""
    try:
        data = json.loads(_AEGRA_JSON.read_text())
        graphs = data.get("graphs", {})
        if graphs:
            return str(next(iter(graphs)))
    except Exception:
        logger.debug("Could not read aegra.json for graph name", exc_info=True)
    return "agent"


_cached_namespace_prefix: str | None = None
_cached_namespace_ts: float = 0.0
_NAMESPACE_TTL = 60.0


async def _resolve_namespace_prefix(user_id: str) -> str:
    """Resolve the store namespace prefix that the runtime actually uses.

    The StoreBackend namespace depends on whether the Aegra auth middleware
    provides server_info. With auth: (assistant_uuid, user_identity). Without
    auth or in local dev: ("default",). We detect which is in use by checking
    the store table for existing entries.

    Result is cached for 60 seconds to allow re-detection after startup.
    """
    global _cached_namespace_prefix, _cached_namespace_ts  # noqa: PLW0603
    now = time.monotonic()
    if (
        _cached_namespace_prefix is not None
        and (now - _cached_namespace_ts) < _NAMESPACE_TTL
    ):
        return _cached_namespace_prefix

    try:
        import asyncpg

        graph_name = _get_default_graph_name()
        async with asyncpg.create_pool(
            settings.database_uri, min_size=1, max_size=2
        ) as pool:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT assistant_id::text FROM assistant WHERE graph_id = $1 LIMIT 1",
                    graph_name,
                )
                assistant_uuid = row["assistant_id"] if row else None

                if assistant_uuid:
                    check = await conn.fetchval(
                        "SELECT 1 FROM store WHERE prefix LIKE $1 LIMIT 1",
                        f"{assistant_uuid}.%",
                    )
                    if check:
                        _cached_namespace_prefix = "uuid"
                        _cached_namespace_ts = now
                        return "uuid"

                check_default = await conn.fetchval(
                    "SELECT 1 FROM store WHERE prefix = 'default' LIMIT 1",
                )
                if check_default:
                    _cached_namespace_prefix = "default"
                    _cached_namespace_ts = now
                    return "default"
    except Exception:
        logger.debug("Could not detect namespace prefix", exc_info=True)

    return "default"


async def _get_store_namespace(user_id: str) -> tuple[str, ...]:
    """Build the LangGraph Store namespace matching StoreBackend at runtime.

    Detects whether the runtime uses auth-based namespaces (assistant_uuid, user_id)
    or the fallback ("default",).
    """
    prefix_mode = await _resolve_namespace_prefix(user_id)

    if prefix_mode == "uuid":
        graph_name = _get_default_graph_name()
        try:
            import asyncpg

            async with asyncpg.create_pool(
                settings.database_uri, min_size=1, max_size=2
            ) as pool:
                async with pool.acquire() as conn:
                    row = await conn.fetchrow(
                        "SELECT assistant_id::text FROM assistant WHERE graph_id = $1 LIMIT 1",
                        graph_name,
                    )
                    if row:
                        return (row["assistant_id"], user_id)
        except Exception:
            pass

    return ("default",)


def _get_repo() -> PersonalizationRepository:
    """Return a ``PersonalizationRepository`` instance for rules."""
    return PersonalizationRepository(settings.database_uri)


_store_instance: Any = None
_store_lock = asyncio.Lock()


async def _get_store() -> Any:
    """Return a shared AsyncPostgresStore backed by a connection pool."""
    global _store_instance  # noqa: PLW0603
    if _store_instance is not None:
        return _store_instance
    async with _store_lock:
        if _store_instance is None:
            from langgraph.store.postgres.aio import AsyncPostgresStore
            from psycopg.rows import dict_row
            from psycopg_pool import AsyncConnectionPool

            pool = AsyncConnectionPool(
                settings.database_uri,
                min_size=2,
                max_size=10,
                kwargs={
                    "autocommit": True,
                    "prepare_threshold": 0,
                    "row_factory": dict_row,
                },
                open=False,
            )
            await pool.open()
            store = AsyncPostgresStore(conn=pool)
            await store.setup()
            _store_instance = store
    return _store_instance


async def _invalidate_cache(user_id: str, request: Request | None = None) -> None:
    """Best-effort Redis cache + graph cache invalidation after a write."""
    try:
        from deep_agent.aegra.graph import invalidate_graph_cache
        from deep_agent.src.cache.personalization_cache import invalidate

        ids_to_invalidate: set[str] = {user_id}

        if request is not None:
            header_id = request.headers.get("x-user-id")
            if header_id:
                ids_to_invalidate.add(header_id)
            user = getattr(request.state, "user", None)
            if user is not None:
                identity = getattr(user, "identity", None)
                if identity:
                    ids_to_invalidate.add(identity)

        for uid in ids_to_invalidate:
            await invalidate(uid)

        invalidate_graph_cache()
    except Exception:
        logger.debug("Cache invalidation failed for %s", user_id, exc_info=True)


def _parse_store_items_to_memories(items: list) -> list[MemoryItemOut]:
    """Parse LangGraph Store items into individual memory facts.

    Each store item contains a ``content`` list of lines. We split on
    newlines / bullet markers so every distinct fact becomes its own
    ``MemoryItemOut`` with a stable id of ``<store_key>:<line_index>``.
    """
    import hashlib
    import re

    results: list[MemoryItemOut] = []
    for item in items:
        key: str = item.key
        value: dict = item.value
        content_lines: list[str] = value.get("content", [])
        created_at = value.get("created_at", "")

        raw_text = "\n".join(content_lines)
        facts = re.split(r"\n+", raw_text)

        for idx, fact in enumerate(facts):
            cleaned = re.sub(r"^[-*•]\s*", "", fact).strip()
            if not cleaned:
                continue
            stable_id = hashlib.sha256(f"{key}:{idx}:{cleaned}".encode()).hexdigest()[
                :12
            ]
            results.append(
                MemoryItemOut(
                    id=stable_id,
                    content=cleaned,
                    created_at=created_at,
                )
            )
    return results


# ── Memory endpoints (individual facts from LangGraph Store) ──────────


class MemoryClusterOut(BaseModel):
    """A group of semantically related memory facts."""

    cluster_id: int
    facts: list[MemoryItemOut]


class DeduplicateResultOut(BaseModel):
    """Result of a deduplication operation."""

    removed: int
    remaining: int


@router.get("/memories", response_model=list[MemoryItemOut])
async def list_memories(
    request: Request,
    deduplicate: bool = False,
) -> list[MemoryItemOut]:
    """List individual memory facts parsed from the LangGraph Store.

    Args:
        request: The incoming HTTP request.
        deduplicate: If true, removes near-duplicate facts from the response
            (keeps the longest representative from each duplicate group).
    """
    _require_memory()
    user_id = _get_user_id(request)
    namespace = await _get_store_namespace(user_id)

    store = await _get_store()
    items = await store.asearch(namespace, limit=100)

    memories = _parse_store_items_to_memories(items)

    if deduplicate and len(memories) >= 2:
        from deep_agent.src.memory.clustering import cluster_memories

        contents = [m.content for m in memories]
        clusters = cluster_memories(contents)
        indices_to_remove: set[int] = set()
        for group in clusters:
            longest_idx = max(group, key=lambda i: len(contents[i]))
            for idx in group:
                if idx != longest_idx:
                    indices_to_remove.add(idx)
        memories = [m for i, m in enumerate(memories) if i not in indices_to_remove]

    return memories


@router.get("/memories/clustered", response_model=list[MemoryClusterOut])
async def list_memories_clustered(request: Request) -> list[MemoryClusterOut]:
    """List memory facts grouped by semantic similarity.

    Returns clusters of related facts. Singletons (unique facts) are
    each returned in their own single-item cluster.
    """
    _require_memory()
    from deep_agent.src.memory.clustering import cluster_memories

    user_id = _get_user_id(request)
    namespace = await _get_store_namespace(user_id)

    store = await _get_store()
    items = await store.asearch(namespace, limit=100)

    memories = _parse_store_items_to_memories(items)
    if not memories:
        return []

    contents = [m.content for m in memories]
    clusters = cluster_memories(contents)

    clustered_indices: set[int] = set()
    result: list[MemoryClusterOut] = []

    for cluster_id, group in enumerate(clusters):
        clustered_indices.update(group)
        result.append(
            MemoryClusterOut(
                cluster_id=cluster_id,
                facts=[memories[i] for i in group],
            )
        )

    singleton_id = len(clusters)
    for i, mem in enumerate(memories):
        if i not in clustered_indices:
            result.append(MemoryClusterOut(cluster_id=singleton_id, facts=[mem]))
            singleton_id += 1

    return result


@router.post(
    "/memories/deduplicate",
    response_model=DeduplicateResultOut,
)
async def deduplicate_memories(request: Request) -> DeduplicateResultOut:
    """Remove near-duplicate facts from the memory store.

    Finds clusters of similar facts and keeps only the longest
    (most informative) fact from each group. Modifies the store in-place.
    """
    _require_memory()
    import re

    from deep_agent.src.memory.clustering import cluster_memories

    user_id = _get_user_id(request)
    namespace = await _get_store_namespace(user_id)

    store = await _get_store()
    items = await store.asearch(namespace, limit=200)

    all_facts: list[tuple[str, str, int]] = []
    for item in items:
        key: str = item.key
        value: dict = item.value
        content_lines: list[str] = value.get("content", [])
        raw_text = "\n".join(content_lines)
        for idx, fact in enumerate(re.split(r"\n+", raw_text)):
            cleaned = re.sub(r"^[-*•]\s*", "", fact).strip()
            if cleaned:
                all_facts.append((key, cleaned, idx))

    if len(all_facts) < 2:
        return DeduplicateResultOut(removed=0, remaining=len(all_facts))

    contents = [f[1] for f in all_facts]
    clusters = cluster_memories(contents)

    facts_to_remove: set[str] = set()
    for group in clusters:
        longest_idx = max(group, key=lambda i: len(contents[i]))
        for idx in group:
            if idx != longest_idx:
                facts_to_remove.add(all_facts[idx][1])

    if not facts_to_remove:
        return DeduplicateResultOut(removed=0, remaining=len(all_facts))

    removed = 0
    for item in items:
        key = item.key
        value = item.value
        content_lines = value.get("content", [])
        raw_text = "\n".join(content_lines)

        original_facts = []
        remaining_facts = []
        for fact in re.split(r"\n+", raw_text):
            cleaned = re.sub(r"^[-*•]\s*", "", fact).strip()
            if not cleaned:
                continue
            original_facts.append(cleaned)
            if cleaned in facts_to_remove:
                removed += 1
            else:
                remaining_facts.append(cleaned)

        if len(remaining_facts) < len(original_facts):
            if remaining_facts:
                now = datetime.now(tz=timezone.utc).isoformat()
                new_value = {
                    "content": remaining_facts,
                    "created_at": value.get("created_at", now),
                    "modified_at": now,
                }
                await store.aput(namespace, key, new_value)
            else:
                await store.adelete(namespace, key)

    remaining = len(all_facts) - removed
    logger.info(
        "Deduplicated memories for user %s: removed %d, remaining %d",
        user_id,
        removed,
        remaining,
    )
    return DeduplicateResultOut(removed=removed, remaining=remaining)


@router.delete("/memories", status_code=status.HTTP_204_NO_CONTENT)
async def delete_all_memories(request: Request) -> None:
    """Delete all memory store files for the authenticated user."""
    _require_memory()
    user_id = _get_user_id(request)
    namespace = await _get_store_namespace(user_id)

    store = await _get_store()
    items = await store.asearch(namespace, limit=100)
    if items:
        await asyncio.gather(*(store.adelete(namespace, item.key) for item in items))

    logger.info("All memories deleted for user %s", user_id)


@router.delete("/memories/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(request: Request, memory_id: str) -> None:
    """Delete a single memory fact by its id.

    The fact is removed from the store file content. If no facts remain
    the file itself is deleted.
    """
    _require_memory()
    import hashlib
    import re

    user_id = _get_user_id(request)
    namespace = await _get_store_namespace(user_id)

    store = await _get_store()
    items = await store.asearch(namespace, limit=100)

    found = False
    for item in items:
        key: str = item.key
        value: dict = item.value
        content_lines: list[str] = value.get("content", [])
        raw_text = "\n".join(content_lines)
        facts = re.split(r"\n+", raw_text)

        remaining: list[str] = []
        for idx, fact in enumerate(facts):
            cleaned = re.sub(r"^[-*•]\s*", "", fact).strip()
            if not cleaned:
                continue
            fact_id = hashlib.sha256(f"{key}:{idx}:{cleaned}".encode()).hexdigest()[:12]
            if fact_id == memory_id:
                found = True
            else:
                remaining.append(cleaned)

        if found:
            if remaining:
                now = datetime.now(tz=timezone.utc).isoformat()
                new_value = {
                    "content": remaining,
                    "created_at": value.get("created_at", now),
                    "modified_at": now,
                }
                await store.aput(namespace, key, new_value)
            else:
                await store.adelete(namespace, key)
            break

    if not found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory '{memory_id}' not found",
        )

    logger.info("Memory %s deleted for user %s", memory_id, user_id)


# ── Preferences endpoints ─────────────────────────────────────────────


class PreferencesOut(BaseModel):
    """User preferences returned to the client."""

    memory_enabled: bool


class PreferencesUpdate(BaseModel):
    """Request body for updating user preferences."""

    memory_enabled: bool | None = None


@router.get("/preferences", response_model=PreferencesOut)
async def get_preferences(request: Request) -> PreferencesOut:
    """Return the authenticated user's feature preferences."""
    user_id = _get_user_id(request)
    repo = _get_repo()
    prefs = await repo.get_preferences(user_id)
    return PreferencesOut(memory_enabled=prefs.memory_enabled)


@router.put("/preferences", response_model=PreferencesOut)
async def update_preferences(
    request: Request, body: PreferencesUpdate
) -> PreferencesOut:
    """Update the authenticated user's feature preferences."""
    user_id = _get_user_id(request)
    repo = _get_repo()
    prefs = await repo.update_preferences(user_id, memory_enabled=body.memory_enabled)
    await _invalidate_cache(user_id, request)
    logger.info(
        "Preferences updated for user %s: memory_enabled=%s",
        user_id,
        prefs.memory_enabled,
    )
    return PreferencesOut(memory_enabled=prefs.memory_enabled)


# ── Rule endpoints (PersonalizationRepository) ───────────────────────


@router.get("/rules", response_model=list[RuleOut])
async def list_rules(request: Request) -> list[RuleOut]:
    """List all rules for the authenticated user."""
    _require_rules()
    user_id = _get_user_id(request)
    repo = _get_repo()
    rules = await repo.list_rules(user_id, active_only=False)
    return [
        RuleOut(
            id=r.id,
            user_id=r.user_id,
            content=r.content,
            is_active=r.is_active,
            created_at=r.created_at.isoformat(),
            updated_at=r.updated_at.isoformat(),
        )
        for r in rules
    ]


@router.post(
    "/rules",
    response_model=RuleOut,
    status_code=status.HTTP_201_CREATED,
)
async def create_rule(request: Request, body: RuleCreate) -> RuleOut:
    """Create or upsert a rule for the authenticated user."""
    _require_rules()
    user_id = _get_user_id(request)
    repo = _get_repo()
    try:
        rule = await repo.upsert_rule(
            user_id,
            body.content,
            is_active=body.is_active,
            max_rules=MAX_RULES_PER_USER,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    await _invalidate_cache(user_id, request)
    logger.info("Rule created for user %s: %s", user_id, rule.id)
    return RuleOut(
        id=rule.id,
        user_id=rule.user_id,
        content=rule.content,
        is_active=rule.is_active,
        created_at=rule.created_at.isoformat(),
        updated_at=rule.updated_at.isoformat(),
    )


@router.delete("/rules", status_code=status.HTTP_204_NO_CONTENT)
async def delete_all_rules(request: Request) -> None:
    """Delete all rules for the authenticated user."""
    _require_rules()
    user_id = _get_user_id(request)
    repo = _get_repo()
    removed = await repo.delete_all_rules(user_id)
    await _invalidate_cache(user_id, request)
    logger.info("All rules deleted for user %s: %d removed", user_id, removed)


@router.delete("/rules/{rule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_rule(request: Request, rule_id: UUID) -> None:
    """Delete a specific rule by ID."""
    _require_rules()
    user_id = _get_user_id(request)
    repo = _get_repo()
    deleted = await repo.delete_rule(user_id, rule_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Rule not found",
        )
    await _invalidate_cache(user_id, request)
    logger.info("Rule deleted for user %s: %s", user_id, rule_id)
