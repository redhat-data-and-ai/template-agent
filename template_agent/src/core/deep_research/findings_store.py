"""Cross-chat research findings store with semantic search.

Persists research findings to the LangGraph memory store so they can be
retrieved across conversations via semantic similarity. Follows the same
pattern as plan_store.py.

Multi-pod safe: all reads/writes go through the store -- no in-memory state.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from template_agent.src.core.deep_research.state import Finding
from template_agent.src.core.deep_research.utils import get_setting as _get_setting
from template_agent.utils.pylogger import get_python_logger

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore

logger = get_python_logger()

_CROSS_CHAT_MAX_FINDINGS = 10


def _normalize_subquery(subquery: str) -> str:
    """Normalize a subquery for comparison."""
    text = (subquery or "").lower().strip()
    text = re.sub(r"[?.!,;:]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def hash_subquery(subquery: str) -> str:
    """Create a hash of a normalized subquery.

    Returns:
        SHA256 hash prefix of the normalized subquery.
    """
    normalized = _normalize_subquery(subquery)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


FINDINGS_NAMESPACE_SUFFIX = ("deep_research", "findings")


def _finding_store_key(thread_id: str, subquery_hash: str) -> str:
    return f"{thread_id}_{subquery_hash}"


def _build_content_text(subquery: str, answer: str) -> str:
    """Build the text indexed for semantic search."""
    return f"Q: {subquery}\nA: {answer[:1000]}"


def _store_supports_deletion(store: Any) -> bool:
    """Return True if store has asearch and adelete methods."""
    return hasattr(store, "asearch") and hasattr(store, "adelete")


def _get_item_value_and_key(item: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Extract value and key from a store search result item."""
    value = getattr(item, "value", item) if hasattr(item, "value") else item
    key = getattr(item, "key", None)
    if not isinstance(value, dict):
        return None, key
    return value, key


async def save_findings_to_store(
    store: BaseStore,
    user_id: str | None,
    thread_id: str | None,
    query: str,
    findings: list[Finding],
) -> int:
    """Persist findings to the centralized store for cross-chat reuse.

    Each finding is stored as a separate entry keyed by
    ``{thread_id}_{subquery_hash}`` so updates are idempotent.

    Returns the number of findings successfully persisted.
    """
    if not user_id or not thread_id or not findings:
        return 0
    if not hasattr(store, "aput"):
        logger.debug("Store does not support aput, skipping findings persistence")
        return 0

    namespace = (user_id, *FINDINGS_NAMESPACE_SUFFIX)
    saved = 0
    now = datetime.now(timezone.utc).isoformat()

    for finding in findings:
        subquery = finding.get("subquery", "")
        answer = finding.get("answer", "")
        if not subquery or not answer:
            continue
        if finding.get("error") and not answer.strip():
            continue

        sq_hash = hash_subquery(subquery)
        key = _finding_store_key(thread_id, sq_hash)

        value: dict[str, Any] = {
            "thread_id": thread_id,
            "query": query,
            "subquery": subquery,
            "answer": answer,
            "resources_used": finding.get("resources_used", [])
            or finding.get("data_products_used", []),
            "created_at": now,
            "content": _build_content_text(subquery, answer),
        }

        try:
            await store.aput(namespace=namespace, key=key, value=value)
            saved += 1
        except Exception as e:
            logger.warning("Failed to persist finding '%s': %s", subquery[:60], e)

    if saved:
        logger.info(
            "Persisted %d/%d findings to cross-chat store for user=%s thread=%s",
            saved,
            len(findings),
            user_id,
            thread_id,
        )
    return saved


async def search_cross_chat_findings(
    store: BaseStore,
    user_id: str | None,
    query: str,
    limit: int | None = None,
    exclude_thread_id: str | None = None,
) -> list[dict[str, Any]]:
    """Semantic search for findings across all conversations for a user.

    Args:
        store: LangGraph BaseStore backed by pgvector or similar.
        user_id: Scope search to this user's namespace.
        query: The current user query to match against.
        limit: Max findings to return (defaults to setting).
        exclude_thread_id: Optionally exclude findings from this thread.

    Returns:
        List of finding value dicts sorted by semantic relevance.
    """
    if not user_id:
        return []
    if not hasattr(store, "asearch"):
        logger.debug("Store does not support asearch, skipping cross-chat search")
        return []

    effective_limit = limit or _get_setting(
        "DEEP_RESEARCH_CROSS_CHAT_MAX_FINDINGS", _CROSS_CHAT_MAX_FINDINGS
    )

    try:
        namespace = (user_id, *FINDINGS_NAMESPACE_SUFFIX)
        results = await store.asearch(namespace, query=query, limit=effective_limit + 5)

        matched = _filter_search_results(results, effective_limit, exclude_thread_id)
        if matched:
            logger.info(
                "Cross-chat search found %d relevant findings for query: %s",
                len(matched),
                query[:80],
            )
        return matched

    except Exception as e:
        logger.warning("Cross-chat findings search failed: %s", e)
        return []


def _filter_search_results(
    results: list[Any],
    limit: int,
    exclude_thread_id: str | None,
) -> list[dict[str, Any]]:
    """Extract and filter finding dicts from store search results."""
    matched: list[dict[str, Any]] = []
    for item in results:
        value = getattr(item, "value", item) if hasattr(item, "value") else item
        if not isinstance(value, dict) or "subquery" not in value:
            continue
        if exclude_thread_id and value.get("thread_id") == exclude_thread_id:
            continue
        matched.append(value)
        if len(matched) >= limit:
            break
    return matched


def cross_chat_to_finding_dict(
    cross_chat_results: list[dict[str, Any]],
) -> dict[str, Finding]:
    """Convert cross-chat store results into the Finding dict format.

    Expected by the triage pipeline (keyed by subquery hash).
    """
    findings: dict[str, Finding] = {}
    for item in cross_chat_results:
        subquery = item.get("subquery", "")
        if not subquery:
            continue
        key = hash_subquery(subquery)
        findings[key] = Finding(
            subquery=subquery,
            answer=item.get("answer", ""),
            tool_results=[],
            error=None,
            cached=True,
            execution_time_ms=None,
        )
    return findings


async def delete_findings_for_thread(
    store: BaseStore,
    user_id: str | None,
    thread_id: str,
) -> int:
    """Delete all cross-chat findings belonging to a specific thread.

    Searches the findings namespace for items whose stored ``thread_id``
    matches, then removes each one via ``adelete``.  Returns the count
    of successfully deleted entries.
    """
    if not user_id or not thread_id:
        return 0
    if not _store_supports_deletion(store):
        logger.debug("Store lacks asearch/adelete – skipping findings cleanup")
        return 0

    namespace = (user_id, *FINDINGS_NAMESPACE_SUFFIX)
    deleted = await _delete_matching_findings(store, namespace, thread_id)

    if deleted:
        logger.info(
            "Deleted %d cross-chat findings for thread=%s user=%s",
            deleted,
            thread_id,
            user_id,
        )
    return deleted


async def _delete_matching_findings(
    store: Any,
    namespace: tuple[str, ...],
    thread_id: str,
) -> int:
    """Delete findings for a thread using server-side metadata filtering."""
    try:
        results = await store.asearch(
            namespace, filter={"thread_id": thread_id}, limit=500
        )
    except Exception:
        logger.warning("Error during findings cleanup for thread deletion")
        return 0

    deleted = 0
    for item in results:
        _, key = _get_item_value_and_key(item)
        if not key:
            continue
        try:
            await store.adelete(namespace, key)
            deleted += 1
        except Exception as exc:
            logger.warning(
                "Failed to delete finding key=%s: %s",
                str(key)[:36] if key else "unknown",
                exc,
            )
    return deleted


def format_cross_chat_findings(
    cross_chat_results: list[dict[str, Any]],
    max_chars: int = 5000,
) -> str:
    """Format cross-chat findings for display / triage context."""
    if not cross_chat_results:
        return ""

    parts: list[str] = []
    total_len = 0
    source_threads: set[str] = set()

    for item in cross_chat_results:
        subquery = item.get("subquery", "")
        answer = item.get("answer", "")
        thread_id = item.get("thread_id", "unknown")
        source_threads.add(thread_id)

        entry = f"### Subquery: {subquery}\n{answer}"
        if total_len + len(entry) > max_chars:
            break
        parts.append(entry)
        total_len += len(entry)

    header = (
        f"[Cross-chat findings: {len(parts)} results "
        f"from {len(source_threads)} previous conversation(s)]\n\n"
    )
    return header + "\n\n".join(parts)
