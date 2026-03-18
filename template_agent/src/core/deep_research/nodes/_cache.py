"""Finding-cache utilities for cross-session finding reuse."""

import hashlib
import re
from typing import Any, Dict, cast

from template_agent.src.core.deep_research.state import Finding
from template_agent.src.core.utils import truncate_text

logger = None

_in_memory_findings: Dict[str, Dict[str, Finding]] = {}
_in_memory_conversation: Dict[str, list[dict]] = {}


def _get_logger():
    global logger
    if logger is None:
        try:
            from template_agent.utils.pylogger import get_python_logger

            logger = get_python_logger()
        except (ImportError, AttributeError):
            import logging

            logger = logging.getLogger(__name__)
    return logger


def normalize_subquery(subquery: str) -> str:
    """Normalize a subquery for comparison."""
    text = (subquery or "").lower().strip()
    text = re.sub(r"[?.!,;:]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def hash_subquery(subquery: str) -> str:
    """Create a hash of a normalized subquery."""
    normalized = normalize_subquery(subquery)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def find_matching_cached_finding(
    subquery: str,
    cached_findings: Dict[str, Finding],
) -> Finding | None:
    """Find a cached finding that matches the given subquery."""
    if not cached_findings:
        return None
    query_hash = hash_subquery(subquery)
    if query_hash in cached_findings:
        return cached_findings[query_hash]
    normalized = normalize_subquery(subquery)
    if not normalized:
        return None
    for finding in cached_findings.values():
        cached_normalized = normalize_subquery(finding.get("subquery", ""))
        if normalized == cached_normalized:
            return finding
        if len(normalized) > 15 and len(cached_normalized) > 15:
            if normalized in cached_normalized or cached_normalized in normalized:
                return finding
    return None


async def load_cached_findings(
    checkpointer: Any,
    thread_id: str | None,
) -> Dict[str, Finding]:
    """Load cached findings from checkpointer metadata."""
    if not checkpointer or not thread_id:
        return {}
    try:
        from langchain_core.runnables import RunnableConfig

        config = RunnableConfig(
            configurable={"thread_id": thread_id, "checkpoint_ns": ""}
        )
        result = await checkpointer.aget_tuple(config)
        if not result or not result.metadata:
            return {}
        findings_list = result.metadata.get("deep_research_findings", [])
        if not isinstance(findings_list, list):
            return {}
        out: Dict[str, Finding] = {}
        for f in findings_list:
            if isinstance(f, dict) and f.get("subquery"):
                h = hash_subquery(f.get("subquery", ""))
                out[h] = Finding(
                    subquery=f.get("subquery", ""),
                    answer=f.get("answer", ""),
                    tool_results=f.get("tool_results", []),
                    error=f.get("error"),
                    cached=f.get("cached", True),
                    execution_time_ms=f.get("execution_time_ms"),
                )
        return out
    except Exception as e:
        _get_logger().warning("Failed to load cached findings: %s", e)
        return {}


async def save_cached_findings(
    checkpointer: Any,
    thread_id: str | None,
    findings_list: list,
) -> None:
    """Save findings to checkpointer metadata. No-op if checkpointer unavailable."""
    if not checkpointer or not thread_id or not findings_list:
        return
    try:
        from langchain_core.runnables import RunnableConfig

        config = RunnableConfig(
            configurable={"thread_id": thread_id, "checkpoint_ns": ""}
        )
        result = await checkpointer.aget_tuple(config)
        metadata = dict(result.metadata) if result and result.metadata else {}
        serialized = []
        for f in findings_list:
            if isinstance(f, dict):
                serialized.append(
                    {
                        "subquery": f.get("subquery", ""),
                        "answer": f.get("answer", ""),
                        "tool_results": f.get("tool_results", []),
                        "error": f.get("error"),
                        "cached": f.get("cached", True),
                        "execution_time_ms": f.get("execution_time_ms"),
                    }
                )
        metadata["deep_research_findings"] = serialized
        if result and result.checkpoint:
            if hasattr(checkpointer, "aput"):
                await checkpointer.aput(config, result.checkpoint, metadata)
            elif hasattr(checkpointer, "aput_tuple"):
                from langgraph.checkpoint.base import (
                    Checkpoint,
                    CheckpointMetadata,
                    CheckpointTuple,
                )

                await checkpointer.aput_tuple(
                    CheckpointTuple(
                        config=config,
                        checkpoint=cast(Checkpoint, result.checkpoint),
                        metadata=cast(CheckpointMetadata, metadata),
                        parent_config=None,
                    )
                )
    except Exception as e:
        _get_logger().warning("Failed to save cached findings: %s", e)


def format_cached_findings_for_prompt(
    findings: Dict[str, Finding],
    max_chars: int = 8000,
) -> str:
    """Format cached findings as context for the planning prompt."""
    if not findings:
        return ""
    parts: list[str] = []
    total_chars = 0
    for finding in findings.values():
        subquery = finding.get("subquery", "")
        answer = finding.get("answer", "")
        if not subquery:
            continue
        answer_preview = truncate_text(answer, 600)
        entry = f"- Q: {subquery}\n  A: {answer_preview}"
        if total_chars + len(entry) > max_chars:
            break
        parts.append(entry)
        total_chars += len(entry) + 1
    return "\n".join(parts)


def save_findings_in_memory(thread_id: str, findings_board: Dict[str, Any]) -> None:
    """Persist findings for a thread in process-level memory.

    Used when no checkpointer/database is available so that follow-up
    queries on the same thread can leverage prior research.
    """
    if not thread_id or not findings_board:
        return
    converted: Dict[str, Finding] = {}
    for key, entry in findings_board.items():
        finding = entry.get("finding", entry) if isinstance(entry, dict) else entry
        if isinstance(finding, dict) and finding.get("subquery"):
            h = hash_subquery(finding["subquery"])
            converted[h] = Finding(
                subquery=finding.get("subquery", ""),
                answer=finding.get("answer", ""),
                tool_results=finding.get("tool_results", []),
                error=finding.get("error"),
                cached=True,
                execution_time_ms=finding.get("execution_time_ms"),
            )
    if converted:
        _in_memory_findings[thread_id] = converted
        _get_logger().info(
            "Saved %d findings in memory for thread %s",
            len(converted),
            thread_id,
        )


def load_findings_in_memory(thread_id: str) -> Dict[str, Finding]:
    """Load findings for a thread from process-level memory."""
    if not thread_id:
        return {}
    return dict(_in_memory_findings.get(thread_id, {}))


def save_conversation_turn(thread_id: str, query: str, answer: str) -> None:
    """Append a user-assistant turn to the in-memory conversation history."""
    if not thread_id or not query:
        return
    _in_memory_conversation.setdefault(thread_id, []).append(
        {"query": query, "answer": answer or ""}
    )


def load_conversation_history(thread_id: str) -> list[dict]:
    """Return all conversation turns for a thread."""
    if not thread_id:
        return []
    return list(_in_memory_conversation.get(thread_id, []))


def format_conversation_for_prompt(
    history: list[dict],
    max_chars: int = 4000,
) -> str:
    """Format conversation turns as ``User: ... / Assistant: ...`` text."""
    if not history:
        return ""
    parts: list[str] = []
    total = 0
    for turn in history:
        entry = f"User: {turn.get('query', '')}\nAssistant: {turn.get('answer', '')}"
        if total + len(entry) > max_chars:
            parts.append("... (earlier conversation truncated)")
            break
        parts.append(entry)
        total += len(entry) + 1
    return "\n\n".join(parts)


def format_cached_findings_for_triage(
    findings: Dict[str, Finding],
    max_chars: int = 30000,
) -> str:
    """Format cached findings with full answers for follow-up routing.

    Unlike ``format_cached_findings_for_prompt`` which truncates answers to
    300 chars, this version keeps them intact so the routing LLM has enough
    context to decide whether a follow-up can be answered directly.
    """
    if not findings:
        return ""
    parts: list[str] = []
    total = 0
    for finding in findings.values():
        subquery = finding.get("subquery", "")
        answer = finding.get("answer", "")
        if not subquery:
            continue
        entry = f"### Subquery: {subquery}\n{answer}"
        if total + len(entry) > max_chars:
            parts.append("... (additional findings truncated)")
            break
        parts.append(entry)
        total += len(entry) + 1
    return "\n\n".join(parts)
