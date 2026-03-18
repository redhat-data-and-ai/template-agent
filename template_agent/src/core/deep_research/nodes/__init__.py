"""Deep research graph nodes package.

Minimal implementations for the template agent. Replace with full implementations
from the dataverse agent for production use.
"""

import hashlib
from typing import Any, Dict, cast

from template_agent.src.core.deep_research.nodes._cache import (
    format_cached_findings_for_triage,
    format_conversation_for_prompt,
    load_conversation_history,
    save_cached_findings,
    save_conversation_turn,
)
from template_agent.src.core.deep_research.nodes.complete import (
    complete_node,
)
from template_agent.src.core.deep_research.nodes.completeness import (
    completeness_evaluator_node,
)
from template_agent.src.core.deep_research.nodes.complexity import (
    assess_complexity_node,
)
from template_agent.src.core.deep_research.nodes.context_answer import (
    context_answer_node,
)
from template_agent.src.core.deep_research.nodes.plan import plan_node
from template_agent.src.core.deep_research.nodes.probe import probe_node
from template_agent.src.core.deep_research.nodes.review import (
    review_node,
)
from template_agent.src.core.deep_research.nodes.supervisor import (
    research_supervisor_node,
)
from template_agent.src.core.deep_research.nodes.synthesize import (
    synthesize_node,
)
from template_agent.src.core.deep_research.nodes.triage import triage_node
from template_agent.src.core.deep_research.nodes.visualize import (
    visualize_node,
)
from template_agent.src.core.deep_research.state import Finding

# ---------------------------------------------------------------------------
# Phase constants
# ---------------------------------------------------------------------------

PHASE_TRIAGE = "triage"
PHASE_PROBE = "probe"
PHASE_PLAN = "plan"
PHASE_SUPERVISOR = "supervisor"
PHASE_COMPLETENESS = "completeness"
PHASE_SYNTHESIZE = "synthesize"
PHASE_COMPLETE = "complete"

# ---------------------------------------------------------------------------
# Cache utilities (used by streaming.py)
# ---------------------------------------------------------------------------


def findings_from_board(board: Any) -> dict:
    """Derive a legacy-shaped findings dict from the findings_board."""
    if not isinstance(board, dict):
        return {}
    return {sq: entry.get("finding") or {} for sq, entry in board.items()}


def format_full_cached_findings_for_triage(
    findings: Dict[str, Finding],
    max_chars: int = 15000,
) -> str:
    """Format cached findings with full answers for triage."""
    if not findings:
        return ""
    parts: list[str] = []
    total_chars = 0
    for finding in findings.values():
        subquery = finding.get("subquery", "")
        answer = finding.get("answer", "")
        if not subquery:
            continue
        entry = f"### Subquery: {subquery}\n{answer}"
        if total_chars + len(entry) > max_chars:
            parts.append("... (additional findings truncated for length)")
            break
        parts.append(entry)
        total_chars += len(entry) + 1
    return "\n\n".join(parts)


async def load_cached_findings(
    checkpointer: Any,
    thread_id: str | None,
) -> Dict[str, Finding]:
    """Load cached findings from checkpointer. Returns empty dict if unavailable."""
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
                h = _hash_subquery(f.get("subquery", ""))
                out[h] = Finding(
                    subquery=f.get("subquery", ""),
                    answer=f.get("answer", ""),
                    tool_results=f.get("tool_results", []),
                    error=f.get("error"),
                    cached=f.get("cached", True),
                    execution_time_ms=f.get("execution_time_ms"),
                )
        return out
    except Exception:
        return {}


def _hash_subquery(subquery: str) -> str:
    """Simple hash for subquery key."""
    return hashlib.sha256((subquery or "").lower().strip().encode()).hexdigest()[:16]


async def _aput_checkpoint(
    checkpointer: Any,
    config: Any,
    checkpoint: dict,
    metadata: dict,
    channel_versions: dict,
) -> None:
    """Put checkpoint with metadata."""
    try:
        if hasattr(checkpointer, "aput"):
            await checkpointer.aput(config, checkpoint, metadata)
        elif hasattr(checkpointer, "aput_tuple"):
            from langgraph.checkpoint.base import (
                Checkpoint,
                CheckpointMetadata,
                CheckpointTuple,
            )

            await checkpointer.aput_tuple(
                CheckpointTuple(
                    config=config,
                    checkpoint=cast(Checkpoint, checkpoint),
                    metadata=cast(CheckpointMetadata, metadata),
                    parent_config=None,
                )
            )
    except Exception:
        pass


__all__ = [
    "findings_from_board",
    "format_cached_findings_for_triage",
    "format_conversation_for_prompt",
    "format_full_cached_findings_for_triage",
    "load_cached_findings",
    "load_conversation_history",
    "save_cached_findings",
    "save_conversation_turn",
    "assess_complexity_node",
    "context_answer_node",
    "triage_node",
    "probe_node",
    "plan_node",
    "research_supervisor_node",
    "completeness_evaluator_node",
    "synthesize_node",
    "visualize_node",
    "review_node",
    "complete_node",
]
