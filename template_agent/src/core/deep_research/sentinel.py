"""Loop Sentinel: global circuit breaker for deep research execution.

Prevents infinite or runaway loops by enforcing global budgets across
all feedback loops (supervisor, completeness, review). Called as a guard
at the top of every loopable node.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from template_agent.src.core.deep_research.cancel import get_cancel_store
from template_agent.utils.pylogger import get_python_logger

if TYPE_CHECKING:
    from template_agent.src.core.deep_research.state import (
        DeepResearchState,
        ResearchContext,
    )

logger = get_python_logger()

# Reasonable defaults when settings are not available
_DEEP_RESEARCH_MAX_SESSION_SECONDS = 600.0
_DEEP_RESEARCH_MAX_TOTAL_SUBQUERIES = 50
_DEEP_RESEARCH_MAX_NODE_TRANSITIONS = 100
_DEEP_RESEARCH_MAX_TOKEN_BUDGET = 0  # 0 = disabled
_DEEP_RESEARCH_MAX_COST_PER_SESSION = 0.0  # 0 = disabled
_DEEP_RESEARCH_STAGNATION_THRESHOLD = 3


def _get_setting(name: str, default: Any) -> Any:
    """Get setting with fallback to default."""
    try:
        from template_agent.src.settings import settings

        return getattr(settings, name, default)
    except Exception:
        return default


PHASE_SYNTHESIZE = "synthesize"
PHASE_COMPLETE = "complete"


def _get_phase_budget(
    max_seconds: float,
    node_name: str,
    ctx: ResearchContext | None = None,
) -> float:
    """Compute a phase-aware time budget so synthesis/review get reserved time."""
    mode_config = ctx.mode_config if ctx else None
    synth_reserve = (
        getattr(mode_config, "synthesis_reserved_seconds", 90) if mode_config else 90
    )
    review_reserve = (
        getattr(mode_config, "review_reserved_seconds", 60) if mode_config else 60
    )

    if node_name in ("supervisor", "completeness"):
        return max_seconds - synth_reserve - review_reserve
    if node_name == "synthesize":
        return max_seconds - review_reserve
    return max_seconds


async def check_cancellation(
    state: DeepResearchState,
) -> tuple[bool, str | None, str | None]:
    """Check if the current thread has been cancelled."""
    thread_id = state.get("thread_id", "")
    if not thread_id:
        return False, None, None
    store = get_cancel_store()
    if await store.is_cancelled(thread_id):
        findings_board = state.get("findings_board", state.get("findings", {}))
        has_findings = _has_valid_findings(findings_board)
        forced_phase = PHASE_SYNTHESIZE if has_findings else PHASE_COMPLETE
        return True, "Cancelled by user", forced_phase
    return False, None, None


def check_loop_sentinel(
    state: DeepResearchState,
    ctx: ResearchContext | None,
    node_name: str,
) -> tuple[bool, str | None, str | None]:
    """Check all global budgets. Called at the top of every loopable node.

    Uses phase-aware time budgets: research nodes stop early enough
    to guarantee time for synthesis and review phases.

    Returns:
        (should_stop, reason, forced_phase)
        - should_stop: True if a budget is exceeded
        - reason: Human-readable explanation of which budget fired
        - forced_phase: PHASE_SYNTHESIZE if findings exist, PHASE_COMPLETE if none
    """
    findings_board = state.get("findings_board", state.get("findings", {}))
    has_findings = _has_valid_findings(findings_board)
    forced_phase = PHASE_SYNTHESIZE if has_findings else PHASE_COMPLETE

    start_time = state.get("execution_start_time", 0.0)
    max_seconds = state.get(
        "max_session_seconds",
        _get_setting(
            "DEEP_RESEARCH_MAX_SESSION_SECONDS", _DEEP_RESEARCH_MAX_SESSION_SECONDS
        ),
    )

    # Hard ceiling (2x budget) always fires
    hard_hit = _check_hard_ceiling(start_time, max_seconds, node_name)
    if hard_hit:
        return True, hard_hit, forced_phase

    if not has_findings and node_name in ("supervisor", "completeness"):
        return False, None, None

    # Soft budget checks
    reason = _check_budget_limits(state, ctx, node_name)
    if reason:
        return True, reason, forced_phase

    # Phase-aware wall clock: research nodes use a tighter budget to
    # leave room for synthesis and review.
    phase_budget = _get_phase_budget(max_seconds, node_name, ctx=ctx)
    reason = _check_wall_clock(start_time, phase_budget, node_name)
    if reason:
        return True, reason, forced_phase

    reason = _check_stagnation(state, node_name)
    if reason:
        return True, reason, forced_phase

    return False, None, None


def _check_hard_ceiling(
    start_time: float,
    max_seconds: float,
    node_name: str,
) -> str | None:
    """Absolute time ceiling (2x budget) -- fires regardless of findings."""
    if start_time <= 0 or max_seconds <= 0:
        return None
    elapsed = time.time() - start_time
    hard_ceiling = max_seconds * 2.0
    if elapsed < hard_ceiling:
        return None
    reason = f"Hard ceiling breached: {elapsed:.0f}s >= {hard_ceiling:.0f}s (node: {node_name})"
    logger.error(reason)
    return reason


def _check_budget_limits(
    state: DeepResearchState,
    ctx: ResearchContext | None,
    node_name: str,
) -> str | None:
    """Check subquery, node-transition, token, and cost budgets."""
    total_sq = state.get("total_subqueries_executed", 0)
    max_sq = state.get(
        "max_total_subqueries",
        _get_setting(
            "DEEP_RESEARCH_MAX_TOTAL_SUBQUERIES", _DEEP_RESEARCH_MAX_TOTAL_SUBQUERIES
        ),
    )
    if max_sq > 0 and total_sq >= max_sq:
        reason = f"Subquery budget exhausted: {total_sq}/{max_sq} executed (node: {node_name})"
        logger.warning(reason)
        return reason

    total_transitions = state.get("total_node_transitions", 0)
    max_transitions = state.get(
        "max_node_transitions",
        _get_setting(
            "DEEP_RESEARCH_MAX_NODE_TRANSITIONS", _DEEP_RESEARCH_MAX_NODE_TRANSITIONS
        ),
    )
    if max_transitions > 0 and total_transitions >= max_transitions:
        reason = f"Node transition budget exhausted: {total_transitions}/{max_transitions} (node: {node_name})"
        logger.warning(reason)
        return reason

    token_reason = _check_token_cost_budget(ctx, node_name)
    return token_reason


def _check_token_cost_budget(
    ctx: ResearchContext | None,
    node_name: str,
) -> str | None:
    """Check token and cost budgets from the token tracker."""
    if ctx is None or ctx.token_tracker is None:
        return None

    max_token_budget = _get_setting(
        "DEEP_RESEARCH_MAX_TOKEN_BUDGET", _DEEP_RESEARCH_MAX_TOKEN_BUDGET
    )
    if max_token_budget > 0:
        total_tokens = getattr(ctx.token_tracker, "total_tokens", 0)
        if total_tokens == 0:
            total = getattr(ctx.token_tracker, "get_total", lambda: None)()
            total_tokens = total.total_tokens if total else 0
        if total_tokens >= max_token_budget:
            reason = f"Token budget exhausted: {total_tokens}/{max_token_budget} tokens (node: {node_name})"
            logger.warning(reason)
            return reason

    max_cost = _get_setting(
        "DEEP_RESEARCH_MAX_COST_PER_SESSION", _DEEP_RESEARCH_MAX_COST_PER_SESSION
    )
    if max_cost > 0:
        estimated_cost = getattr(ctx.token_tracker, "estimated_cost", 0.0)
        if estimated_cost >= max_cost:
            reason = f"Cost budget exceeded: ${estimated_cost:.4f} >= ${max_cost:.4f} (node: {node_name})"
            logger.warning(reason)
            return reason

    return None


def _check_wall_clock(
    start_time: float,
    max_seconds: float,
    node_name: str,
) -> str | None:
    """Soft wall-clock timeout using execution_start_time."""
    if start_time <= 0 or max_seconds <= 0:
        return None
    elapsed = time.time() - start_time
    if elapsed < max_seconds:
        return None
    reason = f"Session timeout: {elapsed:.0f}s >= {max_seconds:.0f}s limit (node: {node_name})"
    logger.warning(reason)
    return reason


def _check_stagnation(state: DeepResearchState, node_name: str) -> str | None:
    """Detect stagnation: findings count unchanged for N consecutive rounds."""
    history = state.get("findings_count_history", [])
    threshold = _get_setting(
        "DEEP_RESEARCH_STAGNATION_THRESHOLD", _DEEP_RESEARCH_STAGNATION_THRESHOLD
    )
    if threshold <= 0 or len(history) < threshold or len(history) <= 1:
        return None
    recent = history[-threshold:]
    if len(set(recent)) != 1:
        return None
    reason = (
        f"Research stagnated: findings count unchanged at {recent[0]} "
        f"for {threshold} consecutive rounds (node: {node_name})"
    )
    logger.warning(reason)
    return reason


def _has_valid_findings(findings_board: dict[str, Any] | list[Any]) -> bool:
    """Check if there is at least one finding without an error."""
    items = (
        findings_board.values() if isinstance(findings_board, dict) else findings_board
    )
    for entry in items:
        finding = (entry.get("finding") or {}) if isinstance(entry, dict) else entry
        if isinstance(finding, dict) and not finding.get("error"):
            return True
        if not isinstance(finding, dict) and hasattr(finding, "get"):
            if not finding.get("error"):
                return True
    return False


def trim_follow_ups(
    follow_ups: list[str],
    total_executed: int,
    max_total: int,
) -> list[str]:
    """Trim follow-up subqueries to stay within the total subquery budget."""
    if max_total <= 0:
        return follow_ups
    remaining = max(0, max_total - total_executed)
    if len(follow_ups) <= remaining:
        return follow_ups
    trimmed = follow_ups[:remaining]
    if len(trimmed) < len(follow_ups):
        logger.info(
            "Trimmed follow-ups from %d to %d (budget: %d/%d)",
            len(follow_ups),
            len(trimmed),
            total_executed,
            max_total,
        )
    return trimmed
