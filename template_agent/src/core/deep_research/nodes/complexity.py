"""Complexity assessment node: calibrate research budgets."""

from typing import Any, Dict, List

from template_agent.src.core.deep_research.events import emit_complexity_assessment
from template_agent.src.core.deep_research.prompts import (
    build_complexity_assessment_prompt,
)
from template_agent.src.core.deep_research.state import (
    DeepResearchState,
    ResearchContext,
)
from template_agent.src.core.deep_research.token_tracker import tracked_invoke
from template_agent.src.core.utils import safe_json_parse, truncate_text
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_VALID_COMPLEXITY_CLASSES = frozenset(
    {"simple", "moderate", "complex", "comprehensive"}
)

# Defaults when no mode config
DEFAULT_MAX_SESSION_SECONDS = 600
DEFAULT_MAX_NODE_TRANSITIONS = 30


def _parse_assessor_response(
    result: dict[str, Any] | None,
) -> tuple[str, int, int, int, str]:
    """Validate and extract assessor response fields."""
    if not result or "complexity_class" not in result:
        raise ValueError("Invalid assessor response — missing complexity_class")
    complexity_class = result["complexity_class"]
    if complexity_class not in _VALID_COMPLEXITY_CLASSES:
        raise ValueError(f"Unknown complexity class: {complexity_class}")
    rec_sq = int(result.get("recommended_subqueries", 10))
    rec_rounds = int(result.get("recommended_supervisor_rounds", 2))
    rec_iters = int(result.get("recommended_review_iterations", 2))
    reasoning = str(result.get("reasoning", ""))
    return complexity_class, rec_sq, rec_rounds, rec_iters, reasoning


def _get_mode_bounds(mode: Any) -> tuple[int, int, int, int, int, float]:
    """Extract floor/ceiling bounds from mode config."""
    sq_floor = mode.min_subqueries if mode else 3
    sq_ceiling = mode.max_subqueries if mode else 12
    rounds_ceiling = mode.max_supervisor_rounds if mode else 2
    iterations_ceiling = mode.max_review_iterations if mode else 2
    node_transitions_ceiling = mode.max_node_transitions if mode else 30
    session_timeout = mode.session_timeout_seconds if mode else 600
    return (
        sq_floor,
        sq_ceiling,
        rounds_ceiling,
        iterations_ceiling,
        node_transitions_ceiling,
        session_timeout,
    )


def _apply_user_overrides(
    ctx: ResearchContext,
    state: DeepResearchState,
    sq_floor: int,
    sq_ceiling: int,
    rounds_ceiling: int,
    iterations_ceiling: int,
) -> tuple[int, int, int, int]:
    """Apply per-request user overrides to ceilings."""
    if ctx.max_subqueries_override:
        sq_ceiling = min(sq_ceiling, ctx.max_subqueries_override)
        sq_floor = min(sq_floor, sq_ceiling)
    user_max_rounds = state.get("_user_max_rounds_override")
    user_max_iterations = state.get("_user_max_iterations_override")
    if user_max_rounds is not None:
        rounds_ceiling = min(rounds_ceiling, user_max_rounds)
    if user_max_iterations is not None:
        iterations_ceiling = min(iterations_ceiling, user_max_iterations)
    return sq_floor, sq_ceiling, rounds_ceiling, iterations_ceiling


async def _invoke_assessor(
    state: DeepResearchState,
    ctx: ResearchContext,
    mode: Any,
    sq_floor: int,
    sq_ceiling: int,
) -> tuple[str, int, int, int, str]:
    """Invoke LLM assessor and parse response."""
    query = state.get("query", "")
    context = state.get("context", "") or ""
    cached_count = len(state.get("findings_board", {}))
    complexity_hint = (
        mode.complexity_hint if mode else "User wants a quick, focused answer."
    )
    prompt = build_complexity_assessment_prompt()
    messages = prompt.format_messages(
        query=query,
        context=truncate_text(context, 500),
        cached_count=str(cached_count),
        complexity_hint=complexity_hint,
        min_subqueries=str(sq_floor),
        max_subqueries=str(sq_ceiling),
        recommended_subqueries=str(mode.recommended_subqueries if mode else 5),
    )
    response = await tracked_invoke(
        ctx.base_model,
        messages,
        ctx.token_tracker,
        "research",
        **ctx.llm_call_kwargs(),
    )
    result = safe_json_parse(str(response.content or ""))
    return _parse_assessor_response(result)


def _compute_session_seconds(
    mode: Any,
    assessed_sq: int,
    assessed_rounds: int,
    session_timeout: float,
) -> float:
    """Compute session timeout proportional to subquery count and supervisor rounds."""
    _BASE_EXECUTION_TIME = 120.0
    per_sq_allowance = getattr(mode, "per_subquery_allowance", 90.0) if mode else 90.0
    synth_reserve = getattr(mode, "synthesis_reserved_seconds", 90.0) if mode else 90.0
    review_reserve = getattr(mode, "review_reserved_seconds", 60.0) if mode else 60.0
    round_multiplier = max(1, (assessed_rounds + 1) // 2)
    research_budget = per_sq_allowance * assessed_sq * round_multiplier
    proportional_budget = (
        _BASE_EXECUTION_TIME + research_budget + synth_reserve + review_reserve
    )
    return min(
        max(proportional_budget, float(session_timeout)),
        DEFAULT_MAX_SESSION_SECONDS,
    )


async def assess_complexity_node(
    state: DeepResearchState,
    ctx: ResearchContext,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Neutral complexity assessor that dynamically sets iteration bounds.

    Runs early in the graph to analyze query complexity and recommend
    how many subqueries, supervisor rounds, and review iterations
    the system should invest.

    Returns:
        Tuple of (state_updates, events)
    """
    events: List[Dict[str, Any]] = []
    mode = ctx.mode_config

    (
        sq_floor,
        sq_ceiling,
        rounds_ceiling,
        iterations_ceiling,
        node_transitions_ceiling,
        session_timeout,
    ) = _get_mode_bounds(mode)
    sq_floor, sq_ceiling, rounds_ceiling, iterations_ceiling = _apply_user_overrides(
        ctx, state, sq_floor, sq_ceiling, rounds_ceiling, iterations_ceiling
    )

    try:
        (
            complexity_class,
            rec_sq,
            rec_rounds,
            rec_iters,
            reasoning,
        ) = await _invoke_assessor(state, ctx, mode, sq_floor, sq_ceiling)
    except Exception as e:
        logger.warning("Complexity assessor failed, using mode config defaults: %s", e)
        rec_sq = mode.recommended_subqueries if mode else 5
        complexity_class = "moderate"
        rec_rounds = rounds_ceiling
        rec_iters = iterations_ceiling
        reasoning = f"Heuristic fallback (assessor error: {e})"

    assessed_sq = max(sq_floor, min(rec_sq, sq_ceiling))
    assessed_rounds = min(rec_rounds, rounds_ceiling)
    assessed_iters = min(rec_iters, iterations_ceiling)
    assessed_node_transitions = min(
        node_transitions_ceiling,
        DEFAULT_MAX_NODE_TRANSITIONS,
    )
    assessed_session_seconds = _compute_session_seconds(
        mode, assessed_sq, assessed_rounds, session_timeout
    )

    event = emit_complexity_assessment(
        complexity_class, assessed_sq, assessed_rounds, assessed_iters, reasoning
    )
    ctx.emit_or_append(event, events)

    state_updates: Dict[str, Any] = {
        "assessed_max_subqueries": assessed_sq,
        "assessed_max_supervisor_rounds": assessed_rounds,
        "assessed_max_review_iterations": assessed_iters,
        "query_complexity_class": complexity_class,
        "complexity_reasoning": reasoning,
        "max_rounds": assessed_rounds,
        "max_iterations": assessed_iters,
        "max_total_subqueries": assessed_sq * assessed_rounds,
        "max_node_transitions": assessed_node_transitions,
        "max_session_seconds": assessed_session_seconds,
    }
    if mode is not None:
        state_updates["_mode_config"] = mode

    return state_updates, events
