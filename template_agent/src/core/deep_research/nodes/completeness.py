"""Completeness evaluator node: assess research coverage."""

from typing import Any, Dict, List

from template_agent.src.core.deep_research.events import (
    emit_completeness_assessment,
    emit_diminishing_returns,
    emit_sentinel_triggered,
    emit_validation_complete,
    emit_validation_conflict,
    emit_validation_start,
)
from template_agent.src.core.deep_research.prompts import (
    build_completeness_evaluation_prompt,
)
from template_agent.src.core.deep_research.sentinel import check_loop_sentinel
from template_agent.src.core.deep_research.state import (
    PHASE_SUPERVISOR,
    PHASE_SYNTHESIZE,
    DeepResearchState,
    ResearchContext,
)
from template_agent.src.core.deep_research.token_tracker import tracked_invoke
from template_agent.src.core.utils import safe_json_parse, truncate_text
from template_agent.utils.pylogger import get_python_logger

from ._event_emitter import NodeEventEmitter
from ._helpers import (
    _format_findings_for_synthesis,
    _summarize_findings_board,
    findings_from_board,
)

DEEP_RESEARCH_COMPLETENESS_THRESHOLD = 70
DEEP_RESEARCH_MAX_SUPERVISOR_ROUNDS = 3

logger = get_python_logger()


def _get_findings_summary_and_count(
    findings_board: dict,
    findings: dict,
) -> tuple[str, int]:
    """Build findings summary text and count successful findings."""
    if findings_board:
        summary = _summarize_findings_board(findings_board)
        count = sum(
            1
            for entry in findings_board.values()
            if not (entry.get("finding") or {}).get("error")
        )
    else:
        summary = _format_findings_for_synthesis(
            findings, findings_board if findings_board else None
        )
        count = sum(
            1
            for f in findings.values()
            if not f.get("error") and not f.get("access_denied")
        )
    return summary, count


def _get_completeness_threshold(state: DeepResearchState) -> float:
    """Resolve completeness threshold from mode config or default."""
    _mc = state.get("_mode_config")
    if _mc and hasattr(_mc, "completeness_threshold"):
        return _mc.completeness_threshold
    return DEEP_RESEARCH_COMPLETENESS_THRESHOLD


def _make_fallback_eval_result(
    round_num: int,
    max_rounds: int,
    reason: str,
) -> dict:
    """Build fail-safe eval result when LLM parse or call fails."""
    fallback_decision = (
        "needs_more_research" if round_num < max_rounds else "ready_for_synthesis"
    )
    return {
        "coverage_pct": 0,
        "uncovered_aspects": [],
        "contradictions": [],
        "numeric_issues": [],
        "decision": fallback_decision,
        "follow_up_subqueries": [],
        "reasoning": reason,
    }


async def _run_llm_completeness_evaluation(
    query: str,
    findings_summary: str,
    completeness_threshold: float,
    round_num: int,
    max_rounds: int,
    ctx: ResearchContext,
    emitter: NodeEventEmitter,
) -> tuple[dict, bool]:
    """Run LLM completeness evaluation; return (eval_result, used_fallback)."""
    try:
        completeness_prompt = build_completeness_evaluation_prompt()
        messages = completeness_prompt.format_messages(
            query=query,
            findings_summary=truncate_text(findings_summary, 4000),
            completeness_threshold=str(completeness_threshold),
        )
        response = await tracked_invoke(
            ctx.base_model,
            messages,
            ctx.token_tracker,
            "research",
            **ctx.llm_call_kwargs(),
        )
        eval_result = safe_json_parse(str(response.content or ""))

        if not eval_result or "decision" not in eval_result:
            emitter.thinking(
                "CompletenessEvaluator",
                "Evaluation parse failed — fail-safe applied",
            )
            return (
                _make_fallback_eval_result(
                    round_num,
                    max_rounds,
                    "Could not parse evaluation — fail-safe applied",
                ),
                True,
            )

        return eval_result, False

    except Exception as e:
        logger.warning("Completeness evaluation failed: %s", e)
        emitter.thinking(
            "CompletenessEvaluator",
            f"Evaluation error — fail-safe applied: {e}",
        )
        return (
            _make_fallback_eval_result(
                round_num, max_rounds, f"Evaluation failed ({e}) — fail-safe applied"
            ),
            True,
        )


def _apply_convergence_tracking(
    coverage_history: list,
    decision: str,
    ctx: ResearchContext,
    events: list,
) -> str:
    """Apply diminishing-returns logic; return updated decision."""
    if len(coverage_history) < 2:
        return decision
    delta = abs(coverage_history[-1] - coverage_history[-2])
    if delta >= 5.0:
        return decision
    ctx.emit_or_append(emit_diminishing_returns(coverage_history, delta), events)
    return "ready_for_synthesis"


def _apply_quality_early_exit(
    coverage_pct: float,
    contradictions: list,
    successful_count: int,
    decision: str,
    emitter: NodeEventEmitter,
) -> str:
    """Apply high-coverage early exit."""
    if (
        coverage_pct >= 90
        and not contradictions
        and successful_count > 0
        and decision != "ready_for_synthesis"
    ):
        emitter.thinking(
            "CompletenessEvaluator",
            f"High coverage ({coverage_pct:.0f}%) with no contradictions — early synthesis",
        )
        return "ready_for_synthesis"
    return decision


def _emit_contradiction_events(
    contradictions: list, events: list, ctx: ResearchContext
) -> None:
    """Emit validation conflict events for each contradiction."""
    for contradiction in contradictions:
        desc = (
            contradiction.get("description", str(contradiction))
            if isinstance(contradiction, dict)
            else str(contradiction)
        )
        ctx.emit_or_append(emit_validation_conflict(desc, [], "low"), events)


def _build_validation_notes(numeric_issues: list, contradictions: list) -> str:
    """Build validation notes string."""
    notes = ""
    if numeric_issues:
        notes += "\nNumeric issues: " + "; ".join(str(i) for i in numeric_issues)
    if contradictions:
        notes += "\nContradictions: " + "; ".join(
            str(c) if isinstance(c, str) else c.get("description", str(c))
            for c in contradictions
        )
    return notes


def _should_route_to_supervisor(
    decision: str,
    round_num: int,
    max_rounds: int,
    follow_ups: list,
    coverage_pct: float,
    completeness_threshold: float,
) -> bool:
    """True when we should route back to supervisor for more research."""
    return (
        decision == "needs_more_research"
        and round_num < max_rounds
        and bool(follow_ups)
        and coverage_pct < completeness_threshold
    )


async def completeness_evaluator_node(
    state: DeepResearchState,
    ctx: ResearchContext,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Completeness Evaluator: LLM-driven check of research coverage.

    Examines all findings against the original query to determine if
    research is sufficient for synthesis. No data product specific metrics.
    """
    events: List[Dict[str, Any]] = []
    emitter = NodeEventEmitter(ctx, events)
    transitions = state.get("total_node_transitions", 0) + 1

    should_stop, reason, forced_phase = check_loop_sentinel(state, ctx, "completeness")
    if should_stop:
        budgets = {
            "total_subqueries": f"{state.get('total_subqueries_executed', 0)}/{state.get('max_total_subqueries', 0)}",
            "node_transitions": f"{transitions}/{state.get('max_node_transitions', 0)}",
        }
        emitter.raw(emit_sentinel_triggered(reason or "", "completeness", budgets))
        return {
            "current_phase": forced_phase,
            "sentinel_triggered": True,
            "sentinel_reason": reason,
            "total_node_transitions": transitions,
        }, events

    query = state.get("query", "")
    findings_board = state.get("findings_board", {})
    findings = findings_from_board(findings_board)
    round_num = state.get("current_round", 1)
    max_rounds = state.get("max_rounds", DEEP_RESEARCH_MAX_SUPERVISOR_ROUNDS)

    emitter.thinking(
        "CompletenessEvaluator",
        "Evaluating research coverage",
    )

    findings_summary, successful_count = _get_findings_summary_and_count(
        findings_board, findings
    )
    ctx.emit_or_append(emit_validation_start(successful_count), events)

    understanding = state.get("understanding", "")

    completeness_threshold = _get_completeness_threshold(state)

    eval_result, used_fallback = await _run_llm_completeness_evaluation(
        query,
        findings_summary,
        completeness_threshold,
        round_num,
        max_rounds,
        ctx,
        emitter,
    )

    fallback_count = state.get("fallback_count", 0)
    if used_fallback:
        fallback_count += 1

    coverage_pct = eval_result.get("coverage_pct", 70)
    gaps = eval_result.get("uncovered_aspects", [])
    contradictions = eval_result.get("contradictions", [])
    numeric_issues = eval_result.get("numeric_issues", [])
    decision = eval_result.get("decision", "ready_for_synthesis")
    follow_ups = eval_result.get("follow_up_subqueries", [])

    coverage_history = list(state.get("coverage_history", []))
    coverage_history.append(float(coverage_pct))
    decision = _apply_convergence_tracking(coverage_history, decision, ctx, events)
    decision = _apply_quality_early_exit(
        coverage_pct, contradictions, successful_count, decision, emitter
    )

    _emit_contradiction_events(contradictions, events, ctx)

    ctx.emit_or_append(
        emit_completeness_assessment(
            coverage_pct=coverage_pct,
            gaps=gaps,
            conflicts=[str(c) for c in contradictions],
            decision=decision,
            threshold=int(completeness_threshold),
        ),
        events,
    )

    validation_notes = _build_validation_notes(numeric_issues, contradictions)
    updated_understanding = (
        understanding + "\n\n## Data Validation Notes" + validation_notes
        if validation_notes
        else understanding
    )

    ctx.emit_or_append(
        emit_validation_complete(len(contradictions), len(numeric_issues)), events
    )

    if _should_route_to_supervisor(
        decision,
        round_num,
        max_rounds,
        follow_ups,
        coverage_pct,
        completeness_threshold,
    ):
        emitter.decision(
            "CompletenessEvaluator",
            "Coverage insufficient - routing back to supervisor",
            f"Coverage: {coverage_pct:.0f}%, threshold: {completeness_threshold:.0f}%, gaps: {len(gaps)}",
        )
        pending = [
            sq
            for sq in follow_ups[:3]
            if sq not in state.get("completed_subqueries", [])
        ]
        return {
            "understanding": updated_understanding,
            "pending_subqueries": pending,
            "coverage_complete": False,
            "current_phase": PHASE_SUPERVISOR,
            "total_node_transitions": transitions,
            "coverage_history": coverage_history,
            "fallback_count": fallback_count,
        }, events

    emitter.decision(
        "CompletenessEvaluator",
        "Coverage sufficient - proceeding to synthesis",
        f"Coverage: {coverage_pct:.0f}% | Ready for report generation",
    )

    return {
        "understanding": updated_understanding,
        "coverage_complete": True,
        "current_phase": PHASE_SYNTHESIZE,
        "total_node_transitions": transitions,
        "coverage_history": coverage_history,
        "fallback_count": fallback_count,
    }, events
