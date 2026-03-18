"""Review node: multi-persona quality review with Answer Quality Matrix."""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, cast

from template_agent.src.core.deep_research.events import (
    emit_agent_decision,
    emit_agent_thinking,
    emit_consensus_result,
    emit_consensus_vote,
    emit_reliability_update,
    emit_review_complete,
    emit_review_start,
    emit_reviewer_feedback,
    emit_reviewer_score,
    emit_sentinel_triggered,
)
from template_agent.src.core.deep_research.prompts import (
    REVIEWER_PERSONAS,
    build_review_prompt,
)
from template_agent.src.core.deep_research.sentinel import check_loop_sentinel
from template_agent.src.core.deep_research.state import (
    DEFAULT_MAX_ITERATIONS,
    PHASE_COMPLETE,
    PHASE_SUPERVISOR,
    PHASE_SYNTHESIZE,
    AgentMessage,
    DeepResearchState,
    ResearchContext,
)
from template_agent.src.core.deep_research.token_tracker import tracked_invoke
from template_agent.src.core.deep_research.utils import sanitize_error_for_client
from template_agent.utils.pylogger import get_python_logger

from ._helpers import (
    _format_findings_for_synthesis,
    _parse_review_result,
    check_node_cancelled,
    findings_from_board,
)

# String literals for actions (no SQL/data product specific)
APPROVE = "approve"
REVISE = "revise"
RESEARCH_MORE = "research_more"

DEEP_RESEARCH_REVIEW_MIN_SCORE = 60

logger = get_python_logger()


def _select_reviewers(mode_config: Any | None, time_pressure: bool) -> list:
    """Select reviewer personas based on mode."""
    if mode_config is None:
        return list(REVIEWER_PERSONAS)
    default_count = getattr(
        mode_config, "default_reviewer_count", len(REVIEWER_PERSONAS)
    )
    min_count = getattr(mode_config, "min_reviewer_count", 2)
    target = min_count if time_pressure else default_count

    def _weight(p: dict[str, Any]) -> float:
        w = p.get("weight", 1.0)
        return float(w) if isinstance(w, (int, float)) else 1.0

    sorted_personas = sorted(
        REVIEWER_PERSONAS,
        key=_weight,
        reverse=True,
    )
    return sorted_personas[: min(target, len(REVIEWER_PERSONAS))]


def _is_time_pressured(state: DeepResearchState, mode_config: Any | None) -> bool:
    """True when remaining time is low."""
    start_time = state.get("execution_start_time", 0.0)
    max_seconds = state.get("max_session_seconds", 0.0)
    if start_time <= 0 or max_seconds <= 0:
        return False
    elapsed = time.time() - start_time
    remaining = max_seconds - elapsed
    synth_reserve = getattr(mode_config, "synthesis_reserved_seconds", 90)
    review_reserve = getattr(mode_config, "review_reserved_seconds", 60)
    return remaining < 1.2 * (synth_reserve + review_reserve)


def _compute_quality_matrix(
    review_results: List[Dict],
    _mode_config: Any | None,
) -> Dict[str, Any]:
    """Aggregate review scores into quality matrix.

    Gate results:
        - "pass": weighted score >= 0.6 (answer is good enough)
        - "revise": 0.4 <= weighted score < 0.6 (text needs revision)
        - "research_more": weighted score < 0.4 (insufficient research)
    """
    if not review_results:
        return {
            "dimensions": {},
            "weighted_score": 0.5,
            "mandatory_pass": True,
            "gate_result": "pass",
            "gate_violations": [],
        }
    scores = [r.get("score", 50) for r in review_results]
    avg = sum(scores) / len(scores) if scores else 50
    weighted_score = avg / 100.0
    if weighted_score >= 0.6:
        gate_result = "pass"
    elif weighted_score >= 0.4:
        gate_result = "revise"
    else:
        gate_result = "research_more"
    return {
        "dimensions": {},
        "weighted_score": round(weighted_score, 3),
        "mandatory_pass": True,
        "gate_result": gate_result,
        "gate_violations": [],
    }


async def _invoke_single_reviewer(
    state: DeepResearchState,
    ctx: ResearchContext,
    findings_text: str,
    persona: str,
    focus: str,
) -> tuple[Dict | None, bool, str]:
    """Invoke LLM for one reviewer."""
    review_prompt = build_review_prompt()
    mode_review_instruction = (
        ctx.mode_config.review_instruction if ctx.mode_config else ""
    )
    review_messages = review_prompt.format_messages(
        query=state.get("query", ""),
        draft_answer=state.get("draft_answer", ""),
        findings=findings_text,
        persona=persona,
        focus=focus,
        mode_instruction=mode_review_instruction,
    )
    try:
        response = await tracked_invoke(
            ctx.base_model,
            review_messages,
            ctx.token_tracker,
            "review",
            **ctx.llm_call_kwargs(),
        )
        review_text = str(response.content or "")
        parsed = _parse_review_result(review_text, persona)
        return parsed, False, ""
    except Exception as e:
        return None, False, sanitize_error_for_client(e)


async def _run_reviewer_loop(
    state: DeepResearchState,
    ctx: ResearchContext,
    findings_text: str,
    selected_personas: list,
    events: List[Dict[str, Any]],
    review_results: List[Dict],
) -> tuple[List[Dict], list, list, List[str]]:
    """Run multi-persona review loop in parallel."""
    personas_info = [(str(pc["persona"]), str(pc["focus"])) for pc in selected_personas]
    for persona, focus in personas_info:
        ctx.emit_or_append(emit_review_start(persona), events)
        ctx.emit_or_append(
            emit_agent_thinking(f"Reviewer:{persona}", f"Evaluating: {focus}"),
            events,
        )

    llm_results = await asyncio.gather(
        *(
            _invoke_single_reviewer(state, ctx, findings_text, persona, focus)
            for persona, focus in personas_info
        ),
        return_exceptions=True,
    )

    all_scores: list = []
    all_actions: list = []
    follow_ups: List[str] = []

    for (persona, focus), result in zip(personas_info, llm_results):
        if isinstance(result, BaseException):
            ctx.emit_or_append(
                emit_agent_thinking(
                    f"Reviewer:{persona}",
                    f"Review failed ({sanitize_error_for_client(result)})",
                ),
                events,
            )
            continue
        parsed, _, _ = result
        if parsed is None:
            continue
        ctx.emit_or_append(
            emit_reviewer_feedback(
                persona,
                parsed.get("reason", ""),
                strengths=None,
                weaknesses=parsed.get("weaknesses"),
            ),
            events,
        )
        ctx.emit_or_append(
            emit_reviewer_score(persona, parsed.get("score", 50), max_score=100),
            events,
        )
        ctx.emit_or_append(
            emit_consensus_vote(
                persona,
                parsed.get("action", "approve"),
                parsed.get("score", 70) / 100.0,
                (parsed.get("feedback") or parsed.get("reason", ""))[:200],
            ),
            events,
        )
        ctx.emit_or_append(
            emit_review_complete(
                parsed.get("action", "approve"),
                parsed.get("score", 50),
                parsed.get("reason", ""),
            ),
            events,
        )
        review_results.append(parsed)
        all_scores.append(parsed.get("score", 50))
        all_actions.append(parsed.get("action", APPROVE))
        follow_ups.extend(parsed.get("follow_up_subqueries", []))

    return review_results, all_scores, all_actions, follow_ups


def _determine_review_action(
    gate_result: str,
    mandatory_pass: bool,
    avg_score: int,
    approve_threshold: int,
    approve_votes: int,
    total_voters: int,
    iteration: int,
    max_iterations: int,
    follow_ups: List[str],
    state: DeepResearchState,
) -> tuple[str, str, List[str]]:
    """Determine overall_action, next_phase, and subqueries."""
    subqueries = state.get("subqueries", [])

    if (
        gate_result == "pass"
        and mandatory_pass
        and avg_score >= approve_threshold
        and approve_votes >= total_voters / 2
    ):
        return APPROVE, PHASE_COMPLETE, subqueries

    if gate_result == "research_more" and iteration < max_iterations and follow_ups:
        current = state.get("subqueries", [])
        new_sq = [sq for sq in follow_ups[:3] if sq not in current]
        return RESEARCH_MORE, PHASE_SUPERVISOR, current + new_sq

    if gate_result in ("revise", "research_more") and iteration < max_iterations:
        return REVISE, PHASE_SYNTHESIZE, subqueries

    if avg_score >= approve_threshold and approve_votes >= total_voters / 2:
        return APPROVE, PHASE_COMPLETE, subqueries

    if iteration >= max_iterations - 1:
        return APPROVE, PHASE_COMPLETE, subqueries

    return REVISE, PHASE_SYNTHESIZE, subqueries


async def review_node(
    state: DeepResearchState,
    ctx: ResearchContext,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Review node: multi-persona quality review with Answer Quality Matrix.

    Keeps reviewer personas and quality logic. No SQL/data product specific criteria.
    """
    cancelled = await check_node_cancelled(state.get("thread_id"), "review", state)
    if cancelled is not None:
        return cancelled, []

    events: List[Dict[str, Any]] = []
    transitions = state.get("total_node_transitions", 0) + 1

    # Sentinel check
    existing_reviews = state.get("review_results", [])
    if existing_reviews:
        should_stop, reason, _ = check_loop_sentinel(state, ctx, "review")
        if should_stop:
            budgets = {
                "total_subqueries": f"{state.get('total_subqueries_executed', 0)}/{state.get('max_total_subqueries', 0)}",
                "node_transitions": f"{transitions}/{state.get('max_node_transitions', 0)}",
            }
            ctx.emit_or_append(
                emit_sentinel_triggered(reason or "", "review", budgets), events
            )
            return {
                "current_phase": PHASE_COMPLETE,
                "current_review": {
                    "action": "approve",
                    "score": 0,
                    "reason": f"Auto-approved: {reason}",
                },
                "sentinel_triggered": True,
                "sentinel_reason": reason,
                "total_node_transitions": transitions,
            }, events

    draft = state.get("draft_answer", "")
    if not draft or draft.startswith("Research could not"):
        return {
            "current_review": {
                "action": "approve",
                "score": 0,
                "reason": "No draft to review",
            },
            "current_phase": PHASE_COMPLETE,
            "total_node_transitions": transitions,
        }, events

    review_results = list(state.get("review_results", []))
    ctx.emit_or_append(
        emit_agent_thinking(
            "ReviewCoordinator",
            "Starting multi-persona review of the synthesized answer",
        ),
        events,
    )

    findings_board = state.get("findings_board", {})
    findings = findings_from_board(findings_board)
    findings_text = _format_findings_for_synthesis(
        findings, findings_board if findings_board else None
    )

    time_pressure = _is_time_pressured(state, ctx.mode_config)
    selected_personas = _select_reviewers(ctx.mode_config, time_pressure)

    review_results, all_scores, all_actions, follow_ups = await _run_reviewer_loop(
        state=state,
        ctx=ctx,
        findings_text=findings_text,
        selected_personas=selected_personas,
        events=events,
        review_results=review_results,
    )

    if len(all_actions) < 2:
        return {
            "review_results": review_results,
            "current_review": {
                "action": REVISE,
                "score": 0,
                "reason": f"Only {len(all_actions)} reviewer(s) succeeded",
                "persona": "system",
            },
            "current_phase": PHASE_SYNTHESIZE,
            "iteration": state.get("iteration", 0) + 1,
            "total_node_transitions": transitions,
        }, events

    quality_matrix = _compute_quality_matrix(review_results, ctx.mode_config)
    avg_score = sum(all_scores) // len(all_scores) if all_scores else 50
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", DEFAULT_MAX_ITERATIONS)
    approve_votes = sum(1 for a in all_actions if a == APPROVE)
    total_voters = len(all_actions)
    approve_threshold = (
        max(50, DEEP_RESEARCH_REVIEW_MIN_SCORE - 10)
        if iteration > 0
        else DEEP_RESEARCH_REVIEW_MIN_SCORE
    )

    gate_result = quality_matrix.get("gate_result", "pass")
    gate_violations = quality_matrix.get("gate_violations", [])
    mandatory_pass = quality_matrix.get("mandatory_pass", True)

    overall_action, next_phase, subqueries = _determine_review_action(
        gate_result=gate_result,
        mandatory_pass=mandatory_pass,
        avg_score=avg_score,
        approve_threshold=approve_threshold,
        approve_votes=approve_votes,
        total_voters=total_voters,
        iteration=iteration,
        max_iterations=max_iterations,
        follow_ups=follow_ups,
        state=state,
    )

    if iteration >= max_iterations:
        overall_action, next_phase = APPROVE, PHASE_COMPLETE

    current_review: Dict = {
        "action": overall_action,
        "score": avg_score,
        "reason": f"Aggregate from {len(selected_personas)} reviewers (gate: {gate_result})",
        "feedback": "; ".join(
            r.get("feedback", "") for r in review_results if r.get("feedback")
        ),
        "follow_up_subqueries": follow_ups[:3],
        "persona": "aggregate",
    }

    agent_messages: list[dict[str, Any]] = list(state.get("agent_messages", []))
    ts = datetime.now(timezone.utc).isoformat()
    if overall_action == RESEARCH_MORE:
        agent_messages.append(
            cast(
                Dict[str, Any],
                AgentMessage(
                    from_agent="reviewer:aggregate",
                    to_agent="supervisor",
                    message_type="feedback",
                    content="More research needed",
                    metadata={
                        "follow_up_subqueries": follow_ups[:3],
                        "avg_score": avg_score,
                    },
                    timestamp=ts,
                ),
            )
        )
    elif overall_action == REVISE:
        agent_messages.append(
            cast(
                Dict[str, Any],
                AgentMessage(
                    from_agent="reviewer:aggregate",
                    to_agent="synthesizer",
                    message_type="feedback",
                    content="Revision needed",
                    metadata={
                        "gate_violations": gate_violations,
                        "avg_score": avg_score,
                    },
                    timestamp=ts,
                ),
            )
        )

    approve_count = sum(1 for a in all_actions if a == APPROVE)
    reject_count = sum(1 for a in all_actions if a == RESEARCH_MORE)
    revision_count = sum(1 for a in all_actions if a == REVISE)
    weighted_matrix_score = quality_matrix.get("weighted_score", 0.5)
    confidence_score = min(0.95, weighted_matrix_score)

    ctx.emit_or_append(
        emit_consensus_result(
            approved=(overall_action == APPROVE),
            approve_count=approve_count,
            reject_count=reject_count,
            revision_count=revision_count,
            overall_confidence=confidence_score,
            summary=f"Action: {overall_action} | Score: {avg_score}/100",
        ),
        events,
    )

    if overall_action == APPROVE:
        ctx.emit_or_append(
            emit_agent_decision(
                "ReviewCoordinator",
                "Answer approved by reviewers",
                f"Score: {avg_score}/100 - Moving to completion",
            ),
            events,
        )
    elif overall_action == REVISE:
        ctx.emit_or_append(
            emit_agent_decision(
                "ReviewCoordinator",
                "Answer needs revision",
                f"Score: {avg_score}/100 - Sending back to synthesis",
            ),
            events,
        )
    else:
        ctx.emit_or_append(
            emit_agent_decision(
                "ReviewCoordinator",
                "More research needed",
                f"Adding {len(follow_ups[:3])} follow-up queries",
            ),
            events,
        )

    reliability_metrics = {
        "confidence_score": round(confidence_score, 3),
        "weighted_avg_score": avg_score,
        "quality_gate_result": gate_result,
        "total_reviewers": total_voters,
    }
    ctx.emit_or_append(emit_reliability_update(reliability_metrics, []), events)

    return {
        "review_results": review_results,
        "current_review": current_review,
        "quality_matrix": quality_matrix,
        "subqueries": subqueries,
        "pending_subqueries": follow_ups[:3] if overall_action == RESEARCH_MORE else [],
        "agent_messages": agent_messages,
        "iteration": iteration + 1,
        "current_phase": next_phase,
        "total_node_transitions": transitions,
    }, events
