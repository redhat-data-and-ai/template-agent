"""Supervisor node: orchestrate parallel research workers."""

import asyncio
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from template_agent.src.core.deep_research.agents import (
    execute_with_research_agent,
)
from template_agent.src.core.deep_research.cancel import get_cancel_store
from template_agent.src.core.deep_research.events import (
    emit_agent_decision,
    emit_agent_thinking,
    emit_cancelled,
    emit_empty_plan,
    emit_research_complete,
    emit_research_failed,
    emit_research_start,
    emit_sentinel_triggered,
    emit_subquery_cached,
    emit_subquery_complete,
    emit_subquery_error,
    emit_subquery_start,
    emit_supervisor_delegating,
    emit_supervisor_follow_up,
    emit_supervisor_reflection,
    emit_supervisor_round_start,
    emit_worker_progress,
    emit_worker_reformulation,
    emit_worker_self_evaluation,
)
from template_agent.src.core.deep_research.prompts import (
    CONFLICT_DETECTOR_SYSTEM_PROMPT,
    build_alternative_approach_prompt,
    build_conflict_resolution_prompt,
    build_plausibility_check_prompt,
    build_supervisor_reflection_prompt,
    build_worker_context_prefix,
    build_worker_execution_instruction,
    build_worker_mode_instruction,
    build_worker_self_evaluation_prompt,
)
from template_agent.src.core.deep_research.sentinel import (
    check_loop_sentinel,
    trim_follow_ups,
)
from template_agent.src.core.deep_research.state import (
    PHASE_COMPLETE,
    PHASE_COMPLETENESS,
    PHASE_SUPERVISOR,
    PHASE_SYNTHESIZE,
    AgentMessage,
    DeepResearchState,
    Finding,
    FindingCard,
    FindingEntry,
    ImmediateContext,
    ResearchContext,
    ResearchMemory,
    SupervisorRound,
)
from template_agent.src.core.deep_research.token_tracker import tracked_invoke
from template_agent.src.core.utils import safe_json_parse, truncate_text
from template_agent.utils.pylogger import get_python_logger

from ._cache import find_matching_cached_finding
from ._helpers import (
    SubAgentResult,
    _classify_failure,
    _find_related_findings,
    _make_finding,
    _summarize_findings_board,
    compute_data_quality_score,
    findings_from_board,
    should_exclude_from_synthesis,
)

# Reasonable defaults (no dataverse settings)
DEEP_RESEARCH_SUBQUERY_TIMEOUT_SECONDS = 180
DEEP_RESEARCH_SUBAGENT_MAX_RETRIES = 2
DEEP_RESEARCH_SUBAGENT_QUALITY_THRESHOLD = 0.6
DEEP_RESEARCH_MAX_SUPERVISOR_ROUNDS = 3
DEEP_RESEARCH_MAX_TOTAL_SUBQUERIES = 20
DEEP_RESEARCH_MAX_CONCURRENT_WORKERS = 4
DEEP_RESEARCH_COMPLETENESS_THRESHOLD = 70
ENABLE_HIERARCHICAL_CONTEXT = False

logger = get_python_logger()


@dataclass
class SupervisorContinueInput:
    """Grouped params for _try_continue_research."""

    decision: str
    round_num: int
    max_rounds: int
    follow_ups: List[str]
    subqueries: List[str]
    to_execute: List[str]
    completed: List[str]
    supervisor_rounds: List[SupervisorRound]
    total_sq_executed: int
    state: DeepResearchState
    execution_start_update: Dict[str, Any]
    findings_board: Dict[str, FindingEntry]
    agent_messages: List[AgentMessage]
    transitions: int
    immediate_context: Optional[Any]
    finding_cards: list
    research_memory: Optional[Any]
    findings_count_history: List[int]
    fallback_count: int
    ctx: ResearchContext
    events: List[Dict[str, Any]]


@dataclass
class MassFailureInput:
    """Grouped params for _check_mass_failure."""

    to_execute: List[str]
    findings_received: List[str]
    findings_board: Dict[str, FindingEntry]
    round_num: int
    execution_start_update: Dict[str, Any]
    agent_messages: List[AgentMessage]
    supervisor_rounds: List[SupervisorRound]
    completed: List[str]
    transitions: int
    immediate_context: Optional[Any]
    finding_cards: list
    research_memory: Optional[Any]
    total_sq_executed: int
    findings_count_history: List[int]
    ctx: ResearchContext
    events: List[Dict[str, Any]]


@dataclass
class SupervisorStateBase:
    """Base params for _build_supervisor_state."""

    execution_start_update: Dict[str, Any]
    findings_board: Dict[str, FindingEntry]
    agent_messages: List[AgentMessage]
    supervisor_rounds: List[SupervisorRound]
    completed: List[str]
    round_num: int
    transitions: int
    immediate_context: Optional[Any]
    finding_cards: list
    research_memory: Optional[Any]


@dataclass
class WorkerResultContext:
    """Grouped params for _process_and_deposit_worker_result."""

    sq: str
    result: SubAgentResult
    round_num: int
    to_execute: List[str]
    total: int
    findings_board: Dict[str, FindingEntry]
    completed: List[str]
    findings_received: List[str]
    agent_messages: List[AgentMessage]
    immediate_context: Optional[ImmediateContext]
    finding_cards: list
    research_memory: Optional[ResearchMemory]
    events: List[Dict[str, Any]]


@dataclass
class ReflectionRouteInput:
    """Grouped params for _supervisor_do_reflection_and_route."""

    ctx: ResearchContext
    state: DeepResearchState
    findings_board: Dict[str, FindingEntry]
    findings_received: List[str]
    round_num: int
    max_rounds: int
    to_execute: List[str]
    subqueries: List[str]
    completed: List[str]
    supervisor_rounds: List[SupervisorRound]
    total_sq_executed: int
    execution_start_update: Dict[str, Any]
    agent_messages: List[AgentMessage]
    transitions: int
    immediate_context: Optional[ImmediateContext]
    finding_cards: list
    research_memory: Optional[ResearchMemory]
    findings_count_history: List[int]
    events: List[Dict[str, Any]]


@dataclass
class CancellationCheckInput:
    """Grouped params for _supervisor_check_cancellation."""

    thread_id: Optional[str]
    execution_start_update: Dict[str, Any]
    findings_board: Dict[str, FindingEntry]
    agent_messages: List[AgentMessage]
    supervisor_rounds: List[SupervisorRound]
    completed: List[str]
    round_num: int
    transitions: int
    immediate_context: Optional[ImmediateContext]
    finding_cards: list
    research_memory: Optional[ResearchMemory]
    total_sq_executed: int
    findings_count_history: List[int]
    ctx: ResearchContext
    events: List[Dict[str, Any]]


def _apply_plausibility_penalties(
    quality_score: float,
    confidence: str,
    plausibility: Dict[str, Any],
) -> tuple[float, str]:
    """Apply plausibility penalties to quality score and confidence."""
    if plausibility.get("plausible", True):
        return quality_score, confidence
    warnings = plausibility.get("warnings", [])
    severity_penalties = {"minor": 0.1, "major": 0.2, "critical": 0.3}
    max_penalty = (
        max(severity_penalties.get(w.get("severity", "minor"), 0.1) for w in warnings)
        if warnings
        else 0.0
    )
    adjusted_score = max(0.0, quality_score - max_penalty)
    adjusted_confidence = "medium" if confidence == "high" else confidence
    return adjusted_score, adjusted_confidence


async def _get_reformulated_query(
    ctx: ResearchContext,
    current_query: str,
    finding: Finding,
    plausibility: Dict[str, Any],
    eval_result: Dict[str, Any],
) -> Optional[str]:
    """Get reformulated query from implausible warnings or eval result."""
    raw_warnings = finding.get("plausibility_warnings") or []
    implausible_warnings = [w for w in raw_warnings if isinstance(w, dict)]
    if implausible_warnings:
        return await _generate_alternative_approach(
            ctx, current_query, implausible_warnings
        )
    return plausibility.get("suggested_requery") or eval_result.get(
        "reformulated_query"
    )


def _is_non_retryable_failure(
    failure_class: str,
    attempt: int,
    effective_retries: int,
) -> bool:
    """Return True if failure should not be retried."""
    non_retryable = ("access_denied", "tool_timeout")
    return failure_class in non_retryable or attempt >= effective_retries


async def _process_successful_attempt(
    ctx: ResearchContext,
    subquery: str,
    index: int,
    attempt: int,
    effective_retries: int,
    effective_threshold: float,
    enhanced_query: str,
    thread_id: Optional[str],
    cross_context: str,
    original_query: str,
    events: List[Dict[str, Any]],
) -> tuple[Finding, float, str, str, Dict[str, Any], Dict[str, Any]]:
    """Execute agent, evaluate, run plausibility check."""
    result = await asyncio.wait_for(
        execute_with_research_agent(
            ctx,
            enhanced_query,
            f"{thread_id}_sq_{index}_a{attempt}" if thread_id else None,
        ),
        timeout=DEEP_RESEARCH_SUBQUERY_TIMEOUT_SECONDS,
    )
    answer = result.get("answer", "")
    tool_results = result.get("tool_results", [])

    finding = _make_finding(
        subquery,
        answer=answer,
        tool_results=tool_results,
    )

    eval_result = await _self_evaluate_finding(ctx, subquery, answer, cross_context)
    try:
        quality_score = float(eval_result.get("quality_score") or 0.5)
    except (TypeError, ValueError):
        quality_score = 0.5
    confidence = eval_result.get("confidence", "medium")
    summary = eval_result.get("summary", truncate_text(answer, 200))

    if eval_result.get("plausibility_concern"):
        finding["plausibility_concern"] = eval_result["plausibility_concern"]

    ctx.emit_or_append(
        emit_worker_self_evaluation(
            subquery,
            quality_score,
            confidence,
            will_retry=(
                quality_score < effective_threshold and attempt < effective_retries
            ),
            attempt=attempt,
        ),
        events,
    )

    plausibility = await _plausibility_check_finding(
        ctx,
        subquery,
        answer,
        original_query or subquery,
    )
    if not plausibility.get("plausible", True):
        finding["plausibility_warnings"] = plausibility.get("warnings", [])
        quality_score, confidence = _apply_plausibility_penalties(
            quality_score, confidence, plausibility
        )
        logger.info(
            "Plausibility check flagged subquery %d: %d warning(s), score adjusted to %.2f",
            index,
            len(plausibility.get("warnings", [])),
            quality_score,
        )

    return finding, quality_score, confidence, summary, plausibility, eval_result


def _retry_loop_handle_exception(
    e: Exception,
    subquery: str,
    attempt: int,
    effective_retries: int,
    index: int,
    best_finding: Optional[Finding],
    best_score: float,
    prior_failure_class: Optional[str],
) -> tuple[
    Optional[tuple[Optional[Finding], float, str, str, Optional[str]]],
    Optional[Finding],
    Optional[str],
]:
    """Handle exception in retry loop."""
    error_msg = truncate_text(str(e), 200)
    failure_class = _classify_failure(e, error_msg)
    error_finding = _create_error_finding(subquery, error_msg, failure_class)

    if _is_non_retryable_failure(failure_class, attempt, effective_retries):
        return (error_finding, 0.0, "low", error_msg, prior_failure_class), None, None

    new_best = (
        error_finding if (best_finding is None or best_score <= 0) else best_finding
    )
    logger.info(
        "Retryable failure on attempt %d/%d for subquery %d: %s",
        attempt,
        effective_retries,
        index,
        failure_class,
    )
    return None, new_best, failure_class


def _build_context_prefix(cross_context: str, attempt: int) -> str:
    """Build cross-context prefix for first attempt only.

    Delegates to the central prompts module.
    """
    if attempt != 1:
        return ""
    return build_worker_context_prefix(
        truncate_text(cross_context, 1000) if cross_context else ""
    )


def _create_timeout_finding(subquery: str) -> Finding:
    """Create a finding for timeout errors."""
    finding = _make_finding(subquery, error="Execution timed out")
    finding["failure_class"] = "tool_timeout"
    return finding


def _create_error_finding(
    subquery: str,
    error_msg: str,
    failure_class: str,
) -> Finding:
    """Create a finding for execution errors."""
    finding = _make_finding(subquery, error=error_msg)
    finding["failure_class"] = failure_class
    return finding


async def _try_reformulate_query(
    ctx: ResearchContext,
    current_query: str,
    subquery: str,
    index: int,
    finding: Finding,
    plausibility: Dict[str, Any],
    eval_result: Dict[str, Any],
    effective_retries: int,
    attempt: int,
    events: List[Dict[str, Any]],
) -> Optional[str]:
    """Attempt to get a reformulated query; emit events if found."""
    if attempt >= effective_retries:
        return None
    reformulated = await _get_reformulated_query(
        ctx, current_query, finding, plausibility, eval_result
    )
    if not reformulated:
        return None
    ctx.emit_or_append(
        emit_worker_reformulation(
            subquery,
            current_query,
            reformulated,
            attempt,
            max_retries=effective_retries,
        ),
        events,
    )
    logger.info(
        "Worker pivoting subquery %d (attempt %d): %s",
        index,
        attempt,
        truncate_text(reformulated, 100),
    )
    return reformulated


def _build_worker_mode_instruction(ctx: ResearchContext) -> str:
    """Build mode-specific worker instruction string.

    Delegates to the central prompts module.
    """
    return build_worker_mode_instruction(ctx.mode_config)


async def _execute_retry_loop(
    ctx: ResearchContext,
    subquery: str,
    index: int,
    thread_id: Optional[str],
    cross_context: str,
    original_query: str,
    execution_instruction: str,
    worker_mode_instruction: str,
    effective_retries: int,
    effective_threshold: float,
    events: List[Dict[str, Any]],
) -> tuple[Optional[Finding], float, str, str, Optional[str]]:
    """Run retry loop."""
    current_query = subquery
    best_finding: Optional[Finding] = None
    best_score = 0.0
    best_confidence = "low"
    best_summary = ""
    prior_failure_class: Optional[str] = None

    for attempt in range(1, effective_retries + 1):
        context_prefix = _build_context_prefix(cross_context, attempt)
        enhanced_query = (
            current_query
            + context_prefix
            + execution_instruction
            + worker_mode_instruction
        )

        try:
            (
                finding,
                quality_score,
                confidence,
                summary,
                plausibility,
                eval_result,
            ) = await _process_successful_attempt(
                ctx,
                subquery,
                index,
                attempt,
                effective_retries,
                effective_threshold,
                enhanced_query,
                thread_id,
                cross_context,
                original_query,
                events,
            )

            if quality_score > best_score:
                best_finding = finding
                best_score = quality_score
                best_confidence = confidence
                best_summary = summary

            if quality_score >= effective_threshold:
                return (
                    best_finding,
                    best_score,
                    best_confidence,
                    best_summary,
                    prior_failure_class,
                )

            reformulated = await _try_reformulate_query(
                ctx,
                current_query,
                subquery,
                index,
                finding,
                plausibility,
                eval_result,
                effective_retries,
                attempt,
                events,
            )
            if reformulated:
                current_query = reformulated

        except asyncio.TimeoutError:
            return (
                _create_timeout_finding(subquery),
                0.0,
                "low",
                "Timed out",
                prior_failure_class,
            )

        except Exception as e:
            early_ret, new_best, new_prior = _retry_loop_handle_exception(
                e,
                subquery,
                attempt,
                effective_retries,
                index,
                best_finding,
                best_score,
                prior_failure_class,
            )
            if early_ret is not None:
                return early_ret
            best_finding = new_best
            prior_failure_class = new_prior

    return best_finding, best_score, best_confidence, best_summary, prior_failure_class


def _finalize_best_finding(
    best_finding: Finding,
    prior_failure_class: Optional[str],
    best_score: float,
    subquery: str,
) -> None:
    """Apply final metadata to best finding (mutates in place)."""
    if best_finding.get("plausibility_warnings"):
        best_finding["data_quality_alert"] = True
    if prior_failure_class and not best_finding.get("error"):
        best_finding["failure_class"] = prior_failure_class
    _DROP_THRESHOLD = 0.30
    if best_score < _DROP_THRESHOLD and not best_finding.get("error"):
        best_finding["low_quality_drop"] = True
        logger.info(
            "Dropping low-quality finding (score=%.2f < %.2f): %s",
            best_score,
            _DROP_THRESHOLD,
            truncate_text(subquery, 80),
        )


async def _execute_research_subagent(
    ctx: ResearchContext,
    subquery: str,
    index: int,
    _total: int,
    thread_id: Optional[str],
    cross_context: str = "",
    max_retries: int = 2,
    quality_threshold: float = 0.6,
    original_query: str = "",
) -> SubAgentResult:
    """Execute a research subquery with self-evaluation and retry.

    Uses execute_with_research_agent with a self-reflection loop.
    """
    events: List[Dict[str, Any]] = []
    execution_instruction = build_worker_execution_instruction()

    worker_mode_instruction = _build_worker_mode_instruction(ctx)
    effective_retries = min(max_retries, DEEP_RESEARCH_SUBAGENT_MAX_RETRIES)
    effective_threshold = quality_threshold

    (
        best_finding,
        best_score,
        best_confidence,
        best_summary,
        prior_failure_class,
    ) = await _execute_retry_loop(
        ctx,
        subquery,
        index,
        thread_id,
        cross_context,
        original_query,
        execution_instruction,
        worker_mode_instruction,
        effective_retries,
        effective_threshold,
        events,
    )

    if best_finding is None:
        best_finding = _make_finding(subquery, error="No result produced")
        best_finding["failure_class"] = "unknown"

    _finalize_best_finding(best_finding, prior_failure_class, best_score, subquery)

    return SubAgentResult(
        finding=best_finding,
        quality_score=best_score,
        confidence=best_confidence,
        summary=best_summary,
        events=events,
    )


async def _self_evaluate_finding(
    ctx: ResearchContext,
    subquery: str,
    answer: str,
    cross_context: str = "",
) -> Dict[str, Any]:
    """Use the LLM to self-evaluate a research finding's quality."""
    try:
        eval_prompt = build_worker_self_evaluation_prompt()
        messages = eval_prompt.format_messages(
            subquery=subquery,
            answer=truncate_text(answer, 2000),
            cross_context=truncate_text(cross_context, 500)
            if cross_context
            else "None available yet.",
        )
        response = await tracked_invoke(
            ctx.base_model,
            messages,
            ctx.token_tracker,
            "research",
            **ctx.llm_call_kwargs(),
        )
        result = safe_json_parse(str(response.content or ""))
        if result and "quality_score" in result:
            return result
    except Exception as e:
        logger.warning("Self-evaluation failed for subquery: %s", e)

    has_data = bool(answer and len(answer.strip()) > 100)
    return {
        "quality_score": 0.7 if has_data else 0.3,
        "confidence": "medium" if has_data else "low",
        "has_real_data": has_data,
        "reformulated_query": None,
        "summary": truncate_text(answer, 200) if answer else "No answer produced.",
        "plausibility_concern": None,
    }


async def _plausibility_check_finding(
    ctx: ResearchContext,
    subquery: str,
    answer: str,
    original_query: str,
) -> Dict[str, Any]:
    """Check whether the numeric values in a finding are plausible.

    Uses the LLM's general reasoning to flag values that are
    wildly outside expected ranges. No SQL-specific checks.
    """
    nums_in_answer = re.findall(r"\b(\d[\d,]*(?:\.\d+)?)\b", answer)
    if not nums_in_answer:
        return {"plausible": True, "warnings": [], "suggested_requery": None}

    try:
        prompt = build_plausibility_check_prompt()
        messages = prompt.format_messages(
            query=original_query,
            subquery=subquery,
            answer=truncate_text(answer, 2000),
        )
        response = await tracked_invoke(
            ctx.base_model,
            messages,
            ctx.token_tracker,
            "research",
            **ctx.llm_call_kwargs(),
        )
        result = safe_json_parse(str(response.content or ""))
        if result and "plausible" in result:
            return result
    except Exception as e:
        logger.warning("Plausibility check failed for subquery: %s", e)

    return {"plausible": True, "warnings": [], "suggested_requery": None}


async def _generate_alternative_approach(
    ctx: ResearchContext,
    subquery: str,
    implausible_warnings: List[Dict[str, Any]],
) -> Optional[str]:
    """Generate a fundamentally different query when results are implausible."""
    warnings_text = "\n".join(
        f"- {w.get('metric', '?')}: {w.get('value', '?')} "
        f"(severity: {w.get('severity', '?')}, "
        f"possible cause: {w.get('possible_cause', '?')})"
        for w in implausible_warnings
    )

    sys_content, human_content = build_alternative_approach_prompt(
        subquery, warnings_text
    )
    messages = [
        SystemMessage(content=sys_content),
        HumanMessage(content=human_content),
    ]

    try:
        response = await tracked_invoke(
            ctx.base_model,
            messages,
            ctx.token_tracker,
            "research",
            **ctx.llm_call_kwargs(),
        )
        alternative = str(response.content or "").strip()
        if alternative and len(alternative) > 10:
            return alternative
    except Exception as e:
        logger.warning("Alternative approach generation failed: %s", e)

    return None


async def _supervisor_reflect(
    ctx: ResearchContext,
    query: str,
    findings_board: Dict[str, FindingEntry],
    round_number: int,
    max_rounds: int,
) -> Dict[str, Any]:
    """Supervisor reflects on all findings to assess coverage and gaps."""
    fallback_reason = "unknown error"

    try:
        findings_summary = _summarize_findings_board(findings_board)
        reflection_prompt = build_supervisor_reflection_prompt()
        messages = reflection_prompt.format_messages(
            query=query,
            round_number=str(round_number),
            max_rounds=str(max_rounds),
            findings_summary=truncate_text(findings_summary, 4000),
            completeness_threshold=str(DEEP_RESEARCH_COMPLETENESS_THRESHOLD),
        )
        response = await tracked_invoke(
            ctx.base_model,
            messages,
            ctx.token_tracker,
            "research",
            **ctx.llm_call_kwargs(),
        )
        result = safe_json_parse(str(response.content or ""))

        if result and "decision" in result:
            return result

        fallback_reason = "unparseable LLM response"
        logger.warning(
            "Supervisor reflection returned unparseable result: %s",
            truncate_text(str(response.content), 200),
        )

    except Exception as e:
        fallback_reason = f"{type(e).__name__}: {e}"
        logger.warning("Supervisor reflection failed: %s", e)

    logger.warning(
        "Supervisor operating in fallback mode (round %d/%d): %s",
        round_number,
        max_rounds,
        fallback_reason,
    )
    ctx.emit(
        emit_agent_thinking(
            "Supervisor",
            f"Reflection fallback activated: {fallback_reason}",
        )
    )

    fallback_decision = (
        "continue_research" if round_number < max_rounds else "proceed_to_completeness"
    )
    return {
        "coverage_pct": 0,
        "gaps": [],
        "conflicts": [],
        "decision": fallback_decision,
        "follow_up_subqueries": [],
        "reasoning": f"Reflection failed ({fallback_reason}) — fail-safe applied",
        "is_fallback": True,
    }


def _extract_answers_for_conflict_detection(
    findings_board: Dict[str, FindingEntry],
    recent_findings: List[str],
) -> tuple[Dict[str, str], Dict[str, float]]:
    """Extract answers and quality scores from recent findings."""
    answers: Dict[str, str] = {}
    quality_scores: Dict[str, float] = {}
    for sq in recent_findings:
        entry = findings_board.get(sq)
        if entry:
            finding = entry.get("finding") or {}
            answer = finding.get("answer", "")
            if answer and not finding.get("error"):
                answers[sq] = answer
                quality_scores[sq] = entry.get(
                    "data_quality_score", entry.get("quality_score", 0.5)
                )
    return answers, quality_scores


def _apply_conflict_metadata(
    findings_board: Dict[str, FindingEntry],
    answers: Dict[str, str],
    conflicts: List[Dict[str, Any]],
) -> None:
    """Mark conflicting findings with has_conflict tag."""
    finding_keys = list(answers.keys())
    for conflict in conflicts[:3]:
        for idx in conflict.get("finding_indices", []):
            if 0 < idx <= len(finding_keys):
                sq = finding_keys[idx - 1]
                if sq in findings_board:
                    entry = findings_board[sq]
                    tags = list(entry.get("tags") or [])
                    if "has_conflict" not in tags:
                        tags.append("has_conflict")
                        findings_board[sq] = FindingEntry(**{**entry, "tags": tags})


async def _detect_and_resolve_conflicts(
    ctx: ResearchContext,
    findings_board: Dict[str, FindingEntry],
    recent_findings: List[str],
    query: str,
) -> tuple[Dict[str, FindingEntry], List[Dict[str, Any]]]:
    """Detect conflicts between findings and attempt to resolve them."""
    events: List[Dict[str, Any]] = []

    if len(recent_findings) < 2:
        return findings_board, events

    answers, quality_scores = _extract_answers_for_conflict_detection(
        findings_board, recent_findings
    )
    if len(answers) < 2:
        return findings_board, events

    try:
        findings_with_quality = [
            f"[{i + 1}] {sq} [Quality: {quality_scores.get(sq, 0.5):.2f}]:\n{truncate_text(ans, 500)}"
            for i, (sq, ans) in enumerate(answers.items())
        ]
        conflict_prompt_text = build_conflict_resolution_prompt(
            query, findings_with_quality
        )

        response = await tracked_invoke(
            ctx.base_model,
            [
                SystemMessage(content=CONFLICT_DETECTOR_SYSTEM_PROMPT),
                HumanMessage(content=conflict_prompt_text),
            ],
            ctx.token_tracker,
            "research",
            **ctx.llm_call_kwargs(),
        )

        result = safe_json_parse(str(response.content or ""))

        if isinstance(result, dict) and result.get("has_conflicts"):
            conflicts = result.get("conflicts", [])
            if conflicts:
                ctx.emit_or_append(
                    emit_agent_thinking(
                        "ConflictResolver",
                        f"Detected {len(conflicts)} potential conflicts between findings",
                    ),
                    events,
                )
                logger.info(
                    "Worker debate: detected %d conflicts in findings", len(conflicts)
                )
                for conflict in conflicts[:3]:
                    conflict_type = conflict.get("type", "unknown")
                    desc = conflict.get("description", "No description")
                    resolution = conflict.get("resolution", "")
                    ctx.emit_or_append(
                        emit_agent_decision(
                            "ConflictResolver",
                            f"{conflict_type.upper()} conflict detected",
                            f"{desc} -> {resolution[:100]}",
                        ),
                        events,
                    )
                _apply_conflict_metadata(findings_board, answers, conflicts)
        else:
            logger.debug("No conflicts detected between findings")

    except Exception as e:
        logger.warning("Conflict detection failed: %s", e)

    return findings_board, events


def _get_status_error_result(
    sq: str,
    idx: int,
    total: int,
    status: str,
) -> Optional[SubAgentResult]:
    """Return SubAgentResult for access_denied or no_data status, else None."""
    if status == "access_denied":
        return SubAgentResult(
            finding=_make_finding(sq, error="Access denied"),
            quality_score=0.0,
            confidence="low",
            summary="Access denied",
            events=[emit_subquery_error(idx, total, sq, "Access denied")],
        )
    if status == "no_data_products":
        return SubAgentResult(
            finding=_make_finding(sq, error="No data sources available"),
            quality_score=0.0,
            confidence="low",
            summary="No data sources",
            events=[emit_subquery_error(idx, total, sq, "No data sources")],
        )
    return None


def _get_cached_subagent_result(
    sq: str,
    enriched: Dict[str, Any],
    findings_board: Dict[str, FindingEntry],
    idx: int,
    total: int,
) -> Optional[SubAgentResult]:
    """Return SubAgentResult for cached finding if applicable, else None."""
    cached_findings_dict = findings_from_board(findings_board)
    source_tag = enriched.get("source", "new")
    cached_finding_key = enriched.get("cached_finding_key")

    cached = None
    if source_tag == "cached" and cached_finding_key and cached_findings_dict:
        cached = cached_findings_dict.get(cached_finding_key)
    if cached is None and cached_findings_dict:
        cached = find_matching_cached_finding(sq, cached_findings_dict)

    if not cached:
        return None

    finding = _make_finding(
        sq,
        answer=cached.get("answer", ""),
        tool_results=cached.get("tool_results", []),
        cached=True,
    )
    return SubAgentResult(
        finding=finding,
        quality_score=0.8,
        confidence="high",
        summary=truncate_text(cached.get("answer", ""), 200),
        events=[emit_subquery_cached(idx, total, sq)],
    )


async def _execute_worker_uncached(
    ctx: ResearchContext,
    state: DeepResearchState,
    sq: str,
    idx: int,
    total: int,
    round_num: int,
    existing_context: str,
    findings_board: Dict[str, FindingEntry],
    agent_messages: List[AgentMessage],
    events: List[Dict[str, Any]],
    thread_id: Optional[str],
) -> SubAgentResult:
    """Execute a worker when not cached."""
    worker_id = f"worker_{idx}"
    ctx.emit_or_append(
        emit_supervisor_delegating(
            round_num,
            sq,
            worker_id,
            has_cross_context=bool(
                existing_context and existing_context != "No findings collected yet."
            ),
        ),
        events,
    )

    ts = datetime.now(timezone.utc).isoformat()
    related_context = _find_related_findings(sq, findings_board)
    agent_messages.append(
        AgentMessage(
            from_agent="supervisor",
            to_agent=worker_id,
            message_type="task_assignment",
            content=sq,
            metadata={
                "round": round_num,
                "index": idx,
                "has_cross_context": bool(related_context),
            },
            timestamp=ts,
        )
    )

    await ctx.async_emit_or_append(emit_subquery_start(idx, total, sq), events)
    ctx.emit(
        emit_worker_progress(subquery=sq, idx=idx, total=total, status="executing")
    )

    return await _execute_research_subagent(
        ctx=ctx,
        subquery=sq,
        index=idx,
        _total=total,
        thread_id=thread_id,
        cross_context=related_context or existing_context,
        max_retries=DEEP_RESEARCH_SUBAGENT_MAX_RETRIES,
        quality_threshold=DEEP_RESEARCH_SUBAGENT_QUALITY_THRESHOLD,
        original_query=state.get("query", ""),
    )


def _try_continue_research(
    inp: SupervisorContinueInput,
) -> Optional[tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """Return (state, events) if continuing research with follow-ups, else None."""
    if (
        inp.decision != "continue_research"
        or inp.round_num >= inp.max_rounds
        or not inp.follow_ups
    ):
        return None
    trimmed = trim_follow_ups(
        inp.follow_ups[:3],
        inp.total_sq_executed,
        inp.state.get("max_total_subqueries", DEEP_RESEARCH_MAX_TOTAL_SUBQUERIES),
    )
    de_cycled = _detect_delegation_cycle(inp.supervisor_rounds, trimmed)
    new_subqueries = [
        sq for sq in de_cycled if sq not in inp.completed and sq not in inp.to_execute
    ]
    if not new_subqueries:
        return None
    inp.ctx.emit_or_append(
        emit_supervisor_follow_up(inp.round_num, new_subqueries),
        inp.events,
    )
    all_subqueries = inp.subqueries + new_subqueries
    base = SupervisorStateBase(
        execution_start_update=inp.execution_start_update,
        findings_board=inp.findings_board,
        agent_messages=inp.agent_messages,
        supervisor_rounds=inp.supervisor_rounds,
        completed=inp.completed,
        round_num=inp.round_num,
        transitions=inp.transitions,
        immediate_context=inp.immediate_context,
        finding_cards=inp.finding_cards,
        research_memory=inp.research_memory,
    )
    state_update = _build_supervisor_state(
        base,
        current_phase=PHASE_SUPERVISOR,
        pending_subqueries=new_subqueries,
        total_subqueries_executed=inp.total_sq_executed,
        findings_count_history=inp.findings_count_history,
        fallback_count=inp.fallback_count,
        subqueries=all_subqueries,
    )
    return state_update, inp.events


def _check_mass_failure(
    inp: MassFailureInput,
) -> Optional[tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """Return (state, events) if mass failure detected, else None."""
    if not inp.to_execute:
        return None
    successful = sum(
        1
        for sq in inp.findings_received
        if sq in inp.findings_board
        and not ((inp.findings_board[sq].get("finding") or {}).get("error"))
    )
    success_rate = successful / len(inp.to_execute)
    if success_rate >= 0.2 or inp.round_num < 1:
        return None
    failure_counts: Dict[str, int] = {}
    for sq in inp.findings_received:
        fc = (inp.findings_board.get(sq, {}).get("finding") or {}).get(
            "failure_class"
        ) or "unknown"
        failure_counts[fc] = failure_counts.get(fc, 0) + 1
    abort_reason = (
        f"Mass failure: {successful}/{len(inp.to_execute)} subqueries succeeded "
        f"(rate: {success_rate:.0%})"
    )
    inp.ctx.emit_or_append(
        emit_research_failed(
            abort_reason, failure_counts, len(inp.to_execute), success_rate
        ),
        inp.events,
    )
    base = SupervisorStateBase(
        execution_start_update=inp.execution_start_update,
        findings_board=inp.findings_board,
        agent_messages=inp.agent_messages,
        supervisor_rounds=inp.supervisor_rounds,
        completed=inp.completed,
        round_num=inp.round_num,
        transitions=inp.transitions,
        immediate_context=inp.immediate_context,
        finding_cards=inp.finding_cards,
        research_memory=inp.research_memory,
    )
    state = _build_supervisor_state(
        base,
        current_phase=PHASE_SYNTHESIZE,
        total_subqueries_executed=inp.total_sq_executed,
        findings_count_history=inp.findings_count_history,
        research_abort_reason=abort_reason,
    )
    return state, inp.events


def _build_supervisor_state(
    base: SupervisorStateBase,
    *,
    current_phase: str,
    pending_subqueries: Optional[List[str]] = None,
    total_subqueries_executed: Optional[int] = None,
    findings_count_history: Optional[List[int]] = None,
    fallback_count: Optional[int] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build common supervisor state update dict."""
    out: Dict[str, Any] = {
        **base.execution_start_update,
        "findings_board": base.findings_board,
        "agent_messages": base.agent_messages,
        "supervisor_rounds": base.supervisor_rounds,
        "completed_subqueries": base.completed,
        "pending_subqueries": pending_subqueries
        if pending_subqueries is not None
        else [],
        "current_round": base.round_num,
        "current_phase": current_phase,
        "total_node_transitions": base.transitions,
        "immediate_context": base.immediate_context,
        "finding_cards": base.finding_cards,
        "research_memory": base.research_memory,
    }
    if total_subqueries_executed is not None:
        out["total_subqueries_executed"] = total_subqueries_executed
    if findings_count_history is not None:
        out["findings_count_history"] = findings_count_history
    if fallback_count is not None:
        out["fallback_count"] = fallback_count
    out.update(extra)
    return out


def _build_supervisor_completeness_return(
    execution_start_update: Dict[str, Any],
    findings_board: Dict[str, FindingEntry],
    agent_messages: List[AgentMessage],
    supervisor_rounds: List[SupervisorRound],
    completed: List[str],
    round_num: int,
    transitions: int,
    immediate_context: Optional[ImmediateContext],
    finding_cards: list,
    research_memory: Optional[ResearchMemory],
    events: List[Dict[str, Any]],
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Build standard return when proceeding to completeness phase."""
    base = SupervisorStateBase(
        execution_start_update=execution_start_update,
        findings_board=findings_board,
        agent_messages=agent_messages,
        supervisor_rounds=supervisor_rounds,
        completed=completed,
        round_num=round_num,
        transitions=transitions,
        immediate_context=immediate_context,
        finding_cards=finding_cards,
        research_memory=research_memory,
    )
    return _build_supervisor_state(base, current_phase=PHASE_COMPLETENESS), events


async def _process_and_deposit_worker_result(
    ctx: ResearchContext,
    worker_ctx: WorkerResultContext,
) -> tuple[
    Optional[ImmediateContext],
    list[FindingCard],
    Optional[ResearchMemory],
]:
    """Deposit one worker result on the board, emit events, update context."""
    sq = worker_ctx.sq
    result = worker_ctx.result
    round_num = worker_ctx.round_num
    to_execute = worker_ctx.to_execute
    total = worker_ctx.total
    findings_board = worker_ctx.findings_board
    completed = worker_ctx.completed
    findings_received = worker_ctx.findings_received
    agent_messages = worker_ctx.agent_messages
    immediate_context = worker_ctx.immediate_context
    finding_cards = worker_ctx.finding_cards
    research_memory = worker_ctx.research_memory
    events = worker_ctx.events

    dq_score = compute_data_quality_score(result.finding, result.confidence)
    exclude = should_exclude_from_synthesis(result.finding)
    findings_board[sq] = FindingEntry(
        finding=result.finding,
        deposited_by=f"worker:{sq}",
        round_number=round_num,
        quality_score=result.quality_score,
        confidence=result.confidence,
        related_subqueries=[],
        tags=[],
        data_quality_score=dq_score,
        exclude_from_synthesis=exclude,
    )
    completed.append(sq)
    findings_received.append(sq)

    ts = datetime.now(timezone.utc).isoformat()
    agent_messages.append(
        AgentMessage(
            from_agent=f"worker:{sq}",
            to_agent="supervisor",
            message_type="finding_report",
            content=result.summary,
            metadata={
                "quality_score": result.quality_score,
                "confidence": result.confidence,
            },
            timestamp=ts,
        )
    )

    for evt in result.events:
        await ctx.async_emit_or_append(evt, events)

    sq_idx_lookup = {s: i for i, s in enumerate(to_execute)}
    sq_idx = sq_idx_lookup.get(sq, len(to_execute) - 1) + 1

    ctx.emit(
        emit_worker_progress(
            subquery=sq,
            idx=sq_idx,
            total=total,
            status="done",
        )
    )

    if ENABLE_HIERARCHICAL_CONTEXT:
        pass  # Simplified: no hierarchical context processing

    if not result.finding.get("error"):
        ctx.emit_or_append(
            emit_subquery_complete(
                sq_idx,
                total,
                sq,
                result.finding.get("answer", ""),
            ),
            events,
        )

    return immediate_context, finding_cards, research_memory


def _detect_delegation_cycle(
    supervisor_rounds: List[SupervisorRound],
    proposed_follow_ups: List[str],
) -> List[str]:
    """Filter out follow-up subqueries that repeat gaps from prior rounds."""
    if not supervisor_rounds or not proposed_follow_ups:
        return proposed_follow_ups

    prior_queries: set[str] = set()
    for rnd in supervisor_rounds:
        for sq in rnd.get("delegated_subqueries", []):
            prior_queries.add(sq.strip().lower())
        for gap in rnd.get("gaps_identified", []):
            prior_queries.add(gap.strip().lower())
        for fu in rnd.get("follow_ups_spawned", []):
            prior_queries.add(fu.strip().lower())

    novel: List[str] = []
    for fu in proposed_follow_ups:
        fu_lower = fu.strip().lower()
        is_repeat = any(
            fu_lower in prior or prior in fu_lower
            for prior in prior_queries
            if len(prior) > 10
        )
        if is_repeat:
            logger.info("Cycle detection: dropping repeated follow-up %r", fu[:80])
        else:
            novel.append(fu)

    return novel


def _supervisor_check_sentinel(
    state: DeepResearchState,
    ctx: ResearchContext,
    transitions: int,
    events: List[Dict[str, Any]],
) -> Optional[tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """Return (state_updates, events) if sentinel triggered, else None."""
    should_stop, reason, forced_phase = check_loop_sentinel(state, ctx, "supervisor")
    if not should_stop:
        return None
    budgets = {
        "total_subqueries": f"{state.get('total_subqueries_executed', 0)}/{state.get('max_total_subqueries', 0)}",
        "node_transitions": f"{transitions}/{state.get('max_node_transitions', 0)}",
    }
    ctx.emit_or_append(
        emit_sentinel_triggered(reason or "", "supervisor", budgets), events
    )
    return {
        "current_phase": forced_phase,
        "sentinel_triggered": True,
        "sentinel_reason": reason,
        "total_node_transitions": transitions,
    }, events


def _supervisor_check_empty_plan(
    pending: List[str],
    completed: List[str],
    round_num: int,
    execution_start_update: Dict[str, Any],
    transitions: int,
    ctx: ResearchContext,
    events: List[Dict[str, Any]],
) -> Optional[tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """Return (state_updates, events) if empty plan, else None."""
    if pending or completed or round_num != 1:
        return None
    ctx.emit_or_append(emit_empty_plan(), events)
    return {
        **execution_start_update,
        "current_phase": PHASE_COMPLETENESS,
        "current_round": round_num,
        "total_node_transitions": transitions,
    }, events


def _supervisor_check_all_completed(
    to_execute: List[str],
    round_num: int,
    execution_start_update: Dict[str, Any],
    findings_board: Dict[str, FindingEntry],
    agent_messages: List[AgentMessage],
    supervisor_rounds: List[SupervisorRound],
    completed: List[str],
    transitions: int,
    immediate_context: Optional[ImmediateContext],
    finding_cards: list,
    research_memory: Optional[ResearchMemory],
    ctx: ResearchContext,
    events: List[Dict[str, Any]],
) -> Optional[tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """Return (state_updates, events) if all subqueries done and round > 1, else None."""
    if to_execute or round_num <= 1:
        return None
    ctx.emit_or_append(
        emit_agent_thinking(
            "Supervisor",
            "All subqueries completed. Proceeding to completeness evaluation.",
        ),
        events,
    )
    return _build_supervisor_completeness_return(
        execution_start_update,
        findings_board,
        agent_messages,
        supervisor_rounds,
        completed,
        round_num,
        transitions,
        immediate_context,
        finding_cards,
        research_memory,
        events,
    )


async def _supervisor_run_debate(
    ctx: ResearchContext,
    findings_board: Dict[str, FindingEntry],
    findings_received: List[str],
    query: str,
    events: List[Dict[str, Any]],
) -> Dict[str, FindingEntry]:
    """Run worker debate if 2+ findings; return updated findings_board."""
    if len(findings_received) < 2:
        return findings_board
    ctx.emit_or_append(
        emit_agent_thinking(
            "Supervisor",
            "Checking for conflicts between worker findings...",
        ),
        events,
    )
    board, debate_events = await _detect_and_resolve_conflicts(
        ctx, findings_board, findings_received, query
    )
    events.extend(debate_events)
    return board


async def _supervisor_do_reflection_and_route(
    inp: ReflectionRouteInput,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Reflect, record round, try continue; return (state_updates, events)."""
    inp.ctx.emit_or_append(
        emit_agent_thinking(
            "Supervisor",
            f"Reflecting on {len(inp.findings_board)} total findings after round {inp.round_num}",
        ),
        inp.events,
    )
    reflection = await _supervisor_reflect(
        inp.ctx,
        inp.state.get("query", ""),
        inp.findings_board,
        inp.round_num,
        inp.max_rounds,
    )
    fallback_count = inp.state.get("fallback_count", 0)
    if reflection.pop("is_fallback", False):
        fallback_count += 1
    try:
        coverage_pct = int(reflection.get("coverage_pct") or 50)
    except (TypeError, ValueError):
        coverage_pct = 50
    gaps = reflection.get("gaps", [])
    decision = reflection.get("decision", "proceed_to_completeness")
    follow_ups = reflection.get("follow_up_subqueries", [])
    inp.ctx.emit_or_append(
        emit_supervisor_reflection(
            inp.round_num,
            coverage_pct,
            gaps,
            decision,
            findings_count=len(inp.findings_board),
        ),
        inp.events,
    )
    inp.supervisor_rounds.append(
        SupervisorRound(
            round_number=inp.round_num,
            delegated_subqueries=inp.to_execute,
            findings_received=inp.findings_received,
            gaps_identified=gaps,
            follow_ups_spawned=follow_ups if decision == "continue_research" else [],
            coverage_assessment=reflection.get("reasoning", ""),
        )
    )
    continue_return = _try_continue_research(
        SupervisorContinueInput(
            decision=decision,
            round_num=inp.round_num,
            max_rounds=inp.max_rounds,
            follow_ups=follow_ups,
            subqueries=inp.subqueries,
            to_execute=inp.to_execute,
            completed=inp.completed,
            supervisor_rounds=inp.supervisor_rounds,
            total_sq_executed=inp.total_sq_executed,
            state=inp.state,
            execution_start_update=inp.execution_start_update,
            findings_board=inp.findings_board,
            agent_messages=inp.agent_messages,
            transitions=inp.transitions,
            immediate_context=inp.immediate_context,
            finding_cards=inp.finding_cards,
            research_memory=inp.research_memory,
            findings_count_history=inp.findings_count_history,
            fallback_count=fallback_count,
            ctx=inp.ctx,
            events=inp.events,
        )
    )
    if continue_return is not None:
        return continue_return
    inp.ctx.emit_or_append(
        emit_agent_decision(
            "Supervisor",
            f"Research round {inp.round_num} complete",
            f"Coverage: {coverage_pct}% | Proceeding to completeness evaluation",
        ),
        inp.events,
    )
    base = SupervisorStateBase(
        execution_start_update=inp.execution_start_update,
        findings_board=inp.findings_board,
        agent_messages=inp.agent_messages,
        supervisor_rounds=inp.supervisor_rounds,
        completed=inp.completed,
        round_num=inp.round_num,
        transitions=inp.transitions,
        immediate_context=inp.immediate_context,
        finding_cards=inp.finding_cards,
        research_memory=inp.research_memory,
    )
    return (
        _build_supervisor_state(
            base,
            current_phase=PHASE_COMPLETENESS,
            total_subqueries_executed=inp.total_sq_executed,
            findings_count_history=inp.findings_count_history,
            fallback_count=fallback_count,
        ),
        inp.events,
    )


def _supervisor_check_cancellation(
    inp: CancellationCheckInput,
) -> Optional[tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """Return (state_updates, events) if thread cancelled, else None."""
    if not inp.thread_id:
        return None
    return None
    inp.ctx.emit_or_append(emit_cancelled(inp.thread_id), inp.events)
    base = SupervisorStateBase(
        execution_start_update=inp.execution_start_update,
        findings_board=inp.findings_board,
        agent_messages=inp.agent_messages,
        supervisor_rounds=inp.supervisor_rounds,
        completed=inp.completed,
        round_num=inp.round_num,
        transitions=inp.transitions,
        immediate_context=inp.immediate_context,
        finding_cards=inp.finding_cards,
        research_memory=inp.research_memory,
    )
    return _build_supervisor_state(
        base,
        current_phase=PHASE_COMPLETE,
        total_subqueries_executed=inp.total_sq_executed,
        findings_count_history=inp.findings_count_history,
    ), inp.events


async def research_supervisor_node(
    state: DeepResearchState,
    ctx: ResearchContext,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Research Supervisor: orchestrate parallel research workers.

    Delegates to workers, reflects on findings, spawns follow-ups.
    Returns (state_updates, events).
    """
    events: List[Dict[str, Any]] = []
    transitions = state.get("total_node_transitions", 0) + 1

    early = _supervisor_check_sentinel(state, ctx, transitions, events)
    if early is not None:
        return early

    execution_start_update: Dict[str, Any] = {}
    if state.get("execution_start_time", 0.0) <= 0:
        execution_start_update["execution_start_time"] = time.time()

    round_num = state.get("current_round", 0) + 1
    max_rounds = state.get("max_rounds", DEEP_RESEARCH_MAX_SUPERVISOR_ROUNDS)
    findings_board: Dict[str, FindingEntry] = dict(state.get("findings_board", {}))
    agent_messages: List[AgentMessage] = list(state.get("agent_messages", []))
    supervisor_rounds: List[SupervisorRound] = list(state.get("supervisor_rounds", []))
    immediate_context = state.get("immediate_context")
    finding_cards: list = list(state.get("finding_cards", []))
    research_memory = state.get("research_memory")
    subqueries = state.get("subqueries", [])
    enriched_subqueries = state.get("enriched_subqueries", [])
    thread_id = state.get("thread_id")
    completed = list(state.get("completed_subqueries", []))
    pending = list(dict.fromkeys(state.get("pending_subqueries", [])))

    # Check for cancellation
    if thread_id:
        store = get_cancel_store()
        if await store.is_cancelled(thread_id):
            logger.info(f"Supervisor: cancellation detected for thread {thread_id}")
            return {
                "current_phase": PHASE_COMPLETE,
                "should_stop": True,
                "sentinel_triggered": True,
                "sentinel_reason": "Cancelled by user",
                "final_answer": "Research was cancelled by the user.",
            }, [emit_cancelled(thread_id)]

    early = _supervisor_check_empty_plan(
        pending, completed, round_num, execution_start_update, transitions, ctx, events
    )
    if early is not None:
        return early

    completed_keys = set(findings_board.keys())
    to_execute = [
        sq for sq in pending if sq not in completed_keys and sq not in completed
    ]

    early = _supervisor_check_all_completed(
        to_execute,
        round_num,
        execution_start_update,
        findings_board,
        agent_messages,
        supervisor_rounds,
        completed,
        transitions,
        immediate_context,
        finding_cards,
        research_memory,
        ctx,
        events,
    )
    if early is not None:
        return early

    total = len(to_execute)
    ctx.emit_or_append(
        emit_supervisor_round_start(round_num, to_execute, max_rounds=max_rounds),
        events,
    )
    ctx.emit_or_append(
        emit_agent_thinking(
            "Supervisor",
            f"Round {round_num}: Orchestrating {total} research tasks with cross-context sharing",
        ),
        events,
    )
    ctx.emit_or_append(emit_research_start(total), events)

    existing_context = _summarize_findings_board(findings_board)

    enriched_lookup: Dict[str, Dict[str, Any]] = {}
    for eq in enriched_subqueries:
        q = eq.get("query", "")
        enriched_lookup[q] = eq

    max_concurrent = DEEP_RESEARCH_MAX_CONCURRENT_WORKERS
    semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_worker(sq: str, idx: int) -> tuple[str, SubAgentResult]:
        async with semaphore:
            enriched = enriched_lookup.get(sq, {})
            status = enriched.get("status", "ready")

            status_result = _get_status_error_result(sq, idx, total, status)
            if status_result is not None:
                return sq, status_result

            cached_result = _get_cached_subagent_result(
                sq, enriched, findings_board, idx, total
            )
            if cached_result is not None:
                return sq, cached_result

            return sq, await _execute_worker_uncached(
                ctx=ctx,
                state=state,
                sq=sq,
                idx=idx,
                total=total,
                round_num=round_num,
                existing_context=existing_context,
                findings_board=findings_board,
                agent_messages=agent_messages,
                events=events,
                thread_id=thread_id,
            )

    tasks = [
        asyncio.create_task(execute_worker(sq, i), name=f"dr-worker-{sq[:40]}")
        for i, sq in enumerate(to_execute, 1)
    ]

    findings_received: List[str] = []
    try:
        for coro in asyncio.as_completed(tasks):
            sq, result = await coro
            worker_ctx = WorkerResultContext(
                sq=sq,
                result=result,
                round_num=round_num,
                to_execute=to_execute,
                total=total,
                findings_board=findings_board,
                completed=completed,
                findings_received=findings_received,
                agent_messages=agent_messages,
                immediate_context=immediate_context,
                finding_cards=finding_cards,
                research_memory=research_memory,
                events=events,
            )
            (
                immediate_context,
                finding_cards,
                research_memory,
            ) = await _process_and_deposit_worker_result(ctx=ctx, worker_ctx=worker_ctx)
    except (asyncio.CancelledError, Exception):
        for t in tasks:
            if not t.done():
                t.cancel()
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        raise

    ctx.emit_or_append(emit_research_complete(len(findings_received)), events)

    total_sq_executed = state.get("total_subqueries_executed", 0) + len(to_execute)

    findings_count_history = list(state.get("findings_count_history", []))
    findings_count_history.append(len(findings_board))

    early = _supervisor_check_cancellation(
        CancellationCheckInput(
            thread_id=thread_id,
            execution_start_update=execution_start_update,
            findings_board=findings_board,
            agent_messages=agent_messages,
            supervisor_rounds=supervisor_rounds,
            completed=completed,
            round_num=round_num,
            transitions=transitions,
            immediate_context=immediate_context,
            finding_cards=finding_cards,
            research_memory=research_memory,
            total_sq_executed=total_sq_executed,
            findings_count_history=findings_count_history,
            ctx=ctx,
            events=events,
        )
    )
    if early is not None:
        return early

    mass_failure_return = _check_mass_failure(
        MassFailureInput(
            to_execute=to_execute,
            findings_received=findings_received,
            findings_board=findings_board,
            round_num=round_num,
            execution_start_update=execution_start_update,
            agent_messages=agent_messages,
            supervisor_rounds=supervisor_rounds,
            completed=completed,
            transitions=transitions,
            immediate_context=immediate_context,
            finding_cards=finding_cards,
            research_memory=research_memory,
            total_sq_executed=total_sq_executed,
            findings_count_history=findings_count_history,
            ctx=ctx,
            events=events,
        )
    )
    if mass_failure_return is not None:
        return mass_failure_return

    findings_board = await _supervisor_run_debate(
        ctx, findings_board, findings_received, state.get("query", ""), events
    )

    return await _supervisor_do_reflection_and_route(
        ReflectionRouteInput(
            ctx=ctx,
            state=state,
            findings_board=findings_board,
            findings_received=findings_received,
            round_num=round_num,
            max_rounds=max_rounds,
            to_execute=to_execute,
            subqueries=subqueries,
            completed=completed,
            supervisor_rounds=supervisor_rounds,
            total_sq_executed=total_sq_executed,
            execution_start_update=execution_start_update,
            agent_messages=agent_messages,
            transitions=transitions,
            immediate_context=immediate_context,
            finding_cards=finding_cards,
            research_memory=research_memory,
            findings_count_history=findings_count_history,
            events=events,
        )
    )
