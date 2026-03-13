"""Synthesis node: generate research reports from findings."""

import asyncio
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from template_agent.src.core.deep_research.context_manager import (
    estimate_tokens,
    get_max_context_tokens,
)
from template_agent.src.core.deep_research.events import (
    emit_agent_decision,
    emit_agent_message,
    emit_agent_thinking,
    emit_data_aggregation_complete,
    emit_data_aggregation_start,
    emit_fact_check_complete,
    emit_fact_check_start,
    emit_no_valid_findings,
    emit_report_generation_complete,
    emit_report_generation_start,
    emit_revision_complete,
    emit_revision_start,
    emit_sentinel_triggered,
    emit_synthesis_complete,
    emit_synthesis_start,
)
from template_agent.src.core.deep_research.prompts import (
    QueryType,
    build_data_aggregation_prompt,
    build_revision_prompt,
    build_synthesis_prompt,
)
from template_agent.src.core.deep_research.sentinel import check_loop_sentinel
from template_agent.src.core.deep_research.state import (
    PHASE_COMPLETE,
    PHASE_VISUALIZE,
    DeepResearchState,
    Finding,
    ResearchContext,
)
from template_agent.src.core.deep_research.token_tracker import tracked_invoke
from template_agent.src.core.prompt import get_current_date
from template_agent.src.core.utils import (
    safe_json_parse,
    strip_annotation_tags,
    truncate_text,
)
from template_agent.utils.pylogger import get_python_logger

from ._helpers import (
    _assess_data_quality,
    _build_fallback_synthesis,
    _format_findings_for_synthesis,
    _looks_like_tool_recommendation,
    check_node_cancelled,
    check_structural_compliance,
    detect_redundancy,
    findings_from_board,
)

logger = get_python_logger()


def _extract_numbers_from_answer(answer: str) -> List[str]:
    """Extract numeric values from answer text."""
    return re.findall(r"\b(\d[\d,]*(?:\.\d+)?)\b", answer)


def _extract_numbers_from_tool_results(tool_results: List) -> List[str]:
    """Extract numeric values from tool results (no SQL-specific logic)."""
    numbers: List[str] = []
    for tr in tool_results[:5]:
        if isinstance(tr, str):
            numbers.extend(_extract_numbers_from_answer(tr))
        elif isinstance(tr, dict):
            for v in tr.values():
                sv = str(v)
                if re.match(r"^\d[\d,]*(?:\.\d+)?$", sv):
                    numbers.append(sv)
    return numbers


def _extract_source_numbers_from_findings(findings: Dict[str, Finding]) -> List[str]:
    """Build inventory of source numbers from findings for fact-checking."""
    source_numbers_parts: List[str] = []
    for subquery, finding in findings.items():
        if finding.get("error") or finding.get("access_denied"):
            continue
        answer = finding.get("answer", "")
        nums = _extract_numbers_from_answer(answer)
        if nums:
            source_numbers_parts.append(
                f"Subquery: {subquery}\nNumbers: {', '.join(nums[:40])}"
            )
        tool_results = finding.get("tool_results", [])
        for sv in _extract_numbers_from_tool_results(tool_results):
            source_numbers_parts.append(f"Tool data: {sv}")
    return source_numbers_parts


def _build_plausibility_warnings_parts(findings: Dict[str, Finding]) -> List[str]:
    """Build list of plausibility warning strings from findings."""
    parts: List[str] = []
    for finding in findings.values():
        for w in finding.get("plausibility_warnings") or []:
            if not isinstance(w, dict):
                continue
            parts.append(
                f"- Metric '{w.get('metric', '?')}' with value "
                f"'{w.get('value', '?')}': {w.get('reasoning', 'potentially implausible')} "
                f"(severity: {w.get('severity', 'unknown')})"
            )
    return parts


async def _run_stage1_fact_check(
    ctx: ResearchContext,
    draft_answer: str,
    source_numbers_parts: List[str],
) -> str:
    """Run stage 1 fact-check via LLM."""
    messages = [
        SystemMessage(
            content="""You are a fact-checker for a research report. Your job is to silently FIX issues.
Compare every number in the draft report against the source data numbers.
For each number: if it matches a source, KEEP IT. If it contradicts, SILENTLY replace.
Do NOT add tags like [UNVERIFIED]. Return the COMPLETE report."""
        ),
        HumanMessage(
            content=(
                "SOURCE DATA NUMBERS:\n"
                + "\n".join(source_numbers_parts[:100])
                + f"\n\n---\n\nDRAFT REPORT TO FACT-CHECK:\n{draft_answer}"
            )
        ),
    ]
    response = await tracked_invoke(
        ctx.base_model,
        messages,
        ctx.token_tracker,
        "synthesis",
        **ctx.llm_call_kwargs(),
    )
    checked_draft = str(response.content or "").strip()
    if len(checked_draft) < len(draft_answer) * 0.6:
        return draft_answer
    return strip_annotation_tags(checked_draft)


async def _apply_plausibility_pass(
    ctx: ResearchContext,
    checked_draft: str,
    findings: Dict[str, Finding],
) -> str:
    """Apply plausibility caveats if findings have warnings."""
    parts = _build_plausibility_warnings_parts(findings)
    if not parts:
        return checked_draft
    try:
        plausibility_messages = [
            SystemMessage(
                content="You are a report editor. Add appropriate uncertainty language "
                "for flagged values. Do NOT remove numbers. Integrate caveats naturally."
            ),
            HumanMessage(
                content="FLAGGED CONCERNS:\n"
                + "\n".join(parts)
                + f"\n\nREPORT:\n{checked_draft}"
            ),
        ]
        response = await tracked_invoke(
            ctx.base_model,
            plausibility_messages,
            ctx.token_tracker,
            "synthesis",
            **ctx.llm_call_kwargs(),
        )
        result = str(response.content or "").strip()
        if len(result) >= len(checked_draft) * 0.6:
            return strip_annotation_tags(result)
    except Exception as e:
        logger.warning("Plausibility pass failed: %s", e)
    return checked_draft


async def _fact_check_draft(
    ctx: ResearchContext,
    draft_answer: str,
    findings: Dict[str, Finding],
) -> str:
    """Verify numbers in draft trace to source findings."""
    source_numbers_parts = _extract_source_numbers_from_findings(findings)
    if not source_numbers_parts:
        return draft_answer
    checked_draft = await _run_stage1_fact_check(
        ctx, draft_answer, source_numbers_parts
    )
    return await _apply_plausibility_pass(ctx, checked_draft, findings)


async def _run_data_aggregation(
    ctx: ResearchContext,
    query: str,
    findings: Dict[str, Finding],
    fallback_on_error: str = "Data aggregation failed – synthesize directly from findings.",
) -> tuple[str, int, int]:
    """Run data aggregation (tool results only, no SQL). Returns (data_summary, data_points, conflicts)."""
    data_summary = ""
    agg_data_points = 0
    agg_conflicts = 0
    try:
        aggregation_messages = build_data_aggregation_prompt(query, findings)
        agg_response = await tracked_invoke(
            ctx.base_model,
            aggregation_messages,
            ctx.token_tracker,
            "synthesis",
            **ctx.llm_call_kwargs(),
        )
        data_summary = str(agg_response.content or "").strip()
        agg_parsed = safe_json_parse(data_summary)
        if agg_parsed:
            agg_data_points = len(agg_parsed.get("data_points", []))
            agg_conflicts = len(agg_parsed.get("conflicts", []))
    except Exception as e:
        logger.warning("Data aggregation failed: %s", e)
        data_summary = fallback_on_error
    return data_summary, agg_data_points, agg_conflicts


def _collect_reviewer_feedback(review_results: list) -> str:
    """Collect reviewer feedback into a single text string."""
    feedback_parts = []
    for r in review_results:
        persona = r.get("persona", "Reviewer")
        action = r.get("action", "")
        score = r.get("score", 0)
        reason = r.get("reason", "")
        fb = r.get("feedback", "")
        fb_entry = f"[{persona}] (score: {score}, action: {action}): {reason}"
        if fb:
            fb_entry += f"\nFeedback: {fb}"
        feedback_parts.append(fb_entry)
    return (
        "\n\n".join(feedback_parts) if feedback_parts else "General improvement needed."
    )


def _resolve_query_type(state: DeepResearchState) -> QueryType:
    """Resolve QueryType from state."""
    query_type_str = state.get("query_type")
    if not query_type_str:
        return QueryType.COMPREHENSIVE
    try:
        return QueryType(query_type_str)
    except ValueError:
        return QueryType.COMPREHENSIVE


def _check_ends_mid_sentence(draft: str) -> List[str]:
    """Return ['ends mid-sentence'] if draft appears truncated, else []."""
    stripped = draft.rstrip()
    if not stripped or stripped[-1] in '.!?)"]|`' or stripped.endswith("```"):
        return []
    return ["ends mid-sentence"]


def _check_headings(draft: str) -> List[str]:
    """Return heading-related issues."""
    headings = re.findall(r"^#{1,3}\s+.+", draft, re.MULTILINE)
    if not headings:
        return ["no headings found"]
    return []


def _check_conclusion(draft: str) -> List[str]:
    """Return ['no conclusion section'] if long draft lacks conclusion, else []."""
    if re.search(r"(?i)##?\s+(conclusion|key\s+takeaways|takeaways|summary\b)", draft):
        return []
    if len(draft.split()) <= 300:
        return []
    return ["no conclusion section"]


def _check_min_words(draft: str, _mode_name: str) -> List[str]:
    """Return word-count issue if draft is too short, else []."""
    word_count = len(draft.split())
    if word_count >= 100:
        return []
    return [f"too short ({word_count} words, min 100)"]


def _verify_answer_completeness(
    draft: str,
    mode_name: str,
    _query_type: str,
) -> tuple[bool, list[str]]:
    """Check that the draft answer is structurally complete."""
    issues: List[str] = []
    issues.extend(_check_ends_mid_sentence(draft))
    issues.extend(_check_headings(draft))
    issues.extend(_check_conclusion(draft))
    issues.extend(_check_min_words(draft, mode_name))
    return len(issues) == 0, issues


async def _run_synthesis_llm(
    ctx: ResearchContext,
    synthesis_messages: List,
    events: List[Dict[str, Any]],
    findings: Dict[str, Finding],
) -> str:
    """Run synthesis LLM, retrying if tool recommendations detected."""
    try:
        response = await tracked_invoke(
            ctx.base_model,
            synthesis_messages,
            ctx.token_tracker,
            "synthesis",
            **ctx.llm_call_kwargs(),
        )
        draft_answer = str(response.content or "").strip()

        if _looks_like_tool_recommendation(draft_answer):
            logger.warning("Synthesis produced tool recommendations, retrying")
            events.append(
                emit_agent_message(
                    "Synthesizer",
                    "LLM",
                    "First attempt produced recommendations, retrying",
                    "request",
                )
            )
            stricter = synthesis_messages + [
                (
                    "system",
                    "IMPORTANT: You MUST synthesize the findings into a report. "
                    "DO NOT recommend tools. If no data exists, state that clearly.",
                )
            ]
            response = await tracked_invoke(
                ctx.base_model,
                stricter,
                ctx.token_tracker,
                "synthesis",
                **ctx.llm_call_kwargs(),
            )
            retry_draft = str(response.content or "").strip()
            if not _looks_like_tool_recommendation(retry_draft):
                draft_answer = retry_draft
    except Exception as e:
        error_detail = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
        logger.warning(
            "LLM synthesis failed (attempt 1): %s", error_detail, exc_info=True
        )
        events.append(
            emit_agent_message(
                "Synthesizer", "LLM", "Synthesis failed, retrying...", "request"
            )
        )
        try:
            await asyncio.sleep(2)
            response = await tracked_invoke(
                ctx.base_model,
                synthesis_messages,
                ctx.token_tracker,
                "synthesis",
                **ctx.llm_call_kwargs(),
            )
            draft_answer = str(response.content or "").strip()
        except Exception as e2:
            logger.warning("LLM synthesis retry also failed: %s", e2, exc_info=True)
            draft_answer = f"Synthesis failed: {error_detail[:100] or 'Unknown error'}"
    return draft_answer


async def _apply_structural_resynth_and_completion(
    ctx: ResearchContext,
    state: DeepResearchState,
    draft_answer: str,
    findings_text: str,
    query_type: QueryType,
    events: List[Dict[str, Any]],
) -> str:
    """Apply structural checks and auto-completion if needed."""
    mode_name = ctx.mode_config.name if ctx.mode_config else "fast"
    query_type_val = query_type.value if query_type else "comprehensive"
    struct_score, struct_violations = check_structural_compliance(
        draft_answer, query_type_val, mode_name
    )
    redundancy_score, redundancy_issues = detect_redundancy(draft_answer)

    is_complete, completeness_issues = _verify_answer_completeness(
        draft_answer, mode_name, query_type_val
    )
    if not is_complete:
        try:
            completion_messages = [
                SystemMessage(
                    content="Write ONLY the missing conclusion. Do NOT repeat existing content."
                ),
                HumanMessage(content=f"Incomplete report:\n{draft_answer[-2000:]}"),
            ]
            completion = await tracked_invoke(
                ctx.base_model,
                completion_messages,
                ctx.token_tracker,
                "synthesis",
                **ctx.llm_call_kwargs(),
            )
            conclusion_text = str(completion.content or "").strip()
            if (
                conclusion_text
                and len(conclusion_text) > 20
                and "## Conclusion" not in draft_answer
            ):
                draft_answer += "\n\n## Conclusion\n\n" + conclusion_text
        except Exception as e:
            logger.warning("Auto-completion failed: %s", e)
    return draft_answer


async def _run_first_pass_synthesis(
    state: DeepResearchState,
    ctx: ResearchContext,
    findings: Dict[str, Finding],
    findings_board: Dict[str, Any],
    active_board: Dict[str, Any] | None,
    events: List[Dict[str, Any]],
    iteration: int,
    transitions: int,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Execute full first-pass synthesis: aggregation, report gen, fact-check."""
    events.append(
        emit_agent_thinking(
            "Synthesizer",
            f"Creating comprehensive answer from {len(findings)} research findings",
        )
    )

    events.append(emit_data_aggregation_start())
    data_summary, agg_data_points, agg_conflicts = await _run_data_aggregation(
        ctx, state.get("query", ""), findings
    )
    events.append(emit_data_aggregation_complete(agg_data_points, agg_conflicts))

    events.append(emit_report_generation_start())
    findings_text = _format_findings_for_synthesis(
        findings, active_board if findings_board else None
    )

    data_quality = _assess_data_quality(findings)
    events.append(
        emit_agent_thinking(
            "Synthesizer",
            f"Stage 2: Generating report (data quality: {data_quality})",
        )
    )
    query_type = _resolve_query_type(state)
    synthesis_prompt = build_synthesis_prompt(query_type)
    synthesis_context = state.get("context", "") or "None."
    mode_synthesis_instruction = (
        ctx.mode_config.synthesis_instruction if ctx.mode_config else ""
    )
    mode_synthesis_instruction += "\n\nYour response must be complete. Include a conclusion and actionable recommendations."

    max_ctx_tokens = get_max_context_tokens(ctx.model_name)
    output_reserve = (
        getattr(ctx.mode_config, "synthesis_max_output_tokens", None) or 16384
    )
    overhead_tokens = 4000 + sum(
        estimate_tokens(p, ctx.model_name)
        for p in [
            synthesis_context,
            state.get("query", ""),
            state.get("understanding", ""),
            data_summary,
            mode_synthesis_instruction,
        ]
    )
    available_for_findings = max(
        max_ctx_tokens - output_reserve - overhead_tokens,
        10_000,
    )
    findings_tokens = estimate_tokens(findings_text, ctx.model_name)
    if findings_tokens > available_for_findings:
        target_chars = int(
            len(findings_text) * (available_for_findings / findings_tokens) * 0.9
        )
        findings_text = _format_findings_for_synthesis(
            findings, active_board if findings_board else None, max_chars=target_chars
        )

    synthesis_messages = synthesis_prompt.format_messages(
        current_date=get_current_date(),
        context=synthesis_context,
        query=state.get("query", ""),
        understanding=state.get("understanding", ""),
        data_summary=data_summary,
        findings=findings_text,
        mode_instruction=mode_synthesis_instruction,
    )

    draft_answer = await _run_synthesis_llm(ctx, synthesis_messages, events, findings)
    events.append(emit_report_generation_complete())

    if draft_answer.startswith("Synthesis failed:"):
        logger.warning(
            "Synthesis LLM failed after retry, using fallback synthesis from raw findings"
        )
        draft_answer = _build_fallback_synthesis(
            findings_board,
            state.get("query", ""),
            state=state,
            mode_name=ctx.mode_config.name if ctx.mode_config else "fast",
        )
        events.append(emit_synthesis_complete())
        events.append(
            emit_agent_decision(
                "Synthesizer",
                "Draft answer created (fallback)",
                f"Preview: {truncate_text(draft_answer, 200)}",
            )
        )
        return {
            "draft_answer": draft_answer,
            "synthesis_iteration": iteration,
            "current_phase": PHASE_VISUALIZE,
            "total_node_transitions": transitions,
        }, events

    events.append(emit_fact_check_start())
    try:
        draft_answer = await _fact_check_draft(ctx, draft_answer, findings)
    except Exception as e:
        logger.warning("Fact-checking failed: %s", e)
    events.append(emit_fact_check_complete(1))

    draft_answer = await _apply_structural_resynth_and_completion(
        ctx, state, draft_answer, findings_text, query_type, events
    )

    events.append(emit_synthesis_complete())
    events.append(
        emit_agent_decision(
            "Synthesizer",
            "Draft answer created",
            f"Preview: {truncate_text(draft_answer, 200)}",
        )
    )

    return {
        "draft_answer": draft_answer,
        "synthesis_iteration": iteration,
        "current_phase": PHASE_VISUALIZE,
        "total_node_transitions": transitions,
    }, events


async def _run_revision_path(
    state: DeepResearchState,
    ctx: ResearchContext,
    findings: Dict[str, Finding],
    findings_board: Dict[str, Any],
    active_board: Dict[str, Any] | None,
    events: List[Dict[str, Any]],
    iteration: int,
    transitions: int,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Execute revision path: feedback + data aggregation + revision LLM."""
    events.append(
        emit_agent_thinking(
            "Synthesizer",
            f"Revising answer based on reviewer feedback (iteration {iteration})",
        )
    )
    feedback_text = _collect_reviewer_feedback(state.get("review_results") or [])

    events.append(emit_data_aggregation_start())
    data_summary, agg_data_points, agg_conflicts = await _run_data_aggregation(
        ctx, state.get("query", ""), findings, fallback_on_error=""
    )
    events.append(emit_data_aggregation_complete(agg_data_points, agg_conflicts))

    findings_text = _format_findings_for_synthesis(
        findings, active_board if findings_board else None
    )

    events.append(emit_revision_start(iteration))
    mode_synthesis_instruction = (
        ctx.mode_config.synthesis_instruction if ctx.mode_config else ""
    )
    mode_synthesis_instruction += "\n\nYour response must be complete. Include a conclusion and actionable recommendations."

    revision_prompt = build_revision_prompt()
    revision_messages = revision_prompt.format_messages(
        query=state.get("query", ""),
        draft_answer=state.get("draft_answer", ""),
        feedback=feedback_text,
        findings=findings_text,
        data_summary=data_summary,
        mode_instruction=mode_synthesis_instruction,
    )

    try:
        response = await tracked_invoke(
            ctx.base_model,
            revision_messages,
            ctx.token_tracker,
            "synthesis",
            **ctx.llm_call_kwargs(),
        )
        draft_answer = str(response.content or "").strip()
    except Exception as e:
        logger.warning("Revision failed: %s", e)
        draft_answer = state.get("draft_answer", "")

    events.append(emit_revision_complete(iteration))
    draft_answer = strip_annotation_tags(draft_answer)
    events.append(emit_synthesis_complete())

    return {
        "draft_answer": draft_answer,
        "synthesis_iteration": iteration,
        "current_phase": PHASE_VISUALIZE,
        "total_node_transitions": transitions,
    }, events


async def synthesize_node(
    state: DeepResearchState,
    ctx: ResearchContext,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Synthesize node: Two-stage synthesis for comprehensive reports.

    First pass: aggregation (tool results), report generation, fact-check.
    Revision pass: uses reviewer feedback and data summary.
    """
    cancelled = check_node_cancelled(state.get("thread_id"), "synthesize", state)
    if cancelled is not None:
        return cancelled, []

    events: List[Dict[str, Any]] = []
    transitions = state.get("total_node_transitions", 0) + 1
    iteration = state.get("synthesis_iteration", 0) + 1

    if iteration > 1:
        should_stop, reason, _ = check_loop_sentinel(state, ctx, "synthesize")
        if should_stop:
            budgets = {
                "total_subqueries": f"{state.get('total_subqueries_executed', 0)}/{state.get('max_total_subqueries', 0)}",
                "node_transitions": f"{transitions}/{state.get('max_node_transitions', 0)}",
            }
            ctx.emit_or_append(
                emit_sentinel_triggered(reason or "", "synthesize", budgets), events
            )
            previous_draft = state.get("draft_answer", "")
            if previous_draft:
                return {
                    "current_phase": PHASE_VISUALIZE,
                    "sentinel_triggered": True,
                    "sentinel_reason": reason,
                    "total_node_transitions": transitions,
                }, events
            fallback = _build_fallback_synthesis(
                state.get("findings_board", {}),
                state.get("query", ""),
                state=state,
                mode_name=ctx.mode_config.name if ctx.mode_config else "fast",
            )
            return {
                "draft_answer": fallback,
                "current_phase": PHASE_VISUALIZE,
                "sentinel_triggered": True,
                "sentinel_reason": reason,
                "total_node_transitions": transitions,
            }, events

    events.append(emit_synthesis_start(iteration))

    findings_board = state.get("findings_board", {})
    findings = findings_from_board(findings_board)

    valid_count = sum(
        1
        for entry in findings_board.values()
        if not (entry.get("finding") or {}).get("error")
    )
    if not valid_count:
        ctx.emit_or_append(
            emit_no_valid_findings(len(findings_board), len(findings_board)), events
        )
        return {
            "final_answer": "Research could not retrieve valid data for your query.",
            "current_phase": PHASE_COMPLETE,
            "total_node_transitions": transitions,
        }, events

    active_board = None
    if findings_board:
        active_board = {
            sq: entry
            for sq, entry in findings_board.items()
            if not entry.get("exclude_from_synthesis", False)
        }
        findings = {
            sq: entry.get("finding") or {} for sq, entry in active_board.items()
        }
    elif not findings:
        findings = {
            sq: entry.get("finding") or {} for sq, entry in findings_board.items()
        }
        active_board = findings_board

    current_review = state.get("current_review")
    previous_draft = state.get("draft_answer", "")

    if iteration > 1 and current_review and previous_draft:
        return await _run_revision_path(
            state,
            ctx,
            findings,
            findings_board,
            active_board,
            events,
            iteration,
            transitions,
        )

    return await _run_first_pass_synthesis(
        state,
        ctx,
        findings,
        findings_board,
        active_board,
        events,
        iteration,
        transitions,
    )
