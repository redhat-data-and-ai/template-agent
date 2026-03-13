"""Event types and formatters for deep research streaming.

This module defines the event types used for UI updates during
the deep research pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from template_agent.src.core.utils import truncate_text as _truncate_text


def _simplify_error(error: str) -> str:
    """Truncate error message for display (max 500 chars)."""
    if len(error) <= 500:
        return error
    return error[:497] + "..."


class DeepResearchEventType(str, Enum):
    """Event types for deep research streaming."""

    # Initialization
    STARTED = "started"
    CONTEXT_LOADED = "context_loaded"  # Conversation context from previous messages
    TOOL_DISCOVERY = "tool_discovery"

    # Probe Phase
    PROBE_START = "probe_start"
    PROBE_COMPLETE = "probe_complete"

    # Planning Phase
    UNDERSTANDING = "understanding"
    PLAN_GENERATED = "plan_generated"
    PLAN_PENDING = "plan_pending"  # Awaiting user approval

    # Research Phase
    RESEARCH_START = "research_start"
    SUBQUERY_START = "subquery_start"
    SUBQUERY_CACHED = "subquery_cached"
    SUBQUERY_COMPLETE = "subquery_complete"
    SUBQUERY_DRILL_DOWN = "subquery_drill_down"
    SUBQUERY_ERROR = "subquery_error"
    RESEARCH_COMPLETE = "research_complete"

    # Validation Phase (between research and synthesis)
    VALIDATION_START = "validation_start"
    VALIDATION_ANALYZING = "validation_analyzing"
    VALIDATION_CONFLICT = "validation_conflict"
    VALIDATION_COMPLETE = "validation_complete"

    # Synthesis Phase
    SYNTHESIS_START = "synthesis_start"
    DATA_AGGREGATION_START = "data_aggregation_start"
    DATA_AGGREGATION_COMPLETE = "data_aggregation_complete"
    REPORT_GENERATION_START = "report_generation_start"
    REPORT_GENERATION_COMPLETE = "report_generation_complete"
    REVISION_START = "revision_start"
    REVISION_COMPLETE = "revision_complete"
    FACT_CHECK_START = "fact_check_start"
    FACT_CHECK_COMPLETE = "fact_check_complete"
    SYNTHESIS_COMPLETE = "synthesis_complete"

    # Visualization Phase
    VISUALIZATION_START = "visualization_start"
    VISUALIZATION_ANALYSIS = "visualization_analysis"
    VISUALIZATION_CREATED = "visualization_created"
    VISUALIZATION_COMPLETE = "visualization_complete"
    VISUALIZATION_SKIPPED = "visualization_skipped"

    # Review Phase
    REVIEW_START = "review_start"
    REVIEW_COMPLETE = "review_complete"

    # Agent Conversation Events (detailed logs)
    AGENT_THINKING = "agent_thinking"  # What an agent is considering
    AGENT_DECISION = "agent_decision"  # Decision made by an agent
    AGENT_MESSAGE = "agent_message"  # Agent-to-agent communication
    REVIEWER_FEEDBACK = "reviewer_feedback"  # Detailed reviewer comments
    REVIEWER_SCORE = "reviewer_score"  # Score from a reviewer
    CONSENSUS_VOTE = "consensus_vote"  # Individual vote in consensus
    CONSENSUS_RESULT = "consensus_result"  # Final consensus outcome

    # Supervisor Phase (multi-agent orchestration)
    SUPERVISOR_ROUND_START = "supervisor_round_start"
    SUPERVISOR_DELEGATING = "supervisor_delegating"
    SUPERVISOR_REFLECTION = "supervisor_reflection"
    SUPERVISOR_FOLLOW_UP = "supervisor_follow_up"

    # Worker Self-Evaluation
    WORKER_SELF_EVALUATION = "worker_self_evaluation"
    WORKER_REFORMULATION = "worker_reformulation"

    # Completeness Evaluation
    COMPLETENESS_ASSESSMENT = "completeness_assessment"

    # Inter-Agent Communication
    INTER_AGENT_MESSAGE = "inter_agent_message"

    # Subquery Validation (planning phase)
    SUBQUERY_VALIDATION = "subquery_validation"

    # Live Progress Events (real-time heartbeat during long-running phases)
    WORKER_PROGRESS = "worker_progress"

    # Triage Phase (follow-up optimization)
    TRIAGE_DECISION = "triage_decision"

    # Cross-chat findings reuse
    CROSS_CHAT_FINDINGS_LOADED = "cross_chat_findings_loaded"

    # Token Usage
    TOKEN_USAGE_UPDATE = "token_usage_update"

    # Context Usage (hierarchical context window monitoring)
    CONTEXT_USAGE_UPDATE = "context_usage_update"

    # Reliability Metrics
    RELIABILITY_UPDATE = "reliability_update"

    # Completion
    FINAL_ANSWER = "final_answer"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class DeepResearchEvent:
    """A structured event for deep research streaming."""

    event_type: DeepResearchEventType
    message: str
    display_text: str
    ui_visible: bool = True
    details: dict[str, Any] | None = None
    stage: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for streaming."""
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        return {
            "type": "deep_research_status",
            "content": {
                "stage": self.stage or self.event_type.value,
                "event_type": self.event_type.value,
                "message": self.message,
                "display_text": self.display_text,
                "log_entry": f"[{ts}] {self.event_type.value} | {self.message}",
                "ui_visible": self.ui_visible,
                "details": self.details or {},
            },
        }


def emit_event(
    event_type: DeepResearchEventType,
    message: str,
    display_text: str | None = None,
    details: dict[str, Any] | None = None,
    ui_visible: bool = True,
    stage: str | None = None,
) -> dict[str, Any]:
    """Create a deep research event dict.

    Args:
        event_type: The type of event.
        message: Brief message describing the event.
        display_text: Text to show in UI (defaults to message).
        details: Additional structured data.
        ui_visible: Whether to show in UI.
        stage: Override stage name.

    Returns:
        Event dict ready for streaming.
    """
    event = DeepResearchEvent(
        event_type=event_type,
        message=message,
        display_text=display_text or message,
        ui_visible=ui_visible,
        details=details,
        stage=stage,
    )
    return event.to_dict()


# Convenience functions for common events


def emit_started() -> dict[str, Any]:
    """Emit pipeline started event."""
    return emit_event(
        DeepResearchEventType.STARTED,
        "Deep research pipeline started",
        "Starting deep research analysis...",
    )


def emit_context_loaded(
    message_count: int,
    context_preview: str,
    has_context: bool = True,
) -> dict[str, Any]:
    """Emit conversation context loaded event.

    Shows what previous conversation context is being used for follow-up questions.

    Args:
        message_count: Number of previous messages loaded.
        context_preview: Preview of the conversation context.
        has_context: Whether any context was found.

    Returns:
        Event dict ready for streaming.
    """
    if not has_context or message_count == 0:
        return emit_event(
            DeepResearchEventType.CONTEXT_LOADED,
            "No previous conversation context found",
            "📝 Context: Starting fresh conversation (no previous context)",
            details={
                "has_context": False,
                "message_count": 0,
                "context_preview": "",
            },
            stage="context",
        )

    preview_display = _truncate_text(context_preview, 500)

    return emit_event(
        DeepResearchEventType.CONTEXT_LOADED,
        f"Loaded {message_count} previous messages as context",
        f"📝 Context: Using {message_count} previous messages\n{preview_display}",
        details={
            "has_context": True,
            "message_count": message_count,
            "context_preview": _truncate_text(context_preview, 1000, suffix=""),
        },
        stage="context",
    )


def emit_cross_chat_findings_loaded(
    count: int,
    source_thread_count: int,
) -> dict[str, Any]:
    """Emit event when cross-chat findings are loaded from previous conversations."""
    if count == 0:
        return emit_event(
            DeepResearchEventType.CROSS_CHAT_FINDINGS_LOADED,
            "No relevant findings from previous conversations",
            "No prior research found for this query",
            details={"count": 0, "source_threads": 0},
            stage="context",
        )

    return emit_event(
        DeepResearchEventType.CROSS_CHAT_FINDINGS_LOADED,
        f"Found {count} relevant findings from {source_thread_count} previous conversation(s)",
        f"Found {count} relevant findings from {source_thread_count} previous conversation(s)",
        details={"count": count, "source_threads": source_thread_count},
        stage="context",
    )


def emit_tool_discovery(tool_count: int, tool_names: list[str]) -> dict[str, Any]:
    """Emit tool discovery event.

    Note: This event is hidden from UI as it shows internal MCP tools,
    not user-relevant data products. The data_product_discovery events
    are what users care about.
    """
    return emit_event(
        DeepResearchEventType.TOOL_DISCOVERY,
        f"Discovered {tool_count} available tools",
        f"Tool Discovery: Found {tool_count} MCP tools",
        details={"tool_count": tool_count, "tool_names": tool_names[:10]},
        ui_visible=False,  # Hide from UI - not user relevant
    )


def emit_probe_start() -> dict[str, Any]:
    """Emit probe start event.

    Note: Hidden from UI - internal implementation detail.
    """
    return emit_event(
        DeepResearchEventType.PROBE_START,
        "Probing available tools for query capabilities",
        "Probing: Discovering which tools can help...",
        ui_visible=False,  # Hide from UI - internal detail
    )


def emit_probe_complete(summary: str) -> dict[str, Any]:
    """Emit probe complete event.

    Note: Hidden from UI - internal implementation detail.
    """
    preview = _truncate_text(summary, 200)
    return emit_event(
        DeepResearchEventType.PROBE_COMPLETE,
        "Tool probe completed",
        f"Probe Complete: {preview}",
        details={"probe_summary": _truncate_text(summary, 500, suffix="")},
        ui_visible=False,  # Hide from UI - internal detail
    )


def emit_understanding(understanding: str) -> dict[str, Any]:
    """Emit query understanding event."""
    return emit_event(
        DeepResearchEventType.UNDERSTANDING,
        "Query analyzed",
        f"Understanding:\n{understanding}",
        details={"understanding": understanding},
    )


def emit_plan_generated(subqueries: list[str]) -> dict[str, Any]:
    """Emit plan generated event."""
    subq_preview = "\n".join(f"  {i}. {sq}" for i, sq in enumerate(subqueries, 1))
    return emit_event(
        DeepResearchEventType.PLAN_GENERATED,
        f"Generated {len(subqueries)} research subqueries",
        f"Plan Generated:\n{subq_preview}",
        details={"subqueries": subqueries, "count": len(subqueries)},
    )


def emit_plan_pending(
    subqueries: list[str],
    understanding: str | None = None,
) -> dict[str, Any]:
    """Emit plan pending approval event (plan awaiting user approval)."""
    subq_preview = "\n".join(f"  {i}. {sq}" for i, sq in enumerate(subqueries, 1))
    display_parts = [f"Research Plan:\n{subq_preview}"]
    if understanding:
        display_parts.append(f"\nUnderstanding:\n{_truncate_text(understanding, 500)}")
    return emit_event(
        DeepResearchEventType.PLAN_PENDING,
        f"Plan ready: {len(subqueries)} subqueries awaiting approval",
        "\n".join(display_parts),
        details={
            "subqueries": subqueries,
            "count": len(subqueries),
            "requires_approval": True,
            "query_understanding": understanding,
        },
    )


def emit_plan_pending_enriched(
    enriched_subqueries: list[dict[str, Any]],
    discovered_products: list[dict[str, Any]] | None = None,
    understanding: str | None = None,
) -> dict[str, Any]:
    """Emit plan pending approval event with enriched subqueries.

    Generic version without data product / mart tags. Each enriched subquery
    has at least: query, status, source (cached|new).
    """
    lines = []
    for i, eq in enumerate(enriched_subqueries, 1):
        query = eq.get("query", "")
        status = eq.get("status", "ready")
        source = eq.get("source", "new")
        lines.append(f"  {i}. [{source}] {query} (status: {status})")
    plan_preview = "\n".join(lines)
    cached_count = sum(1 for eq in enriched_subqueries if eq.get("source") == "cached")
    new_count = len(enriched_subqueries) - cached_count
    return emit_event(
        DeepResearchEventType.PLAN_PENDING,
        f"Plan ready: {len(enriched_subqueries)} subqueries awaiting approval",
        f"Research Plan:\n{plan_preview}",
        details={
            "enriched_subqueries": enriched_subqueries,
            "count": len(enriched_subqueries),
            "requires_approval": True,
            "query_understanding": understanding,
            "cached_count": cached_count,
            "new_count": new_count,
        },
    )


def emit_research_start(total: int) -> dict[str, Any]:
    """Emit research start event."""
    return emit_event(
        DeepResearchEventType.RESEARCH_START,
        f"Starting research with {total} subqueries",
        f"Research: Executing {total} subqueries...",
        details={"total_subqueries": total},
    )


def emit_subquery_start(index: int, total: int, subquery: str) -> dict[str, Any]:
    """Emit subquery start event."""
    return emit_event(
        DeepResearchEventType.SUBQUERY_START,
        f"Worker {index}/{total} starting",
        f"Worker {index}/{total}: {_truncate_text(subquery, 200)}",
        details={"index": index, "total": total, "subquery": subquery},
    )


def emit_subquery_cached(index: int, total: int, subquery: str) -> dict[str, Any]:
    """Emit subquery cached (skipped) event."""
    return emit_event(
        DeepResearchEventType.SUBQUERY_CACHED,
        f"Worker {index}/{total} using cached answer",
        f"Worker {index}/{total}: Reusing cached answer",
        details={"index": index, "total": total, "subquery": subquery, "cached": True},
    )


def emit_subquery_complete(
    index: int, total: int, subquery: str, answer: str
) -> dict[str, Any]:
    """Emit subquery complete event."""
    display_preview = _truncate_text(answer, 300)
    return emit_event(
        DeepResearchEventType.SUBQUERY_COMPLETE,
        f"Worker {index}/{total} completed",
        f"Worker {index}/{total}: {display_preview}",
        details={
            "index": index,
            "total": total,
            "subquery": subquery,
            "answer_preview": _truncate_text(answer, 500, suffix=""),
            "answer_full": answer,
        },
    )


def emit_subquery_error(
    index: int, total: int, subquery: str, error: str
) -> dict[str, Any]:
    """Emit subquery error event."""
    safe_error = _simplify_error(error)
    return emit_event(
        DeepResearchEventType.SUBQUERY_ERROR,
        f"Worker {index}/{total} failed: {safe_error}",
        f"Worker {index}/{total}: Error - {safe_error}",
        details={
            "index": index,
            "total": total,
            "subquery": subquery,
            "error": safe_error,
        },
    )


def emit_research_complete(findings_count: int) -> dict[str, Any]:
    """Emit research complete event."""
    return emit_event(
        DeepResearchEventType.RESEARCH_COMPLETE,
        f"Research completed with {findings_count} findings",
        f"Research Complete: Collected {findings_count} findings",
        details={"findings_count": findings_count},
    )


def emit_subquery_drill_down(
    index: int, total: int, subquery: str, reason: str
) -> dict[str, Any]:
    """Emit subquery drill-down (follow-up) event."""
    return emit_event(
        DeepResearchEventType.SUBQUERY_DRILL_DOWN,
        f"Worker {index}/{total} running follow-up query",
        f"Worker {index}/{total}: Drilling deeper – {reason}",
        details={
            "index": index,
            "total": total,
            "subquery": subquery,
            "reason": reason,
        },
    )


# ============================================================================
# VALIDATION EVENTS
# ============================================================================


def emit_validation_start(findings_count: int) -> dict[str, Any]:
    """Emit validation phase start event."""
    return emit_event(
        DeepResearchEventType.VALIDATION_START,
        f"Validating data consistency across {findings_count} findings",
        f"Validation: Cross-checking {findings_count} findings for numeric consistency...",
        details={"findings_count": findings_count},
    )


def emit_validation_analyzing(number_count: int) -> dict[str, Any]:
    """Emit event when LLM analysis of numeric consistency begins."""
    return emit_event(
        DeepResearchEventType.VALIDATION_ANALYZING,
        f"Analyzing {number_count} numeric values for cross-finding consistency",
        f"Validation: Sending {number_count} values to LLM for conflict detection...",
        details={"number_count": number_count},
    )


def emit_validation_conflict(
    metric: str, values: list[dict[str, str]], severity: str
) -> dict[str, Any]:
    """Emit data conflict detected event."""
    vals_str = ", ".join(
        f"{v.get('value', '?')} (from {v.get('source', '?')})" for v in values[:3]
    )
    return emit_event(
        DeepResearchEventType.VALIDATION_CONFLICT,
        f"Data conflict ({severity}): {metric}",
        f"Conflict [{severity.upper()}]: {metric} – values: {vals_str}",
        details={"metric": metric, "values": values, "severity": severity},
    )


def emit_validation_complete(
    conflicts_found: int, verified_count: int
) -> dict[str, Any]:
    """Emit validation phase complete event."""
    if conflicts_found > 0:
        return emit_event(
            DeepResearchEventType.VALIDATION_COMPLETE,
            f"Validation complete: {conflicts_found} conflicts found, {verified_count} metrics verified",
            f"Validation Complete: {conflicts_found} conflicts detected, {verified_count} metrics verified",
            details={
                "conflicts_found": conflicts_found,
                "verified_count": verified_count,
            },
        )
    return emit_event(
        DeepResearchEventType.VALIDATION_COMPLETE,
        f"Validation complete: {verified_count} metrics verified, no conflicts",
        f"Validation Complete: All {verified_count} metrics consistent",
        details={"conflicts_found": 0, "verified_count": verified_count},
    )


# ============================================================================
# SYNTHESIS EVENTS (granular stages)
# ============================================================================


def emit_synthesis_start(iteration: int) -> dict[str, Any]:
    """Emit synthesis start event."""
    return emit_event(
        DeepResearchEventType.SYNTHESIS_START,
        f"Synthesizing answer (iteration {iteration})",
        "Synthesizing: Creating comprehensive answer...",
        details={"iteration": iteration},
    )


def emit_data_aggregation_start() -> dict[str, Any]:
    """Emit data aggregation (Stage 1 of synthesis) start."""
    return emit_event(
        DeepResearchEventType.DATA_AGGREGATION_START,
        "Extracting and validating all numeric data from findings",
        "Stage 1: Aggregating data – extracting numbers, detecting conflicts...",
    )


def emit_data_aggregation_complete(data_points: int, conflicts: int) -> dict[str, Any]:
    """Emit data aggregation complete."""
    return emit_event(
        DeepResearchEventType.DATA_AGGREGATION_COMPLETE,
        f"Data aggregation complete: {data_points} data points extracted, {conflicts} conflicts",
        f"Stage 1 Complete: {data_points} data points extracted"
        + (f", {conflicts} conflicts flagged" if conflicts > 0 else ""),
        details={"data_points": data_points, "conflicts": conflicts},
    )


def emit_report_generation_start() -> dict[str, Any]:
    """Emit report generation (Stage 2 of synthesis) start."""
    return emit_event(
        DeepResearchEventType.REPORT_GENERATION_START,
        "Generating comprehensive report from validated data",
        "Stage 2: Writing report – combining data summary with source findings...",
    )


def emit_report_generation_complete() -> dict[str, Any]:
    """Emit report generation complete."""
    return emit_event(
        DeepResearchEventType.REPORT_GENERATION_COMPLETE,
        "Report draft generated",
        "Stage 2 Complete: Report draft ready for fact-checking",
    )


def emit_revision_start(iteration: int) -> dict[str, Any]:
    """Emit revision pass start (iteration 2+)."""
    return emit_event(
        DeepResearchEventType.REVISION_START,
        f"Revising report based on reviewer feedback (iteration {iteration})",
        f"Revision: Applying reviewer feedback to refine report (iteration {iteration})...",
        details={"iteration": iteration},
    )


def emit_revision_complete(iteration: int) -> dict[str, Any]:
    """Emit revision pass complete."""
    return emit_event(
        DeepResearchEventType.REVISION_COMPLETE,
        f"Revision complete (iteration {iteration})",
        f"Revision Complete: Report updated based on feedback (iteration {iteration})",
        details={"iteration": iteration},
    )


def emit_fact_check_start() -> dict[str, Any]:
    """Emit fact-checking pass start."""
    return emit_event(
        DeepResearchEventType.FACT_CHECK_START,
        "Fact-checking: Verifying every number traces to source data",
        "Stage 3: Fact-checking – verifying all numbers against source findings...",
    )


def emit_fact_check_complete(corrections: int) -> dict[str, Any]:
    """Emit fact-checking pass complete."""
    if corrections > 0:
        return emit_event(
            DeepResearchEventType.FACT_CHECK_COMPLETE,
            f"Fact-check complete: {corrections} corrections applied",
            f"Stage 3 Complete: {corrections} numbers corrected or flagged",
            details={"corrections": corrections},
        )
    return emit_event(
        DeepResearchEventType.FACT_CHECK_COMPLETE,
        "Fact-check complete: All numbers verified",
        "Stage 3 Complete: All numbers verified against source data",
        details={"corrections": 0},
    )


def emit_synthesis_complete() -> dict[str, Any]:
    """Emit synthesis complete event."""
    return emit_event(
        DeepResearchEventType.SYNTHESIS_COMPLETE,
        "Draft answer synthesized",
        "Synthesis Complete: Draft ready for review",
    )


# ============================================================================
# VISUALIZATION EVENTS
# ============================================================================


def emit_visualization_start() -> dict[str, Any]:
    """Emit visualization phase start event."""
    return emit_event(
        DeepResearchEventType.VISUALIZATION_START,
        "Starting visualization analysis",
        "📊 Visualization: Analyzing data for visualizable content...",
        stage="visualization",
    )


def emit_visualization_analysis(
    has_numeric_data: bool,
    recommended_charts: list[str],
) -> dict[str, Any]:
    """Emit visualization analysis result."""
    if has_numeric_data:
        charts_str = ", ".join(recommended_charts[:3]) if recommended_charts else "auto"
        return emit_event(
            DeepResearchEventType.VISUALIZATION_ANALYSIS,
            f"Found visualizable data, recommending: {charts_str}",
            f"📊 Visualization: Data can be visualized as {charts_str}",
            details={
                "has_numeric_data": has_numeric_data,
                "recommended_charts": recommended_charts,
            },
            stage="visualization",
        )
    else:
        return emit_event(
            DeepResearchEventType.VISUALIZATION_ANALYSIS,
            "No visualizable numeric data found",
            "📊 Visualization: No suitable numeric data for charts",
            details={"has_numeric_data": False},
            stage="visualization",
        )


def emit_visualization_created(
    chart_type: str,
    title: str,
    has_image: bool = False,
    image_url: str | None = None,
    plotly_json: str | None = None,
    labels: list[str] | None = None,
    values: list[Any] | None = None,
) -> dict[str, Any]:
    """Emit visualization created event with chart data."""
    details: dict[str, Any] = {
        "chart_type": chart_type,
        "title": title,
        "has_image": has_image,
    }
    if image_url:
        details["image_url"] = image_url
    if plotly_json:
        details["plotly_json"] = plotly_json
    if labels:
        details["labels"] = labels
    if values:
        details["values"] = values

    return emit_event(
        DeepResearchEventType.VISUALIZATION_CREATED,
        f"Created {chart_type} chart: {title}",
        f"📈 Created: {chart_type} - {title}",
        details=details,
        stage="visualization",
    )


def emit_visualization_complete(chart_count: int) -> dict[str, Any]:
    """Emit visualization phase complete event."""
    if chart_count > 0:
        return emit_event(
            DeepResearchEventType.VISUALIZATION_COMPLETE,
            f"Visualization complete: {chart_count} chart(s) created",
            f"📊 Visualization Complete: {chart_count} chart(s) added to answer",
            details={"chart_count": chart_count},
            stage="visualization",
        )
    else:
        return emit_event(
            DeepResearchEventType.VISUALIZATION_COMPLETE,
            "Visualization complete: No charts created",
            "📊 Visualization Complete: No charts needed",
            details={"chart_count": 0},
            stage="visualization",
        )


def emit_visualization_skipped(reason: str) -> dict[str, Any]:
    """Emit visualization skipped event."""
    return emit_event(
        DeepResearchEventType.VISUALIZATION_SKIPPED,
        f"Visualization skipped: {reason}",
        f"📊 Visualization Skipped: {reason}",
        details={"reason": reason},
        stage="visualization",
    )


def emit_review_start(persona: str) -> dict[str, Any]:
    """Emit review start event."""
    return emit_event(
        DeepResearchEventType.REVIEW_START,
        f"Reviewing as {persona}",
        f"Review: {persona} evaluating answer...",
        details={"persona": persona},
    )


def emit_review_complete(action: str, score: int, reason: str) -> dict[str, Any]:
    """Emit review complete event."""
    return emit_event(
        DeepResearchEventType.REVIEW_COMPLETE,
        f"Review complete: {action} (score: {score})",
        f"Review: {action.upper()} - {_truncate_text(reason, 250)}",
        details={"action": action, "score": score, "reason": reason},
    )


def emit_final_answer(
    answer: str,
    visualizations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Emit final answer event with visualizations."""
    details: dict[str, Any] = {"final_answer": answer}
    if visualizations:
        details["visualizations"] = visualizations
        details["visualization_count"] = len(visualizations)
    return emit_event(
        DeepResearchEventType.FINAL_ANSWER,
        "Deep research answer ready",
        f"Answer:\n{answer}",
        details=details,
    )


def emit_completed(
    effective_elapsed_seconds: float = 0.0,
    pre_plan_elapsed_seconds: float = 0.0,
) -> dict[str, Any]:
    """Emit pipeline completed event with fair execution timing."""
    if effective_elapsed_seconds > 0:
        minutes = int(effective_elapsed_seconds // 60)
        seconds = int(effective_elapsed_seconds % 60)
        time_str = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
        display = f"Deep research complete — {time_str} computation time"
    else:
        display = "Deep research analysis complete"

    details: dict[str, Any] = {}
    if effective_elapsed_seconds > 0:
        details["effective_elapsed_seconds"] = round(effective_elapsed_seconds, 2)
    if pre_plan_elapsed_seconds > 0:
        details["pre_plan_elapsed_seconds"] = round(pre_plan_elapsed_seconds, 2)

    return emit_event(
        DeepResearchEventType.COMPLETED,
        "Deep research completed",
        display,
        details=details if details else None,
    )


def emit_token_usage_update(
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    llm_calls: int,
    estimated_cost_usd: float,
    per_phase: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit token usage update event.

    Args:
        input_tokens: Total input tokens used.
        output_tokens: Total output tokens generated.
        total_tokens: Combined input + output tokens.
        llm_calls: Number of LLM calls made.
        estimated_cost_usd: Estimated cost in USD.
        per_phase: Optional breakdown by pipeline phase.

    Returns:
        Event dict ready for streaming.
    """
    cost_formatted = f"${estimated_cost_usd:.4f}"
    return emit_event(
        DeepResearchEventType.TOKEN_USAGE_UPDATE,
        f"Token usage: {total_tokens:,} tokens ({llm_calls} calls, ~{cost_formatted})",
        f"Token Usage: {total_tokens:,} tokens (~{cost_formatted})",
        details={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "llm_calls": llm_calls,
            "estimated_cost_usd": round(estimated_cost_usd, 6),
            "per_phase": per_phase or {},
        },
        stage="token_usage",
        ui_visible=True,
    )


def emit_context_usage_update(
    current_tokens: int,
    max_tokens: int,
    usage_percent: float,
    status: str,
    stage: str | None = None,
) -> dict[str, Any]:
    """Emit context usage update event for hierarchical context monitoring.

    Args:
        current_tokens: Current estimated token usage across all context levels.
        max_tokens: Maximum token limit (e.g., 200000 for Claude).
        usage_percent: Percentage of context window used (0-100).
        status: One of "normal", "warning", "critical".
        stage: Optional pipeline stage name.

    Returns:
        Event dict ready for streaming.
    """
    status_labels = {
        "normal": "Normal",
        "warning": "High usage",
        "critical": "Near limit",
    }
    status_label = status_labels.get(status, status)

    return emit_event(
        DeepResearchEventType.CONTEXT_USAGE_UPDATE,
        f"Context usage: {current_tokens:,}/{max_tokens:,} tokens ({usage_percent:.1f}%)",
        f"Context: {usage_percent:.1f}% used ({status_label})",
        details={
            "current_tokens": current_tokens,
            "max_tokens": max_tokens,
            "usage_percent": round(usage_percent, 2),
            "status": status,
        },
        stage=stage or "context_usage",
        ui_visible=True,
    )


def emit_error(error: str) -> dict[str, Any]:
    """Emit error event."""
    safe_error = _simplify_error(error)
    return emit_event(
        DeepResearchEventType.ERROR,
        f"Error: {safe_error}",
        f"Error: {safe_error}",
        details={"error": safe_error},
        stage="error",
    )


def emit_triage_decision(
    decision: str,
    reasoning: str,
    cached_findings_count: int = 0,
    context_message_count: int = 0,
) -> dict[str, Any]:
    """Emit triage decision event for follow-up query optimization.

    Shows which path the follow-up query will take based on prior research.

    Args:
        decision: One of "context_sufficient", "partial_research", "full_research".
        reasoning: Brief explanation of why this path was chosen.
        cached_findings_count: Number of cached findings available.
        context_message_count: Number of conversation messages available.

    Returns:
        Event dict ready for streaming.
    """
    decision_labels = {
        "context_sufficient": "⚡ Answering from existing research",
        "partial_research": "🔍 Partial research needed (extending prior data)",
        "full_research": "🔬 Full research required (new topic)",
    }
    display = decision_labels.get(decision, f"Triage: {decision}")

    return emit_event(
        DeepResearchEventType.TRIAGE_DECISION,
        f"Triage: {decision} — {reasoning}",
        f"{display}\n{reasoning}",
        details={
            "decision": decision,
            "reasoning": reasoning,
            "cached_findings_count": cached_findings_count,
            "context_message_count": context_message_count,
        },
        stage="triage_decision",
    )


def emit_heartbeat() -> dict[str, Any]:
    """Emit a heartbeat event to keep the SSE connection alive.

    This is sent periodically during long-running operations to prevent
    proxy timeouts (e.g., nginx default 60s).
    """
    return emit_event(
        DeepResearchEventType.STARTED,  # Reuse started type as it's benign
        "Processing...",
        ".",  # Minimal display
        details={"heartbeat": True},
        stage="heartbeat",
        ui_visible=False,  # Don't show in UI timeline
    )


# ============================================================================
# DETAILED AGENT CONVERSATION EVENTS
# These events provide full natural language logs of agent interactions
# ============================================================================


def emit_agent_thinking(agent_name: str, thought: str) -> dict[str, Any]:
    """Emit what an agent is thinking/considering."""
    return emit_event(
        DeepResearchEventType.AGENT_THINKING,
        f"[{agent_name}] Thinking: {thought}",
        f"💭 {agent_name}: {thought}",
        details={"agent": agent_name, "thought": thought},
        stage="agent_conversation",
    )


def emit_agent_decision(
    agent_name: str, decision: str, reasoning: str = ""
) -> dict[str, Any]:
    """Emit a decision made by an agent."""
    full_msg = f"[{agent_name}] Decision: {decision}"
    display = f"✅ {agent_name} decided: {decision}"
    if reasoning:
        full_msg += f" | Reason: {reasoning}"
        display += f" — {reasoning}"
    return emit_event(
        DeepResearchEventType.AGENT_DECISION,
        full_msg,
        display,
        details={"agent": agent_name, "decision": decision, "reasoning": reasoning},
        stage="agent_conversation",
    )


def emit_agent_message(
    from_agent: str, to_agent: str, message: str, message_type: str = "request"
) -> dict[str, Any]:
    """Emit agent-to-agent communication."""
    arrow = "→" if message_type == "request" else "←"
    return emit_event(
        DeepResearchEventType.AGENT_MESSAGE,
        f"[{from_agent} {arrow} {to_agent}] {message}",
        f"💬 {from_agent} {arrow} {to_agent}: {message}",
        details={
            "from_agent": from_agent,
            "to_agent": to_agent,
            "message": message,
            "message_type": message_type,
        },
        stage="agent_conversation",
    )


def emit_reviewer_feedback(
    reviewer_name: str,
    feedback: str,
    strengths: list[str] | None = None,
    weaknesses: list[str] | None = None,
) -> dict[str, Any]:
    """Emit detailed feedback from a reviewer."""
    details_dict: dict[str, Any] = {
        "reviewer": reviewer_name,
        "feedback": feedback,
    }
    if strengths:
        details_dict["strengths"] = strengths
    if weaknesses:
        details_dict["weaknesses"] = weaknesses

    display = f"📝 {reviewer_name} feedback: {feedback}"
    if strengths:
        display += f"\n   Strengths: {', '.join(strengths)}"
    if weaknesses:
        display += f"\n   Weaknesses: {', '.join(weaknesses)}"

    return emit_event(
        DeepResearchEventType.REVIEWER_FEEDBACK,
        f"[{reviewer_name}] {feedback}",
        display,
        details=details_dict,
        stage="review",
    )


def emit_reviewer_score(
    reviewer_name: str,
    score: int,
    max_score: int = 100,
    criteria: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Emit score from a reviewer."""
    details_dict: dict[str, Any] = {
        "reviewer": reviewer_name,
        "score": score,
        "max_score": max_score,
    }
    if criteria:
        details_dict["criteria"] = criteria

    criteria_str = ""
    if criteria:
        criteria_str = " | " + ", ".join(f"{k}: {v}" for k, v in criteria.items())

    return emit_event(
        DeepResearchEventType.REVIEWER_SCORE,
        f"[{reviewer_name}] Score: {score}/{max_score}{criteria_str}",
        f"⭐ {reviewer_name} scored: {score}/{max_score}{criteria_str}",
        details=details_dict,
        stage="review",
    )


def _vote_emoji(vote: str) -> str:
    """Return emoji for consensus vote."""
    if vote == "approve":
        return "👍"
    if vote == "reject":
        return "👎"
    return "🔄"


def emit_consensus_vote(
    agent_name: str,
    vote: str,  # "approve", "reject", "needs_revision"
    confidence: float,
    reasoning: str,
) -> dict[str, Any]:
    """Emit a vote in the consensus process."""
    vote_emoji = _vote_emoji(vote)
    return emit_event(
        DeepResearchEventType.CONSENSUS_VOTE,
        f"[{agent_name}] Vote: {vote} (confidence: {confidence:.0%})",
        f"{vote_emoji} {agent_name} votes {vote.upper()} ({confidence:.0%}): {reasoning}",
        details={
            "agent": agent_name,
            "vote": vote,
            "confidence": confidence,
            "reasoning": reasoning,
        },
        stage="consensus",
    )


def emit_consensus_result(
    approved: bool,
    approve_count: int,
    reject_count: int,
    revision_count: int,
    overall_confidence: float,
    summary: str,
) -> dict[str, Any]:
    """Emit final consensus result."""
    result = "APPROVED" if approved else "NEEDS REVISION"
    result_emoji = "✅" if approved else "🔄"
    return emit_event(
        DeepResearchEventType.CONSENSUS_RESULT,
        f"Consensus: {result} ({approve_count} approve, {reject_count} reject, {revision_count} revise)",
        f"{result_emoji} Consensus {result}: {summary}\n   Votes: {approve_count} approve, {reject_count} reject, {revision_count} revise\n   Overall confidence: {overall_confidence:.0%}",
        details={
            "approved": approved,
            "approve_count": approve_count,
            "reject_count": reject_count,
            "revision_count": revision_count,
            "overall_confidence": overall_confidence,
            "summary": summary,
        },
        stage="consensus",
    )


# --- Multi-Agent Supervisor Events ---


def emit_supervisor_round_start(
    round_number: int,
    subqueries_delegated: list[str],
    max_rounds: int = 3,
) -> dict[str, Any]:
    """Emit start of a supervisor delegation round."""
    count = len(subqueries_delegated)
    return emit_event(
        DeepResearchEventType.SUPERVISOR_ROUND_START,
        f"Supervisor round {round_number}/{max_rounds}: delegating {count} subqueries to workers",
        f"Round {round_number}/{max_rounds}: Delegating {count} research tasks to worker agents",
        details={
            "round_number": round_number,
            "max_rounds": max_rounds,
            "subqueries_delegated": subqueries_delegated,
            "worker_count": count,
        },
        stage="supervisor",
    )


def emit_supervisor_delegating(
    round_number: int,
    subquery: str,
    worker_id: str,
    has_cross_context: bool = False,
) -> dict[str, Any]:
    """Emit when supervisor assigns a task to a worker."""
    preview = _truncate_text(subquery, 200)
    context_note = " (with cross-context)" if has_cross_context else ""
    return emit_event(
        DeepResearchEventType.SUPERVISOR_DELEGATING,
        f"Supervisor assigns to {worker_id}: {preview}",
        f"Assigning to {worker_id}{context_note}: {preview}",
        details={
            "round_number": round_number,
            "subquery": subquery,
            "worker_id": worker_id,
            "has_cross_context": has_cross_context,
        },
        stage="supervisor",
    )


def emit_supervisor_reflection(
    round_number: int,
    coverage_pct: int,
    gaps: list[str],
    decision: str,
    findings_count: int = 0,
) -> dict[str, Any]:
    """Emit supervisor reflection on collected findings."""
    gap_count = len(gaps)
    gap_summary = f" | {gap_count} gaps identified" if gap_count > 0 else " | No gaps"
    return emit_event(
        DeepResearchEventType.SUPERVISOR_REFLECTION,
        f"Supervisor reflection (round {round_number}): {coverage_pct}% coverage, decision={decision}",
        f"Supervisor assessed coverage: {coverage_pct}%{gap_summary} | Decision: {decision}",
        details={
            "round_number": round_number,
            "coverage_pct": coverage_pct,
            "gaps": gaps,
            "gap_count": gap_count,
            "decision": decision,
            "findings_count": findings_count,
        },
        stage="supervisor",
    )


def emit_supervisor_follow_up(
    round_number: int,
    follow_up_subqueries: list[str],
) -> dict[str, Any]:
    """Emit when supervisor spawns follow-up subqueries."""
    count = len(follow_up_subqueries)
    return emit_event(
        DeepResearchEventType.SUPERVISOR_FOLLOW_UP,
        f"Supervisor spawning {count} follow-up subqueries for round {round_number + 1}",
        f"Spawning {count} follow-up research tasks based on gap analysis",
        details={
            "round_number": round_number,
            "follow_up_subqueries": follow_up_subqueries,
            "count": count,
        },
        stage="supervisor",
    )


def emit_worker_self_evaluation(
    subquery: str,
    quality_score: float,
    confidence: str,
    will_retry: bool,
    attempt: int,
) -> dict[str, Any]:
    """Emit worker self-evaluation of its finding quality."""
    preview = _truncate_text(subquery, 200)
    retry_note = " - will retry with reformulation" if will_retry else " - accepted"
    return emit_event(
        DeepResearchEventType.WORKER_SELF_EVALUATION,
        f"Worker self-eval for '{preview}': quality={quality_score:.2f}, confidence={confidence}{retry_note}",
        f"Self-evaluation (attempt {attempt}): quality {quality_score:.0%}, confidence {confidence}{retry_note}",
        details={
            "subquery": subquery,
            "quality_score": quality_score,
            "confidence": confidence,
            "will_retry": will_retry,
            "attempt": attempt,
        },
        stage="research",
    )


def emit_worker_reformulation(
    subquery: str,
    original_query: str,
    reformulated_query: str,
    attempt: int,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Emit when a worker reformulates its query after low self-evaluation."""
    return emit_event(
        DeepResearchEventType.WORKER_REFORMULATION,
        f"Worker reformulating query (attempt {attempt}/{max_retries})",
        f"Reformulating query (attempt {attempt}/{max_retries}): {_truncate_text(reformulated_query, 200)}",
        details={
            "subquery": subquery,
            "original_query": original_query,
            "reformulated_query": reformulated_query,
            "attempt": attempt,
            "max_retries": max_retries,
        },
        stage="research",
    )


def emit_completeness_assessment(
    coverage_pct: int,
    gaps: list[str],
    conflicts: list[str],
    decision: str,
    threshold: int = 75,
) -> dict[str, Any]:
    """Emit completeness evaluator assessment."""
    gap_info = f", {len(gaps)} gaps" if gaps else ""
    conflict_info = f", {len(conflicts)} conflicts" if conflicts else ""
    passed = decision in ("proceed_to_synthesis", "sufficient")
    return emit_event(
        DeepResearchEventType.COMPLETENESS_ASSESSMENT,
        f"Completeness: {coverage_pct}% coverage{gap_info}{conflict_info}, decision={decision}",
        f"Completeness check: {coverage_pct}% coverage{gap_info}{conflict_info} | Decision: {decision}",
        details={
            "coverage_pct": coverage_pct,
            "threshold": threshold,
            "passed": passed,
            "gaps": gaps,
            "conflicts": conflicts,
            "decision": decision,
        },
        stage="completeness",
    )


def emit_inter_agent_message(
    from_agent: str,
    to_agent: str,
    message_type: str,
    summary: str,
) -> dict[str, Any]:
    """Emit an inter-agent communication event for UI observability."""
    return emit_event(
        DeepResearchEventType.INTER_AGENT_MESSAGE,
        f"[{from_agent} -> {to_agent}] ({message_type}): {_truncate_text(summary, 200)}",
        f"{from_agent} -> {to_agent}: {_truncate_text(summary, 300)}",
        details={
            "from_agent": from_agent,
            "to_agent": to_agent,
            "message_type": message_type,
            "summary": summary,
        },
        stage="coordination",
    )


def emit_subquery_validation(
    total_subqueries: int,
    valid_count: int,
    reformulated_count: int,
    removed_count: int,
) -> dict[str, Any]:
    """Emit subquery validation results from the planner."""
    return emit_event(
        DeepResearchEventType.SUBQUERY_VALIDATION,
        f"Validated {total_subqueries} subqueries: {valid_count} valid, {reformulated_count} reformulated, {removed_count} removed",
        f"Plan validation: {valid_count} valid, {reformulated_count} reformulated, {removed_count} unanswerable",
        details={
            "total_subqueries": total_subqueries,
            "valid_count": valid_count,
            "reformulated_count": reformulated_count,
            "removed_count": removed_count,
        },
        stage="planning",
    )


# ── Live Progress Events (real-time heartbeat during long-running phases) ─


def emit_worker_progress(
    subquery: str,
    idx: int,
    total: int,
    status: str,
) -> dict[str, Any]:
    """Emit per-worker execution progress during research phase."""
    status_label = "completed" if status == "done" else status
    return emit_event(
        DeepResearchEventType.WORKER_PROGRESS,
        f"Research worker {idx}/{total}: "
        f"'{_truncate_text(subquery, 200)}' — {status_label}",
        f"Research worker {idx}/{total} — {status_label}",
        details={
            "subquery": subquery,
            "idx": idx,
            "total": total,
            "status": status,
        },
        stage="research",
    )


# ---------------------------------------------------------------------------
# Complexity Assessor events
# ---------------------------------------------------------------------------


def emit_complexity_assessment(
    complexity_class: str,
    assessed_subqueries: int,
    assessed_rounds: int,
    assessed_iterations: int,
    reasoning: str,
) -> dict[str, Any]:
    """Emit query complexity assessment result."""
    return emit_event(
        DeepResearchEventType.AGENT_DECISION,
        f"Query complexity: {complexity_class} "
        f"(subqueries={assessed_subqueries}, rounds={assessed_rounds}, "
        f"iterations={assessed_iterations})",
        f"Query analysis: {complexity_class} complexity",
        details={
            "complexity_class": complexity_class,
            "assessed_subqueries": assessed_subqueries,
            "assessed_rounds": assessed_rounds,
            "assessed_iterations": assessed_iterations,
            "reasoning": reasoning,
        },
        stage="complexity_assessment",
    )


# ---------------------------------------------------------------------------
# Loop Sentinel events
# ---------------------------------------------------------------------------


def emit_cancelled(thread_id: str) -> dict[str, Any]:
    """Emit cancellation event when user cancels the research session."""
    return emit_event(
        DeepResearchEventType.CANCELLED,
        "Research cancelled by user",
        "Research was cancelled",
        details={"thread_id": thread_id},
        stage="cancelled",
    )


def emit_sentinel_triggered(
    reason: str,
    node_name: str,
    budgets_status: dict[str, Any],
) -> dict[str, Any]:
    """Emit sentinel circuit-breaker event explaining why research stopped."""
    return emit_event(
        DeepResearchEventType.AGENT_DECISION,
        f"Research stopped: {reason}",
        "Research limit reached",
        details={
            "sentinel_reason": reason,
            "triggered_at_node": node_name,
            "budgets": budgets_status,
        },
        stage="sentinel_triggered",
    )


# ---------------------------------------------------------------------------
# Research Loop Contract events
# ---------------------------------------------------------------------------


def emit_research_failed(
    abort_reason: str,
    failure_breakdown: dict[str, int],
    total_attempted: int,
    success_rate: float,
) -> dict[str, Any]:
    """Emit mass-failure abort event."""
    return emit_event(
        DeepResearchEventType.ERROR,
        f"Research failed: {abort_reason}",
        f"Research stopped due to high failure rate ({success_rate:.0%})",
        details={
            "abort_reason": abort_reason,
            "failure_breakdown": failure_breakdown,
            "total_attempted": total_attempted,
            "success_rate": success_rate,
        },
        stage="research",
    )


def emit_diminishing_returns(
    coverage_history: list[float],
    delta: float,
) -> dict[str, Any]:
    """Emit convergence-based early stop event."""
    return emit_event(
        DeepResearchEventType.AGENT_DECISION,
        f"Diminishing returns detected: coverage delta {delta:.1f}%",
        "Research converged — proceeding to synthesis",
        details={
            "coverage_history": coverage_history,
            "last_delta": delta,
        },
        stage="completeness",
    )


# ---------------------------------------------------------------------------
# Transition Guard events
# ---------------------------------------------------------------------------


def emit_no_valid_findings(
    total_attempted: int,
    error_count: int,
) -> dict[str, Any]:
    """Emit event when synthesis is skipped due to zero valid findings."""
    return emit_event(
        DeepResearchEventType.ERROR,
        f"No valid findings: {error_count}/{total_attempted} queries failed",
        "Research could not retrieve valid data",
        details={
            "total_attempted": total_attempted,
            "error_count": error_count,
        },
        stage="synthesis",
    )


def emit_empty_plan() -> dict[str, Any]:
    """Emit event when supervisor receives an empty plan."""
    return emit_event(
        DeepResearchEventType.AGENT_DECISION,
        "Empty plan detected — no subqueries to execute",
        "No research queries in plan",
        details={},
        stage="research",
    )


def emit_reliability_update(
    metrics: dict[str, Any],
    violations: list[str] | None = None,
) -> dict[str, Any]:
    """Emit aggregated HAL reliability metrics update.

    Args:
        metrics: Full reliability metrics dict with all 7 metric scores.
        violations: Compliance violations found, if any.

    Returns:
        Event dict ready for streaming.
    """
    compliance = metrics.get("compliance_score", 1.0)
    fault_rec = metrics.get("fault_recovery_rate", 1.0)
    resource_con = metrics.get("resource_consistency", 1.0)
    plan_align = metrics.get("plan_alignment", 1.0)

    flags: list[str] = []
    if compliance < 0.6:
        flags.append(f"compliance={compliance:.2f}")
    if fault_rec < 0.3:
        flags.append(f"fault_recovery={fault_rec:.2f}")
    if resource_con < 0.3:
        flags.append(f"resource_consistency={resource_con:.2f}")
    if plan_align < 0.4:
        flags.append(f"plan_alignment={plan_align:.2f}")

    if flags:
        summary = f"Reliability warnings: {', '.join(flags)}"
    else:
        summary = "All reliability metrics within acceptable range"

    details = dict(metrics)
    if violations:
        details["compliance_violations"] = violations

    return emit_event(
        DeepResearchEventType.RELIABILITY_UPDATE,
        summary,
        summary,
        details=details,
        stage="review",
    )
