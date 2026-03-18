"""Complete node: finalize research output."""

from typing import Any

from template_agent.src.core.deep_research.events import (
    emit_completed,
    emit_final_answer,
    emit_token_usage_update,
)
from template_agent.src.core.deep_research.state import (
    PHASE_COMPLETE,
    DeepResearchState,
    ResearchContext,
)
from template_agent.src.core.utils import (
    sanitize_markdown_tables,
    strip_annotation_tags,
)
from template_agent.utils.pylogger import get_python_logger

from ._cache import save_cached_findings, save_findings_in_memory
from ._helpers import findings_from_board

logger = get_python_logger()


async def complete_node(
    state: DeepResearchState,
    ctx: ResearchContext,
) -> tuple[dict, list]:
    """Complete node: Finalize and return the answer.

    Saves findings to cache, emits final answer and completion events.
    No dataverse-specific persistence.
    """
    events: list[dict[str, Any]] = []
    final_answer = state.get("draft_answer") or state.get("final_answer") or ""
    final_answer = strip_annotation_tags(final_answer)
    final_answer = sanitize_markdown_tables(final_answer)
    visualizations = state.get("visualizations", [])

    # Save findings to cache for future follow-ups
    findings_board = state.get("findings_board", {})
    findings = findings_from_board(findings_board)
    findings_list = list(findings.values())
    await save_cached_findings(
        ctx.checkpointer,
        state.get("thread_id"),
        findings_list,
    )
    save_findings_in_memory(state.get("thread_id") or "", findings_board)

    # Persist token usage if tracker available
    if ctx.token_tracker is not None:
        try:
            if hasattr(ctx.token_tracker, "persist_to_db"):
                ctx.token_tracker.persist_to_db(
                    state.get("thread_id") or "",
                    getattr(ctx, "user_id", None),
                )
            if hasattr(ctx.token_tracker, "flush_to_langfuse") and ctx.root_tracer:
                ctx.token_tracker.flush_to_langfuse(ctx.root_tracer)
        except Exception as e:
            logger.debug("Token tracker persist failed: %s", e)

    # Flush tracing client if available
    try:
        from template_agent.utils.tracing import client as tracing_client

        if tracing_client is not None:
            tracing_client.flush()
    except Exception as e:
        logger.debug("Tracing client flush failed: %s", e)

    ctx.emit_or_append(emit_final_answer(final_answer, visualizations), events)

    if ctx.token_tracker is not None:
        usage_summary = ctx.token_tracker.get_summary()
        total = usage_summary.get("total", {})
        per_phase = usage_summary.get("per_phase", {})
        ctx.emit_or_append(
            emit_token_usage_update(
                input_tokens=total.get("input_tokens", 0),
                output_tokens=total.get("output_tokens", 0),
                total_tokens=total.get("total_tokens", 0),
                llm_calls=total.get("llm_calls", 0),
                estimated_cost_usd=total.get("estimated_cost_usd", 0.0),
                per_phase=per_phase,
            ),
            events,
        )

    import time as _time

    start_time = state.get(
        "research_start_time", state.get("execution_start_time", 0.0)
    )
    effective_elapsed = round(_time.time() - start_time, 2) if start_time > 0 else 0.0
    pre_plan = state.get("pre_plan_elapsed_seconds", 0.0)

    ctx.emit_or_append(
        emit_completed(
            effective_elapsed_seconds=effective_elapsed,
            pre_plan_elapsed_seconds=pre_plan,
        ),
        events,
    )

    return {
        "final_answer": final_answer,
        "visualizations": visualizations,
        "current_phase": PHASE_COMPLETE,
        "should_stop": True,
    }, events
