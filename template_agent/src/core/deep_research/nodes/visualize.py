"""Visualization node: generates Mermaid diagrams from research data."""

import re
from typing import Any

from template_agent.src.core.deep_research.events import (
    emit_agent_decision,
    emit_visualization_complete,
    emit_visualization_created,
    emit_visualization_skipped,
    emit_visualization_start,
)
from template_agent.src.core.deep_research.prompts import build_visualization_prompt
from template_agent.src.core.deep_research.state import (
    PHASE_REVIEW,
    DeepResearchState,
    ResearchContext,
)
from template_agent.src.core.deep_research.token_tracker import tracked_invoke
from template_agent.utils.pylogger import get_python_logger

from ._helpers import check_node_cancelled

logger = get_python_logger()

_MERMAID_BLOCK_RE = re.compile(
    r"```mermaid\s*\n(.*?)```",
    re.DOTALL,
)


def _extract_mermaid_blocks(text: str) -> list[dict[str, Any]]:
    """Extract Mermaid code blocks and their preceding headings from LLM output."""
    charts: list[dict[str, Any]] = []
    lines = text.split("\n")
    current_heading = ""

    for line in lines:
        if line.strip().startswith("##"):
            current_heading = line.strip().lstrip("#").strip()

    for match in _MERMAID_BLOCK_RE.finditer(text):
        code = match.group(1).strip()
        if not code:
            continue

        before_text = text[: match.start()]
        heading_lines = [
            ln.strip().lstrip("#").strip()
            for ln in before_text.split("\n")
            if ln.strip().startswith("##")
        ]
        title = heading_lines[-1] if heading_lines else current_heading or "Chart"

        chart_type = "graph"
        code_lower = code.lower()
        if code_lower.startswith("pie"):
            chart_type = "pie"
        elif "xychart" in code_lower:
            chart_type = "bar"
        elif "timeline" in code_lower:
            chart_type = "timeline"

        charts.append(
            {
                "chart_type": chart_type,
                "title": title,
                "mermaid_code": code,
            }
        )

    return charts


async def visualize_node(
    state: DeepResearchState,
    ctx: ResearchContext,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Generate Mermaid visualizations from the draft research report."""
    cancelled = await check_node_cancelled(state.get("thread_id"), "visualize", state)
    if cancelled is not None:
        return cancelled, []

    events: list[dict[str, Any]] = []
    ctx.emit_or_append(emit_visualization_start(), events)

    draft_answer = state.get("draft_answer", "")

    viz_split = re.split(r"\n+---\n+## Visualizations\b", draft_answer, maxsplit=1)
    draft_answer = viz_split[0]

    if not draft_answer:
        ctx.emit_or_append(
            emit_visualization_skipped("No draft answer available to visualize"), events
        )
        ctx.emit_or_append(emit_visualization_complete(0), events)
        return {
            "visualizations": [],
            "visualization_attempted": True,
            "current_phase": PHASE_REVIEW,
        }, events

    ctx.emit_or_append(
        emit_agent_decision(
            "Visualizer",
            "Analyzing report for visualizable data",
            "Scanning for numeric, relational, and temporal data",
        ),
        events,
    )

    try:
        messages = build_visualization_prompt(draft_answer[:8000])
        response = await tracked_invoke(
            ctx.base_model,
            messages,
            ctx.token_tracker,
            "visualization",
            **ctx.get_llm_kwargs(timeout=120),
        )

        response_text = getattr(response, "content", str(response)).strip()

        if "NO_CHARTS" in response_text:
            ctx.emit_or_append(
                emit_visualization_skipped("LLM found no chart-worthy data"), events
            )
            ctx.emit_or_append(emit_visualization_complete(0), events)
            return {
                "visualizations": [],
                "visualization_attempted": True,
                "current_phase": PHASE_REVIEW,
            }, events

        charts = _extract_mermaid_blocks(response_text)
        if not charts:
            ctx.emit_or_append(
                emit_visualization_skipped("No valid Mermaid diagrams generated"),
                events,
            )
            ctx.emit_or_append(emit_visualization_complete(0), events)
            return {
                "visualizations": [],
                "visualization_attempted": True,
                "current_phase": PHASE_REVIEW,
            }, events

        viz_section_parts = ["\n\n---\n\n## Visualizations\n"]
        for chart in charts:
            viz_section_parts.append(f"\n### {chart['title']}\n")
            viz_section_parts.append(f"\n```mermaid\n{chart['mermaid_code']}\n```\n")
            ctx.emit_or_append(
                emit_visualization_created(
                    chart_type=chart["chart_type"],
                    title=chart["title"],
                ),
                events,
            )

        updated_draft = draft_answer + "".join(viz_section_parts)

        ctx.emit_or_append(emit_visualization_complete(len(charts)), events)
        logger.info("Visualization node generated %d chart(s)", len(charts))

        return {
            "draft_answer": updated_draft,
            "visualizations": charts,
            "visualization_attempted": True,
            "current_phase": PHASE_REVIEW,
        }, events

    except Exception as e:
        logger.warning("Visualization LLM call failed: %s", e)
        ctx.emit_or_append(
            emit_visualization_skipped(f"Visualization failed: {type(e).__name__}"),
            events,
        )
        ctx.emit_or_append(emit_visualization_complete(0), events)
        return {
            "visualizations": [],
            "visualization_attempted": True,
            "current_phase": PHASE_REVIEW,
        }, events
