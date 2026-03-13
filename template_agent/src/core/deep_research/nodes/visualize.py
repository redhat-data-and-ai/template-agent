"""Visualization node: lightweight/optional skeleton for domain-specific charts."""

from typing import Any, Dict, List

from template_agent.src.core.deep_research.events import (
    emit_agent_decision,
    emit_visualization_complete,
    emit_visualization_skipped,
    emit_visualization_start,
)
from template_agent.src.core.deep_research.state import (
    PHASE_REVIEW,
    DeepResearchState,
    ResearchContext,
)
from template_agent.utils.pylogger import get_python_logger

from ._helpers import check_node_cancelled, findings_from_board

logger = get_python_logger()

# Tool name patterns that suggest visualization capability
_VISUALIZATION_TOOL_PATTERNS = (
    "chart",
    "plot",
    "visualize",
    "graph",
    "create_bar",
    "create_pie",
    "create_line",
)


def _has_visualization_tools(ctx: ResearchContext) -> bool:
    """Check if ctx.tools contains any visualization-capable tools."""
    if not ctx.tools:
        return False
    tool_names = [getattr(t, "name", str(t)).lower() for t in ctx.tools]
    return any(
        any(p in name for p in _VISUALIZATION_TOOL_PATTERNS) for name in tool_names
    )


async def visualize_node(
    state: DeepResearchState,
    ctx: ResearchContext,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Visualization node: lightweight skeleton.

    Checks if any visualization tools are available in ctx.tools.
    If none found, skips visualization and proceeds to review.
    If found, attempts to create visualizations (domain-specific implementations
    can extend this).
    """
    cancelled = check_node_cancelled(state.get("thread_id"), "visualize", state)
    if cancelled is not None:
        return cancelled, []

    events: List[Dict[str, Any]] = []
    events.append(emit_visualization_start())
    events.append(
        emit_agent_decision(
            "Visualizer",
            "Checking for visualization tools",
            "Scanning available tools",
        )
    )

    if not _has_visualization_tools(ctx):
        logger.info(
            "No visualization tools found in ctx.tools — skipping visualization"
        )
        events.append(
            emit_visualization_skipped("No visualization tools available in context")
        )
        events.append(emit_visualization_complete(0))
        return {
            "visualizations": [],
            "visualization_attempted": True,
            "current_phase": PHASE_REVIEW,
        }, events

    findings = findings_from_board(state.get("findings_board", {}))
    draft_answer = state.get("draft_answer", "")

    if not findings and not draft_answer:
        events.append(emit_visualization_skipped("No findings or draft to visualize"))
        events.append(emit_visualization_complete(0))
        return {
            "visualizations": [],
            "visualization_attempted": True,
            "current_phase": PHASE_REVIEW,
        }, events

    # Skeleton: domain-specific implementations would invoke visualization
    # tools here. For the generic template, we skip actual chart creation.
    events.append(
        emit_visualization_skipped(
            "Visualization tools present but chart creation is domain-specific"
        )
    )
    events.append(emit_visualization_complete(0))

    return {
        "visualizations": [],
        "visualization_attempted": True,
        "current_phase": PHASE_REVIEW,
    }, events
