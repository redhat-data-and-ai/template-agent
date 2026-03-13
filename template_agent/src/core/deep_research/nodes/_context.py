"""Hierarchical context management helpers for deep research nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from template_agent.src.core.deep_research.context_manager import (
    create_context_manager,
    estimate_state_tokens,
    get_max_context_tokens,
)

if TYPE_CHECKING:
    from template_agent.src.core.deep_research.context_manager import (
        FindingCard,
        ImmediateContext,
        ResearchMemory,
    )
from template_agent.src.core.deep_research.events import emit_context_usage_update
from template_agent.src.core.deep_research.state import (
    DeepResearchState,
    Finding,
    ResearchContext,
)
from template_agent.utils.pylogger import get_python_logger

from ._helpers import findings_from_board


def _get_setting(name: str, default: Any) -> Any:
    """Get setting with fallback to default."""
    try:
        from template_agent.src.settings import settings

        return getattr(settings, name, default)
    except Exception:
        return default


logger = get_python_logger(_get_setting("PYTHON_LOG_LEVEL", "INFO"))


async def _process_finding_hierarchical(
    ctx: ResearchContext,
    finding: Finding,
    state: DeepResearchState | dict[str, Any],
) -> tuple["ImmediateContext | None", list["FindingCard"], "ResearchMemory | None"]:
    """Process a new finding through hierarchical context management.

    Updates the three-level context hierarchy:
    - Level 1 (Immediate): Recent findings in full detail
    - Level 2 (Working Memory): Compressed FindingCards
    - Level 3 (Research Memory): High-level research state

    Args:
        ctx: Research context with base_model for compression.
        finding: The new finding to process.
        state: Current deep research state.

    Returns:
        Tuple of (immediate_context, finding_cards, research_memory)
    """
    if not _get_setting("ENABLE_HIERARCHICAL_CONTEXT", False):
        return None, [], None

    immediate_context = state.get("immediate_context") or {
        "recent_findings": [],
        "recent_subqueries": [],
        "window_size": _get_setting("CONTEXT_WINDOW_SIZE", 8),
        "slide_step": _get_setting("CONTEXT_SLIDE_STEP", 4),
    }
    finding_cards = list(state.get("finding_cards", []))
    research_memory = state.get("research_memory")

    try:
        mgr = create_context_manager(ctx)
        return await mgr.process_new_finding(
            finding,
            immediate_context,
            finding_cards,
            research_memory,
        )
    except Exception as e:
        logger.warning("Hierarchical context processing failed: %s", e)
        return immediate_context, finding_cards, research_memory


def _emit_context_usage(
    ctx: ResearchContext,
    state: DeepResearchState | dict[str, Any],
    current_phase: str,
) -> None:
    """Emit context usage update event if hierarchical context is enabled.

    Calculates current token usage and emits an event for UI display.

    Args:
        ctx: Research context for emitting events.
        state: Current deep research state with context data.
        current_phase: Current pipeline phase name.
    """
    if not _get_setting("ENABLE_HIERARCHICAL_CONTEXT", False):
        return

    try:
        board = state.get("findings_board", {})
        findings = findings_from_board(board) if board else state.get("findings", {})
        immediate_context = state.get("immediate_context")
        finding_cards = state.get("finding_cards", [])
        research_memory = state.get("research_memory")

        current_tokens = estimate_state_tokens(
            findings,
            immediate_context,
            finding_cards,
            research_memory,
        )
        max_tokens = get_max_context_tokens(ctx.model_name)
        usage_pct = (current_tokens / max_tokens) * 100 if max_tokens else 0

        if usage_pct > 90:
            status = "critical"
        elif usage_pct > 70:
            status = "warning"
        else:
            status = "normal"

        ctx.emit(
            emit_context_usage_update(
                current_tokens=current_tokens,
                max_tokens=max_tokens,
                usage_percent=usage_pct,
                status=status,
                stage=current_phase,
            )
        )
    except Exception as e:
        logger.warning("Failed to emit context usage: %s", e)


def _format_hierarchical_context_for_synthesis(
    state: DeepResearchState,
    ctx: ResearchContext,
) -> str:
    """Format hierarchical context for synthesis prompt.

    When hierarchical context is enabled, combines all three levels:
    - Level 3: Research overview and key insights
    - Level 2: Finding summaries from compressed cards
    - Level 1: Detailed recent findings

    Args:
        state: Current deep research state.
        ctx: Research context.

    Returns:
        Formatted context string for synthesis.
    """
    if not _get_setting("ENABLE_HIERARCHICAL_CONTEXT", False):
        return ""

    immediate_context = state.get("immediate_context")
    if not immediate_context:
        return ""

    try:
        mgr = create_context_manager(ctx)
        return mgr.format_for_synthesis(
            immediate_context,
            state.get("finding_cards", []),
            state.get("research_memory"),
            state.get("query", ""),
        )
    except Exception as e:
        logger.warning("Failed to format hierarchical context: %s", e)
        return ""
