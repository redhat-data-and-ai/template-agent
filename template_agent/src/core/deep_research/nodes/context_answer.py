"""Context answer node: direct response for simple queries."""

from typing import Any, Dict, List

from template_agent.src.core.deep_research.events import (
    emit_agent_decision,
    emit_agent_thinking,
    emit_synthesis_complete,
    emit_synthesis_start,
)
from template_agent.src.core.deep_research.prompts import build_context_answer_prompt
from template_agent.src.core.deep_research.state import (
    PHASE_REVIEW,
    DeepResearchState,
    Finding,
    ResearchContext,
)
from template_agent.src.core.deep_research.token_tracker import tracked_invoke
from template_agent.src.core.prompt import get_current_date
from template_agent.src.core.utils import (
    simplify_error_for_display,
    strip_annotation_tags,
)
from template_agent.utils.pylogger import get_python_logger

from ._cache import load_cached_findings

logger = get_python_logger()


async def context_answer_node(
    state: DeepResearchState,
    ctx: ResearchContext,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Context answer node: Fast-path synthesis from existing research data.

    Used when the triage node determined the follow-up question can be
    fully answered from cached findings and conversation context.

    Also populates ``findings`` from cached data so the review node
    can validate the draft answer against actual research evidence.

    Returns:
        Tuple of (state_updates, events)
    """
    events: List[Dict[str, Any]] = []
    query = state.get("query", "")
    context = state.get("context", "") or "None."
    cached_findings_text = state.get("cached_findings_text", "") or ""

    ctx.emit_or_append(
        emit_agent_thinking(
            "SynthesisAgent",
            "Answering from existing research data (no new queries needed)",
        ),
        events,
    )
    ctx.emit_or_append(emit_synthesis_start(iteration=1), events)

    cached_findings: Dict[str, Finding] = {}
    try:
        cached_findings = await load_cached_findings(
            ctx.checkpointer, state.get("thread_id")
        )
    except Exception as e:
        logger.warning("Failed to load cached findings for context answer: %s", e)

    prompt = build_context_answer_prompt()
    prompt_messages = prompt.format_messages(
        query=query,
        context=context,
        cached_findings=cached_findings_text,
        current_date=get_current_date(),
    )

    try:
        response = await tracked_invoke(
            ctx.base_model,
            prompt_messages,
            ctx.token_tracker,
            "synthesis",
            **ctx.llm_call_kwargs(),
        )
        draft_answer = str(response.content or "").strip()
    except Exception as e:
        logger.error("Context answer synthesis failed: %s", e, exc_info=True)
        draft_answer = f"Failed to synthesize answer from existing data: {simplify_error_for_display(str(e))}"

    draft_answer = strip_annotation_tags(draft_answer)

    ctx.emit_or_append(emit_synthesis_complete(), events)
    ctx.emit_or_append(
        emit_agent_decision(
            "SynthesisAgent",
            "Draft answer created from existing research",
            f"{len(draft_answer)} chars",
        ),
        events,
    )

    return {
        "draft_answer": draft_answer,
        "findings": cached_findings,
        "synthesis_iteration": 1,
        "current_phase": PHASE_REVIEW,
    }, events
