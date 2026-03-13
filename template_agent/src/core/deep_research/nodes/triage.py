"""Triage node: decide between direct answer and deep research."""

from typing import Any, Dict, List

from template_agent.src.core.deep_research.events import (
    emit_agent_decision,
    emit_agent_thinking,
    emit_triage_decision,
)
from template_agent.src.core.deep_research.prompts import build_triage_prompt
from template_agent.src.core.deep_research.state import (
    PHASE_PLAN,
    PHASE_PROBE,
    PHASE_SYNTHESIZE,
    DeepResearchState,
    ResearchContext,
)
from template_agent.src.core.deep_research.token_tracker import tracked_invoke
from template_agent.src.core.utils import safe_json_parse, truncate_text
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


async def triage_node(
    state: DeepResearchState,
    ctx: ResearchContext,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Triage node: Classify follow-up queries for optimal routing.

    Determines whether a follow-up query can be answered from existing
    research data (context_sufficient), needs partial new research
    (partial_research), or requires full research (full_research).

    Returns:
        Tuple of (state_updates, events)
    """
    events: List[Dict[str, Any]] = []
    query = state.get("query", "")
    context = state.get("context", "") or ""
    cached_findings_text = state.get("cached_findings_text", "") or ""

    if not cached_findings_text and not context:
        logger.info("Triage: no cached findings or context, routing to full_research")
        ctx.emit_or_append(
            emit_triage_decision(
                decision="full_research",
                reasoning="No previous research data available — starting fresh.",
                cached_findings_count=0,
                context_message_count=0,
            ),
            events,
        )
        return {
            "triage_decision": "full_research",
            "current_phase": PHASE_PROBE,
        }, events

    context_msg_count = context.count("User:") + context.count("Assistant:")
    findings_count = cached_findings_text.count("- Q: ")

    ctx.emit_or_append(
        emit_agent_thinking(
            "TriageAgent",
            f"Evaluating if follow-up can be answered from "
            f"{findings_count} cached findings and {context_msg_count} context messages",
        ),
        events,
    )

    triage_prompt = build_triage_prompt()
    triage_messages = triage_prompt.format_messages(
        query=query,
        context=context or "None.",
        cached_findings=cached_findings_text or "None.",
    )

    decision = "full_research"
    reasoning = "Default: uncertain classification"

    try:
        response = await tracked_invoke(
            ctx.base_model,
            triage_messages,
            ctx.token_tracker,
            "triage",
            **ctx.llm_call_kwargs(),
        )
        response_text = str(response.content or "").strip()

        parsed = safe_json_parse(response_text)
        if parsed:
            raw_decision = str(parsed.get("decision", "")).strip().lower()
            reasoning = str(parsed.get("reasoning", reasoning))

            valid_decisions = {
                "context_sufficient",
                "partial_research",
                "full_research",
            }
            if raw_decision in valid_decisions:
                decision = raw_decision
            else:
                logger.warning(
                    "Triage: invalid decision '%s', defaulting to full_research",
                    raw_decision,
                )
                decision = "full_research"
                reasoning = f"Invalid LLM decision '{raw_decision}' — defaulting to full research"
        else:
            logger.warning(
                "Triage: no JSON found in response: %s",
                truncate_text(response_text, 200),
            )
            reasoning = "Could not parse LLM response — defaulting to full research"

    except Exception as e:
        logger.warning("Triage LLM call failed: %s", e)
        reasoning = f"LLM call failed: {truncate_text(str(e), 100)} — defaulting to full research"

    logger.info("Triage decision: %s — %s", decision, reasoning)

    ctx.emit_or_append(
        emit_agent_decision("TriageAgent", f"Decision: {decision}", reasoning), events
    )
    ctx.emit_or_append(
        emit_triage_decision(
            decision=decision,
            reasoning=reasoning,
            cached_findings_count=findings_count,
            context_message_count=context_msg_count,
        ),
        events,
    )

    phase_map = {
        "context_sufficient": PHASE_SYNTHESIZE,
        "partial_research": PHASE_PLAN,
        "full_research": PHASE_PROBE,
    }

    return {
        "triage_decision": decision,
        "current_phase": phase_map.get(decision, PHASE_PROBE),
    }, events
