"""Probe node: initial tool discovery."""

import asyncio
from typing import Any, Dict, List

from template_agent.src.core.deep_research.agents import execute_with_research_agent
from template_agent.src.core.deep_research.events import (
    emit_probe_complete,
    emit_probe_start,
    emit_tool_discovery,
)
from template_agent.src.core.deep_research.prompts import build_probe_prompt
from template_agent.src.core.deep_research.state import (
    PHASE_PLAN,
    DeepResearchState,
    ResearchContext,
)
from template_agent.src.core.utils import simplify_error_for_display
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

# Default probe timeout when no settings available
DEFAULT_PROBE_TIMEOUT_SECONDS = 60


async def probe_node(
    state: DeepResearchState,
    ctx: ResearchContext,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Probe node: Discover available tools via the research agent.

    Runs a probe query through the research agent to understand what tools
    are available and how they can help with the query.

    Returns:
        Tuple of (state_updates, events)
    """
    events: List[Dict[str, Any]] = []
    query = state.get("query", "")
    context = state.get("context", "") or "None."

    tool_names = ctx.get_tool_names()
    tool_inventory = ctx.format_tool_inventory()
    events.append(emit_tool_discovery(len(ctx.tools), tool_names))

    events.append(emit_probe_start())

    probe_prompt = build_probe_prompt()
    probe_query = probe_prompt.format_messages(
        query=query,
        context=context,
        tool_inventory=tool_inventory,
    )
    raw_content = probe_query[-1].content if probe_query else query
    probe_query_text = raw_content if isinstance(raw_content, str) else str(raw_content)

    timeout = getattr(
        ctx,
        "probe_timeout_seconds",
        DEFAULT_PROBE_TIMEOUT_SECONDS,
    )

    try:
        result = await asyncio.wait_for(
            execute_with_research_agent(ctx, probe_query_text, state.get("thread_id")),
            timeout=timeout,
        )
        probe_result = (
            result.get("answer", "") if isinstance(result, dict) else str(result)
        )
    except asyncio.TimeoutError:
        probe_result = "Probe timed out. Proceeding with available tool inventory."
        logger.warning("Tool discovery probe timed out")
    except Exception as e:
        probe_result = f"Probe failed: {simplify_error_for_display(str(e))}"
        logger.warning("Tool discovery probe failed: %s", e)

    events.append(emit_probe_complete(probe_result))

    return {
        "tool_inventory": tool_inventory,
        "tool_names": tool_names,
        "probe_result": probe_result,
        "current_phase": PHASE_PLAN,
    }, events
