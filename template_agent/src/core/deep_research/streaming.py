"""Deep research streaming agent.

This module provides the DeepResearchAgent that orchestrates the
hierarchical multi-agent deep research pipeline with proper streaming.
"""

from __future__ import annotations

import asyncio
import copy
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, AsyncGenerator, Literal

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from template_agent.src.core.deep_research.agents import get_research_context
from template_agent.src.core.deep_research.cancel import get_cancel_store
from template_agent.src.core.deep_research.events import (
    DeepResearchEventType,
    emit_context_loaded,
    emit_cross_chat_findings_loaded,
    emit_error,
    emit_event,
    emit_heartbeat,
    emit_started,
)
from template_agent.src.core.deep_research.nodes import (
    assess_complexity_node,
    complete_node,
    completeness_evaluator_node,
    context_answer_node,
    findings_from_board,
    format_full_cached_findings_for_triage,
    load_cached_findings,
    plan_node,
    probe_node,
    research_supervisor_node,
    review_node,
    save_cached_findings,
    synthesize_node,
    triage_node,
    visualize_node,
)
from template_agent.src.core.deep_research.plan_store import (
    clear_plan,
    get_plan_enrichment,
    get_plan_with_load,
    set_plan,
)
from template_agent.src.core.deep_research.state import (
    PHASE_AWAIT_APPROVAL,
    PHASE_COMPLETE,
    PHASE_COMPLETENESS,
    PHASE_PROBE,
    PHASE_SUPERVISOR,
    PHASE_SYNTHESIZE,
    PHASE_TRIAGE,
    DeepResearchState,
    Finding,
    ResearchContext,
)
from template_agent.src.core.deep_research.utils import (
    GIBBERISH_RESPONSE,
    aput_checkpoint,
    classify_input_quality,
    get_raw_checkpointer,
    sanitize_error_for_client,
)
from template_agent.src.core.utils import safe_json_parse, truncate_text
from template_agent.src.settings import settings as app_settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


DEEP_RESEARCH_MAX_SELECTED_FINDINGS = 10
DEEP_RESEARCH_FINDING_SELECTION_THRESHOLD = 8
DEEP_RESEARCH_GRAPH_RECURSION_LIMIT = 100
DEEP_RESEARCH_MAX_SUPERVISOR_ROUNDS = 3
DEEP_RESEARCH_MAX_ITERATIONS = 3
DEEP_RESEARCH_MAX_SUBQUERIES = 10
DEEP_RESEARCH_MAX_TOTAL_SUBQUERIES = 20
DEEP_RESEARCH_MAX_NODE_TRANSITIONS = 50
DEEP_RESEARCH_MAX_SESSION_SECONDS: float = float(
    app_settings.DEEP_RESEARCH_MAX_SESSION_SECONDS
)
DEEP_RESEARCH_HEARTBEAT_INTERVAL_SECONDS = 30.0
DEEP_RESEARCH_EVENT_QUEUE_MAXSIZE = 2000
DEEP_RESEARCH_MAX_FULL_CONTEXT_MESSAGES = 6
DEEP_RESEARCH_CROSS_CHAT_FINDINGS_ENABLED = False

_CONTEXT_USER_PREFIX = "User:"
_CONTEXT_ASSISTANT_PREFIX = "Assistant:"


def _build_context_loaded_event(context: str) -> dict[str, Any]:
    """Build context_loaded event from conversation context string."""
    if context:
        message_count = context.count(_CONTEXT_USER_PREFIX) + context.count(
            _CONTEXT_ASSISTANT_PREFIX
        )
        return emit_context_loaded(
            message_count=message_count,
            context_preview=context,
            has_context=True,
        )
    return emit_context_loaded(
        message_count=0,
        context_preview="",
        has_context=False,
    )


def _build_indexed_summaries(
    findings: dict[str, Finding],
) -> tuple[list[tuple[str, Finding]], list[str]]:
    """Build indexed list and summaries for LLM selection."""
    indexed: list[tuple[str, Finding]] = []
    summaries_for_llm: list[str] = []
    for idx, (fhash, finding) in enumerate(findings.items()):
        subquery = finding.get("subquery", "")
        if not subquery:
            continue
        answer_preview = (finding.get("answer", "") or "")[:200]
        indexed.append((fhash, finding))
        summaries_for_llm.append(f"[{idx}] Q: {subquery}\n    A: {answer_preview}...")
    return indexed, summaries_for_llm


def _parse_valid_llm_indices(
    response_text: str,
    indexed: list[tuple[str, Finding]],
    max_findings: int,
) -> list[int]:
    """Parse LLM response and return valid indices."""
    selected_indices = safe_json_parse(response_text, pattern=r"\[[\d\s,]+\]")
    if not isinstance(selected_indices, list):
        raise ValueError(
            f"No JSON array in response: {truncate_text(response_text, 200)}"
        )
    seen: set[int] = set()
    valid_indices: list[int] = []
    for idx in selected_indices:
        resolved = int(idx) if isinstance(idx, (int, float)) else None
        if (
            resolved is not None
            and 0 <= resolved < len(indexed)
            and resolved not in seen
        ):
            seen.add(resolved)
            valid_indices.append(resolved)
        if len(valid_indices) >= max_findings:
            break
    if not valid_indices:
        raise ValueError("LLM returned no valid indices")
    return valid_indices


def _format_selected_findings(
    indexed: list[tuple[str, Finding]],
    valid_indices: list[int],
    max_chars: int,
) -> str:
    """Format selected findings with truncation."""
    parts: list[str] = []
    total_chars = 0
    for idx in valid_indices:
        _, finding = indexed[idx]
        subquery = finding.get("subquery", "")
        answer = finding.get("answer", "")
        entry = f"### Subquery: {subquery}\n{answer}"
        if total_chars + len(entry) > max_chars:
            parts.append("... (additional findings truncated for length)")
            break
        parts.append(entry)
        total_chars += len(entry) + 1
    return "\n\n".join(parts)


async def select_relevant_findings(
    model: Any,
    query: str,
    findings: dict[str, Finding],
    max_findings: int | None = None,
    max_chars: int = 20000,
) -> str:
    """Select the most relevant cached findings for a follow-up query."""
    if not findings:
        return ""
    if max_findings is None:
        max_findings = DEEP_RESEARCH_MAX_SELECTED_FINDINGS
    threshold = DEEP_RESEARCH_FINDING_SELECTION_THRESHOLD

    if len(findings) <= threshold:
        return format_full_cached_findings_for_triage(findings, max_chars=max_chars)

    indexed, summaries_for_llm = _build_indexed_summaries(findings)
    if not summaries_for_llm:
        return ""

    try:
        from langchain_core.messages import HumanMessage as HM
        from langchain_core.messages import SystemMessage as SM

        summaries_text = "\n".join(summaries_for_llm)
        system_text = (
            "You are a relevance scorer. Given a follow-up question and a "
            "numbered list of prior research finding summaries, return a JSON "
            "array of the indices (integers) of the findings most relevant to "
            "the question, ordered by relevance (most relevant first).\n\n"
            f"Return at most {max_findings} indices.\n\n"
            "Return ONLY a JSON array of integers, e.g. [3, 0, 7, 1]."
        )
        human_text = (
            f"Follow-up question: {query}\n\nFinding summaries:\n{summaries_text}"
        )
        response = await model.ainvoke(
            [SM(content=system_text), HM(content=human_text)]
        )
        response_text = str(response.content or "").strip()
        valid_indices = _parse_valid_llm_indices(response_text, indexed, max_findings)
        return _format_selected_findings(indexed, valid_indices, max_chars)
    except Exception as e:
        logger.warning("LLM finding selection failed: %s, falling back", e)
        return format_full_cached_findings_for_triage(findings, max_chars=max_chars)


def _node_error_updates(exc: Exception) -> dict[str, Any]:
    """Build error state updates for node failures."""
    logger.error("Deep research node error: %s", exc, exc_info=True)
    safe_msg = sanitize_error_for_client(exc)
    error_event = emit_event(
        DeepResearchEventType.ERROR,
        f"Research error: {safe_msg}",
        f"A research step failed: {safe_msg}",
        ui_visible=True,
    )
    return {
        "final_answer": f"Research encountered an error: {safe_msg}",
        "current_phase": PHASE_COMPLETE,
        "_pending_events": [error_event],
    }


def _span_open(ctx: ResearchContext, node_name: str, state: DeepResearchState) -> Any:
    """Open a Langfuse span for a deep research node."""
    root_tracer = getattr(ctx, "root_tracer", None)
    if root_tracer is None or not hasattr(root_tracer, "span"):
        return None
    try:
        return root_tracer.span(
            name=f"deep_research.{node_name}",
            input={
                "query": state.get("query", ""),
                "phase": state.get("current_phase", ""),
            },
        )
    except Exception:
        return None


def _span_end_ok(span: Any, updates: dict[str, Any], start: float) -> None:
    """End a Langfuse span with success status."""
    if span is None:
        return
    try:
        duration_ms = (time.monotonic() - start) * 1000
        output_keys = [k for k in updates if k != "_pending_events"]
        span.end(
            output={"updated_keys": output_keys},
            metadata={"duration_ms": round(duration_ms, 2)},
        )
    except Exception:
        pass


def _span_end_error(span: Any, exc: Exception) -> None:
    """End a Langfuse span with error status."""
    if span is None:
        return
    try:
        span.update(level="ERROR", status_message=str(exc))
        span.end()
    except Exception:
        pass


def _wrap_node(node_fn: Any, ctx: ResearchContext, node_name: str) -> Any:
    """Wrap a node with error boundary, event injection, and Langfuse spans."""

    async def _wrapped(state: DeepResearchState) -> dict[str, Any]:
        span = _span_open(ctx, node_name, state)
        start = time.monotonic()
        try:
            updates, events = await node_fn(state, ctx)
            updates["_pending_events"] = events
            _span_end_ok(span, updates, start)
            return updates
        except Exception as e:
            _span_end_error(span, e)
            return _node_error_updates(e)

    return _wrapped


def _after_review_route(
    state: DeepResearchState,
) -> Literal["synthesize", "supervisor", "complete"]:
    """Determine next node after review."""
    current_review = state.get("current_review")
    if not current_review:
        return "complete"
    action = current_review.get("action", "approve")
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)
    if iteration >= max_iterations:
        return "complete"
    if action == "research_more":
        return "supervisor"
    if action == "revise":
        return "synthesize"
    return "complete"


def _after_router(
    state: DeepResearchState,
) -> Literal["complete", "assess_complexity", "supervisor"]:
    """Route after router node."""
    current_phase = state.get("current_phase", PHASE_PROBE)
    if current_phase == PHASE_COMPLETE:
        return "complete"
    if current_phase == PHASE_SUPERVISOR:
        return "supervisor"
    return "assess_complexity"


def _after_assess_complexity(
    state: DeepResearchState,
) -> Literal["triage", "probe", "supervisor"]:
    """Route after assess_complexity node."""
    if state.get("cached_findings_text"):
        return "triage"
    if state.get("current_phase") == PHASE_SUPERVISOR:
        return "supervisor"
    return "probe"


def _after_triage(
    state: DeepResearchState,
) -> Literal["context_answer", "plan", "probe"]:
    """Route after triage node."""
    decision = state.get("triage_decision", "full_research")
    if decision == "context_sufficient":
        return "context_answer"
    if decision == "partial_research":
        return "plan"
    return "probe"


def _after_plan(state: DeepResearchState) -> Literal["await_approval", "supervisor"]:
    """Route after plan node."""
    return "supervisor" if state.get("plan_approved") else "await_approval"


def _after_await_approval(state: DeepResearchState) -> str:
    """Route after await_approval node."""
    return "supervisor" if state.get("plan_approved") else "plan_rejected"


def _after_supervisor(
    state: DeepResearchState,
) -> Literal["completeness", "supervisor"]:
    """Route after supervisor node."""
    phase = state.get("current_phase", PHASE_COMPLETENESS)
    return "supervisor" if phase == PHASE_SUPERVISOR else "completeness"


def _after_completeness(
    state: DeepResearchState,
) -> Literal["synthesize", "supervisor"]:
    """Route after completeness node."""
    phase = state.get("current_phase", PHASE_SYNTHESIZE)
    return "supervisor" if phase == PHASE_SUPERVISOR else "synthesize"


def build_research_graph(
    ctx: ResearchContext,
    checkpointer: Any = None,
) -> CompiledStateGraph:
    """Build the deep research state machine graph."""
    from langgraph.graph import END, StateGraph

    probe = _wrap_node(probe_node, ctx, "probe")
    triage = _wrap_node(triage_node, ctx, "triage")
    plan = _wrap_node(plan_node, ctx, "plan")
    supervisor = _wrap_node(research_supervisor_node, ctx, "supervisor")
    completeness = _wrap_node(completeness_evaluator_node, ctx, "completeness")
    synthesize = _wrap_node(synthesize_node, ctx, "synthesize")
    visualize = _wrap_node(visualize_node, ctx, "visualize")
    review = _wrap_node(review_node, ctx, "review")
    complete = _wrap_node(complete_node, ctx, "complete")
    assess_complexity = _wrap_node(assess_complexity_node, ctx, "assess_complexity")

    def await_approval(state: DeepResearchState) -> dict[str, Any]:
        if state.get("plan_approved"):
            return {"current_phase": PHASE_SUPERVISOR, "_pending_events": []}
        return {"current_phase": PHASE_AWAIT_APPROVAL, "_pending_events": []}

    async def router(state: DeepResearchState) -> dict[str, Any]:
        _MIN_QUERY_LENGTH = 3
        _MAX_QUERY_LENGTH = 10000
        try:
            query = str(state.get("query") or "").strip()

            if len(query) < _MIN_QUERY_LENGTH:
                return {
                    "final_answer": "Please provide a more detailed question for deep research.",
                    "current_phase": PHASE_COMPLETE,
                    "_pending_events": [
                        emit_error("Query too short for deep research")
                    ],
                }

            if len(query) > _MAX_QUERY_LENGTH:
                return {
                    "final_answer": "Query is too long. Please shorten your question.",
                    "current_phase": PHASE_COMPLETE,
                    "_pending_events": [emit_error("Query exceeds maximum length")],
                }

            classification = await classify_input_quality(query, ctx.base_model)
            if classification == "gibberish":
                logger.info("Router: input classified as gibberish, short-circuiting")
                return {
                    "final_answer": GIBBERISH_RESPONSE,
                    "current_phase": PHASE_COMPLETE,
                    "_pending_events": [],
                }

            logger.info(f"Router: accepted query ({len(query)} chars)")
            return {
                "node_transitions": state.get("node_transitions", 0) + 1,
                "_pending_events": [emit_started()],
            }
        except Exception as e:
            return _node_error_updates(e)

    def plan_rejected(state: DeepResearchState) -> dict[str, Any]:
        from template_agent.src.core.deep_research.events import (
            DeepResearchEvent,
            DeepResearchEventType,
        )

        event = DeepResearchEvent(
            event_type=DeepResearchEventType.COMPLETED,
            message="Research plan was not approved. You can modify the plan and resubmit.",
            display_text="Plan rejected",
            details={"reason": "user_rejected"},
        )
        return {
            "final_answer": "The research plan was not approved. Please modify the plan and resubmit.",
            "current_phase": PHASE_COMPLETE,
            "_pending_events": [event.to_dict()],
        }

    context_answer = _wrap_node(context_answer_node, ctx, "context_answer")

    graph = StateGraph(DeepResearchState)
    graph.add_node("router", router)
    graph.add_node("assess_complexity", assess_complexity)
    graph.add_node("triage", triage)
    graph.add_node("probe", probe)
    graph.add_node("plan", plan)
    graph.add_node("await_approval", await_approval)
    graph.add_node("context_answer", context_answer)
    graph.add_node("supervisor", supervisor)
    graph.add_node("completeness", completeness)
    graph.add_node("synthesize", synthesize)
    graph.add_node("visualize", visualize)
    graph.add_node("review", review)
    graph.add_node("complete", complete)

    graph.add_conditional_edges("router", _after_router)
    graph.add_conditional_edges("assess_complexity", _after_assess_complexity)
    graph.add_conditional_edges("triage", _after_triage)
    graph.add_edge("context_answer", "review")
    graph.add_edge("probe", "plan")
    graph.add_conditional_edges("plan", _after_plan)
    graph.add_node("plan_rejected", plan_rejected)
    graph.add_conditional_edges("await_approval", _after_await_approval)
    graph.add_edge("plan_rejected", END)
    graph.add_conditional_edges("supervisor", _after_supervisor)
    graph.add_conditional_edges("completeness", _after_completeness)
    graph.add_edge("synthesize", "visualize")
    graph.add_edge("visualize", "review")
    graph.add_conditional_edges("review", _after_review_route)
    graph.add_edge("complete", END)
    graph.set_entry_point("router")

    return graph.compile(checkpointer=checkpointer)


def _get_thread_id(metadata: dict, configurable: dict) -> str | None:
    return metadata.get("thread_id") or configurable.get("thread_id")


def _get_plan_flags_from_metadata(metadata: dict) -> tuple[Any, bool, bool]:
    return (
        metadata.get("deep_research_plan"),
        metadata.get("deep_research_plan_approved", False),
        metadata.get("deep_research_require_plan_approval", True),
    )


def _get_metadata_and_configurable(config: Any) -> tuple[dict, dict]:
    if isinstance(config, dict):
        return (
            config.get("metadata", {}) or {},
            config.get("configurable", {}) or {},
        )
    return (
        (getattr(config, "metadata", None) or {}) if config else {},
        (getattr(config, "configurable", None) or {}) if config else {},
    )


def _event_fingerprint(evt: dict[str, Any]) -> int:
    content = evt.get("content", {})
    details = content.get("details", {}) if isinstance(content, dict) else {}
    if not isinstance(details, dict):
        details = {}
    key = (
        evt.get("type", ""),
        content.get("stage", "") if isinstance(content, dict) else "",
        content.get("event_type", "") if isinstance(content, dict) else "",
        str(details.get("idx", details.get("index", ""))),
        str(details.get("status", "")),
        str(details.get("subquery", ""))[:80] if isinstance(details, dict) else "",
        content.get("message", "")[:120] if isinstance(content, dict) else "",
    )
    return hash(key)


def _extract_msg_type_from_dict(msg: dict[str, Any]) -> str | None:
    return (
        msg.get("type")
        or msg.get("_type")
        or msg.get("message_type")
        or msg.get("lc_id", "").split("/")[-1]
    )


def _extract_content_from_dict(msg: dict[str, Any]) -> str:
    return (
        msg.get("content")
        or msg.get("text")
        or msg.get("message")
        or (msg.get("kwargs") or {}).get("content", "")
        or ""
    )


def _extract_msg_type_from_obj(msg: Any) -> str | None:
    msg_type = getattr(msg, "type", None) or getattr(msg, "_type", None)
    if msg_type:
        return msg_type
    if not hasattr(msg, "lc_id"):
        return None
    lc_id = getattr(msg, "lc_id", "")
    if "human" in lc_id.lower():
        return "human"
    if "ai" in lc_id.lower() and "tool" not in lc_id.lower():
        return "ai"
    return None


def _normalize_message_type(msg_type: str | None) -> str | None:
    if not msg_type or not isinstance(msg_type, str):
        return msg_type
    low = msg_type.lower()
    if "human" in low and "tool" not in low:
        return "human"
    if "ai" in low and "tool" not in low:
        return "ai"
    return None


async def _run_graph_get_next_item(
    output_queue: asyncio.Queue,
    heartbeat_interval: float,
) -> tuple[dict[str, Any] | None, bool]:
    try:
        item = await asyncio.wait_for(output_queue.get(), timeout=heartbeat_interval)
        return item, False
    except asyncio.TimeoutError:
        return None, True


async def _relay_events_to_output(
    event_queue: asyncio.Queue,
    output_queue: asyncio.Queue,
) -> None:
    while True:
        item = await event_queue.get()
        if item.get("_sentinel"):
            break
        await output_queue.put(item)


async def _run_graph_astream(
    graph: Any,
    current_state: dict[str, Any],
    output_queue: asyncio.Queue,
    event_queue: asyncio.Queue,
    node_output_prefix: str,
    sentinel: dict[str, Any],
    thread_id: str | None = None,
    callbacks: list | None = None,
    extra_config: dict[str, Any] | None = None,
) -> None:
    try:
        config: dict[str, Any] = {
            "recursion_limit": DEEP_RESEARCH_GRAPH_RECURSION_LIMIT,
        }
        configurable: dict[str, Any] = {}
        if thread_id:
            configurable["thread_id"] = thread_id
        if extra_config:
            if "configurable" in extra_config:
                configurable.update(extra_config["configurable"])
            for k, v in extra_config.items():
                if k not in ("configurable", "callbacks"):
                    config[k] = v
        if configurable:
            config["configurable"] = configurable
        if callbacks:
            config["callbacks"] = callbacks
        async for node_output in graph.astream(current_state, config=config):
            await output_queue.put({node_output_prefix: True, "data": node_output})
    except Exception as exc:
        await output_queue.put({node_output_prefix: True, "error": exc})
    finally:
        await event_queue.put(sentinel)
        await output_queue.put(sentinel)


def _extract_pending_events_and_clean_state(
    node_state: Any,
) -> tuple[list[dict[str, Any]], Any]:
    if isinstance(node_state, dict):
        pending = node_state.get("_pending_events", [])
        clean = {k: v for k, v in node_state.items() if k != "_pending_events"}
        return pending, clean
    return [], node_state


def _track_dedup_and_should_yield(
    fp: int, recent_hashes: set[int], max_size: int
) -> bool:
    if fp in recent_hashes:
        return False
    recent_hashes.add(fp)
    if len(recent_hashes) > max_size:
        recent_hashes.clear()
    return True


def _process_single_node_output(
    node_state: Any,
    current_state: dict[str, Any],
    recent_hashes: set[int],
    max_dedup_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    if node_state is None:
        return [], current_state, False
    pending_events, node_state_clean = _extract_pending_events_and_clean_state(
        node_state
    )
    events = []
    for event in pending_events:
        fp = _event_fingerprint(event) if isinstance(event, dict) else hash(str(event))
        if _track_dedup_and_should_yield(fp, recent_hashes, max_dedup_size):
            events.append(event)
    new_state = (
        {**current_state, **node_state_clean}
        if isinstance(node_state_clean, dict)
        else current_state
    )
    return events, new_state, bool(new_state.get("should_stop"))


def _reconcile_enrichment(
    plan: list[str],
    cached_enriched: list[dict],
) -> list[dict]:
    lookup: dict[str, dict] = {eq.get("query", ""): eq for eq in cached_enriched}
    result: list[dict] = []
    for sq in plan:
        cached = lookup.get(sq)
        if cached:
            result.append(cached)
        else:
            result.append({"query": sq, "status": "ready", "data_products": []})
    return result


def _get_starting_phase(
    skip_to_research: bool,
    plan_override: list[str] | None,
    cached_findings_text: str,
) -> str:
    if skip_to_research and plan_override:
        return PHASE_SUPERVISOR
    if cached_findings_text:
        return PHASE_TRIAGE
    return PHASE_PROBE


async def _get_restored_context(
    skip_to_research: bool,
    thread_id: str | None,
    user_id: str | None = None,
) -> dict:
    if not skip_to_research or not thread_id:
        return {}
    try:
        from template_agent.src.core.deep_research.plan_store import get_plan_context

        return await get_plan_context(thread_id, user_id=user_id) or {}
    except Exception:
        return {}


def _should_pause_for_plan_approval(
    event: dict[str, Any],
    require_approval: bool,
    plan_approved: bool,
) -> bool:
    if event.get("content", {}).get("stage") != "plan_pending":
        return False
    return require_approval and not plan_approved


async def create_initial_state(
    query: str,
    context: str = "",
    thread_id: str | None = None,
    plan_override: list[str] | None = None,
    plan_approved: bool = False,
    max_iterations: int = 3,
    max_rounds_override: int | None = None,
    max_iterations_override: int | None = None,
    skip_to_research: bool = False,
    cached_findings_text: str = "",
    enriched_subqueries: list[dict] | None = None,
    discovered_data_products: list[dict] | None = None,
    pre_plan_elapsed_seconds: float = 0.0,
    user_id: str | None = None,
) -> DeepResearchState:
    """Create the initial state for a deep research run."""
    starting_phase = _get_starting_phase(
        skip_to_research, plan_override, cached_findings_text
    )
    restored_context = await _get_restored_context(
        skip_to_research, thread_id, user_id=user_id
    )

    return {
        "query": query,
        "context": context,
        "thread_id": thread_id,
        "tool_inventory": restored_context.get("tool_inventory", ""),
        "tool_names": [],
        "probe_result": restored_context.get("probe_result", ""),
        "understanding": restored_context.get("understanding", ""),
        "subqueries": plan_override or [],
        "enriched_subqueries": enriched_subqueries or [],
        "discovered_data_products": discovered_data_products or [],
        "plan_approved": plan_approved or bool(plan_override),
        "plan_modified": False,
        "findings": {},
        "pending_subqueries": plan_override.copy() if plan_override else [],
        "completed_subqueries": [],
        "findings_board": {},
        "agent_messages": [],
        "supervisor_rounds": [],
        "current_round": 0,
        "max_rounds": max_rounds_override or DEEP_RESEARCH_MAX_SUPERVISOR_ROUNDS,
        "_user_max_rounds_override": max_rounds_override,
        "_user_max_iterations_override": max_iterations_override,
        "coverage_complete": False,
        "draft_answer": "",
        "synthesis_iteration": 0,
        "visualizations": [],
        "visualization_attempted": False,
        "review_results": [],
        "current_review": None,
        "final_answer": "",
        "current_phase": starting_phase,
        "iteration": 0,
        "max_iterations": max_iterations,
        "error": None,
        "should_stop": False,
        "triage_decision": None,
        "cached_findings_text": cached_findings_text,
        "assessed_max_subqueries": restored_context.get(
            "assessed_max_subqueries", DEEP_RESEARCH_MAX_SUBQUERIES
        ),
        "assessed_max_supervisor_rounds": restored_context.get(
            "assessed_max_supervisor_rounds", DEEP_RESEARCH_MAX_SUPERVISOR_ROUNDS
        ),
        "assessed_max_review_iterations": restored_context.get(
            "assessed_max_review_iterations", DEEP_RESEARCH_MAX_ITERATIONS
        ),
        "query_complexity_class": restored_context.get("query_complexity_class"),
        "complexity_reasoning": None,
        "total_node_transitions": 0,
        "research_start_time": time.time(),
        "execution_start_time": 0.0,
        "pre_plan_elapsed_seconds": pre_plan_elapsed_seconds,
        "human_wait_seconds": 0.0,
        "total_subqueries_executed": 0,
        "max_total_subqueries": DEEP_RESEARCH_MAX_TOTAL_SUBQUERIES,
        "max_node_transitions": restored_context.get(
            "max_node_transitions", DEEP_RESEARCH_MAX_NODE_TRANSITIONS
        ),
        "max_session_seconds": restored_context.get(
            "max_session_seconds", DEEP_RESEARCH_MAX_SESSION_SECONDS
        ),
        "sentinel_triggered": False,
        "sentinel_reason": None,
        "findings_count_history": [],
        "coverage_history": [],
        "research_abort_reason": None,
        "_pending_events": [],
    }


class DeepResearchAgent:
    """Agent that orchestrates hierarchical deep research with streaming."""

    def __init__(self, ctx: ResearchContext, checkpointer: Any = None):
        self.ctx = ctx
        self.checkpointer = checkpointer
        self.graph = build_research_graph(ctx, checkpointer=checkpointer)

    def _astream_extract_config(
        self, config: Any, kwargs: dict[str, Any]
    ) -> tuple[dict, dict, str | None, dict, Any, bool, bool]:
        config = config if config is not None else kwargs.get("config", {})
        metadata, configurable = _get_metadata_and_configurable(config)
        thread_id = _get_thread_id(metadata, configurable)
        run_config_metadata = self._extract_run_metadata_for_thread_listing(
            metadata, configurable
        )
        plan_override, plan_approved, require_approval = _get_plan_flags_from_metadata(
            metadata
        )
        return (
            metadata,
            configurable,
            thread_id,
            run_config_metadata,
            plan_override,
            plan_approved,
            require_approval,
        )

    def _astream_extract_query(self, input: dict[str, Any] | None) -> str:
        if input is None:
            return ""
        messages = input.get("messages", [])
        if not messages:
            return ""
        last_msg = messages[-1]
        if isinstance(last_msg, dict):
            return str(last_msg.get("content", "") or "")
        return str(getattr(last_msg, "content", "") or "")

    async def _astream_resolve_plan(
        self,
        thread_id: str | None,
        user_id: str | None,
        plan_override: Any,
        plan_approved: bool,
        metadata: dict,
    ) -> tuple[Any, bool]:
        stored_plan = (
            await get_plan_with_load(thread_id, None, user_id) if thread_id else None
        )
        if plan_override:
            if thread_id:
                await set_plan(thread_id, plan_override, store=None, user_id=user_id)
            return plan_override, True
        if stored_plan and metadata.get("deep_research_resume"):
            return stored_plan, True
        return plan_override, plan_approved

    async def _astream_load_enrichment(
        self, plan_override: Any, thread_id: str | None, user_id: str | None = None
    ) -> tuple[list[dict] | None, list[dict] | None]:
        if not plan_override or not thread_id:
            return None, None
        enrichment = await get_plan_enrichment(thread_id, user_id=user_id)
        if not enrichment:
            return None, None
        cached_eq = enrichment.get("enriched_subqueries", [])
        discovered_data_products = enrichment.get("discovered_data_products")
        enriched_subqueries = _reconcile_enrichment(plan_override, cached_eq)
        return enriched_subqueries, discovered_data_products

    async def _astream_merge_cross_chat_findings(
        self,
        cached_findings: dict[str, Any],
        user_id: str | None,
        query: str,
        thread_id: str | None,
    ) -> tuple[dict[str, Any], int, int]:
        if not DEEP_RESEARCH_CROSS_CHAT_FINDINGS_ENABLED:
            return cached_findings, 0, 0
        return cached_findings, 0, 0

    def _run_graph_process_queue_item(
        self,
        item: dict[str, Any] | None,
        current_state: dict[str, Any],
        state_ref: dict[str, dict[str, Any]],
        node_output_prefix: str,
        recent_hashes: set[int],
        max_dedup_size: int,
    ) -> tuple[list[dict[str, Any]], bool, bool]:
        if item is None:
            return [], False, False
        if item.get("_sentinel"):
            return [], False, True
        events, new_state, should_stop = self._run_graph_process_one_item(
            item, current_state, node_output_prefix, recent_hashes, max_dedup_size
        )
        state_ref["current"] = new_state
        return events, should_stop, False

    async def _run_graph_consume_loop(
        self,
        output_queue: asyncio.Queue,
        state_ref: dict[str, dict[str, Any]],
        heartbeat_interval: float,
        node_output_prefix: str,
        recent_hashes: set[int],
        max_dedup_size: int,
    ) -> AsyncGenerator[dict[str, Any]]:
        current_state = state_ref["current"]
        while True:
            item, should_heartbeat = await _run_graph_get_next_item(
                output_queue, heartbeat_interval
            )
            if should_heartbeat:
                yield emit_heartbeat()
                continue
            events, should_stop, is_sentinel = self._run_graph_process_queue_item(
                item,
                current_state,
                state_ref,
                node_output_prefix,
                recent_hashes,
                max_dedup_size,
            )
            current_state = state_ref["current"]
            if is_sentinel:
                break
            for evt in events:
                yield evt
            if should_stop:
                break

    def _run_graph_process_one_item(
        self,
        item: dict[str, Any],
        current_state: dict[str, Any],
        node_output_prefix: str,
        recent_hashes: set[int],
        max_dedup_size: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
        if item.get(node_output_prefix):
            if "error" in item:
                raise item["error"]
            node_output = item["data"]
            events, new_state, should_stop = self._run_graph_process_node_output(
                node_output, current_state, recent_hashes, max_dedup_size
            )
            return events, new_state, should_stop
        fp = _event_fingerprint(item)
        if not _track_dedup_and_should_yield(fp, recent_hashes, max_dedup_size):
            return [], current_state, False
        return [item], current_state, False

    def _run_graph_process_node_output(
        self,
        node_output: dict[str, Any],
        current_state: dict[str, Any],
        recent_hashes: set[int],
        max_dedup_size: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
        events: list[dict[str, Any]] = []
        state = dict(current_state)
        for node_name, node_state in node_output.items():
            node_events, state, should_stop = _process_single_node_output(
                node_state, state, recent_hashes, max_dedup_size
            )
            events.extend(node_events)
            if should_stop:
                return events, state, True
        return events, state, False

    def _extract_run_metadata_for_thread_listing(
        self, metadata: dict[str, Any], configurable: dict[str, Any]
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in (
            "user_id",
            "thread_title",
            "thread_created_at",
            "thread_updated_at",
            "project_id",
            "deep_research_model",
            "deep_research_max_mode",
        ):
            val = metadata.get(key) or configurable.get(key)
            if val is not None:
                out[key] = val
        return out

    async def _run_graph_emit_final_answer_block(
        self,
        current_state: dict[str, Any],
        thread_id: str | None,
        run_config_metadata: dict[str, Any] | None,
        final_answer_emitted: bool,
    ) -> tuple[dict[str, Any] | None, bool]:
        final_answer = str(current_state.get("final_answer") or "")
        if not final_answer or final_answer_emitted:
            return None, final_answer_emitted
        query_str = str(current_state.get("query") or "")
        raw_board = current_state.get("findings_board", {})
        findings_dict = findings_from_board(
            raw_board if isinstance(raw_board, dict) else {}
        )
        raw_viz = current_state.get("visualizations")
        viz_list = list(raw_viz) if isinstance(raw_viz, list) else []
        await self._persist_history(
            thread_id,
            query_str,
            final_answer,
            findings_dict,
            run_config_metadata,
            visualizations=viz_list if viz_list else None,
        )
        if thread_id:
            uid = run_config_metadata.get("user_id") if run_config_metadata else None
            await clear_plan(thread_id, user_id=uid)
        ai_message_content: dict[str, Any] = {"type": "ai", "content": final_answer}
        if viz_list:
            ai_message_content["additional_kwargs"] = {"visualizations": viz_list}
        return {"type": "message", "content": ai_message_content}, True

    @staticmethod
    def _extract_message_type_and_content(msg: Any) -> tuple[str | None, str]:
        if isinstance(msg, HumanMessage):
            content = getattr(msg, "content", "")
            return "human", str(content) if content else ""
        if isinstance(msg, AIMessage):
            content = getattr(msg, "content", "")
            return "ai", str(content) if content else ""
        if isinstance(msg, dict):
            msg_type = _normalize_message_type(_extract_msg_type_from_dict(msg))
            content = _extract_content_from_dict(msg)
        else:
            msg_type = _normalize_message_type(_extract_msg_type_from_obj(msg))
            content = getattr(msg, "content", "") or getattr(msg, "text", "") or ""
        return msg_type, content if isinstance(content, str) else ""

    def _format_messages_as_text(
        self, messages: list, *, truncate_user: int = 500
    ) -> list[str]:
        parts: list[str] = []
        for msg in messages:
            msg_type, content = self._extract_message_type_and_content(msg)
            if not content:
                continue
            if msg_type == "human":
                parts.append(f"{_CONTEXT_USER_PREFIX} {content[:truncate_user]}")
            elif msg_type == "ai":
                parts.append(f"{_CONTEXT_ASSISTANT_PREFIX} {content}")
        return parts

    async def _load_conversation_context(self, thread_id: str | None) -> str:
        if not self.checkpointer or not thread_id:
            return ""
        try:
            config = RunnableConfig(
                configurable={"thread_id": thread_id, "checkpoint_ns": ""}
            )
            raw_checkpointer = get_raw_checkpointer(self.checkpointer)
            result = await raw_checkpointer.aget_tuple(config)
            if not result or not result.checkpoint:
                return ""
            messages = result.checkpoint.get("channel_values", {}).get(
                "messages", result.checkpoint.get("messages", [])
            )
            if not messages:
                return ""
            return "\n".join(self._format_messages_as_text(messages))
        except Exception:
            return ""

    def _persist_prepare_messages(
        self,
        query: str,
        final_answer: str,
        visualizations: list[dict[str, Any]] | None,
        existing: Any,
        run_id: str,
    ) -> list:
        ai_kwargs: dict[str, Any] = {"run_id": run_id}
        if visualizations:
            ai_kwargs["visualizations"] = visualizations
        if existing and existing.checkpoint:
            return [AIMessage(content=final_answer, additional_kwargs=ai_kwargs)]
        return [
            HumanMessage(content=query, additional_kwargs={"run_id": run_id}),
            AIMessage(content=final_answer, additional_kwargs=ai_kwargs),
        ]

    async def _persist_history(
        self,
        thread_id: str | None,
        query: str,
        final_answer: str,
        findings: dict[str, Any],
        run_config_metadata: dict[str, Any] | None = None,
        visualizations: list[dict[str, Any]] | None = None,
    ) -> None:
        if not self.checkpointer or not thread_id or not query or not final_answer:
            return
        try:
            import uuid

            config = RunnableConfig(
                configurable={"thread_id": thread_id, "checkpoint_ns": ""}
            )
            existing = await self.checkpointer.aget_tuple(config)
            run_id = str(uuid.uuid4())
            messages = self._persist_prepare_messages(
                query, final_answer, visualizations, existing, run_id
            )
            if existing and existing.checkpoint:
                checkpoint = copy.deepcopy(existing.checkpoint)
                existing_msgs = checkpoint.get("channel_values", {}).get("messages", [])
                if "channel_values" not in checkpoint:
                    checkpoint["channel_values"] = {}
                checkpoint["channel_values"]["messages"] = existing_msgs + messages
                metadata = copy.deepcopy(existing.metadata) if existing.metadata else {}
            else:
                checkpoint = {
                    "v": 1,
                    "id": str(uuid.uuid4()),
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "channel_values": {"messages": messages},
                    "channel_versions": {},
                    "versions_seen": {},
                    "pending_sends": [],
                }
                metadata = {}
            metadata["deep_research_persisted"] = True
            if run_config_metadata:
                for k, v in run_config_metadata.items():
                    if v is not None:
                        metadata[k] = v
            metadata["thread_updated_at"] = datetime.now(timezone.utc).isoformat()
            await aput_checkpoint(self.checkpointer, config, checkpoint, metadata)
            findings_list = list(findings.values())
            if findings_list:
                await save_cached_findings(self.checkpointer, thread_id, findings_list)
        except Exception as e:
            logger.error("Failed to persist history: %s", e)

    async def _ensure_checkpoint_exists(
        self,
        thread_id: str | None,
        query: str,
        run_config_metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.checkpointer or not thread_id:
            return
        try:
            import uuid

            config = RunnableConfig(
                configurable={"thread_id": thread_id, "checkpoint_ns": ""}
            )
            existing = await self.checkpointer.aget_tuple(config)
            if existing and existing.checkpoint:
                return
            checkpoint = {
                "v": 1,
                "id": str(uuid.uuid4()),
                "ts": datetime.now(timezone.utc).isoformat(),
                "channel_values": {"messages": [HumanMessage(content=query)]},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            }
            metadata: dict[str, Any] = {"deep_research_plan_pending": True}
            if run_config_metadata:
                for k, v in run_config_metadata.items():
                    if v is not None:
                        metadata[k] = v
            await aput_checkpoint(self.checkpointer, config, checkpoint, metadata)
        except Exception as exc:
            logger.warning(
                "Failed to ensure checkpoint for thread %s: %s", thread_id, exc
            )

    async def astream(
        self,
        input: dict[str, Any] | None = None,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> AsyncGenerator[tuple[str, dict[str, Any]]]:
        """Stream deep research execution with events."""
        import time as _time

        (
            metadata,
            _,
            thread_id,
            run_config_metadata,
            plan_override,
            plan_approved,
            require_approval,
        ) = self._astream_extract_config(config, kwargs)
        query = self._astream_extract_query(input)
        if not query:
            yield ("custom", emit_error("No query provided"))
            return

        user_id = run_config_metadata.get("user_id") if run_config_metadata else None
        plan_override, plan_approved = await self._astream_resolve_plan(
            thread_id, user_id, plan_override, plan_approved, metadata
        )

        (
            enriched_subqueries,
            discovered_data_products,
        ) = await self._astream_load_enrichment(
            plan_override, thread_id, user_id=user_id
        )

        context = await self._load_conversation_context(thread_id)
        cached_findings = await load_cached_findings(self.checkpointer, thread_id)
        (
            cached_findings,
            cross_chat_count,
            cross_chat_thread_count,
        ) = await self._astream_merge_cross_chat_findings(
            cached_findings, user_id, query, thread_id
        )

        cached_findings_text = await select_relevant_findings(
            self.ctx.base_model, query, cached_findings
        )

        stream_start_time = _time.time()
        yield ("custom", emit_started())
        event = _build_context_loaded_event(context)
        yield ("custom", event)
        if cross_chat_count > 0:
            yield (
                "custom",
                emit_cross_chat_findings_loaded(
                    count=cross_chat_count,
                    source_thread_count=cross_chat_thread_count,
                ),
            )

        is_resume = bool(plan_override and plan_approved)
        pre_plan_elapsed = float(
            metadata.get("deep_research_pre_plan_elapsed_seconds", 0.0)
        )
        max_rounds_override = metadata.get("deep_research_max_supervisor_rounds")
        max_iterations_override = metadata.get("deep_research_max_review_iterations")
        state = await create_initial_state(
            query=query,
            context=context,
            thread_id=thread_id,
            plan_override=plan_override,
            plan_approved=plan_approved or not require_approval,
            max_iterations=(
                max_iterations_override
                if max_iterations_override is not None
                else DEEP_RESEARCH_MAX_ITERATIONS
            ),
            max_rounds_override=max_rounds_override,
            max_iterations_override=max_iterations_override,
            skip_to_research=is_resume,
            cached_findings_text=cached_findings_text,
            enriched_subqueries=enriched_subqueries,
            discovered_data_products=discovered_data_products,
            pre_plan_elapsed_seconds=pre_plan_elapsed,
            user_id=self.ctx.user_id,
        )

        try:
            async for event in self._run_graph_with_events(
                state, thread_id, run_config_metadata
            ):
                yield ("custom", event)
                if _should_pause_for_plan_approval(
                    event, require_approval, plan_approved
                ):
                    pre_plan_secs = round(_time.time() - stream_start_time, 2)
                    yield (
                        "custom",
                        emit_event(
                            DeepResearchEventType.PLAN_PENDING,
                            "Waiting for plan approval",
                            "Execution paused — waiting for plan approval",
                            details={"pre_plan_elapsed_seconds": pre_plan_secs},
                            stage="plan_pending",
                            ui_visible=False,
                        ),
                    )
                    await self._ensure_checkpoint_exists(
                        thread_id, query, run_config_metadata
                    )
                    return

        except Exception as e:
            logger.error("Deep research failed: %s", e, exc_info=True)
            yield ("custom", emit_error(sanitize_error_for_client(e)))

    async def _run_graph_with_events(
        self,
        initial_state: DeepResearchState,
        thread_id: str | None,
        run_config_metadata: dict[str, Any] | None = None,
        run_config: dict[str, Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any]]:
        current_state = dict(initial_state)
        event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=DEEP_RESEARCH_EVENT_QUEUE_MAXSIZE
        )
        self.ctx.event_queue = event_queue
        sentinel = {"_sentinel": True}
        node_output_prefix = "_node_output"
        output_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=DEEP_RESEARCH_EVENT_QUEUE_MAXSIZE
        )
        recent_hashes: set[int] = set()
        state_ref: dict[str, dict[str, Any]] = {"current": current_state}

        relay_task = asyncio.create_task(
            _relay_events_to_output(event_queue, output_queue)
        )
        from template_agent.utils.tracing import langfuse_handler as _lf_handler

        callbacks = [_lf_handler] if _lf_handler else None
        graph_task = asyncio.create_task(
            _run_graph_astream(
                self.graph,
                current_state,
                output_queue,
                event_queue,
                node_output_prefix,
                sentinel,
                thread_id=thread_id,
                callbacks=callbacks,
                extra_config=run_config,
            )
        )
        final_answer_emitted = False
        try:
            async for evt in self._run_graph_consume_loop(
                output_queue,
                state_ref,
                DEEP_RESEARCH_HEARTBEAT_INTERVAL_SECONDS,
                "_node_output",
                recent_hashes,
                500,
            ):
                yield evt
            current_state = state_ref["current"]
            event, final_answer_emitted = await self._run_graph_emit_final_answer_block(
                current_state, thread_id, run_config_metadata, final_answer_emitted
            )
            if event:
                yield event
        except Exception as e:
            logger.error("Graph execution error: %s", e, exc_info=True)
            yield emit_error(sanitize_error_for_client(e))
        finally:
            for task in (relay_task, graph_task):
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
            thread_id_for_cancel = initial_state.get("thread_id", "")
            if thread_id_for_cancel:
                cancel_store = get_cancel_store()
                await cancel_store.clear(thread_id_for_cancel)
            self.ctx.event_queue = None


@asynccontextmanager
async def get_deep_research_agent(
    user_id: str | None = None,
    model_override: str | None = None,
    model_name: str | None = None,
    max_subqueries_override: int | None = None,
    max_mode: bool = False,
    root_tracer: Any = None,
    checkpointer: Any = None,
    event_queue: Any = None,
    token_tracker: Any = None,
) -> AsyncGenerator[DeepResearchAgent]:
    """Get an initialized DeepResearchAgent."""
    effective_model = model_name or model_override
    async with get_research_context(
        model_name=effective_model,
        checkpointer=checkpointer,
        user_id=user_id,
        max_subqueries_override=max_subqueries_override,
        max_mode=max_mode,
        token_tracker=token_tracker,
        event_queue=event_queue,
    ) as ctx:
        if root_tracer is not None:
            ctx.root_tracer = root_tracer
        agent = DeepResearchAgent(ctx=ctx, checkpointer=ctx.checkpointer)
        yield agent
