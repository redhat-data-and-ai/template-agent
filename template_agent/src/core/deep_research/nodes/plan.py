"""Plan node: generate and validate research subqueries."""

import asyncio
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage

from template_agent.src.core.deep_research.events import (
    emit_agent_decision,
    emit_agent_thinking,
    emit_plan_generated,
    emit_plan_pending_enriched,
    emit_subquery_validation,
    emit_understanding,
)
from template_agent.src.core.deep_research.prompts import (
    PLAN_REVIEWER_PERSONAS,
    QueryType,
    build_plan_review_prompt,
    build_planning_prompt,
    build_query_type_detection_prompt,
    build_subquery_validation_prompt,
    build_understanding_prompt,
)
from template_agent.src.core.deep_research.state import (
    PHASE_AWAIT_APPROVAL,
    DeepResearchState,
    ResearchContext,
)
from template_agent.src.core.deep_research.token_tracker import tracked_invoke
from template_agent.src.core.utils import (
    is_model_config_error,
    safe_json_parse,
    simplify_error_for_display,
    truncate_text,
)
from template_agent.utils.pylogger import get_python_logger

from ._cache import (
    find_matching_cached_finding,
    format_cached_findings_for_prompt,
    hash_subquery,
    load_cached_findings,
)
from ._helpers import (
    _format_enriched_plan,
    _is_identity_subquery,
    _parse_subqueries,
    _strip_sql_from_subqueries,
)

logger = get_python_logger()


async def _load_similar_plans_context(
    ctx: ResearchContext,
    query: str,
    events: List[Dict[str, Any]],
) -> str:
    """Load similar past plans for context. Returns context string."""
    try:
        from template_agent.src.core.deep_research.plan_store import load_similar_plans
        from template_agent.src.core.persistence.memory import (
            get_contextual_memories,
            get_global_memory_store,
        )
    except ImportError:
        return ""

    try:
        store = await get_global_memory_store()
        if not store or not ctx.user_id:
            return ""

        similar_plans = await load_similar_plans(store, ctx.user_id, query, limit=2)
        similar_plans_context = ""
        if similar_plans:
            plans_text = "\n".join(
                f"- Query: {p.get('query', '')[:100]} -> {len(p.get('plan', []))} subqueries"
                for p in similar_plans
            )
            similar_plans_context = f"\n\nSimilar past research:\n{plans_text}"
            ctx.emit_or_append(
                emit_agent_thinking(
                    "Planner",
                    f"Found {len(similar_plans)} similar past research sessions for reference",
                ),
                events,
            )

        memory_namespace = ("memory", ctx.user_id)
        memories = await get_contextual_memories(
            store, memory_namespace, query, limit=3
        )
        if memories:
            similar_plans_context += f"\n\nUser context:\n{memories[:500]}"
        return similar_plans_context
    except Exception as e:
        logger.warning("Failed to load memory context: %s", e)
        return ""


async def _run_understanding(
    ctx: ResearchContext,
    query: str,
    context: str,
    similar_plans_context: str,
    events: List[Dict[str, Any]],
) -> str:
    """Run query understanding step. Returns understanding string."""
    understanding_prompt = build_understanding_prompt()
    understanding_messages = understanding_prompt.format_messages(
        query=query,
        context=context + similar_plans_context,
    )
    try:
        response = await tracked_invoke(
            ctx.base_model,
            understanding_messages,
            ctx.token_tracker,
            "planning",
            **ctx.llm_call_kwargs(),
        )
        understanding = str(response.content or "").strip()
        ctx.emit_or_append(
            emit_agent_decision(
                "Planner", "Query intent analyzed", truncate_text(understanding, 200)
            ),
            events,
        )
        return understanding
    except Exception as e:
        understanding = f"Query analysis failed: {simplify_error_for_display(str(e))}"
        logger.warning("Query understanding failed: %s", e)
        return understanding


def _compute_subquery_bounds(
    state: DeepResearchState,
    ctx: ResearchContext,
) -> tuple[int, int, int]:
    """Compute min_count, max_count, recommended for subquery generation."""
    is_partial = state.get("triage_decision") == "partial_research"
    mode = ctx.mode_config
    assessed_max = state.get("assessed_max_subqueries", 0)
    if mode:
        min_count = 1 if is_partial else mode.min_subqueries
        max_count = mode.max_subqueries
        if assessed_max > 0:
            recommended = min(assessed_max, mode.recommended_subqueries)
        else:
            recommended = mode.recommended_subqueries
        recommended = max(min_count, min(recommended, max_count))
    else:
        min_count = 1 if is_partial else 3
        max_count = assessed_max if assessed_max > 0 else 7
        recommended = max(min_count, min(assessed_max or 5, max_count))
    return min_count, max_count, recommended


def _build_planning_context(
    context: str,
    cached_context: str,
    is_partial: bool,
) -> str:
    """Build planning context with cached findings."""
    planning_context = context
    if not cached_context:
        return planning_context
    if is_partial:
        planning_context += (
            "\n\n⚠️ PARTIAL RESEARCH MODE — Generate subqueries ONLY "
            "for information NOT already available in the cached findings "
            "below. Do NOT re-query data that has already been gathered.\n"
            f"Previously answered:\n{cached_context}"
        )
    else:
        planning_context += f"\n\nPreviously answered:\n{cached_context}"
    return planning_context


def _resolve_empty_subqueries(subqueries: List[str], query: str) -> List[str]:
    """Return fallback when subqueries list is empty."""
    if subqueries:
        return subqueries
    return [f"Research question: {query}"]


async def _postprocess_raw_subqueries(
    ctx: ResearchContext,
    query: str,
    subqueries: List[str],
    understanding: str,
    min_count: int,
) -> List[str]:
    """Filter identity, expand if needed, strip SQL, resolve empty."""
    original_count = len(subqueries)
    subqueries = [sq for sq in subqueries if not _is_identity_subquery(sq, query)]
    if len(subqueries) < original_count:
        logger.info(
            "Removed %d identity subqueries (restated original query)",
            original_count - len(subqueries),
        )

    if len(subqueries) <= 1 or (subqueries and len(subqueries) < min_count):
        subqueries = await _expand_subqueries(
            ctx, query, subqueries, understanding, min_count
        )

    subqueries = _strip_sql_from_subqueries(subqueries)
    return _resolve_empty_subqueries(subqueries, query)


async def _generate_subqueries(
    ctx: ResearchContext,
    query: str,
    understanding: str,
    planning_context: str,
    available_resources: str,
    min_count: int,
    max_count: int,
    recommended: int,
) -> List[str]:
    """Generate subqueries via LLM."""
    planning_prompt = build_planning_prompt()
    mode_planning_instruction = (
        ctx.mode_config.planning_instruction if ctx.mode_config else ""
    )
    planning_messages = planning_prompt.format_messages(
        query=query,
        context=planning_context,
        available_resources=f"Available tools:\n{available_resources}",
        understanding=understanding,
        min_count=min_count,
        max_count=max_count,
        recommended_count=recommended,
        mode_instruction=mode_planning_instruction,
    )
    try:
        response = await tracked_invoke(
            ctx.base_model,
            planning_messages,
            ctx.token_tracker,
            "planning",
            **ctx.llm_call_kwargs(),
        )
        response_text = str(response.content or "")
        subqueries = _parse_subqueries(response_text)
        return await _postprocess_raw_subqueries(
            ctx, query, subqueries, understanding, min_count
        )
    except Exception as e:
        logger.warning("Subquery generation failed: %s", e)
        return [query]


def _apply_cache_matching(
    enriched_dicts: List[Dict[str, Any]],
    cached_findings: Dict[str, Any],
    ctx: ResearchContext,
    events: List[Dict[str, Any]],
) -> None:
    """Tag each enriched subquery as cached or new."""
    cached_findings_dict = {}
    if cached_findings:
        cached_findings_dict = {
            hash_subquery(f.get("subquery", "")): f for f in cached_findings.values()
        }
    for ed in enriched_dicts:
        sq_text = str(ed.get("query", ""))
        match = (
            find_matching_cached_finding(sq_text, cached_findings_dict)
            if cached_findings_dict
            else None
        )
        if match:
            ed["source"] = "cached"
            ed["cached_finding_key"] = hash_subquery(
                str(match.get("subquery") or sq_text)
            )
        else:
            ed["source"] = "new"


def _format_resources_summary(tool_inventory: str) -> str:
    """Format tool inventory for validation prompt."""
    return tool_inventory or "No tools available."


def _process_validation_entries(
    validated: List[Dict[str, Any]],
    subqueries: List[str],
    enriched_dicts: List[Dict[str, Any]],
) -> tuple[List[str], List[Dict[str, Any]], int, int, int] | None:
    """Process validated entries into new subqueries and enriched dicts."""
    new_subqueries: List[str] = []
    new_enriched: List[Dict[str, Any]] = []
    valid_count = 0
    reformulated_count = 0
    removed_count = 0

    for i, entry in enumerate(validated):
        status = entry.get("status", "answerable")
        if status == "removed":
            removed_count += 1
            continue
        if status == "reformulated" and entry.get("reformulated"):
            new_query = entry["reformulated"]
            reformulated_count += 1
        else:
            new_query = entry.get(
                "original", subqueries[i] if i < len(subqueries) else ""
            )
            valid_count += 1
        new_subqueries.append(new_query)
        if i < len(enriched_dicts):
            enriched = dict(enriched_dicts[i])
            enriched["query"] = new_query
            new_enriched.append(enriched)

    if not new_subqueries:
        return None
    return new_subqueries, new_enriched, valid_count, reformulated_count, removed_count


async def _validate_subqueries(
    ctx: ResearchContext,
    subqueries: List[str],
    enriched_dicts: List[Dict[str, Any]],
    resources_summary: str,
    state: DeepResearchState,
    events: List[Dict[str, Any]],
) -> tuple[List[str], List[Dict[str, Any]]]:
    """Validate subqueries against available resources and tools."""
    if not subqueries:
        return subqueries, enriched_dicts

    sq_text = "\n".join(f"{i + 1}. {sq}" for i, sq in enumerate(subqueries))
    try:
        validation_prompt = build_subquery_validation_prompt()
        messages = validation_prompt.format_messages(
            subqueries=sq_text,
            available_resources=resources_summary,
            tool_inventory=state.get("tool_inventory", ""),
        )
        response = await tracked_invoke(
            ctx.base_model,
            messages,
            ctx.token_tracker,
            "planning",
            **ctx.llm_call_kwargs(),
        )
        result = safe_json_parse(str(response.content or ""))

        if not result or "validated_subqueries" not in result:
            logger.warning(
                "Subquery validation returned unparseable result, keeping originals"
            )
            return subqueries, enriched_dicts

        validated = result["validated_subqueries"]
        processed = _process_validation_entries(validated, subqueries, enriched_dicts)
        if processed is None:
            logger.warning(
                "All subqueries were removed by validation, keeping originals"
            )
            return subqueries, enriched_dicts

        new_subqueries, new_enriched, valid_count, reformulated_count, removed_count = (
            processed
        )
        events.append(
            emit_subquery_validation(
                total_subqueries=len(validated),
                valid_count=valid_count,
                reformulated_count=reformulated_count,
                removed_count=removed_count,
            )
        )
        return new_subqueries, new_enriched

    except Exception as e:
        logger.warning("Subquery validation failed: %s, keeping originals", e)
        return subqueries, enriched_dicts


def _add_redundant_indices_from_result(
    result: Dict[str, Any],
    subqueries: List[str],
    redundant_indices: set[int],
) -> None:
    """Extract redundant subquery indices from review result."""
    for idx in result.get("redundant_subqueries", []):
        if isinstance(idx, int) and 0 <= idx < len(subqueries):
            redundant_indices.add(idx)
        elif isinstance(idx, str) and idx.isdigit():
            idx_int = int(idx) - 1
            if 0 <= idx_int < len(subqueries):
                redundant_indices.add(idx_int)


def _apply_plan_improvements(
    subqueries: List[str],
    enriched_dicts: List[Dict[str, Any]],
    missing_subqueries: List[str],
    redundant_indices: set[int],
    max_subqueries: int,
    events: List[Dict[str, Any]],
) -> tuple[List[str], List[Dict[str, Any]]]:
    """Apply redundant removal and missing additions."""
    if redundant_indices and len(subqueries) - len(redundant_indices) >= 2:
        subqueries = [
            sq for i, sq in enumerate(subqueries) if i not in redundant_indices
        ]
        enriched_dicts = [
            ed for i, ed in enumerate(enriched_dicts) if i not in redundant_indices
        ]
        events.append(
            emit_agent_thinking(
                "Planner",
                f"Removed {len(redundant_indices)} redundant subqueries",
            )
        )

    for new_sq in missing_subqueries[:3]:
        if len(subqueries) >= max_subqueries:
            break
        if new_sq not in subqueries:
            subqueries.append(new_sq)
            enriched_dicts.append(
                {"query": new_sq, "data_products": [], "status": "ready"}
            )
            events.append(
                emit_agent_thinking("Planner", f"Added missing subquery: {new_sq}")
            )
    return subqueries, enriched_dicts


async def _run_persona_reviews(
    ctx: ResearchContext,
    query: str,
    understanding: str,
    resources_text: str,
    sq_text: str,
    subqueries: List[str],
    events: List[Dict[str, Any]],
) -> tuple[List[int], List[str], set[int]]:
    """Run all persona reviews."""
    review_prompt = build_plan_review_prompt()
    missing_subqueries: List[str] = []
    redundant_indices: set[int] = set()
    scores: List[int] = []
    _MAX_CONFIG_FAILURES = 2

    personas_info = [(pc["persona"], pc["focus"]) for pc in PLAN_REVIEWER_PERSONAS]

    for persona, focus in personas_info:
        events.append(
            emit_agent_thinking(
                f"PlanReviewer:{persona}",
                f"Reviewing plan with focus on: {focus}",
            )
        )

    async def _invoke_one(persona: str, focus: str) -> dict | None:
        messages = review_prompt.format_messages(
            persona=persona,
            focus=focus,
            query=query,
            understanding=understanding,
            available_resources=resources_text,
            subqueries=sq_text,
        )
        response = await tracked_invoke(
            ctx.base_model,
            messages,
            ctx.token_tracker,
            "planning",
            **ctx.llm_call_kwargs(),
        )
        review_text = str(response.content or "")
        return safe_json_parse(review_text)

    results = await asyncio.gather(
        *(_invoke_one(p, f) for p, f in personas_info),
        return_exceptions=True,
    )

    consecutive_config_failures = 0
    for (persona, _focus), result in zip(personas_info, results):
        if consecutive_config_failures >= _MAX_CONFIG_FAILURES:
            scores.append(70)
            continue
        if isinstance(result, BaseException):
            if is_model_config_error(result):
                consecutive_config_failures += 1
            else:
                consecutive_config_failures = 0
            logger.warning("Plan review by %s failed: %s", persona, result)
            scores.append(70)
            continue
        consecutive_config_failures = 0
        if isinstance(result, dict):
            score = result.get("score", 70)
            scores.append(score)
            missing_subqueries.extend(result.get("missing_subqueries", []))
            _add_redundant_indices_from_result(result, subqueries, redundant_indices)
            events.append(
                emit_agent_decision(
                    f"PlanReviewer:{persona}",
                    f"Score: {score}/100",
                    f"Issues: {len(result.get('issues', []))}, Suggestions: {len(result.get('suggestions', []))}",
                )
            )
        else:
            scores.append(70)

    return scores, missing_subqueries, redundant_indices


async def _review_research_plan(
    ctx: ResearchContext,
    query: str,
    understanding: str,
    subqueries: List[str],
    enriched_dicts: List[Dict[str, Any]],
) -> tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Review research plan with multiple personas before execution."""
    events: List[Dict[str, Any]] = []

    if not subqueries:
        return subqueries, enriched_dicts, events

    resources_text = ctx.format_tool_inventory() or "No tools available."
    sq_text = "\n".join(f"{i + 1}. {sq}" for i, sq in enumerate(subqueries))

    scores, missing_subqueries, redundant_indices = await _run_persona_reviews(
        ctx, query, understanding, resources_text, sq_text, subqueries, events
    )

    avg_score = sum(scores) // len(scores) if scores else 70
    max_subqueries = ctx.mode_config.max_subqueries if ctx.mode_config else 20

    if avg_score < 75 or missing_subqueries:
        subqueries, enriched_dicts = _apply_plan_improvements(
            subqueries,
            enriched_dicts,
            missing_subqueries,
            redundant_indices,
            max_subqueries,
            events,
        )

    events.append(
        emit_agent_decision(
            "PlanReviewCoordinator",
            f"Plan review complete: {avg_score}/100",
            f"Final plan has {len(subqueries)} subqueries",
        )
    )

    return subqueries, enriched_dicts, events


async def _detect_query_type(
    ctx: ResearchContext,
    query: str,
    understanding: str,
) -> tuple[str, float]:
    """Detect the type of research query for dynamic answer formatting."""
    try:
        prompt = build_query_type_detection_prompt()
        messages = prompt.format_messages(query=query, understanding=understanding)

        response = await tracked_invoke(
            ctx.base_model,
            messages,
            ctx.token_tracker,
            "planning",
            **ctx.llm_call_kwargs(),
        )
        response_text = str(response.content or "").strip()

        result = safe_json_parse(response_text, pattern=r"\{[^{}]*\}")
        if isinstance(result, dict):
            raw_type = result.get("query_type", "COMPREHENSIVE").upper()
            confidence = float(result.get("confidence", 0.5))

            try:
                query_type = QueryType(raw_type.lower())
                return query_type.value, min(max(confidence, 0.0), 1.0)
            except ValueError:
                logger.warning(
                    "Unknown query type '%s', defaulting to COMPREHENSIVE", raw_type
                )
                return QueryType.COMPREHENSIVE.value, 0.5

        logger.warning("Could not parse query type detection response")
        return QueryType.COMPREHENSIVE.value, 0.5

    except Exception as e:
        logger.warning("Query type detection failed: %s", e)
        return QueryType.COMPREHENSIVE.value, 0.5


async def _expand_subqueries(
    ctx: ResearchContext,
    original_query: str,
    current_subqueries: List[str],
    understanding: str,
    target_count: int,
) -> List[str]:
    """Expand subqueries to meet target count."""
    numbered_queries = "\n".join(
        f"{i + 1}. {sq}" for i, sq in enumerate(current_subqueries)
    )
    expansion_prompt = f"""You have generated {len(current_subqueries)} subqueries, but we need at least {target_count}.

ORIGINAL QUERY: {original_query}

CURRENT SUBQUERIES:
{numbered_queries}

QUERY UNDERSTANDING:
{understanding}

TASK: Expand these into {target_count} or more SPECIFIC, FOCUSED subqueries.

Return JSON only: {{"subqueries": ["query 1", "query 2", "query 3", ...]}}"""

    try:
        response = await tracked_invoke(
            ctx.base_model,
            [HumanMessage(content=expansion_prompt)],
            ctx.token_tracker,
            "planning",
            **ctx.llm_call_kwargs(),
        )
        response_text = str(response.content or "")
        expanded = _parse_subqueries(response_text)

        if expanded and len(expanded) >= target_count:
            return expanded
        elif expanded and len(expanded) > len(current_subqueries):
            return expanded
    except Exception as e:
        logger.warning("Subquery expansion failed: %s", e)

    return current_subqueries


async def plan_node(
    state: DeepResearchState,
    ctx: ResearchContext,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Plan node: Analyze query -> generate subqueries -> validate -> present plan.

    Simplified flow (no data product discovery):
    1. Analyze query intent
    2. Generate subqueries with knowledge of available tools
    3. Validate subqueries
    4. Review plan
    5. Present for approval

    Returns:
        Tuple of (state_updates, events)
    """
    events: list[dict[str, Any]] = []
    query = state.get("query", "")
    context = state.get("context", "") or "None."
    thread_id = state.get("thread_id")

    ctx.emit_or_append(
        emit_agent_thinking(
            "Planner",
            f"Analyzing user query: '{truncate_text(query, 100)}'",
        ),
        events,
    )

    similar_plans_context = await _load_similar_plans_context(ctx, query, events)
    understanding = await _run_understanding(
        ctx, query, context, similar_plans_context, events
    )
    ctx.emit_or_append(emit_understanding(understanding), events)

    query_type, query_type_confidence = await _detect_query_type(
        ctx, query, understanding
    )
    ctx.emit_or_append(
        emit_agent_decision(
            "Planner",
            f"Query type: {query_type}",
            f"Confidence: {query_type_confidence:.0%}",
        ),
        events,
    )

    available_resources = ctx.format_tool_inventory()
    cached_findings = await load_cached_findings(
        ctx.checkpointer, state.get("thread_id")
    )
    cached_context = format_cached_findings_for_prompt(cached_findings)

    min_count, max_count, recommended = _compute_subquery_bounds(state, ctx)
    is_partial = state.get("triage_decision") == "partial_research"
    planning_context = _build_planning_context(context, cached_context, is_partial)

    subqueries = await _generate_subqueries(
        ctx,
        query,
        understanding,
        planning_context,
        available_resources,
        min_count,
        max_count,
        recommended,
    )

    enriched_dicts = [
        {"query": sq, "data_products": [], "status": "ready"} for sq in subqueries
    ]

    ctx.emit_or_append(
        emit_agent_thinking(
            "Planner",
            f"Validating {len(subqueries)} subqueries against available tools...",
        ),
        events,
    )
    subqueries, enriched_dicts = await _validate_subqueries(
        ctx,
        subqueries,
        enriched_dicts,
        available_resources,
        state,
        events,
    )

    ctx.emit_or_append(
        emit_agent_thinking(
            "Planner",
            "Validating research plan with review team...",
        ),
        events,
    )
    subqueries, enriched_dicts, plan_review_events = await _review_research_plan(
        ctx,
        query,
        understanding,
        subqueries,
        enriched_dicts,
    )
    events.extend(plan_review_events)

    _apply_cache_matching(enriched_dicts, cached_findings, ctx, events)

    enriched_display = _format_enriched_plan(enriched_dicts)

    events.append(emit_plan_generated(subqueries))
    events.append(
        emit_plan_pending_enriched(
            enriched_dicts,
            discovered_products=[],
            understanding=understanding,
        )
    )

    return {
        "understanding": understanding,
        "query_type": query_type,
        "query_type_confidence": query_type_confidence,
        "subqueries": subqueries,
        "enriched_subqueries": enriched_dicts,
        "discovered_data_products": [],
        "pending_subqueries": subqueries.copy(),
        "completed_subqueries": [],
        "current_phase": PHASE_AWAIT_APPROVAL,
    }, events
