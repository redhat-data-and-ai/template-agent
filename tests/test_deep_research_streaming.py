"""Comprehensive pytest tests for the deep research streaming module."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from template_agent.src.core.deep_research.state import (
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
from template_agent.src.core.deep_research.streaming import (
    DeepResearchAgent,
    _after_assess_complexity,
    _after_await_approval,
    _after_completeness,
    _after_plan,
    _after_review_route,
    _after_router,
    _after_supervisor,
    _after_triage,
    _build_context_loaded_event,
    _build_indexed_summaries,
    _event_fingerprint,
    _extract_content_from_dict,
    _extract_msg_type_from_dict,
    _extract_msg_type_from_obj,
    _extract_pending_events_and_clean_state,
    _format_selected_findings,
    _get_metadata_and_configurable,
    _get_plan_flags_from_metadata,
    _get_restored_context,
    _get_starting_phase,
    _get_thread_id,
    _node_error_updates,
    _normalize_message_type,
    _parse_valid_llm_indices,
    _process_single_node_output,
    _reconcile_enrichment,
    _relay_events_to_output,
    _run_graph_get_next_item,
    _should_pause_for_plan_approval,
    _span_end_error,
    _span_end_ok,
    _span_open,
    _track_dedup_and_should_yield,
    _wrap_node,
    build_research_graph,
    create_initial_state,
    select_relevant_findings,
)


class TestCreateInitialState:
    """Test create_initial_state function."""

    @pytest.mark.asyncio
    async def test_create_initial_state_minimal_query_sets_defaults(self):
        """Minimal query produces state with default values."""
        state = await create_initial_state(query="What is AI?")
        assert state["query"] == "What is AI?"
        assert state["context"] == ""
        assert state["thread_id"] is None
        assert state["current_phase"] == PHASE_PROBE
        assert state["findings"] == {}
        assert state["plan_approved"] is False
        assert state["iteration"] == 0
        assert state["max_iterations"] == 3

    @pytest.mark.asyncio
    async def test_create_initial_state_with_context(self):
        """Context is passed through to state."""
        state = await create_initial_state(
            query="Explain ML",
            context="User: prior question\nAssistant: prior answer",
        )
        assert state["context"] == "User: prior question\nAssistant: prior answer"

    @pytest.mark.asyncio
    async def test_create_initial_state_with_plan_override(self):
        """Plan override with skip_to_research sets subqueries and supervisor phase."""
        plan = ["subq1", "subq2"]
        state = await create_initial_state(
            query="Research X",
            plan_override=plan,
            plan_approved=True,
            skip_to_research=True,
        )
        assert state["subqueries"] == plan
        assert state["pending_subqueries"] == plan
        assert state["plan_approved"] is True
        assert state["current_phase"] == PHASE_SUPERVISOR

    @pytest.mark.asyncio
    async def test_create_initial_state_with_cached_findings_starts_at_triage(self):
        """Cached findings text sets starting phase to triage."""
        state = await create_initial_state(
            query="Follow-up",
            cached_findings_text="Prior findings...",
        )
        assert state["current_phase"] == PHASE_TRIAGE
        assert state["cached_findings_text"] == "Prior findings..."

    @pytest.mark.asyncio
    async def test_create_initial_state_with_max_iterations_override(self):
        """Max iterations override is applied."""
        state = await create_initial_state(
            query="Q",
            max_iterations=5,
            max_iterations_override=7,
        )
        assert state["max_iterations"] == 5
        assert state["_user_max_iterations_override"] == 7

    @pytest.mark.asyncio
    async def test_create_initial_state_with_enriched_subqueries(self):
        """Enriched subqueries and discovered data products are set."""
        enriched = [{"query": "q1", "status": "ready"}]
        discovered = [{"product": "dp1"}]
        state = await create_initial_state(
            query="Q",
            enriched_subqueries=enriched,
            discovered_data_products=discovered,
        )
        assert state["enriched_subqueries"] == enriched
        assert state["discovered_data_products"] == discovered


class TestRoutingFunctions:
    """Test conditional routing logic."""

    def test_after_triage_context_sufficient_routes_to_context_answer(self):
        """Triage decision context_sufficient routes to context_answer."""
        state: DeepResearchState = {"triage_decision": "context_sufficient"}
        assert _after_triage(state) == "context_answer"

    def test_after_triage_partial_research_routes_to_plan(self):
        """Triage decision partial_research routes to plan."""
        state: DeepResearchState = {"triage_decision": "partial_research"}
        assert _after_triage(state) == "plan"

    def test_after_triage_full_research_default_routes_to_probe(self):
        """Triage decision full_research or default routes to probe."""
        state: DeepResearchState = {"triage_decision": "full_research"}
        assert _after_triage(state) == "probe"
        assert _after_triage({}) == "probe"

    def test_after_router_complete_routes_to_complete(self):
        """Phase complete routes to complete node."""
        state: DeepResearchState = {"current_phase": PHASE_COMPLETE}
        assert _after_router(state) == "complete"

    def test_after_router_supervisor_routes_to_supervisor(self):
        """Phase supervisor routes to supervisor node."""
        state: DeepResearchState = {"current_phase": PHASE_SUPERVISOR}
        assert _after_router(state) == "supervisor"

    def test_after_router_default_routes_to_assess_complexity(self):
        """Default phase routes to assess_complexity."""
        state: DeepResearchState = {"current_phase": PHASE_PROBE}
        assert _after_router(state) == "assess_complexity"

    def test_after_assess_complexity_with_cached_findings_routes_to_triage(self):
        """Cached findings text routes to triage."""
        state: DeepResearchState = {"cached_findings_text": "cached..."}
        assert _after_assess_complexity(state) == "triage"

    def test_after_assess_complexity_supervisor_routes_to_supervisor(self):
        """Phase supervisor routes to supervisor."""
        state: DeepResearchState = {"current_phase": PHASE_SUPERVISOR}
        assert _after_assess_complexity(state) == "supervisor"

    def test_after_assess_complexity_default_routes_to_probe(self):
        """Default routes to probe."""
        state: DeepResearchState = {}
        assert _after_assess_complexity(state) == "probe"

    def test_after_plan_approved_routes_to_supervisor(self):
        """Plan approved routes to supervisor."""
        state: DeepResearchState = {"plan_approved": True}
        assert _after_plan(state) == "supervisor"

    def test_after_plan_not_approved_routes_to_await_approval(self):
        """Plan not approved routes to await_approval."""
        state: DeepResearchState = {"plan_approved": False}
        assert _after_plan(state) == "await_approval"

    def test_after_await_approval_approved_routes_to_supervisor(self):
        """Await approval with plan_approved routes to supervisor."""
        state: DeepResearchState = {"plan_approved": True}
        assert _after_await_approval(state) == "supervisor"

    def test_after_await_approval_not_approved_routes_to_plan_rejected(self):
        """Await approval without plan_approved routes to plan_rejected."""
        state: DeepResearchState = {"plan_approved": False}
        assert _after_await_approval(state) == "plan_rejected"

    def test_after_supervisor_phase_supervisor_loops(self):
        """Supervisor phase loops back to supervisor."""
        state: DeepResearchState = {"current_phase": PHASE_SUPERVISOR}
        assert _after_supervisor(state) == "supervisor"

    def test_after_supervisor_default_routes_to_completeness(self):
        """Default routes to completeness."""
        state: DeepResearchState = {"current_phase": PHASE_COMPLETENESS}
        assert _after_supervisor(state) == "completeness"

    def test_after_completeness_phase_supervisor_loops(self):
        """Completeness with supervisor phase loops."""
        state: DeepResearchState = {"current_phase": PHASE_SUPERVISOR}
        assert _after_completeness(state) == "supervisor"

    def test_after_completeness_default_routes_to_synthesize(self):
        """Default routes to synthesize."""
        state: DeepResearchState = {"current_phase": PHASE_SYNTHESIZE}
        assert _after_completeness(state) == "synthesize"

    def test_after_review_route_no_review_returns_complete(self):
        """No current_review returns complete."""
        state: DeepResearchState = {"current_review": None}
        assert _after_review_route(state) == "complete"

    def test_after_review_route_research_more_returns_supervisor(self):
        """Action research_more returns supervisor."""
        state: DeepResearchState = {
            "current_review": {"action": "research_more"},
            "iteration": 1,
            "max_iterations": 3,
        }
        assert _after_review_route(state) == "supervisor"

    def test_after_review_route_revise_returns_synthesize(self):
        """Action revise returns synthesize."""
        state: DeepResearchState = {
            "current_review": {"action": "revise"},
            "iteration": 1,
            "max_iterations": 3,
        }
        assert _after_review_route(state) == "synthesize"

    def test_after_review_route_max_iterations_reached_returns_complete(self):
        """Max iterations reached returns complete."""
        state: DeepResearchState = {
            "current_review": {"action": "research_more"},
            "iteration": 3,
            "max_iterations": 3,
        }
        assert _after_review_route(state) == "complete"


class TestShouldPauseForPlanApproval:
    """Test _should_pause_for_plan_approval."""

    def test_returns_false_when_stage_not_plan_pending(self):
        """Non plan_pending stage returns False."""
        event = {"content": {"stage": "other"}}
        assert _should_pause_for_plan_approval(event, True, False) is False

    def test_returns_false_when_approval_not_required(self):
        """Require approval False returns False."""
        event = {"content": {"stage": "plan_pending"}}
        assert _should_pause_for_plan_approval(event, False, False) is False

    def test_returns_false_when_plan_already_approved(self):
        """Plan already approved returns False."""
        event = {"content": {"stage": "plan_pending"}}
        assert _should_pause_for_plan_approval(event, True, True) is False

    def test_returns_true_when_pending_approval_required(self):
        """Plan pending and approval required returns True."""
        event = {"content": {"stage": "plan_pending"}}
        assert _should_pause_for_plan_approval(event, True, False) is True


class TestGetStartingPhase:
    """Test _get_starting_phase."""

    def test_skip_to_research_with_plan_returns_supervisor(self):
        """Skip to research with plan override returns PHASE_SUPERVISOR."""
        assert _get_starting_phase(True, ["q1"], "") == PHASE_SUPERVISOR

    def test_cached_findings_returns_triage(self):
        """Cached findings returns PHASE_TRIAGE."""
        assert _get_starting_phase(False, None, "cached") == PHASE_TRIAGE

    def test_default_returns_probe(self):
        """Default returns PHASE_PROBE."""
        assert _get_starting_phase(False, None, "") == PHASE_PROBE


class TestBuildResearchGraph:
    """Test build_research_graph."""

    def test_build_research_graph_returns_compiled_graph(self):
        """Graph is compiled and has invoke method."""
        ctx = ResearchContext(tools=[], base_model=MagicMock())
        graph = build_research_graph(ctx)
        assert graph is not None
        assert hasattr(graph, "invoke")
        assert hasattr(graph, "astream")

    def test_build_research_graph_with_checkpointer(self):
        """Graph compiles with optional checkpointer."""
        ctx = ResearchContext(tools=[], base_model=MagicMock())
        checkpointer = MagicMock()
        graph = build_research_graph(ctx, checkpointer=checkpointer)
        assert graph is not None


class TestDeepResearchAgentInit:
    """Test DeepResearchAgent initialization."""

    def test_init_sets_ctx_and_graph(self):
        """Agent stores context and builds graph."""
        ctx = ResearchContext(tools=[], base_model=MagicMock())
        with patch(
            "template_agent.src.core.deep_research.streaming.build_research_graph"
        ) as mock_build:
            mock_graph = MagicMock()
            mock_build.return_value = mock_graph
            agent = DeepResearchAgent(ctx)
            assert agent.ctx is ctx
            assert agent.checkpointer is None
            assert agent.graph is mock_graph
            mock_build.assert_called_once_with(ctx, checkpointer=None)

    def test_init_with_checkpointer(self):
        """Agent accepts optional checkpointer."""
        ctx = ResearchContext(tools=[], base_model=MagicMock())
        checkpointer = MagicMock()
        with patch(
            "template_agent.src.core.deep_research.streaming.build_research_graph"
        ) as mock_build:
            agent = DeepResearchAgent(ctx, checkpointer=checkpointer)
            assert agent.checkpointer is checkpointer
            mock_build.assert_called_once_with(ctx, checkpointer=checkpointer)


class TestHelperFunctions:
    """Test internal helper functions."""

    def test_get_thread_id_from_metadata(self):
        """Thread ID from metadata."""
        assert _get_thread_id({"thread_id": "t1"}, {}) == "t1"

    def test_get_thread_id_from_configurable(self):
        """Thread ID from configurable when metadata empty."""
        assert _get_thread_id({}, {"thread_id": "t2"}) == "t2"

    def test_get_thread_id_prefers_metadata(self):
        """Metadata thread_id takes precedence."""
        assert _get_thread_id({"thread_id": "m"}, {"thread_id": "c"}) == "m"

    def test_get_plan_flags_from_metadata(self):
        """Plan flags extracted from metadata."""
        plan, approved, require = _get_plan_flags_from_metadata(
            {"deep_research_plan": ["q1"], "deep_research_plan_approved": True}
        )
        assert plan == ["q1"]
        assert approved is True
        assert require is True  # default

    def test_get_metadata_and_configurable_from_dict(self):
        """Config as dict extracts metadata and configurable."""
        config = {"metadata": {"k": "v"}, "configurable": {"tid": "x"}}
        meta, conf = _get_metadata_and_configurable(config)
        assert meta == {"k": "v"}
        assert conf == {"tid": "x"}

    def test_get_metadata_and_configurable_from_object(self):
        """Config as object with metadata/configurable attributes."""
        config = MagicMock(metadata={"a": 1}, configurable={"b": 2})
        meta, conf = _get_metadata_and_configurable(config)
        assert meta == {"a": 1}
        assert conf == {"b": 2}

    def test_event_fingerprint_deterministic(self):
        """Same event produces same fingerprint."""
        evt = {"type": "x", "content": {"stage": "s", "message": "m"}}
        fp1 = _event_fingerprint(evt)
        fp2 = _event_fingerprint(evt)
        assert fp1 == fp2

    def test_track_dedup_and_should_yield_first_seen_returns_true(self):
        """First-seen fingerprint yields True."""
        assert _track_dedup_and_should_yield(1, set(), 10) is True

    def test_track_dedup_and_should_yield_duplicate_returns_false(self):
        """Duplicate fingerprint yields False."""
        seen = {1}
        assert _track_dedup_and_should_yield(1, seen, 10) is False

    def test_extract_pending_events_and_clean_state(self):
        """Pending events extracted and state cleaned."""
        node_state = {"a": 1, "_pending_events": [{"e": 1}], "b": 2}
        pending, clean = _extract_pending_events_and_clean_state(node_state)
        assert pending == [{"e": 1}]
        assert "_pending_events" not in clean
        assert clean["a"] == 1 and clean["b"] == 2

    def test_reconcile_enrichment_matches_cached(self):
        """Reconcile enrichment merges cached with plan."""
        plan = ["q1", "q2"]
        cached = [{"query": "q1", "status": "done", "data_products": []}]
        result = _reconcile_enrichment(plan, cached)
        assert len(result) == 2
        assert result[0]["query"] == "q1" and result[0]["status"] == "done"
        assert result[1]["query"] == "q2" and result[1]["status"] == "ready"


class TestBuildContextLoadedEvent:
    """Test _build_context_loaded_event."""

    def test_empty_context_has_zero_messages(self):
        """Empty context produces has_context=False."""
        evt = _build_context_loaded_event("")
        assert evt["content"]["details"]["has_context"] is False
        assert evt["content"]["details"]["message_count"] == 0

    def test_non_empty_context_counts_messages(self):
        """Context with User/Assistant prefixes counted."""
        ctx = "User: hi\nAssistant: hello\nUser: bye"
        evt = _build_context_loaded_event(ctx)
        assert evt["content"]["details"]["has_context"] is True
        assert evt["content"]["details"]["message_count"] == 3


class TestBuildIndexedSummaries:
    """Test _build_indexed_summaries."""

    def test_build_indexed_summaries_skips_empty_subquery(self):
        """Findings without subquery are skipped."""
        findings: dict[str, Finding] = {
            "h1": {"subquery": "q1", "answer": "a1"},
            "h2": {"answer": "a2"},
        }
        indexed, summaries = _build_indexed_summaries(findings)
        assert len(indexed) == 1
        assert len(summaries) == 1
        assert "q1" in summaries[0]

    def test_build_indexed_summaries_truncates_answer_preview(self):
        """Answer preview truncated to 200 chars."""
        long_answer = "x" * 300
        findings: dict[str, Finding] = {"h1": {"subquery": "q", "answer": long_answer}}
        _, summaries = _build_indexed_summaries(findings)
        assert len(summaries[0]) < 250


class TestParseValidLlmIndices:
    """Test _parse_valid_llm_indices."""

    def test_parse_valid_indices_from_json_array(self):
        """Valid JSON array parsed to indices."""
        indexed = [("h1", {"subquery": "q1", "answer": "a1"})]
        result = _parse_valid_llm_indices("[0]", indexed, 10)
        assert result == [0]

    def test_parse_valid_indices_raises_on_no_array(self):
        """Non-array response raises ValueError."""
        indexed = [("h1", {"subquery": "q1", "answer": "a1"})]
        with pytest.raises(ValueError, match="No JSON array"):
            _parse_valid_llm_indices("not json", indexed, 10)

    def test_parse_valid_indices_raises_on_empty_valid(self):
        """Empty valid indices raises ValueError."""
        indexed = [("h1", {"subquery": "q1", "answer": "a1"})]
        with pytest.raises(ValueError, match="no valid indices"):
            _parse_valid_llm_indices("[99]", indexed, 10)


class TestFormatSelectedFindings:
    """Test _format_selected_findings."""

    def test_format_selected_findings_truncates_at_max_chars(self):
        """Output truncated when exceeding max_chars."""
        indexed = [
            ("h1", {"subquery": "q1", "answer": "a" * 100}),
            ("h2", {"subquery": "q2", "answer": "b" * 100}),
        ]
        result = _format_selected_findings(indexed, [0, 1], max_chars=50)
        assert "truncated" in result or len(result) <= 100


class TestSelectRelevantFindings:
    """Test select_relevant_findings with mocked LLM."""

    @pytest.mark.asyncio
    async def test_select_relevant_findings_empty_returns_empty(self):
        """Empty findings returns empty string."""
        model = AsyncMock()
        result = await select_relevant_findings(model, "q", {})
        assert result == ""
        model.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_select_relevant_findings_below_threshold_uses_format_full(
        self,
    ):
        """Findings below threshold use format_full_cached_findings_for_triage."""
        model = AsyncMock()
        findings: dict[str, Finding] = {
            "h1": {"subquery": "q1", "answer": "a1"},
            "h2": {"subquery": "q2", "answer": "a2"},
        }
        with patch(
            "template_agent.src.core.deep_research.streaming.format_full_cached_findings_for_triage",
            return_value="formatted",
        ) as mock_format:
            result = await select_relevant_findings(model, "q", findings)
            assert result == "formatted"
            mock_format.assert_called_once()
            model.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_select_relevant_findings_above_threshold_calls_llm(self):
        """Findings above threshold invoke LLM for selection."""
        model = AsyncMock()
        model.ainvoke.return_value = MagicMock(content="[0, 1, 2]")
        findings: dict[str, Finding] = {
            f"h{i}": {"subquery": f"q{i}", "answer": f"a{i}"} for i in range(15)
        }
        result = await select_relevant_findings(model, "follow-up", findings)
        assert model.ainvoke.called
        assert "q0" in result or "q1" in result or "q2" in result

    @pytest.mark.asyncio
    async def test_select_relevant_findings_llm_failure_falls_back(self):
        """LLM failure falls back to format_full_cached_findings_for_triage."""
        model = AsyncMock()
        model.ainvoke.side_effect = Exception("LLM error")
        findings: dict[str, Finding] = {
            f"h{i}": {"subquery": f"q{i}", "answer": f"a{i}"} for i in range(15)
        }
        with patch(
            "template_agent.src.core.deep_research.streaming.format_full_cached_findings_for_triage",
            return_value="fallback",
        ) as mock_format:
            result = await select_relevant_findings(model, "q", findings)
            assert result == "fallback"
            mock_format.assert_called_once()


class TestMessageExtraction:
    """Test message type and content extraction helpers."""

    def test_extract_msg_type_from_dict_type_key(self):
        """Extract type from 'type' key."""
        assert _extract_msg_type_from_dict({"type": "human"}) == "human"

    def test_extract_content_from_dict_content_key(self):
        """Extract content from 'content' key."""
        assert _extract_content_from_dict({"content": "hello"}) == "hello"

    def test_normalize_message_type_human(self):
        """Human type normalized."""
        assert _normalize_message_type("HumanMessage") == "human"

    def test_normalize_message_type_ai(self):
        """AI type normalized."""
        assert _normalize_message_type("AIMessage") == "ai"


class TestDeepResearchAgentExtractQuery:
    """Test DeepResearchAgent query extraction."""

    def test_astream_extract_query_from_messages(self):
        """Query extracted from last message content."""
        ctx = ResearchContext(tools=[], base_model=MagicMock())
        with patch(
            "template_agent.src.core.deep_research.streaming.build_research_graph",
            return_value=MagicMock(),
        ):
            agent = DeepResearchAgent(ctx)
            input_data = {"messages": [{"content": "What is AI?"}]}
            assert agent._astream_extract_query(input_data) == "What is AI?"

    def test_astream_extract_query_empty_messages_returns_empty(self):
        """Empty messages returns empty string."""
        ctx = ResearchContext(tools=[], base_model=MagicMock())
        with patch(
            "template_agent.src.core.deep_research.streaming.build_research_graph",
            return_value=MagicMock(),
        ):
            agent = DeepResearchAgent(ctx)
            assert agent._astream_extract_query({"messages": []}) == ""
            assert agent._astream_extract_query(None) == ""


# =============================================================================
# Additional tests for improved coverage (target: 65%+)
# =============================================================================


class TestNodeErrorUpdates:
    """Test _node_error_updates."""

    def test_node_error_updates_returns_complete_phase_and_error_event(self):
        """Error produces final_answer, PHASE_COMPLETE, and pending event."""
        exc = ValueError("Test error")
        result = _node_error_updates(exc)
        assert result["current_phase"] == PHASE_COMPLETE
        assert "Research encountered an error" in result["final_answer"]
        assert len(result["_pending_events"]) == 1
        assert result["_pending_events"][0]["content"]["event_type"] == "error"


class TestSpanHelpers:
    """Test _span_open, _span_end_ok, _span_end_error."""

    def test_span_open_returns_none_when_no_root_tracer(self):
        """No root_tracer returns None."""
        ctx = ResearchContext(tools=[], base_model=MagicMock())
        assert _span_open(ctx, "probe", {"query": "q"}) is None

    def test_span_open_returns_none_when_tracer_has_no_span(self):
        """Tracer without span attr returns None."""
        ctx = ResearchContext(tools=[], base_model=MagicMock())
        ctx.root_tracer = MagicMock(spec=[])  # no span attr
        assert _span_open(ctx, "probe", {"query": "q"}) is None

    def test_span_open_returns_span_when_tracer_has_span(self):
        """Tracer with span returns span."""
        ctx = ResearchContext(tools=[], base_model=MagicMock())
        mock_span = MagicMock()
        ctx.root_tracer = MagicMock(span=MagicMock(return_value=mock_span))
        result = _span_open(ctx, "probe", {"query": "q", "current_phase": "probe"})
        assert result is mock_span
        ctx.root_tracer.span.assert_called_once()
        call_kw = ctx.root_tracer.span.call_args[1]
        assert call_kw["name"] == "deep_research.probe"

    def test_span_end_ok_no_op_when_span_none(self):
        """None span does nothing."""
        _span_end_ok(None, {"a": 1}, 0.0)

    def test_span_end_ok_calls_span_end_when_span_present(self):
        """Span present calls end with output."""
        span = MagicMock()
        start = 0.0
        _span_end_ok(span, {"final_answer": "x", "_pending_events": []}, start)
        span.end.assert_called_once()
        call_kw = span.end.call_args[1]
        assert "updated_keys" in call_kw["output"]
        assert "_pending_events" not in call_kw["output"]["updated_keys"]

    def test_span_end_error_no_op_when_span_none(self):
        """None span does nothing."""
        _span_end_error(None, ValueError("err"))

    def test_span_end_error_calls_update_and_end_when_span_present(self):
        """Span present calls update and end."""
        span = MagicMock()
        exc = ValueError("err")
        _span_end_error(span, exc)
        span.update.assert_called_once_with(level="ERROR", status_message="err")
        span.end.assert_called_once()


class TestWrapNode:
    """Test _wrap_node."""

    @pytest.mark.asyncio
    async def test_wrap_node_success_returns_updates_with_events(self):
        """Successful node returns updates with _pending_events."""
        ctx = ResearchContext(tools=[], base_model=MagicMock())

        async def mock_node(state, _ctx):
            return {"query": "x"}, [{"type": "event"}]

        wrapped = _wrap_node(mock_node, ctx, "probe")
        result = await wrapped({"query": "q"})
        assert result["query"] == "x"
        assert result["_pending_events"] == [{"type": "event"}]

    @pytest.mark.asyncio
    async def test_wrap_node_exception_returns_node_error_updates(self):
        """Node exception returns _node_error_updates result."""
        ctx = ResearchContext(tools=[], base_model=MagicMock())

        async def failing_node(_state, _ctx):
            raise RuntimeError("Node failed")

        wrapped = _wrap_node(failing_node, ctx, "probe")
        result = await wrapped({"query": "q"})
        assert result["current_phase"] == PHASE_COMPLETE
        assert "Research encountered an error" in result["final_answer"]
        assert len(result["_pending_events"]) == 1


class TestEventFingerprint:
    """Test _event_fingerprint edge cases."""

    def test_event_fingerprint_content_non_dict_uses_empty_details(self):
        """Content not dict uses empty details."""
        evt = {"type": "x", "content": "not-a-dict"}
        fp = _event_fingerprint(evt)
        assert isinstance(fp, int)

    def test_event_fingerprint_details_non_dict_uses_empty(self):
        """Details not dict uses empty dict."""
        evt = {"type": "x", "content": {"details": "not-dict", "stage": "s"}}
        fp = _event_fingerprint(evt)
        assert isinstance(fp, int)

    def test_event_fingerprint_different_events_different_hashes(self):
        """Different events produce different fingerprints."""
        evt1 = {"type": "a", "content": {"stage": "s1", "message": "m1"}}
        evt2 = {"type": "b", "content": {"stage": "s2", "message": "m2"}}
        assert _event_fingerprint(evt1) != _event_fingerprint(evt2)


class TestExtractMsgTypeFromObj:
    """Test _extract_msg_type_from_obj."""

    def test_extract_msg_type_from_obj_type_attr(self):
        """Object with type attr returns it."""
        obj = MagicMock(type="human")
        assert _extract_msg_type_from_obj(obj) == "human"

    def test_extract_msg_type_from_obj_lc_id_human(self):
        """Object with human in lc_id returns human."""
        obj = MagicMock(
            type=None, _type=None, lc_id="langchain_core/messages/HumanMessage"
        )
        assert _extract_msg_type_from_obj(obj) == "human"

    def test_extract_msg_type_from_obj_lc_id_ai(self):
        """Object with ai in lc_id returns ai."""
        obj = MagicMock(
            type=None, _type=None, lc_id="langchain_core/messages/AIMessage"
        )
        assert _extract_msg_type_from_obj(obj) == "ai"

    def test_extract_msg_type_from_obj_lc_id_ai_tool_returns_none(self):
        """Object with ai and tool in lc_id returns None."""
        obj = MagicMock(
            type=None, _type=None, lc_id="langchain_core/messages/AIToolMessage"
        )
        assert _extract_msg_type_from_obj(obj) is None

    def test_extract_msg_type_from_obj_no_lc_id_returns_none(self):
        """Object without lc_id returns None."""
        obj = MagicMock(type=None, _type=None)
        del obj.lc_id
        assert _extract_msg_type_from_obj(obj) is None


class TestExtractContentFromDict:
    """Test _extract_content_from_dict edge cases."""

    def test_extract_content_from_dict_text_key(self):
        """Extract from text key when content missing."""
        assert _extract_content_from_dict({"text": "hello"}) == "hello"

    def test_extract_content_from_dict_message_key(self):
        """Extract from message key."""
        assert _extract_content_from_dict({"message": "msg"}) == "msg"

    def test_extract_content_from_dict_kwargs_content(self):
        """Extract from kwargs.content."""
        assert _extract_content_from_dict({"kwargs": {"content": "kw"}}) == "kw"

    def test_extract_content_from_dict_empty_kwargs(self):
        """Empty kwargs returns empty string."""
        assert _extract_content_from_dict({"kwargs": None}) == ""


class TestExtractMsgTypeFromDict:
    """Test _extract_msg_type_from_dict edge cases."""

    def test_extract_msg_type_from_dict_lc_id_fallback(self):
        """Extract from lc_id when type keys missing."""
        msg = {"lc_id": "langchain_core/messages/HumanMessage"}
        assert _extract_msg_type_from_dict(msg) == "HumanMessage"

    def test_extract_msg_type_from_dict_message_type_key(self):
        """Extract from message_type key."""
        assert _extract_msg_type_from_dict({"message_type": "human"}) == "human"

    def test_extract_msg_type_from_dict_type_key(self):
        """Extract from type key."""
        assert _extract_msg_type_from_dict({"type": "ai"}) == "ai"


class TestNormalizeMessageType:
    """Test _normalize_message_type edge cases."""

    def test_normalize_message_type_none_returns_none(self):
        """None returns None."""
        assert _normalize_message_type(None) is None

    def test_normalize_message_type_non_string_returns_as_is(self):
        """Non-string returns as-is."""
        assert _normalize_message_type(123) == 123

    def test_normalize_message_type_tool_in_human_returns_none(self):
        """HumanMessage with tool returns None (tool message)."""
        assert _normalize_message_type("HumanToolMessage") is None

    def test_normalize_message_type_tool_in_ai_returns_none(self):
        """AIToolMessage returns None."""
        assert _normalize_message_type("AIToolMessage") is None


class TestRunGraphGetNextItem:
    """Test _run_graph_get_next_item."""

    @pytest.mark.asyncio
    async def test_run_graph_get_next_item_returns_item_when_available(self):
        """Item in queue returned with should_heartbeat=False."""
        queue: asyncio.Queue = asyncio.Queue()
        await queue.put({"event": 1})
        item, heartbeat = await _run_graph_get_next_item(queue, 1.0)
        assert item == {"event": 1}
        assert heartbeat is False

    @pytest.mark.asyncio
    async def test_run_graph_get_next_item_timeout_returns_heartbeat(self):
        """Timeout returns None, True for heartbeat."""
        queue: asyncio.Queue = asyncio.Queue()
        item, heartbeat = await _run_graph_get_next_item(queue, 0.01)
        assert item is None
        assert heartbeat is True


class TestRelayEventsToOutput:
    """Test _relay_events_to_output."""

    @pytest.mark.asyncio
    async def test_relay_events_to_output_relays_until_sentinel(self):
        """Events relayed until sentinel received."""
        event_queue: asyncio.Queue = asyncio.Queue()
        output_queue: asyncio.Queue = asyncio.Queue()
        await event_queue.put({"e": 1})
        await event_queue.put({"e": 2})
        await event_queue.put({"_sentinel": True})

        relay_task = asyncio.create_task(
            _relay_events_to_output(event_queue, output_queue)
        )
        # Put sentinel in output for consume to stop - actually relay puts
        # items in output_queue. The relay loops until it gets _sentinel from
        # event_queue. So we need to run relay and have something consume output.
        # Simpler: run relay, put items, then get from output.
        await relay_task
        out1 = await asyncio.wait_for(output_queue.get(), timeout=0.5)
        out2 = await asyncio.wait_for(output_queue.get(), timeout=0.5)
        assert out1 == {"e": 1}
        assert out2 == {"e": 2}
        # Sentinel stops relay, no more items from relay
        assert output_queue.empty()


class TestProcessSingleNodeOutput:
    """Test _process_single_node_output."""

    def test_process_single_node_output_none_returns_empty(self):
        """None node_state returns empty events."""
        events, state, stop = _process_single_node_output(None, {"a": 1}, set(), 10)
        assert events == []
        assert state == {"a": 1}
        assert stop is False

    def test_process_single_node_output_extracts_and_dedupes_events(self):
        """Pending events extracted and deduplicated."""
        node_state = {
            "_pending_events": [
                {"type": "e1", "content": {"stage": "s1"}},
                {"type": "e1", "content": {"stage": "s1"}},
            ],
            "query": "q",
        }
        events, state, should_stop = _process_single_node_output(
            node_state, {"a": 1}, set(), 10
        )
        assert len(events) == 1
        assert state["query"] == "q"
        assert state["a"] == 1
        assert "_pending_events" not in state
        assert should_stop is False

    def test_process_single_node_output_should_stop_true_when_set(self):
        """should_stop in state returns True."""
        node_state = {"should_stop": True, "_pending_events": []}
        _, _, stop = _process_single_node_output(node_state, {}, set(), 10)
        assert stop is True

    def test_process_single_node_output_non_dict_clean_preserves_current_state(self):
        """Non-dict node_state_clean preserves current_state."""
        # _extract_pending_events_and_clean_state returns [], node_state for
        # non-dict. So we need node_state that is dict but produces non-dict
        # clean - actually it always returns dict for dict input. So we need
        # to pass something that yields non-dict clean. Actually for dict input,
        # clean is always dict. The branch is when node_state is not dict -
        # then we get [], node_state. So we need node_state that is not dict.
        # But then _extract_pending_events returns [], node_state. So
        # node_state_clean = node_state (e.g. a list). Then new_state =
        # current_state since node_state_clean is not dict.
        node_state = ["not", "a", "dict"]  # invalid but tests branch
        events, state, _ = _process_single_node_output(node_state, {"a": 1}, set(), 10)
        assert events == []
        assert state == {"a": 1}


class TestTrackDedupAndShouldYield:
    """Test _track_dedup_and_should_yield edge cases."""

    def test_track_dedup_exceeds_max_size_clears_set(self):
        """When set exceeds max_size, it is cleared and new fp yields True."""
        seen: set[int] = set()
        max_size = 2
        assert _track_dedup_and_should_yield(1, seen, max_size) is True
        assert _track_dedup_and_should_yield(2, seen, max_size) is True
        assert len(seen) == 2
        # Adding 3rd triggers clear (len 3 > max_size 2)
        assert _track_dedup_and_should_yield(3, seen, max_size) is True
        # Set was cleared, so only 3 remains (added before clear)
        assert len(seen) <= 1
        # Now 1 should yield again (set was cleared, 1 is new)
        assert _track_dedup_and_should_yield(1, seen, max_size) is True


class TestExtractPendingEventsAndCleanState:
    """Test _extract_pending_events_and_clean_state edge cases."""

    def test_extract_pending_events_non_dict_returns_empty_and_original(self):
        """Non-dict node_state returns [], node_state."""
        node_state = [1, 2, 3]
        pending, clean = _extract_pending_events_and_clean_state(node_state)
        assert pending == []
        assert clean is node_state


class TestGetRestoredContext:
    """Test _get_restored_context."""

    @pytest.mark.asyncio
    async def test_get_restored_context_skip_false_returns_empty(self):
        """skip_to_research False returns empty dict."""
        result = await _get_restored_context(False, "t1", "u1")
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_restored_context_no_thread_id_returns_empty(self):
        """No thread_id returns empty dict."""
        result = await _get_restored_context(True, None, "u1")
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_restored_context_returns_plan_context_when_available(self):
        """With skip and thread_id, returns plan context."""
        with patch(
            "template_agent.src.core.deep_research.plan_store.get_plan_context",
            new_callable=AsyncMock,
            return_value={"tool_inventory": "tools"},
        ):
            result = await _get_restored_context(True, "t1", "u1")
            assert result == {"tool_inventory": "tools"}

    @pytest.mark.asyncio
    async def test_get_restored_context_exception_returns_empty(self):
        """Exception in get_plan_context returns empty dict."""
        with patch(
            "template_agent.src.core.deep_research.plan_store.get_plan_context",
            new_callable=AsyncMock,
            side_effect=Exception("db error"),
        ):
            result = await _get_restored_context(True, "t1", "u1")
            assert result == {}


class TestBuildResearchGraphRouter:
    """Test router and plan_rejected via graph invocation."""

    @pytest.mark.asyncio
    async def test_graph_router_short_query_returns_error(self):
        """Query too short returns final_answer error."""
        ctx = ResearchContext(tools=[], base_model=MagicMock())
        with patch(
            "template_agent.src.core.deep_research.streaming.classify_input_quality",
            new_callable=AsyncMock,
        ):
            graph = build_research_graph(ctx)
            state = await create_initial_state(query="ab")
            result = None
            async for chunk in graph.astream(state):
                result = chunk
                break
            assert result is not None
            for node_name, node_state in result.items():
                if node_state and isinstance(node_state, dict):
                    fa = node_state.get("final_answer", "")
                    if "more detailed" in fa or "too short" in fa.lower():
                        return
            # Router node output
            assert any(
                "more detailed" in str(v.get("final_answer", ""))
                for v in (result or {}).values()
                if isinstance(v, dict)
            )

    @pytest.mark.asyncio
    async def test_graph_router_long_query_returns_error(self):
        """Query too long returns final_answer error."""
        ctx = ResearchContext(tools=[], base_model=MagicMock())
        with patch(
            "template_agent.src.core.deep_research.streaming.classify_input_quality",
            new_callable=AsyncMock,
        ):
            graph = build_research_graph(ctx)
            state = await create_initial_state(query="x" * 10001)
            result = None
            async for chunk in graph.astream(state):
                result = chunk
                break
            assert result is not None
            has_error = any(
                "too long" in str(v.get("final_answer", "")).lower()
                for v in (result or {}).values()
                if isinstance(v, dict)
            )
            assert has_error

    @pytest.mark.asyncio
    async def test_graph_router_gibberish_short_circuits(self):
        """Gibberish classification short-circuits to complete."""
        ctx = ResearchContext(tools=[], base_model=MagicMock())
        with patch(
            "template_agent.src.core.deep_research.streaming.classify_input_quality",
            new_callable=AsyncMock,
            return_value="gibberish",
        ):
            graph = build_research_graph(ctx)
            state = await create_initial_state(query="valid query here")
            chunks = []
            async for chunk in graph.astream(state):
                chunks.append(chunk)
                if len(chunks) >= 3:
                    break
            assert len(chunks) >= 1
            # First chunk should have router output with gibberish response
            first = chunks[0]
            for node_state in first.values():
                if isinstance(node_state, dict) and node_state.get("final_answer"):
                    fa = node_state["final_answer"].lower()
                    assert "clear" in fa or "question" in fa or "data" in fa
                    break


class TestDeepResearchAgentAstreamExtractConfig:
    """Test DeepResearchAgent._astream_extract_config."""

    def test_astream_extract_config_from_kwargs(self):
        """Config from kwargs when config is None."""
        ctx = ResearchContext(tools=[], base_model=MagicMock())
        with patch(
            "template_agent.src.core.deep_research.streaming.build_research_graph",
            return_value=MagicMock(),
        ):
            agent = DeepResearchAgent(ctx)
            meta, *_ = agent._astream_extract_config(
                None, {"config": {"metadata": {"k": "v"}}}
            )
            assert meta.get("k") == "v"

    def test_astream_extract_config_extracts_plan_flags(self):
        """Plan flags extracted from metadata."""
        ctx = ResearchContext(tools=[], base_model=MagicMock())
        with patch(
            "template_agent.src.core.deep_research.streaming.build_research_graph",
            return_value=MagicMock(),
        ):
            agent = DeepResearchAgent(ctx)
            config = {
                "metadata": {
                    "deep_research_plan": ["q1"],
                    "deep_research_plan_approved": True,
                    "deep_research_require_plan_approval": False,
                },
            }
            _, _, _, _, plan, approved, require = agent._astream_extract_config(
                config, {}
            )
            assert plan == ["q1"]
            assert approved is True
            assert require is False


class TestDeepResearchAgentRunGraphProcess:
    """Test DeepResearchAgent graph processing methods."""

    def test_run_graph_process_queue_item_none_returns_empty(self):
        """None item returns empty."""
        ctx = ResearchContext(tools=[], base_model=MagicMock())
        with patch(
            "template_agent.src.core.deep_research.streaming.build_research_graph",
            return_value=MagicMock(),
        ):
            agent = DeepResearchAgent(ctx)
            events, should_stop, is_sentinel = agent._run_graph_process_queue_item(
                None, {}, {"current": {}}, "_out", set(), 10
            )
            assert events == []
            assert should_stop is False
            assert is_sentinel is False

    def test_run_graph_process_queue_item_sentinel_returns_sentinel_true(self):
        """Sentinel item returns is_sentinel=True."""
        ctx = ResearchContext(tools=[], base_model=MagicMock())
        with patch(
            "template_agent.src.core.deep_research.streaming.build_research_graph",
            return_value=MagicMock(),
        ):
            agent = DeepResearchAgent(ctx)
            events, _should_stop, is_sentinel = agent._run_graph_process_queue_item(
                {"_sentinel": True}, {}, {"current": {}}, "_out", set(), 10
            )
            assert events == []
            assert is_sentinel is True

    def test_run_graph_process_one_item_regular_event_yields_when_not_duplicate(self):
        """Regular event (non-node-output) yields when not duplicate."""
        ctx = ResearchContext(tools=[], base_model=MagicMock())
        with patch(
            "template_agent.src.core.deep_research.streaming.build_research_graph",
            return_value=MagicMock(),
        ):
            agent = DeepResearchAgent(ctx)
            item = {"type": "custom", "content": {"message": "hi"}}
            events, state, stop = agent._run_graph_process_one_item(
                item, {"a": 1}, "_node_output", set(), 10
            )
            assert len(events) == 1
            assert events[0] == item
            assert state == {"a": 1}
            assert stop is False

    def test_run_graph_process_node_output_merges_states(self):
        """Node output merges into current state."""
        ctx = ResearchContext(tools=[], base_model=MagicMock())
        with patch(
            "template_agent.src.core.deep_research.streaming.build_research_graph",
            return_value=MagicMock(),
        ):
            agent = DeepResearchAgent(ctx)
            node_output = {
                "probe": {
                    "_pending_events": [{"type": "e1"}],
                    "query": "q",
                },
            }
            events, state, stop = agent._run_graph_process_node_output(
                node_output, {"a": 1}, set(), 10
            )
            assert len(events) == 1
            assert state["query"] == "q"
            assert state["a"] == 1
            assert stop is False


class TestDeepResearchAgentExtractRunMetadata:
    """Test _extract_run_metadata_for_thread_listing."""

    def test_extract_run_metadata_picks_from_metadata_and_configurable(self):
        """Metadata and configurable keys extracted."""
        ctx = ResearchContext(tools=[], base_model=MagicMock())
        with patch(
            "template_agent.src.core.deep_research.streaming.build_research_graph",
            return_value=MagicMock(),
        ):
            agent = DeepResearchAgent(ctx)
            result = agent._extract_run_metadata_for_thread_listing(
                {"user_id": "u1", "thread_title": "t"},
                {"project_id": "p1"},
            )
            assert result["user_id"] == "u1"
            assert result["thread_title"] == "t"
            assert result["project_id"] == "p1"


class TestDeepResearchAgentExtractMessageTypeAndContent:
    """Test _extract_message_type_and_content."""

    def test_extract_message_type_human_message(self):
        """HumanMessage returns human and content."""
        ctx = ResearchContext(tools=[], base_model=MagicMock())
        with patch(
            "template_agent.src.core.deep_research.streaming.build_research_graph",
            return_value=MagicMock(),
        ):
            agent = DeepResearchAgent(ctx)
            msg = HumanMessage(content="hello")
            msg_type, content = agent._extract_message_type_and_content(msg)
            assert msg_type == "human"
            assert content == "hello"

    def test_extract_message_type_ai_message(self):
        """AIMessage returns ai and content."""
        ctx = ResearchContext(tools=[], base_model=MagicMock())
        with patch(
            "template_agent.src.core.deep_research.streaming.build_research_graph",
            return_value=MagicMock(),
        ):
            agent = DeepResearchAgent(ctx)
            msg = AIMessage(content="response")
            msg_type, content = agent._extract_message_type_and_content(msg)
            assert msg_type == "ai"
            assert content == "response"

    def test_extract_message_type_dict(self):
        """Dict message uses normalize and extract helpers."""
        ctx = ResearchContext(tools=[], base_model=MagicMock())
        with patch(
            "template_agent.src.core.deep_research.streaming.build_research_graph",
            return_value=MagicMock(),
        ):
            agent = DeepResearchAgent(ctx)
            msg = {"type": "human", "content": "dict content"}
            msg_type, content = agent._extract_message_type_and_content(msg)
            assert msg_type == "human"
            assert content == "dict content"


class TestDeepResearchAgentFormatMessagesAsText:
    """Test _format_messages_as_text."""

    def test_format_messages_as_text_truncates_human(self):
        """Human content truncated to truncate_user chars."""
        ctx = ResearchContext(tools=[], base_model=MagicMock())
        with patch(
            "template_agent.src.core.deep_research.streaming.build_research_graph",
            return_value=MagicMock(),
        ):
            agent = DeepResearchAgent(ctx)
            msgs = [
                HumanMessage(content="a" * 1000),
                AIMessage(content="short"),
            ]
            parts = agent._format_messages_as_text(msgs, truncate_user=10)
            assert any("aaaaaaaaaa" in p for p in parts)
            assert any("short" in p for p in parts)

    def test_format_messages_as_text_skips_empty_content(self):
        """Messages with empty content are skipped."""
        ctx = ResearchContext(tools=[], base_model=MagicMock())
        with patch(
            "template_agent.src.core.deep_research.streaming.build_research_graph",
            return_value=MagicMock(),
        ):
            agent = DeepResearchAgent(ctx)
            msgs = [HumanMessage(content=""), AIMessage(content="x")]
            parts = agent._format_messages_as_text(msgs)
            assert len(parts) == 1
            assert "x" in parts[0]


class TestDeepResearchAgentAstream:
    """Test DeepResearchAgent.astream."""

    @pytest.mark.asyncio
    async def test_astream_empty_query_yields_error(self):
        """Empty query yields error event and returns."""
        ctx = ResearchContext(tools=[], base_model=MagicMock())
        with patch(
            "template_agent.src.core.deep_research.streaming.build_research_graph",
            return_value=MagicMock(),
        ):
            agent = DeepResearchAgent(ctx)
            events = []
            async for tag, evt in agent.astream(input={"messages": []}, config={}):
                events.append((tag, evt))
                if len(events) >= 5:
                    break
            assert len(events) >= 1
            assert events[0][0] == "custom"
            assert "error" in str(events[0][1]).lower() or (
                events[0][1].get("content", {}).get("event_type") == "error"
            )

    @pytest.mark.asyncio
    async def test_astream_with_query_yields_started_and_context(self):
        """Query yields started and context_loaded."""
        mock_graph = AsyncMock()
        mock_graph.astream = AsyncMock(return_value=AsyncMock().__aiter__())

        ctx = ResearchContext(tools=[], base_model=MagicMock())
        with (
            patch(
                "template_agent.src.core.deep_research.streaming.build_research_graph",
                return_value=mock_graph,
            ),
            patch(
                "template_agent.src.core.deep_research.streaming.load_cached_findings",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "template_agent.src.core.deep_research.streaming.select_relevant_findings",
                new_callable=AsyncMock,
                return_value="",
            ),
        ):
            agent = DeepResearchAgent(ctx)

            async def mock_astream(*args, **kwargs):
                yield {"router": {"_pending_events": [], "node_transitions": 1}}
                yield {"assess_complexity": {"_pending_events": []}}
                # Simulate completion
                yield {"complete": {"final_answer": "Done", "_pending_events": []}}

            mock_graph.astream = mock_astream

            events = []
            async for tag, evt in agent.astream(
                input={"messages": [{"content": "What is AI?"}]},
                config={"metadata": {}},
            ):
                events.append((tag, evt))
                if len(events) >= 15:
                    break

            assert len(events) >= 2
            tags_and_types = [
                (t, e.get("content", {}).get("event_type", e.get("type", "")))
                for t, e in events
                if isinstance(e, dict)
            ]
            assert any("started" in str(v).lower() for _, v in tags_and_types)
            assert any("context" in str(v).lower() for _, v in tags_and_types)
