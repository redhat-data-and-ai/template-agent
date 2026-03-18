"""Comprehensive pytest tests for the deep research events module."""

from __future__ import annotations

import pytest

from template_agent.src.core.deep_research.events import (
    DeepResearchEvent,
    DeepResearchEventType,
    _simplify_error,
    emit_agent_decision,
    emit_agent_message,
    emit_agent_thinking,
    emit_cancelled,
    emit_completed,
    emit_completeness_assessment,
    emit_complexity_assessment,
    emit_consensus_result,
    emit_consensus_vote,
    emit_context_loaded,
    emit_context_usage_update,
    emit_cross_chat_findings_loaded,
    emit_data_aggregation_complete,
    emit_data_aggregation_start,
    emit_diminishing_returns,
    emit_empty_plan,
    emit_error,
    emit_event,
    emit_fact_check_complete,
    emit_fact_check_start,
    emit_final_answer,
    emit_heartbeat,
    emit_inter_agent_message,
    emit_no_valid_findings,
    emit_plan_generated,
    emit_plan_pending,
    emit_plan_pending_enriched,
    emit_probe_complete,
    emit_probe_start,
    emit_reliability_update,
    emit_report_generation_complete,
    emit_report_generation_start,
    emit_research_complete,
    emit_research_failed,
    emit_research_start,
    emit_review_complete,
    emit_review_start,
    emit_reviewer_feedback,
    emit_reviewer_score,
    emit_revision_complete,
    emit_revision_start,
    emit_sentinel_triggered,
    emit_started,
    emit_subquery_cached,
    emit_subquery_complete,
    emit_subquery_drill_down,
    emit_subquery_error,
    emit_subquery_start,
    emit_subquery_validation,
    emit_supervisor_delegating,
    emit_supervisor_follow_up,
    emit_supervisor_reflection,
    emit_supervisor_round_start,
    emit_synthesis_complete,
    emit_synthesis_start,
    emit_token_usage_update,
    emit_tool_discovery,
    emit_triage_decision,
    emit_understanding,
    emit_validation_analyzing,
    emit_validation_complete,
    emit_validation_conflict,
    emit_validation_start,
    emit_visualization_analysis,
    emit_visualization_complete,
    emit_visualization_created,
    emit_visualization_skipped,
    emit_visualization_start,
    emit_worker_progress,
    emit_worker_reformulation,
    emit_worker_self_evaluation,
)

# ---------------------------------------------------------------------------
# TestEventTypes
# ---------------------------------------------------------------------------


class TestEventTypes:
    """Test cases for DeepResearchEventType enum."""

    def test_started_exists(self):
        """Verify STARTED event type exists."""
        assert DeepResearchEventType.STARTED == "started"

    def test_context_loaded_exists(self):
        """Verify CONTEXT_LOADED event type exists."""
        assert DeepResearchEventType.CONTEXT_LOADED == "context_loaded"

    def test_tool_discovery_exists(self):
        """Verify TOOL_DISCOVERY event type exists."""
        assert DeepResearchEventType.TOOL_DISCOVERY == "tool_discovery"

    def test_probe_start_exists(self):
        """Verify PROBE_START event type exists."""
        assert DeepResearchEventType.PROBE_START == "probe_start"

    def test_plan_generated_exists(self):
        """Verify PLAN_GENERATED event type exists."""
        assert DeepResearchEventType.PLAN_GENERATED == "plan_generated"

    def test_subquery_start_exists(self):
        """Verify SUBQUERY_START event type exists."""
        assert DeepResearchEventType.SUBQUERY_START == "subquery_start"

    def test_subquery_complete_exists(self):
        """Verify SUBQUERY_COMPLETE event type exists."""
        assert DeepResearchEventType.SUBQUERY_COMPLETE == "subquery_complete"

    def test_subquery_error_exists(self):
        """Verify SUBQUERY_ERROR event type exists."""
        assert DeepResearchEventType.SUBQUERY_ERROR == "subquery_error"

    def test_synthesis_complete_exists(self):
        """Verify SYNTHESIS_COMPLETE event type exists."""
        assert DeepResearchEventType.SYNTHESIS_COMPLETE == "synthesis_complete"

    def test_visualization_complete_exists(self):
        """Verify VISUALIZATION_COMPLETE event type exists."""
        assert DeepResearchEventType.VISUALIZATION_COMPLETE == "visualization_complete"

    def test_final_answer_exists(self):
        """Verify FINAL_ANSWER event type exists."""
        assert DeepResearchEventType.FINAL_ANSWER == "final_answer"

    def test_completed_exists(self):
        """Verify COMPLETED event type exists."""
        assert DeepResearchEventType.COMPLETED == "completed"

    def test_error_exists(self):
        """Verify ERROR event type exists."""
        assert DeepResearchEventType.ERROR == "error"

    def test_cancelled_exists(self):
        """Verify CANCELLED event type exists."""
        assert DeepResearchEventType.CANCELLED == "cancelled"

    def test_enum_is_string_subclass(self):
        """Verify enum values are strings."""
        assert isinstance(DeepResearchEventType.STARTED.value, str)


# ---------------------------------------------------------------------------
# TestSimplifyError
# ---------------------------------------------------------------------------


class TestSimplifyError:
    """Test cases for _simplify_error helper function."""

    def test_short_error_unchanged(self):
        """Short error (<=500 chars) is returned unchanged."""
        err = "Something went wrong"
        assert _simplify_error(err) == err

    def test_exactly_500_chars_unchanged(self):
        """Error of exactly 500 chars is returned unchanged."""
        err = "x" * 500
        assert _simplify_error(err) == err

    def test_long_error_truncated(self):
        """Error >500 chars is truncated to 500 with ... suffix."""
        err = "a" * 600
        result = _simplify_error(err)
        assert len(result) == 500
        assert result.endswith("...")
        assert result == "a" * 497 + "..."

    def test_empty_string_unchanged(self):
        """Empty string is returned unchanged."""
        assert _simplify_error("") == ""


# ---------------------------------------------------------------------------
# TestInitializationEvents
# ---------------------------------------------------------------------------


class TestInitializationEvents:
    """Test cases for initialization phase emit functions."""

    def test_emit_started_returns_dict_with_type(self):
        """emit_started returns dict with correct structure."""
        result = emit_started()
        assert "type" in result
        assert result["type"] == "deep_research_status"
        assert result["content"]["event_type"] == "started"

    def test_emit_started_has_expected_message(self):
        """emit_started contains expected message."""
        result = emit_started()
        assert "Deep research pipeline started" in result["content"]["message"]

    def test_emit_context_loaded_no_context(self):
        """emit_context_loaded with no context returns correct event."""
        result = emit_context_loaded(0, "", has_context=False)
        assert result["content"]["event_type"] == "context_loaded"
        assert result["content"]["details"]["has_context"] is False
        assert result["content"]["details"]["message_count"] == 0

    def test_emit_context_loaded_with_context(self):
        """emit_context_loaded with context returns correct event."""
        result = emit_context_loaded(3, "Previous messages...", has_context=True)
        assert result["content"]["event_type"] == "context_loaded"
        assert result["content"]["details"]["has_context"] is True
        assert result["content"]["details"]["message_count"] == 3

    def test_emit_context_loaded_zero_messages_treated_as_no_context(self):
        """emit_context_loaded with message_count=0 returns no-context variant."""
        result = emit_context_loaded(0, "anything", has_context=True)
        assert result["content"]["details"]["has_context"] is False

    def test_emit_cross_chat_findings_loaded_zero(self):
        """emit_cross_chat_findings_loaded with count=0."""
        result = emit_cross_chat_findings_loaded(0, 0)
        assert result["content"]["event_type"] == "cross_chat_findings_loaded"
        assert result["content"]["details"]["count"] == 0

    def test_emit_cross_chat_findings_loaded_with_findings(self):
        """emit_cross_chat_findings_loaded with findings."""
        result = emit_cross_chat_findings_loaded(5, 2)
        assert result["content"]["event_type"] == "cross_chat_findings_loaded"
        assert result["content"]["details"]["count"] == 5
        assert result["content"]["details"]["source_threads"] == 2

    def test_emit_tool_discovery_minimal(self):
        """emit_tool_discovery returns correct event."""
        result = emit_tool_discovery(3, ["tool_a", "tool_b", "tool_c"])
        assert result["content"]["event_type"] == "tool_discovery"
        assert result["content"]["details"]["tool_count"] == 3
        assert result["content"]["details"]["tool_names"] == [
            "tool_a",
            "tool_b",
            "tool_c",
        ]

    def test_emit_tool_discovery_truncates_tool_names(self):
        """emit_tool_discovery truncates tool_names to 10."""
        tools = [f"tool_{i}" for i in range(15)]
        result = emit_tool_discovery(15, tools)
        assert len(result["content"]["details"]["tool_names"]) == 10

    def test_emit_tool_discovery_ui_hidden(self):
        """emit_tool_discovery is hidden from UI."""
        result = emit_tool_discovery(1, ["x"])
        assert result["content"]["ui_visible"] is False


# ---------------------------------------------------------------------------
# TestProbeEvents
# ---------------------------------------------------------------------------


class TestProbeEvents:
    """Test cases for probe phase emit functions."""

    def test_emit_probe_start_returns_dict(self):
        """emit_probe_start returns correct event."""
        result = emit_probe_start()
        assert result["content"]["event_type"] == "probe_start"
        assert "Probing" in result["content"]["message"]

    def test_emit_probe_start_ui_hidden(self):
        """emit_probe_start is hidden from UI."""
        result = emit_probe_start()
        assert result["content"]["ui_visible"] is False

    def test_emit_probe_complete_minimal(self):
        """emit_probe_complete with minimal summary."""
        result = emit_probe_complete("Probe summary here")
        assert result["content"]["event_type"] == "probe_complete"
        assert "Probe summary here" in result["content"]["display_text"]

    def test_emit_probe_complete_includes_details(self):
        """emit_probe_complete includes probe_summary in details."""
        summary = "Found 5 tools"
        result = emit_probe_complete(summary)
        assert "probe_summary" in result["content"]["details"]


# ---------------------------------------------------------------------------
# TestPlanEvents
# ---------------------------------------------------------------------------


class TestPlanEvents:
    """Test cases for planning phase emit functions."""

    def test_emit_understanding_minimal(self):
        """emit_understanding returns correct event."""
        result = emit_understanding("User wants market data")
        assert result["content"]["event_type"] == "understanding"
        assert result["content"]["details"]["understanding"] == "User wants market data"

    def test_emit_plan_generated_minimal(self):
        """emit_plan_generated returns correct event."""
        subqueries = ["q1", "q2"]
        result = emit_plan_generated(subqueries)
        assert result["content"]["event_type"] == "plan_generated"
        assert result["content"]["details"]["subqueries"] == subqueries
        assert result["content"]["details"]["count"] == 2

    def test_emit_plan_pending_minimal(self):
        """emit_plan_pending without understanding."""
        subqueries = ["s1", "s2"]
        result = emit_plan_pending(subqueries)
        assert result["content"]["event_type"] == "plan_pending"
        assert result["content"]["details"]["requires_approval"] is True
        assert result["content"]["details"]["subqueries"] == subqueries

    def test_emit_plan_pending_with_understanding(self):
        """emit_plan_pending with understanding."""
        result = emit_plan_pending(["q1"], understanding="Context here")
        assert result["content"]["details"]["query_understanding"] == "Context here"

    def test_emit_plan_pending_enriched_minimal(self):
        """emit_plan_pending_enriched returns correct event."""
        enriched = [{"query": "q1", "status": "ready", "source": "new"}]
        result = emit_plan_pending_enriched(enriched)
        assert result["content"]["event_type"] == "plan_pending"
        assert result["content"]["details"]["enriched_subqueries"] == enriched
        assert result["content"]["details"]["cached_count"] == 0
        assert result["content"]["details"]["new_count"] == 1

    def test_emit_plan_pending_enriched_with_cached(self):
        """emit_plan_pending_enriched counts cached vs new."""
        enriched = [
            {"query": "q1", "status": "ready", "source": "cached"},
            {"query": "q2", "status": "ready", "source": "new"},
        ]
        result = emit_plan_pending_enriched(enriched)
        assert result["content"]["details"]["cached_count"] == 1
        assert result["content"]["details"]["new_count"] == 1


# ---------------------------------------------------------------------------
# TestResearchEvents
# ---------------------------------------------------------------------------


class TestResearchEvents:
    """Test cases for research phase emit functions."""

    def test_emit_research_start_minimal(self):
        """emit_research_start returns correct event."""
        result = emit_research_start(5)
        assert result["content"]["event_type"] == "research_start"
        assert result["content"]["details"]["total_subqueries"] == 5

    def test_emit_subquery_start_minimal(self):
        """emit_subquery_start returns correct event."""
        result = emit_subquery_start(1, 3, "What is X?")
        assert result["content"]["event_type"] == "subquery_start"
        assert result["content"]["details"]["index"] == 1
        assert result["content"]["details"]["total"] == 3
        assert result["content"]["details"]["subquery"] == "What is X?"

    def test_emit_subquery_cached_minimal(self):
        """emit_subquery_cached returns correct event."""
        result = emit_subquery_cached(2, 4, "Cached query")
        assert result["content"]["event_type"] == "subquery_cached"
        assert result["content"]["details"]["cached"] is True

    def test_emit_subquery_complete_minimal(self):
        """emit_subquery_complete returns correct event."""
        result = emit_subquery_complete(1, 2, "q", "Answer text")
        assert result["content"]["event_type"] == "subquery_complete"
        assert result["content"]["details"]["answer_full"] == "Answer text"

    def test_emit_subquery_error_minimal(self):
        """emit_subquery_error returns correct event."""
        result = emit_subquery_error(1, 2, "q", "Something failed")
        assert result["content"]["event_type"] == "subquery_error"
        assert result["content"]["details"]["error"] == "Something failed"

    def test_emit_subquery_error_truncates_long_error(self):
        """emit_subquery_error truncates long error via _simplify_error."""
        long_err = "x" * 600
        result = emit_subquery_error(1, 1, "q", long_err)
        assert len(result["content"]["details"]["error"]) == 500

    def test_emit_research_complete_minimal(self):
        """emit_research_complete returns correct event."""
        result = emit_research_complete(10)
        assert result["content"]["event_type"] == "research_complete"
        assert result["content"]["details"]["findings_count"] == 10

    def test_emit_subquery_drill_down_minimal(self):
        """emit_subquery_drill_down returns correct event."""
        result = emit_subquery_drill_down(1, 2, "q", "Need more depth")
        assert result["content"]["event_type"] == "subquery_drill_down"
        assert result["content"]["details"]["reason"] == "Need more depth"


# ---------------------------------------------------------------------------
# TestValidationEvents
# ---------------------------------------------------------------------------


class TestValidationEvents:
    """Test cases for validation phase emit functions."""

    def test_emit_validation_start_minimal(self):
        """emit_validation_start returns correct event."""
        result = emit_validation_start(8)
        assert result["content"]["event_type"] == "validation_start"
        assert result["content"]["details"]["findings_count"] == 8

    def test_emit_validation_analyzing_minimal(self):
        """emit_validation_analyzing returns correct event."""
        result = emit_validation_analyzing(12)
        assert result["content"]["event_type"] == "validation_analyzing"
        assert result["content"]["details"]["number_count"] == 12

    def test_emit_validation_conflict_minimal(self):
        """emit_validation_conflict returns correct event."""
        values = [{"value": "10", "source": "A"}, {"value": "20", "source": "B"}]
        result = emit_validation_conflict("revenue", values, "high")
        assert result["content"]["event_type"] == "validation_conflict"
        assert result["content"]["details"]["metric"] == "revenue"
        assert result["content"]["details"]["severity"] == "high"

    def test_emit_validation_complete_no_conflicts(self):
        """emit_validation_complete with no conflicts."""
        result = emit_validation_complete(0, 5)
        assert result["content"]["event_type"] == "validation_complete"
        assert result["content"]["details"]["conflicts_found"] == 0

    def test_emit_validation_complete_with_conflicts(self):
        """emit_validation_complete with conflicts."""
        result = emit_validation_complete(2, 3)
        assert result["content"]["details"]["conflicts_found"] == 2
        assert result["content"]["details"]["verified_count"] == 3


# ---------------------------------------------------------------------------
# TestSynthesisEvents
# ---------------------------------------------------------------------------


class TestSynthesisEvents:
    """Test cases for synthesis phase emit functions."""

    def test_emit_synthesis_start_minimal(self):
        """emit_synthesis_start returns correct event."""
        result = emit_synthesis_start(1)
        assert result["content"]["event_type"] == "synthesis_start"
        assert result["content"]["details"]["iteration"] == 1

    def test_emit_data_aggregation_start_minimal(self):
        """emit_data_aggregation_start returns correct event."""
        result = emit_data_aggregation_start()
        assert result["content"]["event_type"] == "data_aggregation_start"

    def test_emit_data_aggregation_complete_minimal(self):
        """emit_data_aggregation_complete returns correct event."""
        result = emit_data_aggregation_complete(20, 1)
        assert result["content"]["event_type"] == "data_aggregation_complete"
        assert result["content"]["details"]["data_points"] == 20
        assert result["content"]["details"]["conflicts"] == 1

    def test_emit_report_generation_start_minimal(self):
        """emit_report_generation_start returns correct event."""
        result = emit_report_generation_start()
        assert result["content"]["event_type"] == "report_generation_start"

    def test_emit_report_generation_complete_minimal(self):
        """emit_report_generation_complete returns correct event."""
        result = emit_report_generation_complete()
        assert result["content"]["event_type"] == "report_generation_complete"

    def test_emit_revision_start_minimal(self):
        """emit_revision_start returns correct event."""
        result = emit_revision_start(2)
        assert result["content"]["event_type"] == "revision_start"
        assert result["content"]["details"]["iteration"] == 2

    def test_emit_revision_complete_minimal(self):
        """emit_revision_complete returns correct event."""
        result = emit_revision_complete(2)
        assert result["content"]["event_type"] == "revision_complete"

    def test_emit_fact_check_start_minimal(self):
        """emit_fact_check_start returns correct event."""
        result = emit_fact_check_start()
        assert result["content"]["event_type"] == "fact_check_start"

    def test_emit_fact_check_complete_no_corrections(self):
        """emit_fact_check_complete with zero corrections."""
        result = emit_fact_check_complete(0)
        assert result["content"]["details"]["corrections"] == 0

    def test_emit_fact_check_complete_with_corrections(self):
        """emit_fact_check_complete with corrections."""
        result = emit_fact_check_complete(3)
        assert result["content"]["details"]["corrections"] == 3

    def test_emit_synthesis_complete_minimal(self):
        """emit_synthesis_complete returns correct event."""
        result = emit_synthesis_complete()
        assert result["content"]["event_type"] == "synthesis_complete"


# ---------------------------------------------------------------------------
# TestVisualizationEvents
# ---------------------------------------------------------------------------


class TestVisualizationEvents:
    """Test cases for visualization phase emit functions."""

    def test_emit_visualization_start_minimal(self):
        """emit_visualization_start returns correct event."""
        result = emit_visualization_start()
        assert result["content"]["event_type"] == "visualization_start"
        assert result["content"]["stage"] == "visualization"

    def test_emit_visualization_analysis_has_numeric_data(self):
        """emit_visualization_analysis with numeric data."""
        result = emit_visualization_analysis(True, ["bar", "line"])
        assert result["content"]["event_type"] == "visualization_analysis"
        assert result["content"]["details"]["has_numeric_data"] is True
        assert result["content"]["details"]["recommended_charts"] == ["bar", "line"]

    def test_emit_visualization_analysis_no_numeric_data(self):
        """emit_visualization_analysis without numeric data."""
        result = emit_visualization_analysis(False, [])
        assert result["content"]["details"]["has_numeric_data"] is False

    def test_emit_visualization_created_minimal(self):
        """emit_visualization_created returns correct event."""
        result = emit_visualization_created("bar", "Sales Chart")
        assert result["content"]["event_type"] == "visualization_created"
        assert result["content"]["details"]["chart_type"] == "bar"
        assert result["content"]["details"]["title"] == "Sales Chart"

    def test_emit_visualization_created_with_optional_fields(self):
        """emit_visualization_created includes optional fields when provided."""
        result = emit_visualization_created(
            "bar",
            "Chart",
            has_image=True,
            image_url="http://x",
            labels=["A"],
            values=[1],
        )
        assert result["content"]["details"]["image_url"] == "http://x"
        assert result["content"]["details"]["labels"] == ["A"]
        assert result["content"]["details"]["values"] == [1]

    def test_emit_visualization_complete_with_charts(self):
        """emit_visualization_complete with chart count > 0."""
        result = emit_visualization_complete(3)
        assert result["content"]["event_type"] == "visualization_complete"
        assert result["content"]["details"]["chart_count"] == 3

    def test_emit_visualization_complete_no_charts(self):
        """emit_visualization_complete with zero charts."""
        result = emit_visualization_complete(0)
        assert result["content"]["details"]["chart_count"] == 0

    def test_emit_visualization_skipped_minimal(self):
        """emit_visualization_skipped returns correct event."""
        result = emit_visualization_skipped("No numeric data")
        assert result["content"]["event_type"] == "visualization_skipped"
        assert result["content"]["details"]["reason"] == "No numeric data"


# ---------------------------------------------------------------------------
# TestReviewEvents
# ---------------------------------------------------------------------------


class TestReviewEvents:
    """Test cases for review phase emit functions."""

    def test_emit_review_start_minimal(self):
        """emit_review_start returns correct event."""
        result = emit_review_start("Skeptic")
        assert result["content"]["event_type"] == "review_start"
        assert result["content"]["details"]["persona"] == "Skeptic"

    def test_emit_review_complete_minimal(self):
        """emit_review_complete returns correct event."""
        result = emit_review_complete("approve", 85, "Looks good")
        assert result["content"]["event_type"] == "review_complete"
        assert result["content"]["details"]["action"] == "approve"
        assert result["content"]["details"]["score"] == 85
        assert result["content"]["details"]["reason"] == "Looks good"


# ---------------------------------------------------------------------------
# TestSupervisorEvents
# ---------------------------------------------------------------------------


class TestSupervisorEvents:
    """Test cases for supervisor phase emit functions."""

    def test_emit_supervisor_round_start_minimal(self):
        """emit_supervisor_round_start returns correct event."""
        result = emit_supervisor_round_start(1, ["q1", "q2"])
        assert result["content"]["event_type"] == "supervisor_round_start"
        assert result["content"]["details"]["round_number"] == 1
        assert result["content"]["details"]["subqueries_delegated"] == ["q1", "q2"]

    def test_emit_supervisor_round_start_with_max_rounds(self):
        """emit_supervisor_round_start includes max_rounds."""
        result = emit_supervisor_round_start(2, ["q1"], max_rounds=5)
        assert result["content"]["details"]["max_rounds"] == 5

    def test_emit_supervisor_delegating_minimal(self):
        """emit_supervisor_delegating returns correct event."""
        result = emit_supervisor_delegating(1, "Query text", "worker-1")
        assert result["content"]["event_type"] == "supervisor_delegating"
        assert result["content"]["details"]["worker_id"] == "worker-1"
        assert result["content"]["details"]["has_cross_context"] is False

    def test_emit_supervisor_delegating_with_cross_context(self):
        """emit_supervisor_delegating with cross context."""
        result = emit_supervisor_delegating(1, "q", "w1", has_cross_context=True)
        assert result["content"]["details"]["has_cross_context"] is True

    def test_emit_supervisor_reflection_minimal(self):
        """emit_supervisor_reflection returns correct event."""
        result = emit_supervisor_reflection(1, 80, ["gap1"], "continue")
        assert result["content"]["event_type"] == "supervisor_reflection"
        assert result["content"]["details"]["coverage_pct"] == 80
        assert result["content"]["details"]["gaps"] == ["gap1"]
        assert result["content"]["details"]["decision"] == "continue"

    def test_emit_supervisor_follow_up_minimal(self):
        """emit_supervisor_follow_up returns correct event."""
        result = emit_supervisor_follow_up(1, ["f1", "f2"])
        assert result["content"]["event_type"] == "supervisor_follow_up"
        assert result["content"]["details"]["follow_up_subqueries"] == ["f1", "f2"]


# ---------------------------------------------------------------------------
# TestCompletenessEvents
# ---------------------------------------------------------------------------


class TestCompletenessEvents:
    """Test cases for completeness assessment emit functions."""

    def test_emit_completeness_assessment_minimal(self):
        """emit_completeness_assessment returns correct event."""
        result = emit_completeness_assessment(90, [], [], "proceed_to_synthesis")
        assert result["content"]["event_type"] == "completeness_assessment"
        assert result["content"]["details"]["coverage_pct"] == 90
        assert result["content"]["details"]["passed"] is True

    def test_emit_completeness_assessment_with_gaps(self):
        """emit_completeness_assessment with gaps."""
        result = emit_completeness_assessment(60, ["g1"], [], "need_more_research")
        assert result["content"]["details"]["gaps"] == ["g1"]
        assert result["content"]["details"]["passed"] is False


# ---------------------------------------------------------------------------
# TestCompletionEvents
# ---------------------------------------------------------------------------


class TestCompletionEvents:
    """Test cases for completion phase emit functions."""

    def test_emit_final_answer_minimal(self):
        """emit_final_answer returns correct event."""
        result = emit_final_answer("Final answer text")
        assert result["content"]["event_type"] == "final_answer"
        assert result["content"]["details"]["final_answer"] == "Final answer text"

    def test_emit_final_answer_with_visualizations(self):
        """emit_final_answer includes visualizations when provided."""
        viz = [{"type": "bar", "data": []}]
        result = emit_final_answer("Answer", visualizations=viz)
        assert result["content"]["details"]["visualizations"] == viz
        assert result["content"]["details"]["visualization_count"] == 1

    def test_emit_completed_no_timing(self):
        """emit_completed without elapsed seconds."""
        result = emit_completed()
        assert result["content"]["event_type"] == "completed"

    def test_emit_completed_with_timing(self):
        """emit_completed with elapsed seconds."""
        result = emit_completed(effective_elapsed_seconds=120.5)
        assert result["content"]["details"]["effective_elapsed_seconds"] == 120.5

    def test_emit_completed_with_pre_plan_elapsed(self):
        """emit_completed includes pre_plan_elapsed_seconds when provided."""
        result = emit_completed(pre_plan_elapsed_seconds=5.2)
        assert result["content"]["details"]["pre_plan_elapsed_seconds"] == 5.2

    def test_emit_error_minimal(self):
        """emit_error returns correct event."""
        result = emit_error("Something broke")
        assert result["content"]["event_type"] == "error"
        assert result["content"]["details"]["error"] == "Something broke"

    def test_emit_error_truncates_long_error(self):
        """emit_error truncates long error via _simplify_error."""
        long_err = "x" * 600
        result = emit_error(long_err)
        assert len(result["content"]["details"]["error"]) == 500

    def test_emit_cancelled_minimal(self):
        """emit_cancelled returns correct event."""
        result = emit_cancelled("thread-123")
        assert result["content"]["event_type"] == "cancelled"
        assert result["content"]["details"]["thread_id"] == "thread-123"


# ---------------------------------------------------------------------------
# TestAgentConversationEvents
# ---------------------------------------------------------------------------


class TestAgentConversationEvents:
    """Test cases for agent conversation emit functions."""

    def test_emit_agent_thinking_minimal(self):
        """emit_agent_thinking returns correct event."""
        result = emit_agent_thinking("Planner", "Considering options")
        assert result["content"]["event_type"] == "agent_thinking"
        assert result["content"]["details"]["agent"] == "Planner"
        assert result["content"]["details"]["thought"] == "Considering options"

    def test_emit_agent_decision_minimal(self):
        """emit_agent_decision without reasoning."""
        result = emit_agent_decision("Planner", "Proceed")
        assert result["content"]["event_type"] == "agent_decision"
        assert result["content"]["details"]["decision"] == "Proceed"

    def test_emit_agent_decision_with_reasoning(self):
        """emit_agent_decision with reasoning."""
        result = emit_agent_decision("Planner", "Proceed", "Data looks good")
        assert result["content"]["details"]["reasoning"] == "Data looks good"

    def test_emit_agent_message_minimal(self):
        """emit_agent_message returns correct event."""
        result = emit_agent_message("Supervisor", "Worker", "Do task X")
        assert result["content"]["event_type"] == "agent_message"
        assert result["content"]["details"]["from_agent"] == "Supervisor"
        assert result["content"]["details"]["to_agent"] == "Worker"
        assert result["content"]["details"]["message_type"] == "request"

    def test_emit_reviewer_feedback_minimal(self):
        """emit_reviewer_feedback returns correct event."""
        result = emit_reviewer_feedback("Skeptic", "Needs more sources")
        assert result["content"]["event_type"] == "reviewer_feedback"
        assert result["content"]["details"]["reviewer"] == "Skeptic"
        assert result["content"]["details"]["feedback"] == "Needs more sources"

    def test_emit_reviewer_feedback_with_strengths_weaknesses(self):
        """emit_reviewer_feedback includes strengths and weaknesses."""
        result = emit_reviewer_feedback("S", "F", strengths=["s1"], weaknesses=["w1"])
        assert result["content"]["details"]["strengths"] == ["s1"]
        assert result["content"]["details"]["weaknesses"] == ["w1"]

    def test_emit_reviewer_score_minimal(self):
        """emit_reviewer_score returns correct event."""
        result = emit_reviewer_score("Skeptic", 75, 100)
        assert result["content"]["event_type"] == "reviewer_score"
        assert result["content"]["details"]["score"] == 75
        assert result["content"]["details"]["max_score"] == 100

    def test_emit_consensus_vote_minimal(self):
        """emit_consensus_vote returns correct event."""
        result = emit_consensus_vote("Agent1", "approve", 0.9, "Looks good")
        assert result["content"]["event_type"] == "consensus_vote"
        assert result["content"]["details"]["vote"] == "approve"
        assert result["content"]["details"]["confidence"] == pytest.approx(0.9)

    def test_emit_consensus_result_minimal(self):
        """emit_consensus_result returns correct event."""
        result = emit_consensus_result(
            approved=True,
            approve_count=2,
            reject_count=0,
            revision_count=1,
            overall_confidence=0.85,
            summary="Approved",
        )
        assert result["content"]["event_type"] == "consensus_result"
        assert result["content"]["details"]["approved"] is True
        assert result["content"]["details"]["approve_count"] == 2


# ---------------------------------------------------------------------------
# TestWorkerEvents
# ---------------------------------------------------------------------------


class TestWorkerEvents:
    """Test cases for worker emit functions."""

    def test_emit_worker_self_evaluation_minimal(self):
        """emit_worker_self_evaluation returns correct event."""
        result = emit_worker_self_evaluation("q", 0.8, "high", False, 1)
        assert result["content"]["event_type"] == "worker_self_evaluation"
        assert result["content"]["details"]["quality_score"] == pytest.approx(0.8)
        assert result["content"]["details"]["will_retry"] is False

    def test_emit_worker_reformulation_minimal(self):
        """emit_worker_reformulation returns correct event."""
        result = emit_worker_reformulation("q", "orig", "reform", 1)
        assert result["content"]["event_type"] == "worker_reformulation"
        assert result["content"]["details"]["original_query"] == "orig"
        assert result["content"]["details"]["reformulated_query"] == "reform"

    def test_emit_worker_progress_minimal(self):
        """emit_worker_progress returns correct event."""
        result = emit_worker_progress("query", 1, 3, "done")
        assert result["content"]["event_type"] == "worker_progress"
        assert result["content"]["details"]["status"] == "done"


# ---------------------------------------------------------------------------
# TestUtilityEvents
# ---------------------------------------------------------------------------


class TestUtilityEvents:
    """Test cases for utility emit functions."""

    def test_emit_token_usage_update_minimal(self):
        """emit_token_usage_update returns correct event."""
        result = emit_token_usage_update(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            llm_calls=2,
            estimated_cost_usd=0.001,
        )
        assert result["content"]["event_type"] == "token_usage_update"
        assert result["content"]["details"]["total_tokens"] == 150
        assert result["content"]["details"]["llm_calls"] == 2

    def test_emit_context_usage_update_minimal(self):
        """emit_context_usage_update returns correct event."""
        result = emit_context_usage_update(
            current_tokens=50000, max_tokens=200000, usage_percent=25.0, status="normal"
        )
        assert result["content"]["event_type"] == "context_usage_update"
        assert result["content"]["details"]["current_tokens"] == 50000
        assert result["content"]["details"]["status"] == "normal"

    def test_emit_triage_decision_minimal(self):
        """emit_triage_decision returns correct event."""
        result = emit_triage_decision("context_sufficient", "Prior research covers it")
        assert result["content"]["event_type"] == "triage_decision"
        assert result["content"]["details"]["decision"] == "context_sufficient"

    def test_emit_heartbeat_returns_event(self):
        """emit_heartbeat returns event with heartbeat flag."""
        result = emit_heartbeat()
        assert result["content"]["details"]["heartbeat"] is True
        assert result["content"]["ui_visible"] is False

    def test_emit_inter_agent_message_minimal(self):
        """emit_inter_agent_message returns correct event."""
        result = emit_inter_agent_message("A", "B", "delegate", "Summary")
        assert result["content"]["event_type"] == "inter_agent_message"
        assert result["content"]["details"]["from_agent"] == "A"
        assert result["content"]["details"]["to_agent"] == "B"

    def test_emit_subquery_validation_minimal(self):
        """emit_subquery_validation returns correct event."""
        result = emit_subquery_validation(5, 3, 1, 1)
        assert result["content"]["event_type"] == "subquery_validation"
        assert result["content"]["details"]["valid_count"] == 3
        assert result["content"]["details"]["reformulated_count"] == 1
        assert result["content"]["details"]["removed_count"] == 1


# ---------------------------------------------------------------------------
# TestEmitEvent
# ---------------------------------------------------------------------------


class TestEmitEvent:
    """Test cases for the base emit_event function."""

    def test_emit_event_minimal(self):
        """emit_event with minimal args returns correct structure."""
        result = emit_event(
            DeepResearchEventType.STARTED,
            "Test message",
        )
        assert result["type"] == "deep_research_status"
        assert result["content"]["event_type"] == "started"
        assert result["content"]["message"] == "Test message"
        assert result["content"]["display_text"] == "Test message"
        assert result["content"]["ui_visible"] is True

    def test_emit_event_with_display_text_override(self):
        """emit_event uses display_text when provided."""
        result = emit_event(
            DeepResearchEventType.STARTED,
            "msg",
            display_text="Display",
        )
        assert result["content"]["display_text"] == "Display"

    def test_emit_event_with_details(self):
        """emit_event includes details when provided."""
        result = emit_event(
            DeepResearchEventType.STARTED,
            "msg",
            details={"key": "value"},
        )
        assert result["content"]["details"] == {"key": "value"}

    def test_emit_event_ui_visible_false(self):
        """emit_event respects ui_visible=False."""
        result = emit_event(
            DeepResearchEventType.STARTED,
            "msg",
            ui_visible=False,
        )
        assert result["content"]["ui_visible"] is False

    def test_emit_event_with_stage_override(self):
        """emit_event uses stage when provided."""
        result = emit_event(
            DeepResearchEventType.STARTED,
            "msg",
            stage="custom_stage",
        )
        assert result["content"]["stage"] == "custom_stage"


# ---------------------------------------------------------------------------
# TestDeepResearchEvent
# ---------------------------------------------------------------------------


class TestDeepResearchEvent:
    """Test cases for DeepResearchEvent dataclass."""

    def test_to_dict_structure(self):
        """DeepResearchEvent.to_dict returns correct structure."""
        event = DeepResearchEvent(
            event_type=DeepResearchEventType.STARTED,
            message="msg",
            display_text="disp",
        )
        d = event.to_dict()
        assert d["type"] == "deep_research_status"
        assert "content" in d
        assert d["content"]["event_type"] == "started"
        assert d["content"]["message"] == "msg"
        assert d["content"]["display_text"] == "disp"
        assert "log_entry" in d["content"]

    def test_to_dict_includes_stage_from_event_type_when_none(self):
        """When stage is None, content stage defaults to event_type value."""
        event = DeepResearchEvent(
            event_type=DeepResearchEventType.STARTED,
            message="msg",
            display_text="disp",
            stage=None,
        )
        d = event.to_dict()
        assert d["content"]["stage"] == "started"


# ---------------------------------------------------------------------------
# TestEdgeCaseEvents
# ---------------------------------------------------------------------------


class TestEdgeCaseEvents:
    """Test cases for edge-case emit functions."""

    def test_emit_complexity_assessment_returns_agent_decision_type(self):
        """emit_complexity_assessment uses AGENT_DECISION event type."""
        result = emit_complexity_assessment("simple", 3, 1, 1, "reasoning")
        assert result["content"]["event_type"] == "agent_decision"
        assert result["content"]["details"]["complexity_class"] == "simple"

    def test_emit_sentinel_triggered_returns_agent_decision_type(self):
        """emit_sentinel_triggered uses AGENT_DECISION event type."""
        result = emit_sentinel_triggered("budget", "node", {"k": "v"})
        assert result["content"]["event_type"] == "agent_decision"
        assert result["content"]["details"]["sentinel_reason"] == "budget"

    def test_emit_research_failed_returns_error_type(self):
        """emit_research_failed uses ERROR event type."""
        result = emit_research_failed(
            "High failure rate",
            {"timeout": 3},
            10,
            0.3,
        )
        assert result["content"]["event_type"] == "error"
        assert result["content"]["details"]["abort_reason"] == "High failure rate"

    def test_emit_diminishing_returns_returns_agent_decision_type(self):
        """emit_diminishing_returns uses AGENT_DECISION event type."""
        result = emit_diminishing_returns([80.0, 85.0, 86.0], 1.0)
        assert result["content"]["event_type"] == "agent_decision"

    def test_emit_no_valid_findings_returns_error_type(self):
        """emit_no_valid_findings uses ERROR event type."""
        result = emit_no_valid_findings(5, 5)
        assert result["content"]["event_type"] == "error"

    def test_emit_empty_plan_returns_agent_decision_type(self):
        """emit_empty_plan uses AGENT_DECISION event type."""
        result = emit_empty_plan()
        assert result["content"]["event_type"] == "agent_decision"

    def test_emit_reliability_update_minimal(self):
        """emit_reliability_update returns correct event."""
        metrics = {
            "compliance_score": 0.9,
            "fault_recovery_rate": 0.8,
            "resource_consistency": 0.7,
            "plan_alignment": 0.85,
        }
        result = emit_reliability_update(metrics)
        assert result["content"]["event_type"] == "reliability_update"
        assert result["content"]["details"]["compliance_score"] == pytest.approx(0.9)
