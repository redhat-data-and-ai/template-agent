"""Comprehensive pytest tests for the deep research COMPLETENESS node."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from template_agent.src.core.deep_research.nodes.completeness import (
    _apply_convergence_tracking,
    _apply_quality_early_exit,
    _build_validation_notes,
    _get_completeness_threshold,
    _get_findings_summary_and_count,
    _make_fallback_eval_result,
    _should_route_to_supervisor,
    completeness_evaluator_node,
)
from template_agent.src.core.deep_research.state import (
    PHASE_SUPERVISOR,
    PHASE_SYNTHESIZE,
    ResearchContext,
)


def _make_ctx(**overrides):
    """Create minimal ResearchContext for testing."""
    base_model = AsyncMock()
    ctx = ResearchContext(tools=[MagicMock(name="tool1")], base_model=base_model)
    for k, v in overrides.items():
        setattr(ctx, k, v)
    return ctx


def _make_state(**overrides):
    """Create minimal DeepResearchState dict with completeness context."""
    base = {
        "query": "test query",
        "thread_id": "t1",
        "current_phase": "completeness",
        "subqueries": ["sq1", "sq2"],
        "findings_board": {},
        "completed_subqueries": [],
        "current_round": 1,
        "max_rounds": 3,
        "understanding": "",
        "fallback_count": 0,
        "coverage_history": [],
    }
    base.update(overrides)
    return base


class TestGetFindingsSummaryAndCount:
    """Tests for _get_findings_summary_and_count."""

    def test_empty_findings_board_returns_no_findings_summary(self):
        """Empty findings_board yields 'No findings' style summary from findings."""
        findings_board = {}
        findings = {"sq1": {"answer": "a1", "error": None}}

        summary, count = _get_findings_summary_and_count(findings_board, findings)

        assert count == 1
        assert "a1" in summary or "sq1" in summary

    def test_findings_board_with_successful_entries_counts_correctly(self):
        """Successful findings (no error) are counted."""
        findings_board = {
            "sq1": {"finding": {"answer": "a1"}},
            "sq2": {"finding": {"answer": "a2", "error": "failed"}},
        }
        findings = {}

        summary, count = _get_findings_summary_and_count(findings_board, findings)

        assert count == 1
        assert "a1" in summary

    def test_findings_with_error_excluded_from_count(self):
        """Findings with error are excluded from successful count."""
        findings_board = {
            "sq1": {"finding": {"answer": "a1", "error": "failed"}},
        }
        findings = {}

        summary, count = _get_findings_summary_and_count(findings_board, findings)

        assert count == 0


class TestGetCompletenessThreshold:
    """Tests for _get_completeness_threshold."""

    def test_default_threshold_when_no_mode_config(self):
        """Returns default 70 when no _mode_config."""
        state = _make_state()
        assert _get_completeness_threshold(state) == 70

    def test_mode_config_threshold_used_when_present(self):
        """Uses mode_config.completeness_threshold when set."""
        mode_config = MagicMock()
        mode_config.completeness_threshold = 85
        state = _make_state(_mode_config=mode_config)
        assert _get_completeness_threshold(state) == 85


class TestMakeFallbackEvalResult:
    """Tests for _make_fallback_eval_result."""

    def test_round_lt_max_returns_needs_more_research(self):
        """When round < max_rounds, decision is needs_more_research."""
        result = _make_fallback_eval_result(1, 3, "parse failed")
        assert result["decision"] == "needs_more_research"
        assert result["coverage_pct"] == 0
        assert result["reasoning"] == "parse failed"

    def test_round_eq_max_returns_ready_for_synthesis(self):
        """When round >= max_rounds, decision is ready_for_synthesis."""
        result = _make_fallback_eval_result(3, 3, "fallback")
        assert result["decision"] == "ready_for_synthesis"


class TestApplyConvergenceTracking:
    """Tests for _apply_convergence_tracking."""

    def test_history_lt_2_returns_decision_unchanged(self):
        """Coverage history with < 2 entries returns decision unchanged."""
        events = []
        ctx = _make_ctx()
        result = _apply_convergence_tracking([70.0], "needs_more_research", ctx, events)
        assert result == "needs_more_research"

    def test_delta_ge_5_returns_decision_unchanged(self):
        """Delta >= 5 keeps decision."""
        events = []
        ctx = _make_ctx()
        result = _apply_convergence_tracking(
            [70.0, 80.0], "needs_more_research", ctx, events
        )
        assert result == "needs_more_research"

    def test_delta_lt_5_returns_ready_for_synthesis(self):
        """Delta < 5 triggers diminishing returns -> ready_for_synthesis."""
        events = []
        ctx = _make_ctx()
        result = _apply_convergence_tracking(
            [70.0, 72.0], "needs_more_research", ctx, events
        )
        assert result == "ready_for_synthesis"
        assert len(events) >= 1


class TestApplyQualityEarlyExit:
    """Tests for _apply_quality_early_exit."""

    def test_high_coverage_no_contradictions_early_exit(self):
        """Coverage >= 90, no contradictions -> early synthesis."""
        emitter = MagicMock()
        result = _apply_quality_early_exit(
            coverage_pct=92,
            contradictions=[],
            successful_count=3,
            decision="needs_more_research",
            emitter=emitter,
        )
        assert result == "ready_for_synthesis"
        emitter.thinking.assert_called_once()

    def test_contradictions_present_no_early_exit(self):
        """Contradictions prevent early exit."""
        emitter = MagicMock()
        result = _apply_quality_early_exit(
            coverage_pct=95,
            contradictions=[{"description": "conflict"}],
            successful_count=2,
            decision="needs_more_research",
            emitter=emitter,
        )
        assert result == "needs_more_research"

    def test_already_ready_returns_unchanged(self):
        """Already ready_for_synthesis returns unchanged."""
        emitter = MagicMock()
        result = _apply_quality_early_exit(
            coverage_pct=95,
            contradictions=[],
            successful_count=2,
            decision="ready_for_synthesis",
            emitter=emitter,
        )
        assert result == "ready_for_synthesis"
        emitter.thinking.assert_not_called()


class TestBuildValidationNotes:
    """Tests for _build_validation_notes."""

    def test_numeric_issues_included(self):
        """Numeric issues are included in notes."""
        notes = _build_validation_notes(
            numeric_issues=["value too high"],
            contradictions=[],
        )
        assert "Numeric issues" in notes
        assert "value too high" in notes

    def test_contradictions_included(self):
        """Contradictions are included in notes."""
        notes = _build_validation_notes(
            numeric_issues=[],
            contradictions=[{"description": "Revenue mismatch"}],
        )
        assert "Contradictions" in notes
        assert "Revenue mismatch" in notes

    def test_empty_returns_empty_string(self):
        """Empty inputs return empty string."""
        assert _build_validation_notes([], []) == ""


class TestShouldRouteToSupervisor:
    """Tests for _should_route_to_supervisor."""

    def test_needs_more_research_with_follow_ups_routes(self):
        """needs_more_research + follow_ups + below threshold -> True."""
        assert (
            _should_route_to_supervisor(
                decision="needs_more_research",
                round_num=1,
                max_rounds=3,
                follow_ups=["sq3"],
                coverage_pct=50,
                completeness_threshold=70,
            )
            is True
        )

    def test_ready_for_synthesis_does_not_route(self):
        """ready_for_synthesis never routes."""
        assert (
            _should_route_to_supervisor(
                decision="ready_for_synthesis",
                round_num=1,
                max_rounds=3,
                follow_ups=["sq3"],
                coverage_pct=50,
                completeness_threshold=70,
            )
            is False
        )

    def test_no_follow_ups_does_not_route(self):
        """Empty follow_ups does not route."""
        assert (
            _should_route_to_supervisor(
                decision="needs_more_research",
                round_num=1,
                max_rounds=3,
                follow_ups=[],
                coverage_pct=50,
                completeness_threshold=70,
            )
            is False
        )

    def test_coverage_above_threshold_does_not_route(self):
        """Coverage above threshold does not route."""
        assert (
            _should_route_to_supervisor(
                decision="needs_more_research",
                round_num=1,
                max_rounds=3,
                follow_ups=["sq3"],
                coverage_pct=80,
                completeness_threshold=70,
            )
            is False
        )

    def test_max_rounds_reached_does_not_route(self):
        """Round >= max_rounds does not route."""
        assert (
            _should_route_to_supervisor(
                decision="needs_more_research",
                round_num=3,
                max_rounds=3,
                follow_ups=["sq3"],
                coverage_pct=50,
                completeness_threshold=70,
            )
            is False
        )


class TestCompletenessEvaluatorNode:
    """Tests for completeness_evaluator_node (main entry point)."""

    @pytest.mark.asyncio
    async def test_completeness_node_returns_coverage_complete_true_when_ready(self):
        """LLM returns ready_for_synthesis -> coverage_complete=True, PHASE_SYNTHESIZE."""
        state = _make_state(
            findings_board={
                "sq1": {"finding": {"answer": "Answer one"}},
            },
        )
        ctx = _make_ctx()

        with patch(
            "template_agent.src.core.deep_research.nodes.completeness.tracked_invoke",
            new_callable=AsyncMock,
        ) as mock_invoke:
            mock_response = MagicMock()
            mock_response.content = (
                '{"decision": "ready_for_synthesis", "coverage_pct": 85}'
            )
            mock_invoke.return_value = mock_response

            updates, events = await completeness_evaluator_node(state, ctx)

        assert updates["coverage_complete"] is True
        assert updates["current_phase"] == PHASE_SYNTHESIZE

    @pytest.mark.asyncio
    async def test_completeness_node_routes_to_supervisor_when_needs_more(self):
        """LLM returns needs_more_research with follow_ups -> routes to supervisor."""
        state = _make_state(
            findings_board={
                "sq1": {"finding": {"answer": "Answer one"}},
            },
            current_round=1,
            max_rounds=3,
        )
        ctx = _make_ctx()

        with patch(
            "template_agent.src.core.deep_research.nodes.completeness.tracked_invoke",
            new_callable=AsyncMock,
        ) as mock_invoke:
            mock_response = MagicMock()
            mock_response.content = (
                '{"decision": "needs_more_research", "coverage_pct": 50, '
                '"follow_up_subqueries": ["sq3", "sq4"]}'
            )
            mock_invoke.return_value = mock_response

            updates, events = await completeness_evaluator_node(state, ctx)

        assert updates["coverage_complete"] is False
        assert updates["current_phase"] == PHASE_SUPERVISOR
        assert "pending_subqueries" in updates

    @pytest.mark.asyncio
    async def test_completeness_node_fallback_on_parse_failure(self):
        """Unparseable LLM response triggers fallback -> ready at max rounds."""
        state = _make_state(
            findings_board={"sq1": {"finding": {"answer": "a1"}}},
            current_round=3,
            max_rounds=3,
        )
        ctx = _make_ctx()

        with patch(
            "template_agent.src.core.deep_research.nodes.completeness.tracked_invoke",
            new_callable=AsyncMock,
        ) as mock_invoke:
            mock_response = MagicMock()
            mock_response.content = "not valid json"
            mock_invoke.return_value = mock_response

            updates, events = await completeness_evaluator_node(state, ctx)

        assert updates["coverage_complete"] is True
        assert updates["current_phase"] == PHASE_SYNTHESIZE

    @pytest.mark.asyncio
    async def test_completeness_node_sentinel_triggered_returns_early(self):
        """Sentinel triggered returns early with forced phase."""
        state = _make_state(
            total_node_transitions=999,
            max_node_transitions=100,
        )
        ctx = _make_ctx()

        with patch(
            "template_agent.src.core.deep_research.nodes.completeness.check_loop_sentinel"
        ) as mock_sentinel:
            mock_sentinel.return_value = (True, "max transitions", "synthesize")

            updates, events = await completeness_evaluator_node(state, ctx)

        assert updates.get("sentinel_triggered") is True
        assert "current_phase" in updates

    @pytest.mark.asyncio
    async def test_completeness_node_emits_validation_events(self):
        """Node emits validation start and complete events."""
        state = _make_state(
            findings_board={"sq1": {"finding": {"answer": "a1"}}},
        )
        ctx = _make_ctx()

        with patch(
            "template_agent.src.core.deep_research.nodes.completeness.tracked_invoke",
            new_callable=AsyncMock,
        ) as mock_invoke:
            mock_response = MagicMock()
            mock_response.content = (
                '{"decision": "ready_for_synthesis", "coverage_pct": 80}'
            )
            mock_invoke.return_value = mock_response

            _, events = await completeness_evaluator_node(state, ctx)

        assert len(events) >= 2
