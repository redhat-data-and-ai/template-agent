"""Comprehensive pytest tests for the complexity assessment node."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from template_agent.src.core.deep_research.nodes.complexity import (
    _apply_user_overrides,
    _compute_session_seconds,
    _get_mode_bounds,
    _parse_assessor_response,
    assess_complexity_node,
)
from template_agent.src.core.deep_research.state import ResearchContext


def _make_state(**overrides):
    """Create minimal DeepResearchState dict with required fields."""
    base = {"query": "test query", "thread_id": "t1", "current_phase": "plan"}
    base.update(overrides)
    return base


def _make_ctx(**overrides):
    """Create minimal ResearchContext for testing."""
    base_model = AsyncMock()
    ctx = ResearchContext(tools=[MagicMock(name="tool1")], base_model=base_model)
    for k, v in overrides.items():
        setattr(ctx, k, v)
    return ctx


def _make_mode(recommended_subqueries: int = 5, **overrides) -> SimpleNamespace:
    """Create mode config with defaults for _get_mode_bounds."""
    defaults = {
        "min_subqueries": 3,
        "max_subqueries": 12,
        "max_supervisor_rounds": 2,
        "max_review_iterations": 2,
        "max_node_transitions": 30,
        "session_timeout_seconds": 600,
        "recommended_subqueries": recommended_subqueries,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestParseAssessorResponse:
    """Test _parse_assessor_response helper."""

    def test_parse_valid_response_extracts_all_fields(self):
        """Valid response extracts complexity_class, subqueries, rounds, iterations."""
        result = {
            "complexity_class": "moderate",
            "recommended_subqueries": 6,
            "recommended_supervisor_rounds": 2,
            "recommended_review_iterations": 3,
            "reasoning": "Query needs moderate depth",
        }
        cc, sq, rnd, itr, reason = _parse_assessor_response(result)
        assert cc == "moderate"
        assert sq == 6
        assert rnd == 2
        assert itr == 3
        assert reason == "Query needs moderate depth"

    def test_parse_raises_when_missing_complexity_class(self):
        """Raises ValueError when complexity_class is missing."""
        result = {"recommended_subqueries": 5}
        with pytest.raises(ValueError, match="missing complexity_class"):
            _parse_assessor_response(result)

    def test_parse_raises_when_empty_result(self):
        """Raises ValueError when result is None."""
        with pytest.raises(ValueError, match="missing complexity_class"):
            _parse_assessor_response(None)

    def test_parse_raises_when_invalid_complexity_class(self):
        """Raises ValueError for unknown complexity class."""
        result = {"complexity_class": "unknown_class"}
        with pytest.raises(ValueError, match="Unknown complexity class"):
            _parse_assessor_response(result)

    def test_parse_accepts_all_valid_complexity_classes(self):
        """All valid classes (simple, moderate, complex, comprehensive) parse."""
        for cls in ("simple", "moderate", "complex", "comprehensive"):
            result = {"complexity_class": cls}
            cc, _, _, _, _ = _parse_assessor_response(result)
            assert cc == cls


class TestGetModeBounds:
    """Test _get_mode_bounds helper."""

    def test_get_mode_bounds_with_none_uses_defaults(self):
        """When mode is None, returns default bounds."""
        sq_floor, sq_ceiling, rnd, itr, trans, timeout = _get_mode_bounds(None)
        assert sq_floor == 3
        assert sq_ceiling == 12
        assert rnd == 2
        assert itr == 2
        assert trans == 30
        assert timeout == 600

    def test_get_mode_bounds_with_mode_uses_config(self):
        """When mode has values, uses them."""
        mode = MagicMock()
        mode.min_subqueries = 5
        mode.max_subqueries = 15
        mode.max_supervisor_rounds = 3
        mode.max_review_iterations = 4
        mode.max_node_transitions = 40
        mode.session_timeout_seconds = 900
        sq_floor, sq_ceiling, rnd, itr, trans, timeout = _get_mode_bounds(mode)
        assert sq_floor == 5
        assert sq_ceiling == 15
        assert rnd == 3
        assert itr == 4
        assert trans == 40
        assert timeout == 900


class TestApplyUserOverrides:
    """Test _apply_user_overrides helper."""

    def test_apply_overrides_reduces_sq_ceiling_when_override_set(self):
        """max_subqueries_override reduces ceiling."""
        ctx = _make_ctx(max_subqueries_override=5)
        state = _make_state()
        sq_f, sq_c, rnd, _ = _apply_user_overrides(ctx, state, 3, 12, 2, 2)
        assert sq_c == 5
        assert sq_f <= 5

    def test_apply_overrides_respects_user_max_rounds(self):
        """_user_max_rounds_override reduces rounds_ceiling."""
        ctx = _make_ctx()
        state = _make_state(_user_max_rounds_override=1)
        _, _, rnd, _ = _apply_user_overrides(ctx, state, 3, 12, 2, 2)
        assert rnd == 1

    def test_apply_overrides_respects_user_max_iterations(self):
        """_user_max_iterations_override reduces iterations_ceiling."""
        ctx = _make_ctx()
        state = _make_state(_user_max_iterations_override=1)
        _, _, _, itr = _apply_user_overrides(ctx, state, 3, 12, 2, 2)
        assert itr == 1


class TestComputeSessionSeconds:
    """Test _compute_session_seconds helper."""

    def test_compute_session_seconds_returns_proportional_budget(self):
        """Session seconds scale with subqueries and rounds."""
        mode = MagicMock()
        mode.per_subquery_allowance = 90.0
        mode.synthesis_reserved_seconds = 90.0
        mode.review_reserved_seconds = 60.0
        result = _compute_session_seconds(mode, 5, 2, 300.0)
        assert result >= 300.0
        assert result <= 600  # DEFAULT_MAX_SESSION_SECONDS

    def test_compute_session_seconds_with_none_mode_uses_defaults(self):
        """None mode uses default allowances."""
        result = _compute_session_seconds(None, 3, 1, 400.0)
        assert result >= 400.0


class TestAssessComplexityNodeSuccess:
    """Test success paths for assess_complexity_node."""

    @pytest.mark.asyncio
    async def test_assess_complexity_node_returns_state_updates_and_events(self):
        """Node returns (state_updates, events) tuple."""
        state = _make_state()
        ctx = _make_ctx(mode_config=None)

        mock_response = MagicMock()
        mock_response.content = '{"complexity_class":"moderate","recommended_subqueries":5,"recommended_supervisor_rounds":2,"recommended_review_iterations":2,"reasoning":"Test"}'

        with patch(
            "template_agent.src.core.deep_research.nodes.complexity.tracked_invoke",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            updates, events = await assess_complexity_node(state, ctx)

        assert "assessed_max_subqueries" in updates
        assert "query_complexity_class" in updates
        assert updates["query_complexity_class"] == "moderate"
        assert len(events) >= 1

    @pytest.mark.asyncio
    async def test_assess_complexity_node_clamps_subqueries_to_ceiling(self):
        """Recommended subqueries above ceiling are clamped."""
        state = _make_state()
        ctx = _make_ctx(mode_config=None)

        mock_response = MagicMock()
        mock_response.content = '{"complexity_class":"complex","recommended_subqueries":50,"recommended_supervisor_rounds":2,"recommended_review_iterations":2,"reasoning":"Heavy"}'

        with patch(
            "template_agent.src.core.deep_research.nodes.complexity.tracked_invoke",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            updates, _ = await assess_complexity_node(state, ctx)

        assert updates["assessed_max_subqueries"] <= 12

    @pytest.mark.asyncio
    async def test_assess_complexity_node_uses_context_and_cached_count(self):
        """Prompt includes context and cached findings count."""
        state = _make_state(
            context="User: prior question\nAssistant: prior answer",
            findings_board={"sq1": {}},
        )
        ctx = _make_ctx(mode_config=None)

        with patch(
            "template_agent.src.core.deep_research.nodes.complexity.tracked_invoke",
            new_callable=AsyncMock,
        ) as mock_invoke:
            mock_invoke.return_value = MagicMock(
                content='{"complexity_class":"simple","recommended_subqueries":3,"recommended_supervisor_rounds":1,"recommended_review_iterations":1,"reasoning":"Simple"}'
            )
            await assess_complexity_node(state, ctx)

        mock_invoke.assert_awaited_once()


class TestAssessComplexityNodeErrorHandling:
    """Test error handling for assess_complexity_node."""

    @pytest.mark.asyncio
    async def test_assess_complexity_node_fallback_on_llm_exception(self):
        """When LLM raises, falls back to mode defaults."""
        state = _make_state()
        mode = _make_mode(recommended_subqueries=5)
        ctx = _make_ctx(mode_config=mode)

        with patch(
            "template_agent.src.core.deep_research.nodes.complexity.tracked_invoke",
            new_callable=AsyncMock,
            side_effect=RuntimeError("LLM unavailable"),
        ):
            updates, _ = await assess_complexity_node(state, ctx)

        assert updates["query_complexity_class"] == "moderate"
        assert "Heuristic fallback" in updates["complexity_reasoning"]
        assert updates["assessed_max_subqueries"] == 5

    @pytest.mark.asyncio
    async def test_assess_complexity_node_fallback_on_invalid_json(self):
        """When response is invalid JSON, falls back."""
        state = _make_state()
        mode = _make_mode(recommended_subqueries=4)
        ctx = _make_ctx(mode_config=mode)

        mock_response = MagicMock()
        mock_response.content = "not valid json {{{"

        with patch(
            "template_agent.src.core.deep_research.nodes.complexity.tracked_invoke",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            updates, _ = await assess_complexity_node(state, ctx)

        assert updates["query_complexity_class"] == "moderate"
        assert "Heuristic fallback" in updates["complexity_reasoning"]


class TestAssessComplexityNodeEdgeCases:
    """Test edge cases for assess_complexity_node."""

    @pytest.mark.asyncio
    async def test_assess_complexity_node_empty_query_handled(self):
        """Empty query does not raise."""
        state = _make_state(query="")
        ctx = _make_ctx(mode_config=None)

        mock_response = MagicMock()
        mock_response.content = '{"complexity_class":"simple","recommended_subqueries":3,"recommended_supervisor_rounds":1,"recommended_review_iterations":1,"reasoning":""}'

        with patch(
            "template_agent.src.core.deep_research.nodes.complexity.tracked_invoke",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            updates, _ = await assess_complexity_node(state, ctx)

        assert "assessed_max_subqueries" in updates

    @pytest.mark.asyncio
    async def test_assess_complexity_node_stores_mode_config_when_present(self):
        """_mode_config is stored in state when mode is set."""
        state = _make_state()
        mode = _make_mode(recommended_subqueries=5)
        ctx = _make_ctx(mode_config=mode)

        mock_response = MagicMock()
        mock_response.content = '{"complexity_class":"moderate","recommended_subqueries":5,"recommended_supervisor_rounds":2,"recommended_review_iterations":2,"reasoning":"Ok"}'

        with patch(
            "template_agent.src.core.deep_research.nodes.complexity.tracked_invoke",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            updates, _ = await assess_complexity_node(state, ctx)

        assert updates.get("_mode_config") is mode
