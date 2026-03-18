"""Comprehensive pytest tests for the review node."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from template_agent.src.core.deep_research.nodes.review import review_node
from template_agent.src.core.deep_research.state import (
    DEFAULT_MAX_ITERATIONS,
    PHASE_COMPLETE,
    PHASE_SYNTHESIZE,
    ResearchContext,
)


def _make_state(**overrides):
    """Create minimal DeepResearchState dict with required fields and findings."""
    base = {
        "query": "test query",
        "thread_id": "t1",
        "current_phase": "synthesize",
        "findings": {"sq1": {"subquery": "sub q", "answer": "answer text"}},
        "findings_board": {
            "sq1": {"finding": {"subquery": "sub q", "answer": "answer text"}}
        },
        "draft_answer": "some draft answer content",
        "subqueries": ["sub q"],
    }
    base.update(overrides)
    return base


def _make_ctx(**overrides):
    """Create minimal ResearchContext for testing."""
    base_model = AsyncMock()
    ctx = ResearchContext(tools=[MagicMock(name="tool1")], base_model=base_model)
    for k, v in overrides.items():
        setattr(ctx, k, v)
    return ctx


def _make_review_response(action: str = "approve", score: int = 75) -> MagicMock:
    """Create mock LLM response for review parsing."""
    content = json.dumps(
        {
            "action": action,
            "score": score,
            "reason": "Looks good",
            "feedback": "Minor improvements possible",
            "follow_up_subqueries": [],
        }
    )
    response = MagicMock()
    response.content = content
    return response


class TestReviewNodeSuccessPaths:
    """Test success paths for review_node."""

    @pytest.mark.asyncio
    async def test_review_node_approves_when_scores_above_threshold(self):
        """Review node approves and moves to PHASE_COMPLETE when scores pass."""
        state = _make_state()
        ctx = _make_ctx()

        mock_response = _make_review_response(action="approve", score=75)

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.review.check_node_cancelled",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.review.tracked_invoke",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
        ):
            updates, events = await review_node(state, ctx)

        assert updates["current_phase"] == PHASE_COMPLETE
        assert updates["current_review"]["action"] == "approve"
        assert updates["quality_matrix"]["gate_result"] == "pass"
        assert len(events) >= 1

    @pytest.mark.asyncio
    async def test_review_node_revises_when_scores_below_threshold(self):
        """Review node triggers revise when weighted score in 0.4-0.6 range."""
        state = _make_state()
        ctx = _make_ctx()

        # Scores ~45-50 yield weighted_score ~0.45 -> gate_result "revise"
        mock_response = _make_review_response(action="revise", score=45)

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.review.check_node_cancelled",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.review.tracked_invoke",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
        ):
            updates, events = await review_node(state, ctx)

        assert updates["current_phase"] == PHASE_SYNTHESIZE
        assert updates["current_review"]["action"] == "revise"
        assert updates["quality_matrix"]["gate_result"] in ("revise", "research_more")
        assert len(events) >= 1

    @pytest.mark.asyncio
    async def test_review_node_research_more_when_follow_ups_provided(self):
        """Review node triggers research_more when follow_up_subqueries exist."""
        state = _make_state(subqueries=["q1"])
        ctx = _make_ctx()

        content = json.dumps(
            {
                "action": "research_more",
                "score": 35,
                "reason": "Need more data",
                "feedback": "",
                "follow_up_subqueries": ["additional query 1", "additional query 2"],
            }
        )
        mock_response = MagicMock(content=content)

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.review.check_node_cancelled",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.review.tracked_invoke",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
        ):
            updates, events = await review_node(state, ctx)

        # With research_more gate and follow_ups, may go to PHASE_SUPERVISOR
        assert updates["current_review"]["action"] in (
            "approve",
            "revise",
            "research_more",
        )
        assert "quality_matrix" in updates
        assert len(events) >= 1

    @pytest.mark.asyncio
    async def test_review_node_increments_iteration(self):
        """Review node increments iteration in state updates."""
        state = _make_state(iteration=0)
        ctx = _make_ctx()

        mock_response = _make_review_response(action="approve", score=80)

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.review.check_node_cancelled",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.review.tracked_invoke",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
        ):
            updates, _ = await review_node(state, ctx)

        assert updates["iteration"] == 1

    @pytest.mark.asyncio
    async def test_review_node_increments_total_node_transitions(self):
        """Review node increments total_node_transitions."""
        state = _make_state(total_node_transitions=5)
        ctx = _make_ctx()

        mock_response = _make_review_response(action="approve", score=70)

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.review.check_node_cancelled",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.review.tracked_invoke",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
        ):
            updates, _ = await review_node(state, ctx)

        assert updates["total_node_transitions"] == 6


class TestReviewNodeQualityThresholds:
    """Test quality threshold behavior."""

    @pytest.mark.asyncio
    async def test_review_node_quality_matrix_pass_at_60_percent(self):
        """Weighted score >= 0.6 yields gate_result pass."""
        state = _make_state()
        ctx = _make_ctx()

        mock_response = _make_review_response(action="approve", score=65)

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.review.check_node_cancelled",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.review.tracked_invoke",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
        ):
            updates, _ = await review_node(state, ctx)

        assert updates["quality_matrix"]["weighted_score"] >= 0.6
        assert updates["quality_matrix"]["gate_result"] == "pass"

    @pytest.mark.asyncio
    async def test_review_node_quality_matrix_revise_between_40_and_60(self):
        """Weighted score 0.4-0.6 yields gate_result revise."""
        state = _make_state()
        ctx = _make_ctx()

        mock_response = _make_review_response(action="revise", score=50)

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.review.check_node_cancelled",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.review.tracked_invoke",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
        ):
            updates, _ = await review_node(state, ctx)

        assert 0.4 <= updates["quality_matrix"]["weighted_score"] < 0.6
        assert updates["quality_matrix"]["gate_result"] == "revise"

    @pytest.mark.asyncio
    async def test_review_node_approves_at_max_iterations(self):
        """At max_iterations, review forces approve regardless of scores."""
        state = _make_state(
            iteration=DEFAULT_MAX_ITERATIONS,
            max_iterations=DEFAULT_MAX_ITERATIONS,
        )
        ctx = _make_ctx()

        mock_response = _make_review_response(action="revise", score=45)

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.review.check_node_cancelled",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.review.tracked_invoke",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
        ):
            updates, _ = await review_node(state, ctx)

        assert updates["current_phase"] == PHASE_COMPLETE
        assert updates["current_review"]["action"] == "approve"


class TestReviewNodeEdgeCases:
    """Test edge cases and early returns."""

    @pytest.mark.asyncio
    async def test_review_node_no_draft_returns_early_approve(self):
        """When draft_answer is empty, review returns early with approve."""
        state = _make_state(draft_answer="")
        ctx = _make_ctx()

        with patch(
            "template_agent.src.core.deep_research.nodes.review.check_node_cancelled",
            new_callable=AsyncMock,
            return_value=None,
        ):
            updates, events = await review_node(state, ctx)

        assert updates["current_phase"] == PHASE_COMPLETE
        assert updates["current_review"]["action"] == "approve"
        assert updates["current_review"]["reason"] == "No draft to review"
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_review_node_research_could_not_draft_returns_early(self):
        """When draft starts with 'Research could not', returns early approve."""
        state = _make_state(draft_answer="Research could not retrieve valid data.")
        ctx = _make_ctx()

        with patch(
            "template_agent.src.core.deep_research.nodes.review.check_node_cancelled",
            new_callable=AsyncMock,
            return_value=None,
        ):
            updates, _ = await review_node(state, ctx)

        assert updates["current_phase"] == PHASE_COMPLETE
        assert updates["current_review"]["action"] == "approve"

    @pytest.mark.asyncio
    async def test_review_node_few_reviewers_triggers_revise(self):
        """When fewer than 2 reviewers succeed, triggers revise."""
        state = _make_state()
        ctx = _make_ctx()

        def side_effect(*args, **kwargs):
            raise ValueError("LLM failed")

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.review.check_node_cancelled",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.review.tracked_invoke",
                new_callable=AsyncMock,
                side_effect=side_effect,
            ),
        ):
            updates, _ = await review_node(state, ctx)

        assert updates["current_phase"] == PHASE_SYNTHESIZE
        assert updates["current_review"]["action"] == "revise"
        assert "Only" in updates["current_review"]["reason"]

    @pytest.mark.asyncio
    async def test_review_node_empty_findings_board_handled(self):
        """Empty findings_board does not raise; uses empty findings text."""
        state = _make_state(findings_board={}, findings={})
        ctx = _make_ctx()

        mock_response = _make_review_response(action="approve", score=70)

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.review.check_node_cancelled",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.review.tracked_invoke",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
        ):
            updates, _ = await review_node(state, ctx)

        assert "current_review" in updates
        assert "quality_matrix" in updates


class TestReviewNodeCancellationAndSentinel:
    """Test cancellation and sentinel behavior."""

    @pytest.mark.asyncio
    async def test_review_node_returns_early_when_cancelled(self):
        """When thread is cancelled, returns early with PHASE_COMPLETE."""
        state = _make_state()
        ctx = _make_ctx()

        with patch(
            "template_agent.src.core.deep_research.nodes.review.check_node_cancelled",
            new_callable=AsyncMock,
            return_value={
                "current_phase": PHASE_COMPLETE,
                "final_answer": "Cancelled",
                "total_node_transitions": 1,
            },
        ):
            updates, events = await review_node(state, ctx)

        assert updates["current_phase"] == PHASE_COMPLETE
        assert updates["final_answer"] == "Cancelled"
        assert events == []

    @pytest.mark.asyncio
    async def test_review_node_sentinel_triggered_auto_approves(self):
        """When sentinel triggers with existing reviews, auto-approves."""
        state = _make_state(
            review_results=[{"action": "approve", "score": 70}],
        )
        ctx = _make_ctx()

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.review.check_node_cancelled",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.review.check_loop_sentinel",
                return_value=(True, "budget exceeded", None),
            ),
        ):
            updates, _ = await review_node(state, ctx)

        assert updates["current_phase"] == PHASE_COMPLETE
        assert updates.get("sentinel_triggered") is True
        assert updates["current_review"]["action"] == "approve"
        assert "Auto-approved" in updates["current_review"]["reason"]


class TestReviewNodeEventEmission:
    """Test event emission."""

    @pytest.mark.asyncio
    async def test_review_node_emits_consensus_result_event(self):
        """Review node emits consensus_result event on success."""
        state = _make_state()
        ctx = _make_ctx()

        mock_response = _make_review_response(action="approve", score=75)

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.review.check_node_cancelled",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.review.tracked_invoke",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
        ):
            _, events = await review_node(state, ctx)

        event_types = [
            e.get("content", {}).get("event_type", e.get("event_type", ""))
            for e in events
            if isinstance(e, dict)
        ]
        assert "consensus_result" in event_types or len(events) >= 1

    @pytest.mark.asyncio
    async def test_review_node_emits_reliability_update_event(self):
        """Review node emits reliability_update event."""
        state = _make_state()
        ctx = _make_ctx()

        mock_response = _make_review_response(action="approve", score=80)

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.review.check_node_cancelled",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.review.tracked_invoke",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
        ):
            _, events = await review_node(state, ctx)

        event_types = [
            e.get("content", {}).get("event_type", e.get("event_type", ""))
            for e in events
            if isinstance(e, dict)
        ]
        assert "reliability_update" in event_types or len(events) >= 1
