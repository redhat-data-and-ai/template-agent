"""Comprehensive pytest tests for the complete node."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from template_agent.src.core.deep_research.nodes.complete import complete_node
from template_agent.src.core.deep_research.state import (
    PHASE_COMPLETE,
    ResearchContext,
)


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


class TestCompleteNodeSuccess:
    """Test success paths for complete_node."""

    @pytest.mark.asyncio
    async def test_complete_node_returns_final_answer_and_stop(self):
        """Complete node returns final_answer and should_stop=True."""
        state = _make_state(draft_answer="Final synthesized answer")
        ctx = _make_ctx(checkpointer=None)

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.complete.save_cached_findings",
                new_callable=AsyncMock,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.complete.save_findings_in_memory"
            ),
        ):
            updates, events = await complete_node(state, ctx)

        assert updates["final_answer"] == "Final synthesized answer"
        assert updates["current_phase"] == PHASE_COMPLETE
        assert updates["should_stop"] is True

    @pytest.mark.asyncio
    async def test_complete_node_prefers_draft_answer_over_final_answer(self):
        """When both exist, draft_answer takes precedence."""
        state = _make_state(
            draft_answer="Draft content",
            final_answer="Old final content",
        )
        ctx = _make_ctx(checkpointer=None)

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.complete.save_cached_findings",
                new_callable=AsyncMock,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.complete.save_findings_in_memory"
            ),
        ):
            updates, _ = await complete_node(state, ctx)

        assert updates["final_answer"] == "Draft content"

    @pytest.mark.asyncio
    async def test_complete_node_uses_final_answer_when_no_draft(self):
        """Uses final_answer when draft_answer is absent."""
        state = _make_state(final_answer="Pre-existing final")
        ctx = _make_ctx(checkpointer=None)

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.complete.save_cached_findings",
                new_callable=AsyncMock,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.complete.save_findings_in_memory"
            ),
        ):
            updates, _ = await complete_node(state, ctx)

        assert updates["final_answer"] == "Pre-existing final"

    @pytest.mark.asyncio
    async def test_complete_node_includes_visualizations(self):
        """Visualizations are passed through to state updates."""
        viz = [{"type": "chart", "data": {}}]
        state = _make_state(draft_answer="Answer", visualizations=viz)
        ctx = _make_ctx(checkpointer=None)

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.complete.save_cached_findings",
                new_callable=AsyncMock,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.complete.save_findings_in_memory"
            ),
        ):
            updates, _ = await complete_node(state, ctx)

        assert updates["visualizations"] == viz

    @pytest.mark.asyncio
    async def test_complete_node_saves_findings_to_cache(self):
        """Findings are saved to checkpointer when available."""
        state = _make_state(
            draft_answer="Answer",
            findings_board={"sq1": {"finding": {"subquery": "q1", "answer": "a1"}}},
        )
        checkpointer = AsyncMock()
        ctx = _make_ctx(checkpointer=checkpointer)

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.complete.save_cached_findings",
                new_callable=AsyncMock,
            ) as mock_save,
            patch(
                "template_agent.src.core.deep_research.nodes.complete.save_findings_in_memory"
            ),
        ):
            _, _ = await complete_node(state, ctx)

        mock_save.assert_awaited_once()
        call_args = mock_save.call_args
        assert call_args[0][1] == "t1"
        assert isinstance(call_args[0][2], list)

    @pytest.mark.asyncio
    async def test_complete_node_emits_events(self):
        """Complete node emits final_answer and completed events."""
        state = _make_state(draft_answer="Done")
        ctx = _make_ctx(checkpointer=None)

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.complete.save_cached_findings",
                new_callable=AsyncMock,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.complete.save_findings_in_memory"
            ),
        ):
            _, events = await complete_node(state, ctx)

        assert len(events) >= 2
        assert "final_answer" in str(events) or any(
            "answer" in str(e).lower() for e in events
        )


class TestCompleteNodeEdgeCases:
    """Test edge cases for complete_node."""

    @pytest.mark.asyncio
    async def test_complete_node_empty_answer_defaults_to_empty_string(self):
        """Empty draft and final yield empty string."""
        state = _make_state()
        ctx = _make_ctx(checkpointer=None)

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.complete.save_cached_findings",
                new_callable=AsyncMock,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.complete.save_findings_in_memory"
            ),
        ):
            updates, _ = await complete_node(state, ctx)

        assert updates["final_answer"] == ""

    @pytest.mark.asyncio
    async def test_complete_node_strips_annotation_tags(self):
        """Annotation tags like [UNVERIFIED] are stripped from final answer."""
        state = _make_state(draft_answer="Content [UNVERIFIED] more")
        ctx = _make_ctx(checkpointer=None)

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.complete.save_cached_findings",
                new_callable=AsyncMock,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.complete.save_findings_in_memory"
            ),
        ):
            updates, _ = await complete_node(state, ctx)

        assert "[UNVERIFIED]" not in updates["final_answer"]

    @pytest.mark.asyncio
    async def test_complete_node_empty_thread_id_handled(self):
        """Empty thread_id does not raise."""
        state = _make_state(draft_answer="A", thread_id="")
        ctx = _make_ctx(checkpointer=None)

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.complete.save_cached_findings",
                new_callable=AsyncMock,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.complete.save_findings_in_memory"
            ),
        ):
            updates, _ = await complete_node(state, ctx)

        assert updates["should_stop"] is True

    @pytest.mark.asyncio
    async def test_complete_node_token_tracker_persist_swallowed_on_error(self):
        """Token tracker persist errors are logged but do not fail the node."""
        state = _make_state(draft_answer="A")
        tracker = MagicMock()
        tracker.persist_to_db = MagicMock(side_effect=RuntimeError("DB down"))
        tracker.get_summary = MagicMock(return_value={"total": {}, "per_phase": {}})
        ctx = _make_ctx(checkpointer=None, token_tracker=tracker)

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.complete.save_cached_findings",
                new_callable=AsyncMock,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.complete.save_findings_in_memory"
            ),
        ):
            updates, _ = await complete_node(state, ctx)

        assert updates["should_stop"] is True

    @pytest.mark.asyncio
    async def test_complete_node_elapsed_time_computed_when_start_time_set(self):
        """Elapsed time is computed when research_start_time is set."""
        state = _make_state(
            draft_answer="A",
            research_start_time=time.time() - 10.0,
        )
        ctx = _make_ctx(checkpointer=None)

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.complete.save_cached_findings",
                new_callable=AsyncMock,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.complete.save_findings_in_memory"
            ),
        ):
            _, events = await complete_node(state, ctx)

        assert len(events) >= 1
