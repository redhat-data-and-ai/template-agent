"""Comprehensive pytest tests for the context answer node."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from template_agent.src.core.deep_research.nodes.context_answer import (
    context_answer_node,
)
from template_agent.src.core.deep_research.state import (
    PHASE_REVIEW,
    ResearchContext,
)


def _make_state(**overrides):
    """Create minimal DeepResearchState dict with required fields."""
    base = {"query": "test query", "thread_id": "t1", "current_phase": "synthesize"}
    base.update(overrides)
    return base


def _make_ctx(**overrides):
    """Create minimal ResearchContext for testing."""
    base_model = AsyncMock()
    ctx = ResearchContext(tools=[MagicMock(name="tool1")], base_model=base_model)
    for k, v in overrides.items():
        setattr(ctx, k, v)
    return ctx


class TestContextAnswerNodeSuccess:
    """Test success paths for context_answer_node."""

    @pytest.mark.asyncio
    async def test_context_answer_node_returns_state_updates_and_events(self):
        """Context answer node returns (state_updates, events) tuple."""
        state = _make_state(
            cached_findings_text="- Q: q1\n  A: a1",
            context="User: hi\nAssistant: hello",
        )
        ctx = _make_ctx()

        mock_response = MagicMock()
        mock_response.content = "Here is the synthesized answer from cached data."

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.context_answer.load_cached_findings",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.context_answer.tracked_invoke",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
        ):
            updates, events = await context_answer_node(state, ctx)

        assert "draft_answer" in updates
        assert "findings" in updates
        assert updates["current_phase"] == PHASE_REVIEW
        assert updates["synthesis_iteration"] == 1
        assert len(events) >= 3

    @pytest.mark.asyncio
    async def test_context_answer_node_draft_from_llm_response(self):
        """Draft answer comes from LLM response content."""
        state = _make_state(
            cached_findings_text="- Q: q1\n  A: a1",
            context="User: hi",
        )
        ctx = _make_ctx()

        mock_response = MagicMock()
        mock_response.content = "## Summary\n\nThis is the direct answer."

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.context_answer.load_cached_findings",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.context_answer.tracked_invoke",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
        ):
            updates, _ = await context_answer_node(state, ctx)

        assert "Summary" in updates["draft_answer"]
        assert "direct answer" in updates["draft_answer"]

    @pytest.mark.asyncio
    async def test_context_answer_node_loads_cached_findings(self):
        """Cached findings are loaded and passed to state."""
        state = _make_state(
            cached_findings_text="- Q: q1\n  A: a1",
            context="User: hi",
        )
        ctx = _make_ctx(checkpointer=MagicMock())

        cached = {"h1": {"subquery": "q1", "answer": "a1"}}

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.context_answer.load_cached_findings",
                new_callable=AsyncMock,
                return_value=cached,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.context_answer.tracked_invoke",
                new_callable=AsyncMock,
                return_value=MagicMock(content="Answer"),
            ),
        ):
            updates, _ = await context_answer_node(state, ctx)

        assert updates["findings"] == cached

    @pytest.mark.asyncio
    async def test_context_answer_node_emits_synthesis_events(self):
        """Node emits synthesis_start, synthesis_complete, agent_decision."""
        state = _make_state(
            cached_findings_text="- Q: q1\n  A: a1",
            context="User: hi",
        )
        ctx = _make_ctx()

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.context_answer.load_cached_findings",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.context_answer.tracked_invoke",
                new_callable=AsyncMock,
                return_value=MagicMock(content="Answer text"),
            ),
        ):
            _, events = await context_answer_node(state, ctx)

        assert len(events) >= 3


class TestContextAnswerNodeErrorHandling:
    """Test error handling for context answer node."""

    @pytest.mark.asyncio
    async def test_context_answer_node_llm_exception_returns_error_message(self):
        """LLM exception yields error message in draft_answer."""
        state = _make_state(
            cached_findings_text="- Q: q1\n  A: a1",
            context="User: hi",
        )
        ctx = _make_ctx()

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.context_answer.load_cached_findings",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.context_answer.tracked_invoke",
                new_callable=AsyncMock,
                side_effect=RuntimeError("API unavailable"),
            ),
        ):
            updates, _ = await context_answer_node(state, ctx)

        assert "Failed to synthesize" in updates["draft_answer"]
        assert updates["current_phase"] == PHASE_REVIEW

    @pytest.mark.asyncio
    async def test_context_answer_node_load_cached_exception_continues(self):
        """load_cached_findings exception does not fail the node."""
        state = _make_state(
            cached_findings_text="- Q: q1\n  A: a1",
            context="User: hi",
        )
        ctx = _make_ctx(checkpointer=MagicMock())

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.context_answer.load_cached_findings",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Checkpointer error"),
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.context_answer.tracked_invoke",
                new_callable=AsyncMock,
                return_value=MagicMock(content="Answer"),
            ),
        ):
            updates, _ = await context_answer_node(state, ctx)

        assert updates["findings"] == {}
        assert "draft_answer" in updates


class TestContextAnswerNodeEdgeCases:
    """Test edge cases for context answer node."""

    @pytest.mark.asyncio
    async def test_context_answer_node_empty_cached_findings_handled(self):
        """Empty cached_findings_text does not raise."""
        state = _make_state(cached_findings_text="", context="User: hi")
        ctx = _make_ctx()

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.context_answer.load_cached_findings",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.context_answer.tracked_invoke",
                new_callable=AsyncMock,
                return_value=MagicMock(content="Answer from minimal context"),
            ),
        ):
            updates, _ = await context_answer_node(state, ctx)

        assert updates["draft_answer"] == "Answer from minimal context"

    @pytest.mark.asyncio
    async def test_context_answer_node_strips_annotation_tags(self):
        """Annotation tags like [UNVERIFIED] are stripped from draft answer."""
        state = _make_state(
            cached_findings_text="- Q: q1\n  A: a1",
            context="User: hi",
        )
        ctx = _make_ctx()

        mock_response = MagicMock()
        mock_response.content = "Content [UNVERIFIED] more"

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.context_answer.load_cached_findings",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.context_answer.tracked_invoke",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
        ):
            updates, _ = await context_answer_node(state, ctx)

        assert "[UNVERIFIED]" not in updates["draft_answer"]

    @pytest.mark.asyncio
    async def test_context_answer_node_none_thread_id_handled(self):
        """None thread_id passed to load_cached_findings does not raise."""
        state = _make_state(
            thread_id=None,
            cached_findings_text="- Q: q1\n  A: a1",
            context="User: hi",
        )
        ctx = _make_ctx(checkpointer=MagicMock())

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.context_answer.load_cached_findings",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.context_answer.tracked_invoke",
                new_callable=AsyncMock,
                return_value=MagicMock(content="Answer"),
            ),
        ):
            updates, _ = await context_answer_node(state, ctx)

        assert updates["current_phase"] == PHASE_REVIEW

    @pytest.mark.asyncio
    async def test_context_answer_node_empty_llm_response_handled(self):
        """Empty LLM response yields empty draft_answer."""
        state = _make_state(
            cached_findings_text="- Q: q1\n  A: a1",
            context="User: hi",
        )
        ctx = _make_ctx()

        mock_response = MagicMock()
        mock_response.content = ""

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.context_answer.load_cached_findings",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.context_answer.tracked_invoke",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
        ):
            updates, _ = await context_answer_node(state, ctx)

        assert updates["draft_answer"] == ""
        assert updates["current_phase"] == PHASE_REVIEW
