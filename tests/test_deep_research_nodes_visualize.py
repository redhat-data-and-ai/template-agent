"""Comprehensive pytest tests for the deep research visualize node."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from template_agent.src.core.deep_research.nodes.visualize import (
    _extract_mermaid_blocks,
    visualize_node,
)
from template_agent.src.core.deep_research.state import (
    PHASE_COMPLETE,
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


class TestExtractMermaidBlocks:
    """Test _extract_mermaid_blocks helper."""

    def test_extract_single_pie_chart(self):
        """Extract pie chart from mermaid block."""
        text = '## Market Share\n```mermaid\npie title "Share"\n  "A" : 50\n  "B" : 50\n```'
        charts = _extract_mermaid_blocks(text)
        assert len(charts) == 1
        assert charts[0]["chart_type"] == "pie"
        assert charts[0]["title"] == "Market Share"
        assert "pie" in charts[0]["mermaid_code"].lower()

    def test_extract_bar_chart_from_xychart(self):
        """Extract bar chart from xychart-beta block."""
        text = '## Revenue\n```mermaid\nxychart-beta\n  title "Q1-Q4"\n  bar [1,2,3,4]\n```'
        charts = _extract_mermaid_blocks(text)
        assert len(charts) == 1
        assert charts[0]["chart_type"] == "bar"
        assert "xychart" in charts[0]["mermaid_code"].lower()

    def test_extract_timeline_chart(self):
        """Extract timeline chart type."""
        text = "## Events\n```mermaid\ntimeline\n  title History\n  section 2020\n  Event : A\n```"
        charts = _extract_mermaid_blocks(text)
        assert len(charts) == 1
        assert charts[0]["chart_type"] == "timeline"

    def test_extract_graph_default(self):
        """Default chart type is graph for flowchart."""
        text = "## Flow\n```mermaid\ngraph TD\n  A --> B\n```"
        charts = _extract_mermaid_blocks(text)
        assert len(charts) == 1
        assert charts[0]["chart_type"] == "graph"

    def test_extract_multiple_blocks(self):
        """Extract multiple mermaid blocks."""
        text = (
            "## Chart 1\n```mermaid\npie title A\n  X : 1\n```\n\n"
            "## Chart 2\n```mermaid\ngraph TD\n  A --> B\n```"
        )
        charts = _extract_mermaid_blocks(text)
        assert len(charts) == 2
        assert charts[0]["chart_type"] == "pie"
        assert charts[1]["chart_type"] == "graph"

    def test_extract_skips_empty_blocks(self):
        """Skip empty mermaid blocks."""
        text = "## Empty\n```mermaid\n\n```"
        charts = _extract_mermaid_blocks(text)
        assert len(charts) == 0


class TestVisualizeNodeSuccess:
    """Test success paths for visualize_node."""

    @pytest.mark.asyncio
    async def test_visualize_node_generates_charts_from_mermaid_response(self):
        """Node returns charts when LLM produces valid mermaid blocks."""
        state = _make_state(draft_answer="Report with Q1: 10, Q2: 20, Q3: 30")
        ctx = _make_ctx()

        mock_response = MagicMock()
        mock_response.content = (
            "## Revenue by Quarter\n```mermaid\nxychart-beta\n"
            '  title "Revenue"\n  x-axis [Q1, Q2, Q3]\n'
            '  y-axis "M" 0 --> 50\n  bar [10, 20, 30]\n```'
        )

        with patch(
            "template_agent.src.core.deep_research.nodes.visualize.tracked_invoke",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            updates, _ = await visualize_node(state, ctx)

        assert updates["visualizations"]
        assert len(updates["visualizations"]) == 1
        assert updates["visualizations"][0]["chart_type"] == "bar"
        assert updates["visualization_attempted"] is True
        assert updates["current_phase"] == PHASE_REVIEW
        assert "draft_answer" in updates
        assert "## Visualizations" in updates["draft_answer"]

    @pytest.mark.asyncio
    async def test_visualize_node_emits_visualization_created_per_chart(self):
        """Each chart triggers visualization_created event."""
        state = _make_state(draft_answer="Data: A=1, B=2")
        ctx = _make_ctx()

        mock_response = MagicMock()
        mock_response.content = (
            "## Chart 1\n```mermaid\npie title X\n  A : 1\n  B : 2\n```"
        )

        with patch(
            "template_agent.src.core.deep_research.nodes.visualize.tracked_invoke",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            _, events = await visualize_node(state, ctx)

        created_events = [
            e
            for e in events
            if e.get("content", {}).get("event_type") == "visualization_created"
        ]
        assert len(created_events) >= 1


class TestVisualizeNodeNoData:
    """Test paths when no visualizable data exists."""

    @pytest.mark.asyncio
    async def test_visualize_node_empty_draft_skips_visualization(self):
        """Empty draft_answer skips visualization and returns empty list."""
        state = _make_state(draft_answer="")
        ctx = _make_ctx()

        updates, events = await visualize_node(state, ctx)

        assert updates["visualizations"] == []
        assert updates["visualization_attempted"] is True
        assert updates["current_phase"] == PHASE_REVIEW
        skip_events = [e for e in events if "skipped" in str(e).lower()]
        assert len(skip_events) >= 1

    @pytest.mark.asyncio
    async def test_visualize_node_no_charts_response_returns_empty(self):
        """When LLM returns NO_CHARTS, returns empty visualizations."""
        state = _make_state(draft_answer="Report with only prose, no numbers")
        ctx = _make_ctx()

        mock_response = MagicMock()
        mock_response.content = (
            "NO_CHARTS - The report contains no numeric data to visualize."
        )

        with patch(
            "template_agent.src.core.deep_research.nodes.visualize.tracked_invoke",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            updates, _ = await visualize_node(state, ctx)

        assert updates["visualizations"] == []
        assert updates["visualization_attempted"] is True
        assert updates["current_phase"] == PHASE_REVIEW

    @pytest.mark.asyncio
    async def test_visualize_node_no_valid_mermaid_blocks_returns_empty(self):
        """When LLM returns text without valid mermaid blocks, returns empty."""
        state = _make_state(draft_answer="Report with data")
        ctx = _make_ctx()

        mock_response = MagicMock()
        mock_response.content = (
            "I analyzed the report but could not produce valid charts."
        )

        with patch(
            "template_agent.src.core.deep_research.nodes.visualize.tracked_invoke",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            updates, _ = await visualize_node(state, ctx)

        assert updates["visualizations"] == []
        assert updates["visualization_attempted"] is True


class TestVisualizeNodeErrorHandling:
    """Test error handling for visualize_node."""

    @pytest.mark.asyncio
    async def test_visualize_node_handles_llm_exception_gracefully(self):
        """LLM exception returns empty visualizations and emits skipped event."""
        state = _make_state(draft_answer="Report content")
        ctx = _make_ctx()

        with patch(
            "template_agent.src.core.deep_research.nodes.visualize.tracked_invoke",
            new_callable=AsyncMock,
            side_effect=RuntimeError("API unavailable"),
        ):
            updates, events = await visualize_node(state, ctx)

        assert updates["visualizations"] == []
        assert updates["visualization_attempted"] is True
        assert updates["current_phase"] == PHASE_REVIEW
        skip_events = [
            e
            for e in events
            if "skipped" in str(e).lower() or "failed" in str(e).lower()
        ]
        assert len(skip_events) >= 1

    @pytest.mark.asyncio
    async def test_visualize_node_handles_timeout_exception(self):
        """Timeout exception returns empty visualizations."""
        state = _make_state(draft_answer="Report content")
        ctx = _make_ctx()

        with patch(
            "template_agent.src.core.deep_research.nodes.visualize.tracked_invoke",
            new_callable=AsyncMock,
            side_effect=TimeoutError("Request timed out"),
        ):
            updates, _ = await visualize_node(state, ctx)

        assert updates["visualizations"] == []
        assert updates["visualization_attempted"] is True


class TestVisualizeNodeEventEmission:
    """Test event emission during visualization."""

    @pytest.mark.asyncio
    async def test_visualize_node_emits_start_event(self):
        """Node emits visualization_start at beginning."""
        state = _make_state(draft_answer="Report")
        ctx = _make_ctx()

        mock_response = MagicMock()
        mock_response.content = "NO_CHARTS"

        with patch(
            "template_agent.src.core.deep_research.nodes.visualize.tracked_invoke",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            _, events = await visualize_node(state, ctx)

        start_events = [
            e
            for e in events
            if "visualization_start" in str(e).lower() or "Starting" in str(e)
        ]
        assert len(start_events) >= 1

    @pytest.mark.asyncio
    async def test_visualize_node_emits_agent_decision_before_llm_call(self):
        """Node emits agent_decision before invoking LLM."""
        state = _make_state(draft_answer="Report with numbers")
        ctx = _make_ctx()

        mock_response = MagicMock()
        mock_response.content = "## X\n```mermaid\npie title Y\n  A : 1\n```"

        with patch(
            "template_agent.src.core.deep_research.nodes.visualize.tracked_invoke",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            _, events = await visualize_node(state, ctx)

        assert len(events) >= 2
        decision_events = [
            e
            for e in events
            if "agent_decision" in str(e).lower() or "Visualizer" in str(e)
        ]
        assert len(decision_events) >= 1

    @pytest.mark.asyncio
    async def test_visualize_node_emits_complete_event(self):
        """Node emits visualization_complete with chart count."""
        state = _make_state(draft_answer="Report")
        ctx = _make_ctx()

        mock_response = MagicMock()
        mock_response.content = "## C\n```mermaid\npie title D\n  X : 1\n```"

        with patch(
            "template_agent.src.core.deep_research.nodes.visualize.tracked_invoke",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            _, events = await visualize_node(state, ctx)

        complete_events = [e for e in events if "complete" in str(e).lower()]
        assert len(complete_events) >= 1


class TestVisualizeNodeCancellation:
    """Test cancellation handling."""

    @pytest.mark.asyncio
    async def test_visualize_node_returns_early_when_cancelled(self):
        """When thread is cancelled, returns early with PHASE_COMPLETE."""
        state = _make_state(draft_answer="Report", thread_id="cancelled-thread")
        ctx = _make_ctx()

        with patch(
            "template_agent.src.core.deep_research.nodes.visualize.check_node_cancelled",
            new_callable=AsyncMock,
            return_value={
                "current_phase": PHASE_COMPLETE,
                "final_answer": "Cancelled",
                "total_node_transitions": 1,
            },
        ):
            updates, events = await visualize_node(state, ctx)

        assert updates["current_phase"] == PHASE_COMPLETE
        assert "final_answer" in updates
        assert len(events) == 0


class TestVisualizeNodeEdgeCases:
    """Test edge cases for visualize_node."""

    @pytest.mark.asyncio
    async def test_visualize_node_strips_existing_visualizations_section(self):
        """Draft with existing ## Visualizations section uses content before it."""
        state = _make_state(
            draft_answer="Main report\n\n---\n\n## Visualizations\nOld charts here"
        )
        ctx = _make_ctx()

        mock_response = MagicMock()
        mock_response.content = "## New Chart\n```mermaid\npie title X\n  A : 1\n```"

        with patch(
            "template_agent.src.core.deep_research.nodes.visualize.tracked_invoke",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            updates, _ = await visualize_node(state, ctx)

        assert "Main report" in updates["draft_answer"]
        assert "## Visualizations" in updates["draft_answer"]
        assert "New Chart" in updates["draft_answer"]

    @pytest.mark.asyncio
    async def test_visualize_node_response_without_content_attr_uses_str(self):
        """Response without .content attribute uses str(response)."""

        class ResponseWithoutContent:
            def __str__(self):
                return "## X\n```mermaid\npie title Y\n  A : 1\n```"

        state = _make_state(draft_answer="Report")
        ctx = _make_ctx()

        with patch(
            "template_agent.src.core.deep_research.nodes.visualize.tracked_invoke",
            new_callable=AsyncMock,
            return_value=ResponseWithoutContent(),
        ):
            updates, _ = await visualize_node(state, ctx)

        assert "visualizations" in updates
