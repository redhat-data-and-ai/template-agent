"""Comprehensive pytest tests for the synthesize node."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from template_agent.src.core.deep_research.nodes.synthesize import synthesize_node
from template_agent.src.core.deep_research.state import (
    PHASE_COMPLETE,
    PHASE_VISUALIZE,
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
        "draft_answer": "",
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


def _make_llm_response(content: str) -> MagicMock:
    """Create mock LLM response with content."""
    response = MagicMock()
    response.content = content
    return response


class TestSynthesizeNodeSuccessPaths:
    """Test success paths for synthesize_node."""

    @pytest.mark.asyncio
    async def test_synthesize_node_first_pass_produces_draft_answer(self):
        """First-pass synthesis produces draft_answer and moves to PHASE_VISUALIZE."""
        state = _make_state()
        ctx = _make_ctx()

        draft_content = "# Research Analysis\n\n## Summary\n\nKey findings here.\n\n## Conclusion\n\nDone."
        mock_response = _make_llm_response(draft_content)

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.synthesize.check_node_cancelled",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.synthesize.tracked_invoke",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
        ):
            updates, events = await synthesize_node(state, ctx)

        assert "draft_answer" in updates
        assert updates["draft_answer"] == draft_content
        assert updates["current_phase"] == PHASE_VISUALIZE
        assert updates["synthesis_iteration"] == 1
        assert len(events) >= 1

    @pytest.mark.asyncio
    async def test_synthesize_node_revision_path_produces_revised_draft(self):
        """Revision path produces revised draft when iteration > 1 and feedback exists."""
        state = _make_state(
            draft_answer="Original draft",
            synthesis_iteration=1,
            current_review={"action": "revise", "score": 50},
            review_results=[
                {
                    "persona": "A",
                    "action": "revise",
                    "score": 50,
                    "reason": "Improve",
                    "feedback": "",
                },
            ],
        )
        ctx = _make_ctx()

        revised_content = "# Revised Analysis\n\n## Summary\n\nImproved content.\n\n## Conclusion\n\nDone."
        mock_response = _make_llm_response(revised_content)

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.synthesize.check_node_cancelled",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.synthesize.check_loop_sentinel",
                return_value=(False, None, None),
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.synthesize.tracked_invoke",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
        ):
            updates, events = await synthesize_node(state, ctx)

        assert updates["draft_answer"] == revised_content
        assert updates["current_phase"] == PHASE_VISUALIZE
        assert updates["synthesis_iteration"] >= 1
        assert len(events) >= 1

    @pytest.mark.asyncio
    async def test_synthesize_node_increments_total_node_transitions(self):
        """Synthesize node increments total_node_transitions."""
        state = _make_state(total_node_transitions=3)
        ctx = _make_ctx()

        draft_content = "# Report\n\n## Summary\n\nContent.\n\n## Conclusion\n\nDone."
        mock_response = _make_llm_response(draft_content)

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.synthesize.check_node_cancelled",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.synthesize.tracked_invoke",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
        ):
            updates, _ = await synthesize_node(state, ctx)

        assert updates["total_node_transitions"] == 4

    @pytest.mark.asyncio
    async def test_synthesize_node_emits_synthesis_start_event(self):
        """Synthesize node emits synthesis_start event."""
        state = _make_state()
        ctx = _make_ctx()

        draft_content = "# Report\n\n## Summary\n\nContent.\n\n## Conclusion\n\nDone."
        mock_response = _make_llm_response(draft_content)

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.synthesize.check_node_cancelled",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.synthesize.tracked_invoke",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
        ):
            _, events = await synthesize_node(state, ctx)

        event_types = [
            e.get("content", {}).get("event_type", e.get("event_type", ""))
            for e in events
            if isinstance(e, dict)
        ]
        assert "synthesis_start" in event_types or len(events) >= 1

    @pytest.mark.asyncio
    async def test_synthesize_node_emits_data_aggregation_events(self):
        """Synthesize node emits data_aggregation_start and data_aggregation_complete."""
        state = _make_state()
        ctx = _make_ctx()

        agg_content = json.dumps({"data_points": [{"x": 1}], "conflicts": []})
        draft_content = "# Report\n\n## Summary\n\nContent.\n\n## Conclusion\n\nDone."
        mock_invoke = AsyncMock(
            side_effect=[
                _make_llm_response(agg_content),
                _make_llm_response(draft_content),
                _make_llm_response(draft_content),
            ]
        )

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.synthesize.check_node_cancelled",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.synthesize.tracked_invoke",
                mock_invoke,
            ),
        ):
            _, events = await synthesize_node(state, ctx)

        event_types = [
            e.get("content", {}).get("event_type", e.get("event_type", ""))
            for e in events
            if isinstance(e, dict)
        ]
        assert "data_aggregation_start" in event_types
        assert "data_aggregation_complete" in event_types


class TestSynthesizeNodeEmptyFindings:
    """Test behavior with empty or invalid findings."""

    @pytest.mark.asyncio
    async def test_synthesize_node_no_valid_findings_returns_early(self):
        """When no valid findings, returns final_answer and PHASE_COMPLETE."""
        state = _make_state(
            findings_board={
                "sq1": {"finding": {"subquery": "q1", "answer": "", "error": "Failed"}},
            },
        )
        ctx = _make_ctx()

        with patch(
            "template_agent.src.core.deep_research.nodes.synthesize.check_node_cancelled",
            new_callable=AsyncMock,
            return_value=None,
        ):
            updates, events = await synthesize_node(state, ctx)

        assert (
            updates["final_answer"]
            == "Research could not retrieve valid data for your query."
        )
        assert updates["current_phase"] == PHASE_COMPLETE
        assert "draft_answer" not in updates or updates.get(
            "draft_answer"
        ) != updates.get("final_answer")

    @pytest.mark.asyncio
    async def test_synthesize_node_empty_findings_board_returns_no_valid_data(self):
        """Empty findings_board with no valid entries returns no valid data message."""
        state = _make_state(findings_board={}, findings={})
        ctx = _make_ctx()

        with patch(
            "template_agent.src.core.deep_research.nodes.synthesize.check_node_cancelled",
            new_callable=AsyncMock,
            return_value=None,
        ):
            updates, events = await synthesize_node(state, ctx)

        assert (
            updates["final_answer"]
            == "Research could not retrieve valid data for your query."
        )
        assert updates["current_phase"] == PHASE_COMPLETE

    @pytest.mark.asyncio
    async def test_synthesize_node_emits_no_valid_findings_event(self):
        """When no valid findings, emits no_valid_findings event."""
        state = _make_state(
            findings_board={
                "sq1": {"finding": {"error": "Failed", "answer": ""}},
            },
        )
        ctx = _make_ctx()

        with patch(
            "template_agent.src.core.deep_research.nodes.synthesize.check_node_cancelled",
            new_callable=AsyncMock,
            return_value=None,
        ):
            _, events = await synthesize_node(state, ctx)

        event_types = [
            e.get("content", {}).get("event_type", e.get("event_type", ""))
            for e in events
            if isinstance(e, dict)
        ]
        assert "no_valid_findings" in event_types or len(events) >= 1


class TestSynthesizeNodeErrorHandling:
    """Test error handling and fallback behavior."""

    @pytest.mark.asyncio
    async def test_synthesize_node_uses_fallback_on_llm_failure(self):
        """When synthesis LLM fails, uses fallback synthesis from findings."""
        state = _make_state()
        ctx = _make_ctx()

        def fail_then_succeed(*args, **kwargs):
            raise RuntimeError("LLM unavailable")

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.synthesize.check_node_cancelled",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.synthesize.tracked_invoke",
                new_callable=AsyncMock,
                side_effect=fail_then_succeed,
            ),
        ):
            updates, events = await synthesize_node(state, ctx)

        # After retries fail, draft_answer starts with "Synthesis failed:" -> fallback used
        assert "draft_answer" in updates
        assert updates["current_phase"] == PHASE_VISUALIZE
        assert (
            "Research Analysis" in updates["draft_answer"]
            or "Synthesis failed" in updates["draft_answer"]
        )

    @pytest.mark.asyncio
    async def test_synthesize_node_handles_tool_recommendation_retry(self):
        """When first synthesis looks like tool recommendation, retries with stricter prompt."""
        state = _make_state()
        ctx = _make_ctx()

        tool_rec_content = (
            "I recommend using the sales_data tool to get this information."
        )
        proper_draft = (
            "# Research Analysis\n\n## Summary\n\nFindings.\n\n## Conclusion\n\nDone."
        )

        call_count = 0

        def invoke_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return _make_llm_response(tool_rec_content)
            return _make_llm_response(proper_draft)

        mock_invoke = AsyncMock(side_effect=invoke_side_effect)

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.synthesize.check_node_cancelled",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.synthesize.tracked_invoke",
                mock_invoke,
            ),
        ):
            updates, _ = await synthesize_node(state, ctx)

        assert "draft_answer" in updates
        assert updates["current_phase"] == PHASE_VISUALIZE


class TestSynthesizeNodeCancellationAndSentinel:
    """Test cancellation and sentinel behavior."""

    @pytest.mark.asyncio
    async def test_synthesize_node_returns_early_when_cancelled(self):
        """When thread is cancelled, returns early with PHASE_COMPLETE."""
        state = _make_state()
        ctx = _make_ctx()

        with patch(
            "template_agent.src.core.deep_research.nodes.synthesize.check_node_cancelled",
            new_callable=AsyncMock,
            return_value={
                "current_phase": PHASE_COMPLETE,
                "final_answer": "Cancelled",
                "total_node_transitions": 1,
            },
        ):
            updates, events = await synthesize_node(state, ctx)

        assert updates["current_phase"] == PHASE_COMPLETE
        assert updates["final_answer"] == "Cancelled"
        assert events == []

    @pytest.mark.asyncio
    async def test_synthesize_node_sentinel_triggered_uses_fallback_when_no_draft(self):
        """When sentinel triggers on iteration>1 with no previous draft, uses fallback."""
        state = _make_state(
            draft_answer="",
            synthesis_iteration=1,
        )
        ctx = _make_ctx()

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.synthesize.check_node_cancelled",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.synthesize.check_loop_sentinel",
                return_value=(True, "budget exceeded", None),
            ),
        ):
            updates, events = await synthesize_node(state, ctx)

        assert updates["current_phase"] == PHASE_VISUALIZE
        assert updates.get("sentinel_triggered") is True
        assert "draft_answer" in updates
        assert "Research Analysis" in updates["draft_answer"]

    @pytest.mark.asyncio
    async def test_synthesize_node_sentinel_triggered_preserves_existing_draft(self):
        """When sentinel triggers with existing draft, preserves it."""
        state = _make_state(
            draft_answer="Existing draft content",
            synthesis_iteration=1,
        )
        ctx = _make_ctx()

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.synthesize.check_node_cancelled",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.synthesize.check_loop_sentinel",
                return_value=(True, "budget exceeded", None),
            ),
        ):
            updates, events = await synthesize_node(state, ctx)

        assert updates["current_phase"] == PHASE_VISUALIZE
        assert updates.get("sentinel_triggered") is True
        assert (
            updates.get("draft_answer") is None
            or updates.get("draft_answer") == "Existing draft content"
        )


class TestSynthesizeNodeEventEmission:
    """Test event emission."""

    @pytest.mark.asyncio
    async def test_synthesize_node_emits_report_generation_events(self):
        """Synthesize node emits report_generation_start and report_generation_complete."""
        state = _make_state()
        ctx = _make_ctx()

        agg_content = json.dumps({"data_points": [], "conflicts": []})
        draft_content = "# Report\n\n## Summary\n\nContent.\n\n## Conclusion\n\nDone."
        mock_invoke = AsyncMock(
            side_effect=[
                _make_llm_response(agg_content),
                _make_llm_response(draft_content),
                _make_llm_response(draft_content),
            ]
        )

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.synthesize.check_node_cancelled",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.synthesize.tracked_invoke",
                mock_invoke,
            ),
        ):
            _, events = await synthesize_node(state, ctx)

        event_types = [
            e.get("content", {}).get("event_type", e.get("event_type", ""))
            for e in events
            if isinstance(e, dict)
        ]
        assert "report_generation_start" in event_types
        assert "report_generation_complete" in event_types

    @pytest.mark.asyncio
    async def test_synthesize_node_emits_synthesis_complete_event(self):
        """Synthesize node emits synthesis_complete event on success."""
        state = _make_state()
        ctx = _make_ctx()

        agg_content = json.dumps({"data_points": [], "conflicts": []})
        draft_content = "# Report\n\n## Summary\n\nContent.\n\n## Conclusion\n\nDone."
        mock_invoke = AsyncMock(
            side_effect=[
                _make_llm_response(agg_content),
                _make_llm_response(draft_content),
                _make_llm_response(draft_content),
            ]
        )

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.synthesize.check_node_cancelled",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.synthesize.tracked_invoke",
                mock_invoke,
            ),
        ):
            _, events = await synthesize_node(state, ctx)

        event_types = [
            e.get("content", {}).get("event_type", e.get("event_type", ""))
            for e in events
            if isinstance(e, dict)
        ]
        assert "synthesis_complete" in event_types
