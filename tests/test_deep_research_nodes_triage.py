"""Comprehensive pytest tests for the triage node."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from template_agent.src.core.deep_research.nodes.triage import triage_node
from template_agent.src.core.deep_research.state import (
    PHASE_PLAN,
    PHASE_PROBE,
    PHASE_SYNTHESIZE,
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


class TestTriageNodeNoContextPath:
    """Test triage when no cached findings or context."""

    @pytest.mark.asyncio
    async def test_triage_routes_to_full_research_when_no_context(self):
        """No cached findings and no context routes to full_research."""
        state = _make_state(cached_findings_text="", context="")
        ctx = _make_ctx()

        updates, events = await triage_node(state, ctx)

        assert updates["triage_decision"] == "full_research"
        assert updates["current_phase"] == PHASE_PROBE
        assert len(events) >= 1

    @pytest.mark.asyncio
    async def test_triage_routes_to_full_research_when_both_empty(self):
        """Empty cached_findings_text and context yields full_research."""
        state = _make_state()
        ctx = _make_ctx()

        updates, _ = await triage_node(state, ctx)

        assert updates["triage_decision"] == "full_research"
        assert updates["current_phase"] == PHASE_PROBE


class TestTriageNodeWithContext:
    """Test triage when context or cached findings exist."""

    @pytest.mark.asyncio
    async def test_triage_context_sufficient_routes_to_synthesize(self):
        """context_sufficient decision routes to PHASE_SYNTHESIZE."""
        state = _make_state(
            cached_findings_text="- Q: q1\n  A: a1",
            context="User: hi\nAssistant: hello",
        )
        ctx = _make_ctx()

        mock_response = MagicMock()
        mock_response.content = (
            '{"decision":"context_sufficient","reasoning":"Has data"}'
        )

        with patch(
            "template_agent.src.core.deep_research.nodes.triage.tracked_invoke",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            updates, events = await triage_node(state, ctx)

        assert updates["triage_decision"] == "context_sufficient"
        assert updates["current_phase"] == PHASE_SYNTHESIZE
        assert len(events) >= 2

    @pytest.mark.asyncio
    async def test_triage_partial_research_routes_to_plan(self):
        """partial_research decision routes to PHASE_PLAN."""
        state = _make_state(
            cached_findings_text="- Q: q1\n  A: a1",
            context="User: hi",
        )
        ctx = _make_ctx()

        mock_response = MagicMock()
        mock_response.content = (
            '{"decision":"partial_research","reasoning":"Need more"}'
        )

        with patch(
            "template_agent.src.core.deep_research.nodes.triage.tracked_invoke",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            updates, _ = await triage_node(state, ctx)

        assert updates["triage_decision"] == "partial_research"
        assert updates["current_phase"] == PHASE_PLAN

    @pytest.mark.asyncio
    async def test_triage_full_research_routes_to_probe(self):
        """full_research decision routes to PHASE_PROBE."""
        state = _make_state(
            cached_findings_text="- Q: q1\n  A: a1",
            context="User: hi",
        )
        ctx = _make_ctx()

        mock_response = MagicMock()
        mock_response.content = '{"decision":"full_research","reasoning":"Need full"}'

        with patch(
            "template_agent.src.core.deep_research.nodes.triage.tracked_invoke",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            updates, _ = await triage_node(state, ctx)

        assert updates["triage_decision"] == "full_research"
        assert updates["current_phase"] == PHASE_PROBE


class TestTriageNodeErrorHandling:
    """Test error handling for triage node."""

    @pytest.mark.asyncio
    async def test_triage_defaults_to_full_research_on_llm_exception(self):
        """LLM exception defaults to full_research."""
        state = _make_state(
            cached_findings_text="- Q: q1\n  A: a1",
            context="User: hi",
        )
        ctx = _make_ctx()

        with patch(
            "template_agent.src.core.deep_research.nodes.triage.tracked_invoke",
            new_callable=AsyncMock,
            side_effect=RuntimeError("API down"),
        ):
            updates, _ = await triage_node(state, ctx)

        assert updates["triage_decision"] == "full_research"
        assert updates["current_phase"] == PHASE_PROBE

    @pytest.mark.asyncio
    async def test_triage_defaults_to_full_research_on_invalid_json(self):
        """Invalid JSON in response defaults to full_research."""
        state = _make_state(
            cached_findings_text="- Q: q1\n  A: a1",
            context="User: hi",
        )
        ctx = _make_ctx()

        mock_response = MagicMock()
        mock_response.content = "not json at all"

        with patch(
            "template_agent.src.core.deep_research.nodes.triage.tracked_invoke",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            updates, _ = await triage_node(state, ctx)

        assert updates["triage_decision"] == "full_research"
        assert updates["current_phase"] == PHASE_PROBE

    @pytest.mark.asyncio
    async def test_triage_defaults_to_full_research_on_invalid_decision(self):
        """Invalid decision value defaults to full_research."""
        state = _make_state(
            cached_findings_text="- Q: q1\n  A: a1",
            context="User: hi",
        )
        ctx = _make_ctx()

        mock_response = MagicMock()
        mock_response.content = '{"decision":"invalid_choice","reasoning":"oops"}'

        with patch(
            "template_agent.src.core.deep_research.nodes.triage.tracked_invoke",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            updates, _ = await triage_node(state, ctx)

        assert updates["triage_decision"] == "full_research"
        assert updates["current_phase"] == PHASE_PROBE


class TestTriageNodeEdgeCases:
    """Test edge cases for triage node."""

    @pytest.mark.asyncio
    async def test_triage_emits_agent_thinking_and_decision_events(self):
        """Triage emits agent_thinking and triage_decision events."""
        state = _make_state(
            cached_findings_text="- Q: q1\n  A: a1",
            context="User: hi",
        )
        ctx = _make_ctx()

        mock_response = MagicMock()
        mock_response.content = '{"decision":"context_sufficient","reasoning":"Ok"}'

        with patch(
            "template_agent.src.core.deep_research.nodes.triage.tracked_invoke",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            _, events = await triage_node(state, ctx)

        assert len(events) >= 2

    @pytest.mark.asyncio
    async def test_triage_handles_empty_query(self):
        """Empty query does not raise."""
        state = _make_state(
            query="",
            cached_findings_text="- Q: q1\n  A: a1",
            context="User: hi",
        )
        ctx = _make_ctx()

        mock_response = MagicMock()
        mock_response.content = '{"decision":"full_research","reasoning":"Empty query"}'

        with patch(
            "template_agent.src.core.deep_research.nodes.triage.tracked_invoke",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            updates, _ = await triage_node(state, ctx)

        assert "triage_decision" in updates

    @pytest.mark.asyncio
    async def test_triage_handles_none_context(self):
        """None context is treated as empty."""
        state = _make_state(cached_findings_text="", context=None)
        ctx = _make_ctx()

        updates, _ = await triage_node(state, ctx)

        assert updates["triage_decision"] == "full_research"
