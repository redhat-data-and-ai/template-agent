"""Comprehensive pytest tests for the plan node."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from template_agent.src.core.deep_research.events import DeepResearchEventType
from template_agent.src.core.deep_research.nodes.plan import plan_node
from template_agent.src.core.deep_research.state import (
    PHASE_AWAIT_APPROVAL,
    PHASE_SUPERVISOR,
    ResearchContext,
)


def _make_state(**overrides):
    """Create minimal DeepResearchState dict with required fields."""
    base = {
        "query": "test query",
        "thread_id": "t1",
        "current_phase": "plan",
        "probe_result": "some probe data",
    }
    base.update(overrides)
    return base


def _make_ctx(**overrides):
    """Create minimal ResearchContext for testing."""
    tool1 = MagicMock(name="tool1")
    tool1.name = "tool1"
    base_model = AsyncMock()
    ctx = ResearchContext(tools=[tool1], base_model=base_model)
    for k, v in overrides.items():
        setattr(ctx, k, v)
    return ctx


def _make_llm_responses_success():
    """Return list of mock LLM responses for successful plan flow."""
    understanding = "User wants to research a topic with multiple dimensions."
    query_type = '{"query_type": "comprehensive", "confidence": 0.85}'
    subqueries = '{"subqueries": ["What is X?", "What are the benefits of X?", "What are alternatives to X?"]}'
    validation = '{"validated_subqueries": [{"status": "answerable", "original": "What is X?"}, {"status": "answerable", "original": "What are the benefits of X?"}, {"status": "answerable", "original": "What are alternatives to X?"}]}'
    persona_review = '{"score": 85, "issues": [], "suggestions": [], "missing_subqueries": [], "redundant_subqueries": []}'
    return [
        understanding,
        query_type,
        subqueries,
        validation,
        persona_review,
        persona_review,
        persona_review,
    ]


def _mock_tracked_invoke_success(*args: object, **kwargs: object) -> MagicMock:
    """Side-effect for tracked_invoke returning success responses in order."""
    responses = _make_llm_responses_success()
    call_count: int = getattr(_mock_tracked_invoke_success, "_count", 0)
    setattr(_mock_tracked_invoke_success, "_count", call_count + 1)
    if call_count < len(responses):
        content = responses[call_count]
    else:
        content = responses[-1]
    msg = MagicMock()
    msg.content = content
    return msg


setattr(_mock_tracked_invoke_success, "_count", 0)


class TestPlanNodeSuccess:
    """Test success paths for plan_node."""

    @pytest.mark.asyncio
    async def test_plan_node_returns_state_updates_and_events(self):
        """Plan node returns (state_updates, events) tuple."""
        state = _make_state()
        ctx = _make_ctx()

        _mock_tracked_invoke_success._count = 0
        with (
            patch(
                "template_agent.src.core.deep_research.nodes.plan.tracked_invoke",
                new_callable=AsyncMock,
                side_effect=_mock_tracked_invoke_success,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan.load_cached_findings",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan._load_similar_plans_context",
                new_callable=AsyncMock,
                return_value="",
            ),
        ):
            updates, events = await plan_node(state, ctx)

        assert "understanding" in updates
        assert "subqueries" in updates
        assert "enriched_subqueries" in updates
        assert "current_phase" in updates
        assert updates["current_phase"] == PHASE_AWAIT_APPROVAL
        assert len(events) >= 5

    @pytest.mark.asyncio
    async def test_plan_node_subqueries_populated_from_llm_response(self):
        """Subqueries come from parsed LLM planning response."""
        state = _make_state()
        ctx = _make_ctx()

        _mock_tracked_invoke_success._count = 0
        with (
            patch(
                "template_agent.src.core.deep_research.nodes.plan.tracked_invoke",
                new_callable=AsyncMock,
                side_effect=_mock_tracked_invoke_success,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan.load_cached_findings",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan._load_similar_plans_context",
                new_callable=AsyncMock,
                return_value="",
            ),
        ):
            updates, _ = await plan_node(state, ctx)

        assert len(updates["subqueries"]) >= 3
        assert "What is X?" in updates["subqueries"] or any(
            "X" in sq for sq in updates["subqueries"]
        )

    @pytest.mark.asyncio
    async def test_plan_node_enriched_subqueries_have_required_keys(self):
        """Enriched subqueries have query, data_products, status."""
        state = _make_state()
        ctx = _make_ctx()

        _mock_tracked_invoke_success._count = 0
        with (
            patch(
                "template_agent.src.core.deep_research.nodes.plan.tracked_invoke",
                new_callable=AsyncMock,
                side_effect=_mock_tracked_invoke_success,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan.load_cached_findings",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan._load_similar_plans_context",
                new_callable=AsyncMock,
                return_value="",
            ),
        ):
            updates, _ = await plan_node(state, ctx)

        for eq in updates["enriched_subqueries"]:
            assert "query" in eq
            assert "data_products" in eq
            assert "status" in eq

    @pytest.mark.asyncio
    async def test_plan_node_understanding_in_state_updates(self):
        """Understanding from LLM is stored in state updates."""
        state = _make_state()
        ctx = _make_ctx()

        _mock_tracked_invoke_success._count = 0
        with (
            patch(
                "template_agent.src.core.deep_research.nodes.plan.tracked_invoke",
                new_callable=AsyncMock,
                side_effect=_mock_tracked_invoke_success,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan.load_cached_findings",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan._load_similar_plans_context",
                new_callable=AsyncMock,
                return_value="",
            ),
        ):
            updates, _ = await plan_node(state, ctx)

        assert "User wants to research" in updates["understanding"]

    @pytest.mark.asyncio
    async def test_plan_node_plan_approved_skips_await_approval(self):
        """When plan_approved is True, next phase is PHASE_SUPERVISOR."""
        state = _make_state(plan_approved=True)
        ctx = _make_ctx()

        _mock_tracked_invoke_success._count = 0
        with (
            patch(
                "template_agent.src.core.deep_research.nodes.plan.tracked_invoke",
                new_callable=AsyncMock,
                side_effect=_mock_tracked_invoke_success,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan.load_cached_findings",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan._load_similar_plans_context",
                new_callable=AsyncMock,
                return_value="",
            ),
        ):
            updates, _ = await plan_node(state, ctx)

        assert updates["current_phase"] == PHASE_SUPERVISOR


class TestPlanNodeErrorHandling:
    """Test error handling for plan node."""

    @pytest.mark.asyncio
    async def test_plan_node_understanding_failure_continues_with_fallback(self):
        """Understanding LLM failure yields fallback understanding text."""
        state = _make_state()
        ctx = _make_ctx()

        call_count = [0]

        async def mock_invoke(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("LLM timeout")
            msg = MagicMock()
            msg.content = '{"query_type": "comprehensive", "confidence": 0.5}'
            if call_count[0] == 2:
                return msg
            msg2 = MagicMock()
            msg2.content = '{"subqueries": ["Research question: test query"]}'
            if call_count[0] == 3:
                return msg2
            msg3 = MagicMock()
            msg3.content = '{"validated_subqueries": [{"status": "answerable", "original": "Research question: test query"}]}'
            if call_count[0] == 4:
                return msg3
            msg4 = MagicMock()
            msg4.content = '{"score": 70, "issues": [], "suggestions": [], "missing_subqueries": [], "redundant_subqueries": []}'
            return msg4

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.plan.tracked_invoke",
                new_callable=AsyncMock,
                side_effect=mock_invoke,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan.load_cached_findings",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan._load_similar_plans_context",
                new_callable=AsyncMock,
                return_value="",
            ),
        ):
            updates, _ = await plan_node(state, ctx)

        assert (
            "Query analysis failed" in updates["understanding"]
            or "failed" in updates["understanding"].lower()
        )
        assert len(updates["subqueries"]) >= 1

    @pytest.mark.asyncio
    async def test_plan_node_subquery_generation_failure_returns_query_as_fallback(
        self,
    ):
        """Subquery generation failure returns original query as single subquery."""
        state = _make_state(query="my research question")
        ctx = _make_ctx()

        call_count = [0]

        async def mock_invoke(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                msg = MagicMock()
                msg.content = "Understanding: user wants info"
                return msg
            if call_count[0] == 2:
                msg = MagicMock()
                msg.content = '{"query_type": "comprehensive", "confidence": 0.5}'
                return msg
            if call_count[0] == 3:
                raise RuntimeError("Subquery generation failed")
            msg = MagicMock()
            msg.content = '{"validated_subqueries": [{"status": "answerable", "original": "Research question: my research question"}]}'
            if call_count[0] == 4:
                return msg
            msg2 = MagicMock()
            msg2.content = '{"score": 70, "issues": [], "suggestions": [], "missing_subqueries": [], "redundant_subqueries": []}'
            return msg2

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.plan.tracked_invoke",
                new_callable=AsyncMock,
                side_effect=mock_invoke,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan.load_cached_findings",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan._load_similar_plans_context",
                new_callable=AsyncMock,
                return_value="",
            ),
        ):
            updates, _ = await plan_node(state, ctx)

        assert "Research question: my research question" in updates["subqueries"]

    @pytest.mark.asyncio
    async def test_plan_node_validation_all_removed_keeps_originals(self):
        """When validation removes all subqueries, originals are kept."""
        state = _make_state()
        ctx = _make_ctx()

        responses = [
            "Understanding",
            '{"query_type": "comprehensive", "confidence": 0.8}',
            '{"subqueries": ["Q1", "Q2"]}',
            '{"validated_subqueries": [{"status": "removed"}, {"status": "removed"}]}',
            '{"score": 70}',
            '{"score": 70}',
            '{"score": 70}',
        ]
        idx = [0]

        async def mock_invoke(*args, **kwargs):
            i = idx[0]
            idx[0] += 1
            msg = MagicMock()
            msg.content = responses[min(i, len(responses) - 1)]
            return msg

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.plan.tracked_invoke",
                new_callable=AsyncMock,
                side_effect=mock_invoke,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan.load_cached_findings",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan._load_similar_plans_context",
                new_callable=AsyncMock,
                return_value="",
            ),
        ):
            updates, _ = await plan_node(state, ctx)

        assert updates["subqueries"] == ["Q1", "Q2"]

    @pytest.mark.asyncio
    async def test_plan_node_invalid_json_validation_keeps_originals(self):
        """Invalid JSON from validation keeps original subqueries."""
        state = _make_state()
        ctx = _make_ctx()

        responses = [
            "Understanding text",
            '{"query_type": "comprehensive", "confidence": 0.8}',
            '{"subqueries": ["Q1", "Q2"]}',
            "not valid json at all",
            '{"score": 70}',
            '{"score": 70}',
            '{"score": 70}',
        ]
        idx = [0]

        async def mock_invoke(*args, **kwargs):
            i = idx[0]
            idx[0] += 1
            msg = MagicMock()
            msg.content = responses[min(i, len(responses) - 1)]
            return msg

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.plan.tracked_invoke",
                new_callable=AsyncMock,
                side_effect=mock_invoke,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan.load_cached_findings",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan._load_similar_plans_context",
                new_callable=AsyncMock,
                return_value="",
            ),
        ):
            updates, _ = await plan_node(state, ctx)

        assert updates["subqueries"] == ["Q1", "Q2"]


class TestPlanNodeEdgeCases:
    """Test edge cases for plan node."""

    @pytest.mark.asyncio
    async def test_plan_node_empty_query_handled(self):
        """Empty query does not raise; fallback subquery used."""
        state = _make_state(query="")
        ctx = _make_ctx()

        _mock_tracked_invoke_success._count = 0
        responses = [
            "Empty query understanding",
            '{"query_type": "comprehensive", "confidence": 0.5}',
            "{}",
            '{"validated_subqueries": []}',
            '{"score": 70}',
            '{"score": 70}',
            '{"score": 70}',
        ]
        idx = [0]

        async def mock_invoke(*args, **kwargs):
            i = idx[0]
            idx[0] += 1
            msg = MagicMock()
            msg.content = responses[min(i, len(responses) - 1)]
            return msg

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.plan.tracked_invoke",
                new_callable=AsyncMock,
                side_effect=mock_invoke,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan.load_cached_findings",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan._load_similar_plans_context",
                new_callable=AsyncMock,
                return_value="",
            ),
        ):
            updates, _ = await plan_node(state, ctx)

        assert len(updates["subqueries"]) >= 1
        assert "Research question:" in updates["subqueries"][0]

    @pytest.mark.asyncio
    async def test_plan_node_no_probe_result_handled(self):
        """Missing probe_result does not raise."""
        state = _make_state(probe_result=None)
        ctx = _make_ctx()

        _mock_tracked_invoke_success._count = 0
        with (
            patch(
                "template_agent.src.core.deep_research.nodes.plan.tracked_invoke",
                new_callable=AsyncMock,
                side_effect=_mock_tracked_invoke_success,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan.load_cached_findings",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan._load_similar_plans_context",
                new_callable=AsyncMock,
                return_value="",
            ),
        ):
            updates, _ = await plan_node(state, ctx)

        assert "subqueries" in updates
        assert "understanding" in updates

    @pytest.mark.asyncio
    async def test_plan_node_partial_research_mode_uses_min_subqueries(self):
        """Partial research triage uses min_count=1 for subquery bounds."""
        state = _make_state(triage_decision="partial_research")
        ctx = _make_ctx()

        _mock_tracked_invoke_success._count = 0
        with (
            patch(
                "template_agent.src.core.deep_research.nodes.plan.tracked_invoke",
                new_callable=AsyncMock,
                side_effect=_mock_tracked_invoke_success,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan.load_cached_findings",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan._load_similar_plans_context",
                new_callable=AsyncMock,
                return_value="",
            ),
        ):
            updates, _ = await plan_node(state, ctx)

        assert len(updates["subqueries"]) >= 1

    @pytest.mark.asyncio
    async def test_plan_node_numbered_list_subqueries_parsed(self):
        """Numbered list format (1. q1, 2. q2) is parsed correctly."""
        state = _make_state()
        ctx = _make_ctx()

        responses = [
            "Understanding",
            '{"query_type": "comprehensive", "confidence": 0.8}',
            "1. First subquery\n2. Second subquery\n3. Third subquery",
            '{"validated_subqueries": [{"status": "answerable", "original": "First subquery"}, {"status": "answerable", "original": "Second subquery"}, {"status": "answerable", "original": "Third subquery"}]}',
            '{"score": 80}',
            '{"score": 80}',
            '{"score": 80}',
        ]
        idx = [0]

        async def mock_invoke(*args, **kwargs):
            i = idx[0]
            idx[0] += 1
            msg = MagicMock()
            msg.content = responses[min(i, len(responses) - 1)]
            return msg

        with (
            patch(
                "template_agent.src.core.deep_research.nodes.plan.tracked_invoke",
                new_callable=AsyncMock,
                side_effect=mock_invoke,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan.load_cached_findings",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan._load_similar_plans_context",
                new_callable=AsyncMock,
                return_value="",
            ),
        ):
            updates, _ = await plan_node(state, ctx)

        assert "First subquery" in updates["subqueries"]
        assert "Second subquery" in updates["subqueries"]
        assert "Third subquery" in updates["subqueries"]


class TestPlanNodeEventEmission:
    """Test that plan node emits correct events."""

    @pytest.mark.asyncio
    async def test_plan_node_emits_understanding_event(self):
        """Plan node emits understanding event."""
        state = _make_state()
        ctx = _make_ctx()

        _mock_tracked_invoke_success._count = 0
        with (
            patch(
                "template_agent.src.core.deep_research.nodes.plan.tracked_invoke",
                new_callable=AsyncMock,
                side_effect=_mock_tracked_invoke_success,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan.load_cached_findings",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan._load_similar_plans_context",
                new_callable=AsyncMock,
                return_value="",
            ),
        ):
            _, events = await plan_node(state, ctx)

        understanding_events = [
            e
            for e in events
            if e.get("content", {}).get("event_type")
            == DeepResearchEventType.UNDERSTANDING.value
        ]
        assert len(understanding_events) >= 1

    @pytest.mark.asyncio
    async def test_plan_node_emits_plan_generated_event(self):
        """Plan node emits plan_generated event."""
        state = _make_state()
        ctx = _make_ctx()

        _mock_tracked_invoke_success._count = 0
        with (
            patch(
                "template_agent.src.core.deep_research.nodes.plan.tracked_invoke",
                new_callable=AsyncMock,
                side_effect=_mock_tracked_invoke_success,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan.load_cached_findings",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan._load_similar_plans_context",
                new_callable=AsyncMock,
                return_value="",
            ),
        ):
            _, events = await plan_node(state, ctx)

        plan_events = [
            e
            for e in events
            if e.get("content", {}).get("event_type")
            == DeepResearchEventType.PLAN_GENERATED.value
        ]
        assert len(plan_events) >= 1

    @pytest.mark.asyncio
    async def test_plan_node_emits_agent_thinking_events(self):
        """Plan node emits agent_thinking events during planning."""
        state = _make_state()
        ctx = _make_ctx()

        _mock_tracked_invoke_success._count = 0
        with (
            patch(
                "template_agent.src.core.deep_research.nodes.plan.tracked_invoke",
                new_callable=AsyncMock,
                side_effect=_mock_tracked_invoke_success,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan.load_cached_findings",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan._load_similar_plans_context",
                new_callable=AsyncMock,
                return_value="",
            ),
        ):
            _, events = await plan_node(state, ctx)

        thinking_events = [
            e
            for e in events
            if e.get("content", {}).get("event_type")
            == DeepResearchEventType.AGENT_THINKING.value
        ]
        assert len(thinking_events) >= 2

    @pytest.mark.asyncio
    async def test_plan_node_emits_plan_pending_when_not_approved(self):
        """Plan node emits plan_pending when plan_approved is False."""
        state = _make_state(plan_approved=False)
        ctx = _make_ctx()

        _mock_tracked_invoke_success._count = 0
        with (
            patch(
                "template_agent.src.core.deep_research.nodes.plan.tracked_invoke",
                new_callable=AsyncMock,
                side_effect=_mock_tracked_invoke_success,
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan.load_cached_findings",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "template_agent.src.core.deep_research.nodes.plan._load_similar_plans_context",
                new_callable=AsyncMock,
                return_value="",
            ),
        ):
            _, events = await plan_node(state, ctx)

        pending_events = [
            e
            for e in events
            if e.get("content", {}).get("event_type")
            == DeepResearchEventType.PLAN_PENDING.value
        ]
        assert len(pending_events) >= 1
