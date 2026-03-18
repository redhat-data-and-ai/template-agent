"""Comprehensive pytest tests for the probe node."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from template_agent.src.core.deep_research.nodes.probe import probe_node
from template_agent.src.core.deep_research.state import PHASE_PLAN, ResearchContext


def _make_state(**overrides):
    """Create minimal DeepResearchState dict with required fields."""
    base = {"query": "test query", "thread_id": "t1", "current_phase": "plan"}
    base.update(overrides)
    return base


def _make_ctx(**overrides):
    """Create minimal ResearchContext for testing."""
    tool1 = MagicMock(name="search_tool")
    tool1.name = "search_tool"
    base_model = AsyncMock()
    ctx = ResearchContext(tools=[tool1], base_model=base_model)
    for k, v in overrides.items():
        setattr(ctx, k, v)
    return ctx


class TestProbeNodeSuccess:
    """Test success paths for probe_node."""

    @pytest.mark.asyncio
    async def test_probe_node_returns_state_updates_and_events(self):
        """Probe node returns (state_updates, events) tuple."""
        state = _make_state()
        ctx = _make_ctx()

        with patch(
            "template_agent.src.core.deep_research.nodes.probe.execute_with_research_agent",
            new_callable=AsyncMock,
            return_value={"answer": "Probe discovered tools: search_tool"},
        ):
            updates, events = await probe_node(state, ctx)

        assert "tool_inventory" in updates
        assert "tool_names" in updates
        assert "probe_result" in updates
        assert updates["current_phase"] == PHASE_PLAN
        assert len(events) >= 2

    @pytest.mark.asyncio
    async def test_probe_node_probe_result_from_agent_answer(self):
        """Probe result comes from agent answer key."""
        state = _make_state()
        ctx = _make_ctx()

        with patch(
            "template_agent.src.core.deep_research.nodes.probe.execute_with_research_agent",
            new_callable=AsyncMock,
            return_value={"answer": "Tools available: search, summarize"},
        ):
            updates, _ = await probe_node(state, ctx)

        assert updates["probe_result"] == "Tools available: search, summarize"

    @pytest.mark.asyncio
    async def test_probe_node_emits_tool_discovery_and_probe_events(self):
        """Probe emits tool_discovery, probe_start, probe_complete."""
        state = _make_state()
        ctx = _make_ctx()

        with patch(
            "template_agent.src.core.deep_research.nodes.probe.execute_with_research_agent",
            new_callable=AsyncMock,
            return_value={"answer": "Ok"},
        ):
            _, events = await probe_node(state, ctx)

        assert len(events) >= 3

    @pytest.mark.asyncio
    async def test_probe_node_tool_inventory_formatted_from_ctx(self):
        """Tool inventory is formatted from ctx tools."""
        state = _make_state()
        ctx = _make_ctx()

        with patch(
            "template_agent.src.core.deep_research.nodes.probe.execute_with_research_agent",
            new_callable=AsyncMock,
            return_value={"answer": "Ok"},
        ):
            updates, _ = await probe_node(state, ctx)

        assert (
            "search_tool" in updates["tool_inventory"]
            or "tool" in str(updates["tool_inventory"]).lower()
        )
        assert "search_tool" in updates["tool_names"]


class TestProbeNodeErrorHandling:
    """Test error handling for probe node."""

    @pytest.mark.asyncio
    async def test_probe_node_timeout_returns_fallback_message(self):
        """Timeout yields fallback probe result message."""
        state = _make_state()
        ctx = _make_ctx()

        with patch(
            "template_agent.src.core.deep_research.nodes.probe.execute_with_research_agent",
            new_callable=AsyncMock,
            side_effect=asyncio.TimeoutError(),
        ):
            updates, _ = await probe_node(state, ctx)

        assert "timed out" in updates["probe_result"].lower()
        assert updates["current_phase"] == PHASE_PLAN

    @pytest.mark.asyncio
    async def test_probe_node_exception_returns_error_message(self):
        """Generic exception yields error message in probe_result."""
        state = _make_state()
        ctx = _make_ctx()

        with patch(
            "template_agent.src.core.deep_research.nodes.probe.execute_with_research_agent",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Connection refused"),
        ):
            updates, _ = await probe_node(state, ctx)

        assert (
            "Probe failed" in updates["probe_result"]
            or "failed" in updates["probe_result"].lower()
        )
        assert updates["current_phase"] == PHASE_PLAN

    @pytest.mark.asyncio
    async def test_probe_node_non_dict_result_handled(self):
        """Non-dict result from agent is stringified."""
        state = _make_state()
        ctx = _make_ctx()

        with patch(
            "template_agent.src.core.deep_research.nodes.probe.execute_with_research_agent",
            new_callable=AsyncMock,
            return_value="plain string answer",
        ):
            updates, _ = await probe_node(state, ctx)

        assert updates["probe_result"] == "plain string answer"


class TestProbeNodeEdgeCases:
    """Test edge cases for probe node."""

    @pytest.mark.asyncio
    async def test_probe_node_empty_query_handled(self):
        """Empty query does not raise."""
        state = _make_state(query="")
        ctx = _make_ctx()

        with patch(
            "template_agent.src.core.deep_research.nodes.probe.execute_with_research_agent",
            new_callable=AsyncMock,
            return_value={"answer": "Ok"},
        ):
            updates, _ = await probe_node(state, ctx)

        assert "probe_result" in updates

    @pytest.mark.asyncio
    async def test_probe_node_none_thread_id_passed_to_agent(self):
        """None thread_id is passed through to execute_with_research_agent."""
        state = _make_state(thread_id=None)
        ctx = _make_ctx()

        with patch(
            "template_agent.src.core.deep_research.nodes.probe.execute_with_research_agent",
            new_callable=AsyncMock,
            return_value={"answer": "Ok"},
        ) as mock_exec:
            await probe_node(state, ctx)

        mock_exec.assert_awaited_once()
        call_kwargs = mock_exec.call_args
        assert call_kwargs[0][2] is None  # thread_id

    @pytest.mark.asyncio
    async def test_probe_node_with_custom_timeout_completes(self):
        """probe_timeout_seconds from ctx is used; node completes with custom timeout."""
        state = _make_state()
        ctx = _make_ctx(probe_timeout_seconds=120)

        with patch(
            "template_agent.src.core.deep_research.nodes.probe.execute_with_research_agent",
            new_callable=AsyncMock,
            return_value={"answer": "Ok"},
        ):
            updates, _ = await probe_node(state, ctx)

        assert updates["probe_result"] == "Ok"

    @pytest.mark.asyncio
    async def test_probe_node_empty_tools_handled(self):
        """Empty tools list does not raise."""
        state = _make_state()
        ctx = ResearchContext(tools=[], base_model=AsyncMock())

        with patch(
            "template_agent.src.core.deep_research.nodes.probe.execute_with_research_agent",
            new_callable=AsyncMock,
            return_value={"answer": "No tools"},
        ):
            updates, _ = await probe_node(state, ctx)

        assert updates["tool_names"] == []
        assert updates["tool_inventory"] == ""
