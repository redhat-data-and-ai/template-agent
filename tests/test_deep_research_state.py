"""Tests for the deep research state module."""

import asyncio
from unittest.mock import MagicMock

import pytest

from template_agent.src.core.deep_research.state import (
    DeepResearchStateRequired,
    FailureClass,
    Finding,
    ResearchContext,
    ReviewAction,
)


class TestReviewAction:
    """Test cases for ReviewAction enum."""

    def test_review_action_values(self):
        """Test that ReviewAction has expected values."""
        assert ReviewAction.ACCEPT.value == "accept"
        assert ReviewAction.REJECT.value == "reject"
        assert ReviewAction.REVISE.value == "revise"
        assert ReviewAction.DEFER.value == "defer"


class TestFailureClass:
    """Test cases for FailureClass enum."""

    def test_failure_class_values(self):
        """Test that FailureClass has expected values."""
        assert FailureClass.TOOL_ERROR.value == "tool_error"
        assert FailureClass.TIMEOUT.value == "timeout"
        assert FailureClass.INVALID_RESPONSE.value == "invalid_response"
        assert FailureClass.LOW_QUALITY.value == "low_quality"
        assert FailureClass.PLAUSIBILITY_CONCERN.value == "plausibility_concern"
        assert FailureClass.DATA_QUALITY.value == "data_quality"
        assert FailureClass.OTHER.value == "other"


class TestFinding:
    """Test cases for Finding TypedDict."""

    def test_finding_creation_with_minimal_fields(self):
        """Test creating Finding with minimal fields."""
        finding: Finding = {"subquery": "q1", "answer": "a1"}

        assert finding["subquery"] == "q1"
        assert finding["answer"] == "a1"

    def test_finding_creation_with_all_fields(self):
        """Test creating Finding with all optional fields."""
        finding: Finding = {
            "subquery": "q1",
            "answer": "a1",
            "tool_results": [],
            "error": None,
            "failure_class": "tool_error",
            "cached": False,
            "execution_time_ms": 100.0,
            "plausibility_concern": False,
            "plausibility_warnings": [],
            "data_quality_alert": False,
            "low_quality_drop": False,
            "data_quality_score": 0.9,
            "access_denied": False,
            "resources_used": ["tool1"],
        }

        assert finding["subquery"] == "q1"
        assert finding["execution_time_ms"] == 100.0
        assert finding["resources_used"] == ["tool1"]


class TestDeepResearchState:
    """Test cases for DeepResearchState TypedDicts."""

    def test_deep_research_state_required_fields(self):
        """Test that DeepResearchStateRequired has required fields."""
        state: DeepResearchStateRequired = {
            "query": "test query",
            "thread_id": "thread-1",
            "current_phase": "plan",
        }

        assert state["query"] == "test query"
        assert state["thread_id"] == "thread-1"
        assert state["current_phase"] == "plan"


class TestResearchContext:
    """Test cases for ResearchContext dataclass."""

    def test_research_context_creation(self):
        """Test creating ResearchContext with mock tools and model."""
        tools = [MagicMock(name="tool1")]
        base_model = MagicMock()

        ctx = ResearchContext(tools=tools, base_model=base_model)

        assert ctx.tools == tools
        assert ctx.base_model == base_model
        assert ctx.event_queue is None

    def test_research_context_emit_with_queue(self):
        """Test that emit puts event in queue when queue is configured."""
        queue = asyncio.Queue()
        ctx = ResearchContext(
            tools=[MagicMock()],
            base_model=MagicMock(),
            event_queue=queue,
        )
        event = {"type": "test", "data": 1}

        ctx.emit(event)

        assert queue.get_nowait() == event

    def test_research_context_emit_without_queue(self):
        """Test that emit does nothing when queue is None."""
        ctx = ResearchContext(
            tools=[MagicMock()],
            base_model=MagicMock(),
            event_queue=None,
        )

        ctx.emit({"type": "test"})  # Should not raise

    def test_research_context_emit_queue_full_silently_ignored(self):
        """Test that emit catches QueueFull and does not raise."""
        queue = asyncio.Queue(maxsize=0)  # Cannot put anything
        ctx = ResearchContext(
            tools=[MagicMock()],
            base_model=MagicMock(),
            event_queue=queue,
        )

        ctx.emit({"type": "test"})  # Should not raise

    def test_research_context_emit_or_append_appends_and_emits(self):
        """Test that emit_or_append appends to fallback_list and emits."""
        queue = asyncio.Queue()
        fallback_list: list = []
        ctx = ResearchContext(
            tools=[MagicMock()],
            base_model=MagicMock(),
            event_queue=queue,
        )
        event = {"type": "finding", "data": "x"}

        ctx.emit_or_append(event, fallback_list)

        assert fallback_list == [event]
        assert queue.get_nowait() == event

    def test_research_context_emit_or_append_key_value(self):
        """Test emit_or_append with key/value only."""
        queue = asyncio.Queue()
        ctx = ResearchContext(
            tools=[MagicMock()],
            base_model=MagicMock(),
            event_queue=queue,
        )

        ctx.emit_or_append(key="status", value="done")

        assert queue.get_nowait() == {"status": "done"}

    def test_research_context_llm_call_kwargs_without_tracer(self):
        """Test llm_call_kwargs when root_tracer is None."""
        ctx = ResearchContext(
            tools=[MagicMock()],
            base_model=MagicMock(),
            root_tracer=None,
        )

        kwargs = ctx.llm_call_kwargs()

        assert kwargs == {"timeout": 120}

    def test_research_context_llm_call_kwargs_with_tracer(self):
        """Test llm_call_kwargs when root_tracer is set."""
        tracer = MagicMock()
        ctx = ResearchContext(
            tools=[MagicMock()],
            base_model=MagicMock(),
            root_tracer=tracer,
        )

        kwargs = ctx.llm_call_kwargs()

        assert kwargs == {"timeout": 120, "root_tracer": tracer}

    def test_research_context_get_llm_kwargs_with_overrides(self):
        """Test get_llm_kwargs applies overrides."""
        ctx = ResearchContext(
            tools=[MagicMock()],
            base_model=MagicMock(),
        )

        kwargs = ctx.get_llm_kwargs(temperature=0.5, max_tokens=100)

        assert kwargs["timeout"] == 120
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 100

    def test_research_context_get_tool_names(self):
        """Test get_tool_names returns tool names."""
        tool1 = MagicMock()
        tool1.name = "tool1"
        tool2 = MagicMock()
        tool2.name = "tool2"
        ctx = ResearchContext(
            tools=[tool1, tool2],
            base_model=MagicMock(),
        )

        names = ctx.get_tool_names()

        assert names == ["tool1", "tool2"]

    def test_research_context_format_tool_inventory_few_tools(self):
        """Test format_tool_inventory with fewer than max_tools."""
        tools = [MagicMock() for _ in range(3)]
        for i, t in enumerate(tools):
            t.name = f"t{i}"
        ctx = ResearchContext(tools=tools, base_model=MagicMock())

        result = ctx.format_tool_inventory(max_tools=50)

        assert result == "t0, t1, t2"

    def test_research_context_format_tool_inventory_many_tools(self):
        """Test format_tool_inventory truncates when exceeding max_tools."""
        tools = [MagicMock() for _ in range(10)]
        for i, t in enumerate(tools):
            t.name = f"t{i}"
        ctx = ResearchContext(tools=tools, base_model=MagicMock())

        result = ctx.format_tool_inventory(max_tools=3)

        assert result == "t0, t1, t2 ... (+7 more)"

    def test_research_context_max_mode_property(self):
        """Test max_mode property reflects _max_mode."""
        ctx = ResearchContext(
            tools=[MagicMock()],
            base_model=MagicMock(),
            _max_mode=True,
        )

        assert ctx.max_mode is True

        ctx2 = ResearchContext(
            tools=[MagicMock()],
            base_model=MagicMock(),
            _max_mode=False,
        )
        assert ctx2.max_mode is False


class TestResearchContextAsync:
    """Test cases for ResearchContext async methods."""

    @pytest.mark.asyncio
    async def test_research_context_async_emit_or_append_appends_and_emits(self):
        """Test async_emit_or_append appends and emits."""
        queue = asyncio.Queue()
        fallback_list: list = []
        ctx = ResearchContext(
            tools=[MagicMock()],
            base_model=MagicMock(),
            event_queue=queue,
        )
        event = {"type": "async_finding"}

        await ctx.async_emit_or_append(event, fallback_list)

        assert fallback_list == [event]
        assert queue.get_nowait() == event
