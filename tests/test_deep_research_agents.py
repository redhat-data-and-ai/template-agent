"""Comprehensive pytest tests for the deep research agents module."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from template_agent.src.core.deep_research.agents import (
    execute_with_research_agent,
    extract_answer_from_result,
    extract_tool_results,
    get_research_context,
)
from template_agent.src.core.deep_research.state import ResearchContext


class TestGetResearchContext:
    """Test get_research_context async context manager."""

    @pytest.mark.asyncio
    async def test_get_research_context_yields_context_with_tools(self):
        """Context manager yields ResearchContext with tools and model."""
        mock_tools = [MagicMock(name="tool1"), MagicMock(name="tool2")]
        mock_model = MagicMock()

        with (
            patch(
                "template_agent.src.core.deep_research.agents.MultiServerMCPClient"
            ) as mock_client_cls,
            patch(
                "template_agent.src.core.deep_research.agents.ChatGoogleGenerativeAI",
                return_value=mock_model,
            ),
        ):
            mock_client = MagicMock()
            mock_client.get_tools = AsyncMock(return_value=mock_tools)
            mock_client_cls.return_value = mock_client

            async with get_research_context() as ctx:
                assert isinstance(ctx, ResearchContext)
                assert ctx.tools == mock_tools
                assert ctx.base_model is mock_model
                assert ctx.user_id is None
                assert ctx.checkpointer is None

    @pytest.mark.asyncio
    async def test_get_research_context_with_model_name_override(self):
        """Model name override is passed to ChatGoogleGenerativeAI."""
        with (
            patch(
                "template_agent.src.core.deep_research.agents.MultiServerMCPClient"
            ) as mock_client_cls,
            patch(
                "template_agent.src.core.deep_research.agents.ChatGoogleGenerativeAI"
            ) as mock_llm_cls,
        ):
            mock_client = MagicMock()
            mock_client.get_tools = AsyncMock(return_value=[])
            mock_client_cls.return_value = mock_client

            async with get_research_context(model_name="gemini-2.5-pro") as ctx:
                assert ctx.model_name == "gemini-2.5-pro"
                mock_llm_cls.assert_called_once()
                call_kwargs = mock_llm_cls.call_args[1]
                assert call_kwargs.get("model") == "gemini-2.5-pro"

    @pytest.mark.asyncio
    async def test_get_research_context_defaults_to_gemini_flash(self):
        """No model name defaults to gemini-2.5-flash."""
        with (
            patch(
                "template_agent.src.core.deep_research.agents.MultiServerMCPClient"
            ) as mock_client_cls,
            patch(
                "template_agent.src.core.deep_research.agents.ChatGoogleGenerativeAI"
            ) as mock_llm_cls,
        ):
            mock_client = MagicMock()
            mock_client.get_tools = AsyncMock(return_value=[])
            mock_client_cls.return_value = mock_client

            async with get_research_context() as ctx:
                assert ctx.model_name == "gemini-2.5-flash"
                call_args = mock_llm_cls.call_args
                assert call_args[1]["model"] == "gemini-2.5-flash"

    @pytest.mark.asyncio
    async def test_get_research_context_mcp_timeout_uses_empty_tools(self):
        """MCP connection timeout results in empty tools list."""
        with (
            patch(
                "template_agent.src.core.deep_research.agents.MultiServerMCPClient"
            ) as mock_client_cls,
            patch(
                "template_agent.src.core.deep_research.agents.ChatGoogleGenerativeAI"
            ) as mock_llm_cls,
        ):
            mock_client = MagicMock()
            mock_client.get_tools = AsyncMock(
                side_effect=asyncio.TimeoutError("Connection timed out")
            )
            mock_client_cls.return_value = mock_client
            mock_llm_cls.return_value = MagicMock()

            async with get_research_context() as ctx:
                assert ctx.tools == []
                assert ctx.base_model is not None

    @pytest.mark.asyncio
    async def test_get_research_context_mcp_exception_uses_empty_tools(self):
        """MCP exception results in empty tools list."""
        with (
            patch(
                "template_agent.src.core.deep_research.agents.MultiServerMCPClient"
            ) as mock_client_cls,
            patch(
                "template_agent.src.core.deep_research.agents.ChatGoogleGenerativeAI"
            ) as mock_llm_cls,
        ):
            mock_client = MagicMock()
            mock_client.get_tools = AsyncMock(side_effect=Exception("MCP unavailable"))
            mock_client_cls.return_value = mock_client
            mock_llm_cls.return_value = MagicMock()

            async with get_research_context() as ctx:
                assert ctx.tools == []
                assert ctx.base_model is not None

    @pytest.mark.asyncio
    async def test_get_research_context_with_user_id(self):
        """User ID is passed to ResearchContext."""
        with (
            patch(
                "template_agent.src.core.deep_research.agents.MultiServerMCPClient"
            ) as mock_client_cls,
            patch(
                "template_agent.src.core.deep_research.agents.ChatGoogleGenerativeAI"
            ),
        ):
            mock_client = MagicMock()
            mock_client.get_tools = AsyncMock(return_value=[])
            mock_client_cls.return_value = mock_client

            async with get_research_context(user_id="user-123") as ctx:
                assert ctx.user_id == "user-123"

    @pytest.mark.asyncio
    async def test_get_research_context_with_checkpointer(self):
        """Checkpointer is passed to ResearchContext."""
        checkpointer = MagicMock()
        with (
            patch(
                "template_agent.src.core.deep_research.agents.MultiServerMCPClient"
            ) as mock_client_cls,
            patch(
                "template_agent.src.core.deep_research.agents.ChatGoogleGenerativeAI"
            ),
        ):
            mock_client = MagicMock()
            mock_client.get_tools = AsyncMock(return_value=[])
            mock_client_cls.return_value = mock_client

            async with get_research_context(checkpointer=checkpointer) as ctx:
                assert ctx.checkpointer is checkpointer

    @pytest.mark.asyncio
    async def test_get_research_context_with_event_queue(self):
        """Event queue is passed to ResearchContext."""
        queue = MagicMock()
        with (
            patch(
                "template_agent.src.core.deep_research.agents.MultiServerMCPClient"
            ) as mock_client_cls,
            patch(
                "template_agent.src.core.deep_research.agents.ChatGoogleGenerativeAI"
            ),
        ):
            mock_client = MagicMock()
            mock_client.get_tools = AsyncMock(return_value=[])
            mock_client_cls.return_value = mock_client

            async with get_research_context(event_queue=queue) as ctx:
                assert ctx.event_queue is queue


class TestExecuteWithResearchAgent:
    """Test execute_with_research_agent."""

    @pytest.mark.asyncio
    async def test_execute_returns_answer_and_tool_results(self):
        """Successful execution returns answer and tool_results."""
        mock_worker = MagicMock()
        mock_worker.ainvoke = AsyncMock(
            return_value={
                "messages": [
                    HumanMessage(content="query"),
                    AIMessage(content="The answer is 42."),
                ]
            }
        )

        ctx = ResearchContext(tools=[MagicMock()], base_model=MagicMock())

        with patch(
            "template_agent.src.core.deep_research.agents.create_react_agent",
            return_value=mock_worker,
        ):
            result = await execute_with_research_agent(ctx, "What is 6*7?")
            assert result["answer"] == "The answer is 42."
            assert "tool_results" in result
            assert isinstance(result["tool_results"], list)

    @pytest.mark.asyncio
    async def test_execute_with_tool_messages_extracts_tool_results(self):
        """ToolMessage results are extracted and formatted."""
        tool_msg = ToolMessage(
            content="result data", tool_call_id="tc1", name="search_tool"
        )
        mock_worker = MagicMock()
        mock_worker.ainvoke = AsyncMock(
            return_value={
                "messages": [
                    HumanMessage(content="q"),
                    tool_msg,
                    AIMessage(content="Final answer"),
                ]
            }
        )

        ctx = ResearchContext(tools=[MagicMock()], base_model=MagicMock())

        with patch(
            "template_agent.src.core.deep_research.agents.create_react_agent",
            return_value=mock_worker,
        ):
            result = await execute_with_research_agent(ctx, "query")
            assert result["answer"] == "Final answer"
            assert len(result["tool_results"]) >= 1
            assert "search_tool" in result["tool_results"][0]

    @pytest.mark.asyncio
    async def test_execute_timeout_after_retries_returns_error(self):
        """Timeout after max retries returns error dict."""
        mock_worker = MagicMock()
        mock_worker.ainvoke = AsyncMock(side_effect=TimeoutError("timeout"))

        ctx = ResearchContext(tools=[], base_model=MagicMock())

        with patch(
            "template_agent.src.core.deep_research.agents.create_react_agent",
            return_value=mock_worker,
        ):
            result = await execute_with_research_agent(ctx, "query", timeout=1)
            assert "error" in result
            assert (
                "timeout" in result["error"].lower()
                or "timed out" in result["error"].lower()
            )
            assert result["answer"] == ""
            assert result["tool_results"] == []

    @pytest.mark.asyncio
    async def test_execute_retries_on_timeout(self):
        """Execution retries on timeout before giving up."""
        call_count = 0

        async def mock_invoke(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("timeout")
            return {"messages": [HumanMessage(content="q"), AIMessage(content="ok")]}

        mock_worker = MagicMock()
        mock_worker.ainvoke = mock_invoke

        ctx = ResearchContext(tools=[], base_model=MagicMock())

        with patch(
            "template_agent.src.core.deep_research.agents.create_react_agent",
            return_value=mock_worker,
        ):
            result = await execute_with_research_agent(ctx, "query", timeout=60)
            assert result["answer"] == "ok"
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_execute_non_timeout_exception_raises_after_retries(self):
        """Non-TimeoutError exception is re-raised after retries."""
        mock_worker = MagicMock()
        mock_worker.ainvoke = AsyncMock(side_effect=ValueError("bad"))

        ctx = ResearchContext(tools=[], base_model=MagicMock())

        with patch(
            "template_agent.src.core.deep_research.agents.create_react_agent",
            return_value=mock_worker,
        ):
            with pytest.raises(ValueError, match="bad"):
                await execute_with_research_agent(ctx, "query")

    @pytest.mark.asyncio
    async def test_execute_empty_result_returns_empty_answer(self):
        """Non-dict result returns empty answer."""
        mock_worker = MagicMock()
        mock_worker.ainvoke = AsyncMock(return_value=None)

        ctx = ResearchContext(tools=[], base_model=MagicMock())

        with patch(
            "template_agent.src.core.deep_research.agents.create_react_agent",
            return_value=mock_worker,
        ):
            result = await execute_with_research_agent(ctx, "query")
            assert result["answer"] == ""
            assert result["tool_results"] == []


class TestExtractAnswerFromResult:
    """Test extract_answer_from_result."""

    def test_extract_answer_from_last_ai_message(self):
        """Last AIMessage content is extracted."""
        result = {
            "messages": [
                HumanMessage(content="q"),
                AIMessage(content="First"),
                AIMessage(content="Final answer"),
            ]
        }
        assert extract_answer_from_result(result) == "Final answer"

    def test_extract_answer_empty_messages_returns_empty(self):
        """Empty messages returns empty string."""
        assert extract_answer_from_result({"messages": []}) == ""
        assert extract_answer_from_result({}) == ""

    def test_extract_answer_skips_empty_ai_content(self):
        """Empty AI content is skipped in favor of prior AI message."""
        result = {
            "messages": [
                HumanMessage(content="q"),
                AIMessage(content="Real answer"),
                AIMessage(content=""),
            ]
        }
        assert extract_answer_from_result(result) == "Real answer"

    def test_extract_answer_strips_whitespace(self):
        """Answer content is stripped."""
        result = {
            "messages": [
                HumanMessage(content="q"),
                AIMessage(content="  trimmed  "),
            ]
        }
        assert extract_answer_from_result(result) == "trimmed"

    def test_extract_answer_no_ai_message_returns_empty(self):
        """No AIMessage in result returns empty."""
        result = {
            "messages": [
                HumanMessage(content="q"),
                ToolMessage(content="tool", tool_call_id="tc1", name="t"),
            ]
        }
        assert extract_answer_from_result(result) == ""


class TestExtractToolResults:
    """Test extract_tool_results."""

    def test_extract_tool_results_formats_tool_messages(self):
        """ToolMessage entries are formatted with name and content."""
        result = {
            "messages": [
                ToolMessage(content="data1", tool_call_id="tc1", name="tool_a"),
                ToolMessage(content="data2", tool_call_id="tc2", name="tool_b"),
            ]
        }
        tool_results = extract_tool_results(result)
        assert len(tool_results) == 2
        assert "tool_a" in tool_results[0]
        assert "data1" in tool_results[0]
        assert "tool_b" in tool_results[1]

    def test_extract_tool_results_empty_messages_returns_empty_list(self):
        """Empty messages returns empty list."""
        assert extract_tool_results({"messages": []}) == []
        assert extract_tool_results({}) == []

    def test_extract_tool_results_ignores_non_tool_messages(self):
        """Human and AI messages are ignored."""
        result = {
            "messages": [
                HumanMessage(content="q"),
                AIMessage(content="a"),
                ToolMessage(content="t", tool_call_id="tc1", name="tool"),
            ]
        }
        tool_results = extract_tool_results(result)
        assert len(tool_results) == 1
        assert "tool" in tool_results[0]

    def test_extract_tool_results_truncates_long_content(self):
        """Long tool content is truncated to 3000 chars."""
        long_content = "x" * 5000
        result = {
            "messages": [
                ToolMessage(content=long_content, tool_call_id="tc1", name="tool"),
            ]
        }
        tool_results = extract_tool_results(result)
        assert len(tool_results) == 1
        assert len(tool_results[0]) <= 3100  # name + ": " + truncated

    def test_extract_tool_results_skips_empty_content(self):
        """ToolMessage with empty content is skipped."""
        result = {
            "messages": [
                ToolMessage(content="", tool_call_id="tc1", name="tool"),
            ]
        }
        tool_results = extract_tool_results(result)
        assert len(tool_results) == 0

    def test_extract_tool_results_includes_tool_name_in_output(self):
        """Tool name is included in formatted output."""
        msg = ToolMessage(content="data", tool_call_id="tc1", name="my_tool")
        result = {"messages": [msg]}
        tool_results = extract_tool_results(result)
        assert len(tool_results) == 1
        assert "my_tool" in tool_results[0]
        assert "data" in tool_results[0]


class TestGetMessageContent:
    """Test _get_message_content via extract functions."""

    def test_message_content_from_list_extracts_text(self):
        """Content as list of dicts with text key is concatenated."""
        from template_agent.src.core.deep_research.agents import _get_message_content

        msg = MagicMock()
        msg.content = [{"text": "part1"}, {"text": "part2"}]
        assert _get_message_content(msg) == "part1part2"

    def test_message_content_from_dict(self):
        """Content from dict message."""
        from template_agent.src.core.deep_research.agents import _get_message_content

        msg = {"content": "hello"}
        assert _get_message_content(msg) == "hello"

    def test_message_content_fallback_str(self):
        """Non-message object returns str()."""
        from template_agent.src.core.deep_research.agents import _get_message_content

        assert _get_message_content(123) == "123"


class TestResearchContextConfiguration:
    """Test ResearchContext behavior when used with agents."""

    def test_context_format_tool_inventory(self):
        """Context formats tool names for display."""
        tool1 = MagicMock()
        tool1.name = "search"
        tool2 = MagicMock()
        tool2.name = "query"
        ctx = ResearchContext(tools=[tool1, tool2], base_model=MagicMock())
        inv = ctx.format_tool_inventory(max_tools=10)
        assert "search" in inv
        assert "query" in inv

    def test_context_get_tool_names(self):
        """Context returns tool names."""
        tool = MagicMock()
        tool.name = "my_tool"
        ctx = ResearchContext(tools=[tool], base_model=MagicMock())
        assert ctx.get_tool_names() == ["my_tool"]
