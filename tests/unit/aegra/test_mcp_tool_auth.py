"""Unit tests for MCP tool auth wrapping and error handling."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import ToolMessage

from deep_agent.aegra.mcp_auth import NeedsAuthorization
from deep_agent.aegra.mcp_tool_auth import _wrap_single_tool, wrap_mcp_tools_for_auth


def _make_mock_tool(*, name: str = "gitlab_list_issues", coroutine=None):
    """Build a mock tool with the same shape as a StructuredTool."""
    tool = MagicMock()
    tool.name = name
    tool.coroutine = coroutine
    tool.func = None
    tool.args = {}
    tool.ainvoke = AsyncMock(return_value="ok")
    return tool


class TestSafeAinvoke:
    @pytest.mark.asyncio
    async def test_passthrough_on_success(self):
        tool = _make_mock_tool()
        tool.ainvoke = AsyncMock(return_value="success result")
        original = tool.ainvoke

        wrapped = _wrap_single_tool(tool)

        result = await wrapped.ainvoke(
            {"id": "call_1", "name": "gitlab_list_issues", "args": {}}
        )
        assert result == "success result"

    @pytest.mark.asyncio
    async def test_catches_generic_exception(self):
        tool = _make_mock_tool()
        tool.ainvoke = AsyncMock(
            side_effect=RuntimeError("GitLab API error: 403 Forbidden")
        )

        wrapped = _wrap_single_tool(tool)

        result = await wrapped.ainvoke(
            {"id": "call_2", "name": "gitlab_list_issues", "args": {}}
        )
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "403 Forbidden" in result.content
        assert "[TOOL_ERROR]" in result.content
        assert result.tool_call_id == "call_2"
        assert result.name == "gitlab_list_issues"

    @pytest.mark.asyncio
    async def test_catches_mcp_error(self):
        """McpError (transport/protocol failure) is caught like any other exception."""
        try:
            from mcp.shared.exceptions import McpError
            from mcp.types import ErrorData

            exc = McpError(ErrorData(code=-1, message="server returned error"))
        except ImportError:
            exc = Exception("server returned error")

        tool = _make_mock_tool()
        tool.ainvoke = AsyncMock(side_effect=exc)

        wrapped = _wrap_single_tool(tool)

        result = await wrapped.ainvoke(
            {"id": "call_3", "name": "gitlab_list_issues", "args": {}}
        )
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "server returned error" in result.content

    @pytest.mark.asyncio
    async def test_extracts_tool_call_id_from_input(self):
        tool = _make_mock_tool()
        tool.ainvoke = AsyncMock(side_effect=ValueError("bad args"))

        wrapped = _wrap_single_tool(tool)

        result = await wrapped.ainvoke({"id": "tc_abc123"})
        assert isinstance(result, ToolMessage)
        assert result.tool_call_id == "tc_abc123"

    @pytest.mark.asyncio
    async def test_handles_non_dict_input(self):
        tool = _make_mock_tool()
        tool.ainvoke = AsyncMock(side_effect=ValueError("bad"))

        wrapped = _wrap_single_tool(tool)

        result = await wrapped.ainvoke("raw string input")
        assert isinstance(result, ToolMessage)
        assert result.tool_call_id == ""

    @pytest.mark.asyncio
    async def test_error_content_includes_tool_name(self):
        tool = _make_mock_tool(name="google_search_docs")
        tool.ainvoke = AsyncMock(side_effect=TimeoutError("timed out"))

        wrapped = _wrap_single_tool(tool)

        result = await wrapped.ainvoke({"id": "call_4"})
        assert isinstance(result, ToolMessage)
        assert "google_search_docs" in result.content

    @pytest.mark.asyncio
    async def test_reraises_graph_bubble_up(self):
        """GraphBubbleUp (including GraphInterrupt) must not be swallowed."""
        from langgraph.errors import GraphInterrupt

        tool = _make_mock_tool()
        tool.ainvoke = AsyncMock(side_effect=GraphInterrupt())

        wrapped = _wrap_single_tool(tool)

        with pytest.raises(GraphInterrupt):
            await wrapped.ainvoke({"id": "call_5"})


class TestWrapMcpToolsForAuth:
    def test_wraps_all_tools(self):
        tools = [_make_mock_tool(name=f"tool_{i}") for i in range(3)]
        original_ainvokes = [t.ainvoke for t in tools]
        wrapped = wrap_mcp_tools_for_auth(tools)
        assert len(wrapped) == 3
        for i, tool in enumerate(wrapped):
            assert tool.ainvoke is not original_ainvokes[i]

    def test_empty_list(self):
        assert wrap_mcp_tools_for_auth([]) == []
