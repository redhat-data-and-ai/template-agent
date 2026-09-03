"""Unit tests for MCP tool auth wrapping and error handling."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import ToolMessage

from deep_agent.aegra.mcp_auth import NeedsAuthorization
from deep_agent.aegra.mcp_tool_auth import (
    _extract_needs_authorization,
    _wrap_single_tool,
    wrap_mcp_tools_for_auth,
)


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


class TestExtractNeedsAuthorization:
    def test_returns_bare_needs_authorization(self):
        exc = NeedsAuthorization("mcp-x", "/connect")
        assert _extract_needs_authorization(exc) is exc

    def test_returns_none_for_unrelated_exception(self):
        assert _extract_needs_authorization(RuntimeError("boom")) is None

    def test_unwraps_from_exception_group(self):
        inner = NeedsAuthorization("mcp-x", "/connect")
        group = ExceptionGroup("task group", [inner])
        assert _extract_needs_authorization(group) is inner

    def test_unwraps_nested_exception_group(self):
        inner = NeedsAuthorization("mcp-x", "/connect")
        inner_group = ExceptionGroup("inner", [inner])
        outer_group = ExceptionGroup("outer", [RuntimeError("other"), inner_group])
        assert _extract_needs_authorization(outer_group) is inner

    def test_unwraps_from_cause(self):
        inner = NeedsAuthorization("mcp-x", "/connect")
        outer = RuntimeError("wrapper")
        outer.__cause__ = inner
        assert _extract_needs_authorization(outer) is inner

    def test_returns_none_for_empty_group(self):
        group = ExceptionGroup("empty", [ValueError("a"), RuntimeError("b")])
        assert _extract_needs_authorization(group) is None

    def test_unwraps_from_context(self):
        inner = NeedsAuthorization("mcp-x", "/connect")
        outer = RuntimeError("implicit wrapper")
        outer.__context__ = inner
        assert _extract_needs_authorization(outer) is inner


class TestSafeAinvokeExceptionGroupUnwrap:
    @pytest.mark.asyncio
    async def test_exception_group_with_needs_authorization_triggers_interrupt(self):
        """NeedsAuthorization inside ExceptionGroup should trigger interrupt, not [TOOL_ERROR]."""
        from langgraph.errors import GraphInterrupt

        inner = NeedsAuthorization("gitlab-mcp", "/mcp/gitlab-mcp/connect")
        group = ExceptionGroup("unhandled errors in a TaskGroup", [inner])

        tool = _make_mock_tool()
        tool.ainvoke = AsyncMock(side_effect=group)

        wrapped = _wrap_single_tool(tool)

        with patch(
            "deep_agent.aegra.mcp_tool_auth.interrupt",
            side_effect=GraphInterrupt(),
        ) as mock_interrupt:
            with pytest.raises(GraphInterrupt):
                await wrapped.ainvoke({"id": "call_eg", "name": "gitlab_list_issues"})

        mock_interrupt.assert_called_once()
        payload = mock_interrupt.call_args[0][0]
        assert "gitlab-mcp" in payload

    @pytest.mark.asyncio
    async def test_exception_group_with_needs_auth_retries_after_interrupt(self):
        """After interrupt returns normally, safe_ainvoke retries and returns the result."""
        inner = NeedsAuthorization("gitlab-mcp", "/mcp/gitlab-mcp/connect")
        group = ExceptionGroup("unhandled errors in a TaskGroup", [inner])

        tool = _make_mock_tool()
        tool.ainvoke = AsyncMock(side_effect=[group, "retry-ok"])

        wrapped = _wrap_single_tool(tool)

        with patch(
            "deep_agent.aegra.mcp_tool_auth.interrupt",
            return_value=None,
        ):
            result = await wrapped.ainvoke({"id": "call_retry"})

        assert result == "retry-ok"

    @pytest.mark.asyncio
    async def test_exception_group_without_needs_auth_returns_tool_error(self):
        """ExceptionGroup without NeedsAuthorization should still return [TOOL_ERROR]."""
        group = ExceptionGroup("task group", [RuntimeError("connection failed")])

        tool = _make_mock_tool()
        tool.ainvoke = AsyncMock(side_effect=group)

        wrapped = _wrap_single_tool(tool)

        result = await wrapped.ainvoke({"id": "call_eg2"})
        assert isinstance(result, ToolMessage)
        assert "[TOOL_ERROR]" in result.content


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


def _make_coroutine_tool(name, coroutine):
    """Build a mock tool that behaves like a StructuredTool with a coroutine."""
    tool = MagicMock()
    tool.name = name
    tool.coroutine = coroutine
    tool.func = None
    tool.args = {}
    tool.model_copy = MagicMock(side_effect=AttributeError("no pydantic"))
    return tool


def _make_func_tool(name, func):
    """Build a mock tool that behaves like a StructuredTool with a sync func."""
    tool = MagicMock()
    tool.name = name
    tool.coroutine = None
    tool.func = func
    tool.args = {}
    tool.model_copy = MagicMock(side_effect=AttributeError("no pydantic"))
    return tool


class TestWrappedCoroutineExceptionHandling:
    """Test ExceptionGroup/NeedsAuthorization handling in wrapped_coroutine path."""

    @pytest.mark.asyncio
    async def test_needs_authorization_triggers_interrupt(self):
        from langgraph.errors import GraphInterrupt

        exc = NeedsAuthorization("mcp-x", "/connect")

        async def failing_coroutine(**kwargs):
            raise exc

        tool = _make_coroutine_tool("test_tool", failing_coroutine)
        wrapped = _wrap_single_tool(tool)

        with patch(
            "deep_agent.aegra.mcp_tool_auth.interrupt",
            side_effect=GraphInterrupt(),
        ) as mock_interrupt:
            with pytest.raises(GraphInterrupt):
                await wrapped.coroutine()

        mock_interrupt.assert_called_once()
        payload = mock_interrupt.call_args[0][0]
        assert "mcp-x" in payload

    @pytest.mark.asyncio
    async def test_exception_group_with_needs_auth_triggers_interrupt(self):
        from langgraph.errors import GraphInterrupt

        inner = NeedsAuthorization("mcp-y", "/connect")
        group = ExceptionGroup("task group", [inner])

        async def failing_coroutine(**kwargs):
            raise group

        tool = _make_coroutine_tool("test_tool", failing_coroutine)
        wrapped = _wrap_single_tool(tool)

        with patch(
            "deep_agent.aegra.mcp_tool_auth.interrupt",
            side_effect=GraphInterrupt(),
        ) as mock_interrupt:
            with pytest.raises(GraphInterrupt):
                await wrapped.coroutine()

        mock_interrupt.assert_called_once()
        payload = mock_interrupt.call_args[0][0]
        assert "mcp-y" in payload

    @pytest.mark.asyncio
    async def test_exception_group_without_needs_auth_reraises(self):
        group = ExceptionGroup("task group", [RuntimeError("unrelated")])

        async def failing_coroutine(**kwargs):
            raise group

        tool = _make_coroutine_tool("test_tool", failing_coroutine)
        wrapped = _wrap_single_tool(tool)

        with pytest.raises(ExceptionGroup):
            await wrapped.coroutine()


class TestWrappedFuncExceptionHandling:
    """Test ExceptionGroup/NeedsAuthorization handling in wrapped_func path."""

    def test_needs_authorization_triggers_interrupt(self):
        from langgraph.errors import GraphInterrupt

        exc = NeedsAuthorization("mcp-x", "/connect")

        def failing_func(**kwargs):
            raise exc

        tool = _make_func_tool("test_tool", failing_func)
        wrapped = _wrap_single_tool(tool)

        with patch(
            "deep_agent.aegra.mcp_tool_auth.interrupt",
            side_effect=GraphInterrupt(),
        ) as mock_interrupt:
            with pytest.raises(GraphInterrupt):
                wrapped.func()

        mock_interrupt.assert_called_once()
        payload = mock_interrupt.call_args[0][0]
        assert "mcp-x" in payload

    def test_exception_group_with_needs_auth_triggers_interrupt(self):
        from langgraph.errors import GraphInterrupt

        inner = NeedsAuthorization("mcp-z", "/connect")
        group = ExceptionGroup("task group", [inner])

        def failing_func(**kwargs):
            raise group

        tool = _make_func_tool("test_tool", failing_func)
        wrapped = _wrap_single_tool(tool)

        with patch(
            "deep_agent.aegra.mcp_tool_auth.interrupt",
            side_effect=GraphInterrupt(),
        ) as mock_interrupt:
            with pytest.raises(GraphInterrupt):
                wrapped.func()

        mock_interrupt.assert_called_once()
        payload = mock_interrupt.call_args[0][0]
        assert "mcp-z" in payload

    def test_exception_group_without_needs_auth_reraises(self):
        group = ExceptionGroup("task group", [RuntimeError("unrelated")])

        def failing_func(**kwargs):
            raise group

        tool = _make_func_tool("test_tool", failing_func)
        wrapped = _wrap_single_tool(tool)

        with pytest.raises(ExceptionGroup):
            wrapped.func()
