"""Unit tests for MCP tool auth wrapping and error handling."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import ToolMessage

from deep_agent.aegra.mcp_tool_auth import (
    _fix_stringified_json_args,
    _is_http_401,
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


class TestIsHttp401:
    def test_status_401_is_true(self):
        exc = Exception("Unauthorized")
        exc.response = MagicMock(status_code=401)
        assert _is_http_401(exc) is True

    def test_status_403_is_false(self):
        exc = Exception("Forbidden")
        exc.response = MagicMock(status_code=403)
        assert _is_http_401(exc) is False

    def test_standalone_401_in_message_is_true(self):
        assert _is_http_401(RuntimeError("HTTP 401 Unauthorized")) is True

    def test_4012_is_false(self):
        assert _is_http_401(RuntimeError("error 4012")) is False

    def test_1401_is_false(self):
        assert _is_http_401(RuntimeError("error 1401")) is False


class TestSafeAinvoke:
    @pytest.mark.asyncio
    async def test_passthrough_on_success(self):
        tool = _make_mock_tool()
        tool.ainvoke = AsyncMock(return_value="success result")

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
    async def test_http_401_on_oauth_tool_interrupts_and_drops_token(self):
        class Http401(Exception):
            def __init__(self) -> None:
                super().__init__("Unauthorized")
                self.response = MagicMock(status_code=401)

        tool = _make_mock_tool(name="jira_search")
        tool.metadata = {"mcp_server": "jira-mcp"}
        tool.ainvoke = AsyncMock(side_effect=[Http401(), "ok"])
        wrapped = _wrap_single_tool(tool)
        store = AsyncMock()
        with (
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value={"jira-mcp": {"enabled": True, "auth_mode": "dcr"}},
            ),
            patch("deep_agent.aegra.mcp._resolve_mcp_user_id", return_value="user-1"),
            patch("deep_agent.aegra.mcp_auth.get_mcp_credential_resolver") as mock_res,
            patch("deep_agent.aegra.mcp_token_store.McpTokenStore") as mock_store_cls,
            patch(
                "deep_agent.aegra.mcp.invalidate_authenticated_oauth_tools"
            ) as mock_invalidate,
            patch("deep_agent.src.settings.settings") as mock_settings,
            patch("deep_agent.aegra.mcp_tool_auth.interrupt") as mock_int,
        ):
            mock_res.return_value.connect_url.return_value = "/mcp/jira-mcp/connect"
            mock_res.return_value.invalidate_cache = MagicMock()
            mock_store_cls.return_value = store
            mock_settings.database_uri = "postgres://"
            mock_settings.agent_deployment_id = "agent-1"
            result = await wrapped.ainvoke({"id": "call_401", "name": "jira_search"})
        assert result == "ok"
        mock_int.assert_called_once()
        payload = mock_int.call_args[0][0]
        assert "mcp_auth_required" in payload
        assert "jira-mcp" in payload
        store.delete_token.assert_awaited_once()
        mock_res.return_value.invalidate_cache.assert_called_once_with(
            "user-1", "jira-mcp"
        )
        mock_invalidate.assert_called_once_with("user-1", "jira-mcp")

    @pytest.mark.asyncio
    async def test_http_403_on_dcr_tool_stays_tool_error(self):
        class Http403(Exception):
            def __init__(self) -> None:
                super().__init__("Forbidden")
                self.response = MagicMock(status_code=403)

        tool = _make_mock_tool(name="jira_search")
        tool.metadata = {"mcp_server": "jira-mcp"}
        tool.ainvoke = AsyncMock(side_effect=Http403())
        wrapped = _wrap_single_tool(tool)
        with (
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value={"jira-mcp": {"enabled": True, "auth_mode": "dcr"}},
            ),
            patch("deep_agent.aegra.mcp_token_store.McpTokenStore") as mock_store_cls,
            patch("deep_agent.aegra.mcp_tool_auth.interrupt") as mock_int,
        ):
            result = await wrapped.ainvoke({"id": "call_403", "name": "jira_search"})
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "[TOOL_ERROR]" in result.content
        mock_int.assert_not_called()
        mock_store_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_sso_403_with_metadata_stays_tool_error(self):
        tool = _make_mock_tool()
        tool.metadata = {"mcp_server": "gitlab-mcp"}
        tool.ainvoke = AsyncMock(side_effect=RuntimeError("403 Forbidden"))
        wrapped = _wrap_single_tool(tool)
        with patch(
            "deep_agent.aegra.mcp._get_server_configs",
            return_value={"gitlab-mcp": {"enabled": True, "auth_mode": "sso"}},
        ):
            result = await wrapped.ainvoke({"id": "call_sso"})
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "[TOOL_ERROR]" in result.content

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


class TestWrappedCoroutineNeedsAuth:
    """Test that NeedsAuthorization in a wrapped coroutine triggers an interrupt."""

    @pytest.mark.asyncio
    async def test_needs_authorization_triggers_interrupt_async(self):
        from unittest.mock import patch as _patch

        from deep_agent.aegra.mcp_auth import NeedsAuthorization

        exc = NeedsAuthorization("gitlab-mcp", "/mcp/gitlab-mcp/connect")

        async def failing_coroutine(**kwargs):
            raise exc

        tool = _make_mock_tool(coroutine=failing_coroutine)
        tool.func = None
        # Force model_copy to fail so wrapped == tool (coroutine patched in place)
        tool.model_copy = MagicMock(side_effect=TypeError("no model_copy"))

        wrapped = _wrap_single_tool(tool)

        sentinel = RuntimeError("interrupt-called")
        with _patch("deep_agent.aegra.mcp_tool_auth.interrupt", side_effect=sentinel):
            with pytest.raises(RuntimeError, match="interrupt-called"):
                await wrapped.coroutine(query="test")

    @pytest.mark.asyncio
    async def test_needs_authorization_triggers_interrupt_sync(self):
        from unittest.mock import patch as _patch

        from deep_agent.aegra.mcp_auth import NeedsAuthorization

        exc = NeedsAuthorization("gitlab-mcp", "/mcp/gitlab-mcp/connect")

        def failing_func(**kwargs):
            raise exc

        tool = _make_mock_tool()
        tool.coroutine = None
        tool.func = failing_func
        # Force model_copy to fail so wrapped == tool (func patched in place)
        tool.model_copy = MagicMock(side_effect=TypeError("no model_copy"))

        wrapped = _wrap_single_tool(tool)

        sentinel = RuntimeError("interrupt-called")
        with _patch("deep_agent.aegra.mcp_tool_auth.interrupt", side_effect=sentinel):
            with pytest.raises(RuntimeError, match="interrupt-called"):
                wrapped.func(query="test")

    @pytest.mark.asyncio
    async def test_model_copy_fallback_patches_coroutine_directly(self):
        """When model_copy fails, the tool is patched directly."""

        async def ok_coroutine(**kwargs):
            return "result"

        tool = _make_mock_tool(coroutine=ok_coroutine)
        tool.func = None
        # Make model_copy raise so the fallback path is taken
        tool.model_copy = MagicMock(side_effect=TypeError("no model_copy"))

        wrapped = _wrap_single_tool(tool)
        assert wrapped is tool  # same object, patched in place
        assert wrapped.coroutine is not ok_coroutine  # replaced

    def test_model_copy_fallback_patches_func_directly(self):
        """When model_copy fails for a sync tool, func is patched directly."""

        def ok_func(**kwargs):
            return "result"

        tool = _make_mock_tool()
        tool.coroutine = None
        tool.func = ok_func
        tool.model_copy = MagicMock(side_effect=TypeError("no model_copy"))

        wrapped = _wrap_single_tool(tool)
        assert wrapped is tool  # same object, patched in place
        assert wrapped.func is not ok_func  # replaced


class TestFixStringifiedJsonArgs:
    """Test _fix_stringified_json_args edge cases."""

    def test_returns_kwargs_when_args_property_raises(self):
        """When getattr(tool, 'args') raises, kwargs are returned unchanged."""
        tool = MagicMock()
        type(tool).args = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        kwargs = {"query": "test"}
        result = _fix_stringified_json_args(tool, kwargs)
        assert result == kwargs


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
