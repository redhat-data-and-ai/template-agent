"""Unit tests for runtime OAuth/DCR MCP tool attach middleware."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from deep_agent.aegra.mcp_runtime_tools import (
    McpRuntimeToolsMiddleware,
    _apply_live_hitl_decisions,
    _decision_at,
    _interrupt_hitl,
    _is_auth_continue,
    install_compiled_hitl_auth_resume,
    runtime_mcp_attach_filters,
)


def _mcp_tool(name: str, server: str | None = None):
    tool = MagicMock()
    tool.name = name
    tool.metadata = {"mcp_server": server} if server else {}
    return tool


def _model_request(tools: list | None = None):
    req = MagicMock()
    req.tools = tools or []

    def _override(**kwargs):
        new_req = MagicMock()
        new_req.tools = kwargs.get("tools", req.tools)
        new_req.override = _override
        return new_req

    req.override = _override
    return req


def _tool_request(*, name: str, tool=None, call_id: str = "call_1", args=None):
    req = MagicMock()
    req.tool = tool
    req.tool_call = {"name": name, "id": call_id, "args": args or {}}

    def _override(**kwargs):
        new_req = MagicMock()
        new_req.tool = kwargs.get("tool", req.tool)
        new_req.tool_call = kwargs.get("tool_call", req.tool_call)
        new_req.override = _override
        return new_req

    req.override = _override
    return req


def _ai_state(*calls: dict):
    return {"messages": [AIMessage(content="", tool_calls=list(calls))]}


_SERVERS = {
    "jira-mcp": {"enabled": True, "auth_mode": "dcr", "tool_prefix": "jira"},
    "template-mcp-server": {
        "enabled": True,
        "auth_mode": "sso",
        "tool_prefix": "template",
    },
}


class TestAwrapModelCall:
    @pytest.mark.asyncio
    async def test_adds_live_tools_when_placeholder_and_token(self):
        placeholder = _mcp_tool("mcp__jira_mcp")
        live = _mcp_tool("jira_search", "jira-mcp")
        req = _model_request([placeholder])
        handler = AsyncMock(return_value="ok")
        mw = McpRuntimeToolsMiddleware()
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch("deep_agent.aegra.mcp._current_user_id") as mock_ctx,
            patch(
                "deep_agent.aegra.mcp.get_authenticated_oauth_mcp_tools",
                new=AsyncMock(return_value=[live]),
            ),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value={
                    "jira-mcp": {"enabled": True, "auth_mode": "dcr"},
                },
            ),
        ):
            mock_ctx.set = MagicMock()
            result = await mw.awrap_model_call(req, handler)
        assert result == "ok"
        overridden = handler.call_args[0][0]
        names = [t.name for t in overridden.tools]
        assert "jira_search" in names
        assert "mcp__jira_mcp" not in names

    @pytest.mark.asyncio
    async def test_skips_when_no_user(self):
        req = _model_request()
        handler = AsyncMock(return_value="ok")
        mw = McpRuntimeToolsMiddleware()
        with patch("deep_agent.aegra.mcp._resolve_mcp_user_id", return_value=None):
            result = await mw.awrap_model_call(req, handler)
        assert result == "ok"
        handler.assert_awaited_once_with(req)

    @pytest.mark.asyncio
    async def test_does_not_duplicate_already_present_tools(self):
        placeholder = _mcp_tool("mcp__jira_mcp")
        live = _mcp_tool("jira_search", "jira-mcp")
        req = _model_request([placeholder, live])
        handler = AsyncMock(return_value="ok")
        mw = McpRuntimeToolsMiddleware()
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch("deep_agent.aegra.mcp._current_user_id"),
            patch(
                "deep_agent.aegra.mcp.get_authenticated_oauth_mcp_tools",
                new=AsyncMock(return_value=[live]),
            ),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value={
                    "jira-mcp": {"enabled": True, "auth_mode": "dcr"},
                },
            ),
        ):
            await mw.awrap_model_call(req, handler)
        names = [t.name for t in handler.call_args[0][0].tools]
        assert names.count("jira_search") == 1
        assert "mcp__jira_mcp" not in names

    @pytest.mark.asyncio
    async def test_allowlist_drops_live_tools_not_declared(self):
        placeholder = _mcp_tool("mcp__jira_mcp")
        search = _mcp_tool("jira_search", "jira-mcp")
        create = _mcp_tool("jira_create", "jira-mcp")
        req = _model_request([placeholder])
        handler = AsyncMock(return_value="ok")
        mw = McpRuntimeToolsMiddleware(allowed_tool_names=frozenset({"jira_search"}))
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch("deep_agent.aegra.mcp._current_user_id"),
            patch(
                "deep_agent.aegra.mcp.get_authenticated_oauth_mcp_tools",
                new=AsyncMock(return_value=[search, create]),
            ),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value={
                    "jira-mcp": {"enabled": True, "auth_mode": "dcr"},
                },
            ),
        ):
            await mw.awrap_model_call(req, handler)
        names = [t.name for t in handler.call_args[0][0].tools]
        assert "jira_search" in names
        assert "jira_create" not in names
        assert "mcp__jira_mcp" not in names

    @pytest.mark.asyncio
    async def test_drops_live_tool_from_non_first_wins_owner(self):
        placeholder = _mcp_tool("mcp__acme_jira")
        vault_search = _mcp_tool("search", "acme-vault")
        req = _model_request([placeholder])
        handler = AsyncMock(return_value="ok")
        mw = McpRuntimeToolsMiddleware()
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch("deep_agent.aegra.mcp._current_user_id"),
            patch(
                "deep_agent.aegra.mcp.get_authenticated_oauth_mcp_tools",
                new=AsyncMock(return_value=[vault_search]),
            ),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value={
                    "acme-jira": {"enabled": True, "auth_mode": "dcr"},
                    "acme-vault": {"enabled": True, "auth_mode": "dcr"},
                },
            ),
            patch(
                "deep_agent.aegra.mcp.oauth_dcr_server_for_tool_name",
                return_value="acme-jira",
            ),
        ):
            await mw.awrap_model_call(req, handler)
        names = [t.name for t in handler.call_args[0][0].tools]
        assert "search" not in names
        assert "mcp__acme_jira" in names

    @pytest.mark.asyncio
    async def test_mcp_names_fence_skips_unlisted_server_placeholder(self):
        placeholder = MagicMock()
        placeholder.name = "mcp__jira_mcp"
        live = MagicMock()
        live.name = "jira_search"
        req = _model_request([placeholder])
        handler = AsyncMock(return_value="ok")
        mw = McpRuntimeToolsMiddleware(mcp_names=frozenset({"other-mcp"}))
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch(
                "deep_agent.aegra.mcp.get_authenticated_oauth_mcp_tools",
                new=AsyncMock(return_value=[live]),
            ) as mock_live,
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value={
                    "jira-mcp": {"enabled": True, "auth_mode": "dcr"},
                },
            ),
        ):
            await mw.awrap_model_call(req, handler)
        mock_live.assert_not_called()
        handler.assert_awaited_once_with(req)

    @pytest.mark.asyncio
    async def test_keeps_placeholder_when_live_listing_is_empty(self):
        placeholder = _mcp_tool("mcp__jira_mcp")
        req = _model_request([placeholder])
        handler = AsyncMock(return_value="ok")
        mw = McpRuntimeToolsMiddleware()
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch("deep_agent.aegra.mcp._current_user_id"),
            patch(
                "deep_agent.aegra.mcp.get_authenticated_oauth_mcp_tools",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value={
                    "jira-mcp": {"enabled": True, "auth_mode": "dcr"},
                },
            ),
        ):
            await mw.awrap_model_call(req, handler)
        handler.assert_awaited_once_with(req)

    @pytest.mark.asyncio
    async def test_keeps_tagged_placeholder_when_not_connected(self):
        placeholder = _mcp_tool("mcp__jira_mcp", "jira-mcp")
        req = _model_request([placeholder])
        handler = AsyncMock(return_value="ok")
        mw = McpRuntimeToolsMiddleware()
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch("deep_agent.aegra.mcp._current_user_id"),
            patch(
                "deep_agent.aegra.mcp.get_authenticated_oauth_mcp_tools",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value={
                    "jira-mcp": {"enabled": True, "auth_mode": "dcr"},
                },
            ),
        ):
            await mw.awrap_model_call(req, handler)
        tools = handler.call_args[0][0].tools
        assert [t.name for t in tools] == ["mcp__jira_mcp"]


class TestAwrapToolCall:
    def setup_method(self):
        from deep_agent.aegra import mcp as mcp_mod

        mcp_mod._oauth_live_name_index.clear()

    def teardown_method(self):
        from deep_agent.aegra import mcp as mcp_mod

        mcp_mod._oauth_live_name_index.clear()

    @pytest.mark.asyncio
    async def test_compiled_tool_passes_through(self):
        compiled = MagicMock()
        req = _tool_request(name="jira_search", tool=compiled)
        handler = AsyncMock(return_value="ran")
        mw = McpRuntimeToolsMiddleware()
        result = await mw.awrap_tool_call(req, handler)
        assert result == "ran"
        handler.assert_awaited_once_with(req)

    @pytest.mark.asyncio
    async def test_overrides_unregistered_tool_without_hitl(self):
        from deep_agent.aegra.mcp import record_oauth_live_names

        live = MagicMock()
        live.name = "jira_search"
        req = _tool_request(name="jira_search", tool=None)
        handler = AsyncMock(return_value="ran")
        mw = McpRuntimeToolsMiddleware()
        listing = AsyncMock(return_value=[live])
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch("deep_agent.aegra.mcp._current_user_id"),
            patch(
                "deep_agent.aegra.mcp.get_authenticated_oauth_mcp_tools",
                new=listing,
            ),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value={
                    "jira-mcp": {"enabled": True, "auth_mode": "dcr"},
                },
            ),
            patch("deep_agent.aegra.redis.cache_set_persistent", return_value=True),
            patch(
                "deep_agent.aegra.mcp_runtime_tools.interrupt",
            ) as mock_interrupt,
        ):
            record_oauth_live_names("jira-mcp", ["jira_search"])
            result = await mw.awrap_tool_call(req, handler)
        assert result == "ran"
        mock_interrupt.assert_not_called()
        listing.assert_awaited_once()
        assert listing.await_args.kwargs["server_names"] == ["jira-mcp"]
        overridden = handler.call_args[0][0]
        assert overridden.tool is live

    @pytest.mark.asyncio
    async def test_missing_live_tool_returns_error(self):
        from deep_agent.aegra.mcp import record_oauth_live_names

        req = _tool_request(name="jira_search", tool=None)
        handler = AsyncMock(return_value="ran")
        mw = McpRuntimeToolsMiddleware()
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch("deep_agent.aegra.mcp._current_user_id"),
            patch(
                "deep_agent.aegra.mcp.get_authenticated_oauth_mcp_tools",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value={
                    "jira-mcp": {"enabled": True, "auth_mode": "dcr"},
                },
            ),
            patch("deep_agent.aegra.redis.cache_set_persistent", return_value=True),
        ):
            record_oauth_live_names("jira-mcp", ["jira_search"])
            result = await mw.awrap_tool_call(req, handler)
        handler.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "not connected" in result.content

    @pytest.mark.asyncio
    async def test_listing_failure_returns_error(self):
        from deep_agent.aegra.mcp import record_oauth_live_names

        req = _tool_request(name="jira_search", tool=None)
        handler = AsyncMock(return_value="ran")
        mw = McpRuntimeToolsMiddleware()
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch("deep_agent.aegra.mcp._current_user_id"),
            patch(
                "deep_agent.aegra.mcp.get_authenticated_oauth_mcp_tools",
                new=AsyncMock(side_effect=RuntimeError("boom")),
            ),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value={
                    "jira-mcp": {"enabled": True, "auth_mode": "dcr"},
                },
            ),
            patch("deep_agent.aegra.redis.cache_set_persistent", return_value=True),
        ):
            record_oauth_live_names("jira-mcp", ["jira_search"])
            result = await mw.awrap_tool_call(req, handler)
        handler.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "could not be loaded" in result.content

    @pytest.mark.asyncio
    async def test_missing_user_id_returns_error(self):
        from deep_agent.aegra.mcp import record_oauth_live_names

        req = _tool_request(name="jira_search", tool=None)
        handler = AsyncMock(return_value="ran")
        mw = McpRuntimeToolsMiddleware()
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value=None,
            ),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value={
                    "jira-mcp": {"enabled": True, "auth_mode": "dcr"},
                },
            ),
            patch("deep_agent.aegra.redis.cache_set_persistent", return_value=True),
        ):
            record_oauth_live_names("jira-mcp", ["jira_search"])
            result = await mw.awrap_tool_call(req, handler)
        handler.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "not connected" in result.content

    @pytest.mark.asyncio
    async def test_wraps_live_tool_with_guardian(self):
        from deep_agent.aegra.mcp import record_oauth_live_names

        live = MagicMock()
        live.name = "jira_search"
        wrapped = MagicMock()
        wrapped.name = "jira_search"
        req = _tool_request(name="jira_search", tool=None)
        handler = AsyncMock(return_value="ran")
        mw = McpRuntimeToolsMiddleware()
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch("deep_agent.aegra.mcp._current_user_id"),
            patch(
                "deep_agent.aegra.mcp.get_authenticated_oauth_mcp_tools",
                new=AsyncMock(return_value=[live]),
            ),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value={
                    "jira-mcp": {"enabled": True, "auth_mode": "dcr"},
                },
            ),
            patch("deep_agent.aegra.redis.cache_set_persistent", return_value=True),
            patch(
                "deep_agent.src.guardrails.tool_proxy.wrap_tools",
                return_value=[wrapped],
            ) as mock_wrap,
        ):
            record_oauth_live_names("jira-mcp", ["jira_search"])
            result = await mw.awrap_tool_call(req, handler)
        assert result == "ran"
        mock_wrap.assert_called_once_with([live])
        assert handler.call_args[0][0].tool is wrapped

    @pytest.mark.asyncio
    async def test_refuses_live_tool_outside_mcp_fence(self):
        from deep_agent.aegra.mcp import record_oauth_live_names

        live = MagicMock()
        live.name = "read_secret"
        req = _tool_request(name="read_secret", tool=None)
        handler = AsyncMock(return_value="ran")
        mw = McpRuntimeToolsMiddleware(mcp_names=frozenset({"acme-jira"}))
        listing = AsyncMock(return_value=[live])
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch(
                "deep_agent.aegra.mcp.get_authenticated_oauth_mcp_tools",
                new=listing,
            ),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value={
                    "acme-jira": {"enabled": True, "auth_mode": "dcr"},
                    "acme-vault": {"enabled": True, "auth_mode": "dcr"},
                },
            ),
            patch("deep_agent.aegra.redis.cache_set_persistent", return_value=True),
        ):
            record_oauth_live_names("acme-jira", ["search"])
            record_oauth_live_names("acme-vault", ["read_secret"])
            result = await mw.awrap_tool_call(req, handler)
        handler.assert_not_called()
        listing.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "read_secret" in result.content

    @pytest.mark.asyncio
    async def test_duplicate_live_name_runs_first_fenced_server(self):
        from deep_agent.aegra.mcp import record_oauth_live_names

        jira_search = MagicMock()
        jira_search.name = "search"
        vault_search = MagicMock()
        vault_search.name = "search"
        req = _tool_request(name="search", tool=None)
        handler = AsyncMock(return_value="ran")
        mw = McpRuntimeToolsMiddleware(
            mcp_names=frozenset({"acme-jira", "acme-vault"}),
        )

        async def listing(_user_id, server_names=None):
            assert server_names == ["acme-jira"]
            return [jira_search]

        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch("deep_agent.aegra.mcp._current_user_id"),
            patch(
                "deep_agent.aegra.mcp.get_authenticated_oauth_mcp_tools",
                new=listing,
            ),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value={
                    "acme-jira": {"enabled": True, "auth_mode": "dcr"},
                    "acme-vault": {"enabled": True, "auth_mode": "dcr"},
                },
            ),
            patch("deep_agent.aegra.redis.cache_set_persistent", return_value=True),
        ):
            record_oauth_live_names("acme-jira", ["search"])
            record_oauth_live_names("acme-vault", ["search"])
            result = await mw.awrap_tool_call(req, handler)
        assert result == "ran"
        overridden = handler.call_args[0][0]
        assert overridden.tool is jira_search
        assert overridden.tool is not vault_search

    @pytest.mark.asyncio
    async def test_refuses_live_tool_outside_allowlist(self):
        live = MagicMock()
        live.name = "jira_create"
        req = _tool_request(name="jira_create", tool=None)
        handler = AsyncMock(return_value="ran")
        mw = McpRuntimeToolsMiddleware(allowed_tool_names=frozenset({"jira_search"}))
        with (
            patch(
                "deep_agent.aegra.mcp.get_authenticated_oauth_mcp_tools",
                new=AsyncMock(return_value=[live]),
            ) as mock_live,
        ):
            result = await mw.awrap_tool_call(req, handler)
        handler.assert_not_called()
        mock_live.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "jira_create" in result.content


def _hitl_all():
    hitl = MagicMock()
    hitl.enabled = True
    hitl.mode = "all"
    hitl.exclude = []
    resolved = MagicMock()
    resolved.human_approval = hitl
    return resolved


class TestAafterModel:
    def setup_method(self):
        from deep_agent.aegra import mcp as mcp_mod

        mcp_mod._oauth_live_name_index.clear()

    def teardown_method(self):
        from deep_agent.aegra import mcp as mcp_mod

        mcp_mod._oauth_live_name_index.clear()

    @pytest.mark.asyncio
    async def test_auth_interrupt_when_token_missing(self):
        state = _ai_state(
            {
                "name": "mcp__jira_mcp",
                "id": "c1",
                "args": {"query": ""},
            }
        )
        mw = McpRuntimeToolsMiddleware()
        resolve = AsyncMock(side_effect=[None, "tok"])
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch("deep_agent.aegra.mcp._current_user_id"),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value=_SERVERS,
            ),
            patch(
                "deep_agent.aegra.mcp._resolve_connection_token",
                new=resolve,
            ),
            patch(
                "deep_agent.aegra.mcp_auth.get_mcp_credential_resolver",
            ) as mock_resolver,
            patch(
                "deep_agent.aegra.mcp_runtime_tools.interrupt",
                return_value="continue",
            ) as mock_interrupt,
        ):
            mock_resolver.return_value.connect_url.return_value = (
                "/mcp/jira-mcp/connect"
            )
            result = await mw.aafter_model(state, None)
        assert result is None
        mock_interrupt.assert_called_once()
        payload = mock_interrupt.call_args[0][0]
        assert "mcp_auth_required" in payload
        assert "jira-mcp" in payload

    @pytest.mark.asyncio
    async def test_resource_calls_authenticate_one_dcr_server_at_a_time(self):
        state = _ai_state(
            {
                "name": "mcp_list_resources",
                "id": "r1",
                "args": {"mcp_name": "template-mcp-server-dcr-open"},
            },
            {
                "name": "mcp_list_resource_templates",
                "id": "t1",
                "args": {"mcp_name": "template-mcp-server-dcr-open"},
            },
            {
                "name": "mcp_list_resources",
                "id": "r2",
                "args": {"mcp_name": "template-mcp-server-dcr-gated"},
            },
            {
                "name": "mcp_list_resource_templates",
                "id": "t2",
                "args": {"mcp_name": "template-mcp-server-dcr-gated"},
            },
        )
        servers = {
            "template-mcp-server-dcr-open": {
                "enabled": True,
                "auth_mode": "dcr",
            },
            "template-mcp-server-dcr-gated": {
                "enabled": True,
                "auth_mode": "dcr",
            },
        }
        tokens: dict[str, str | None] = {
            "template-mcp-server-dcr-open": None,
            "template-mcp-server-dcr-gated": None,
        }

        async def resolve(name, _entry, _sso, _user_id):
            return tokens.get(name)

        def on_interrupt(payload):
            if "template-mcp-server-dcr-open" in payload:
                tokens["template-mcp-server-dcr-open"] = "tok-open"
            if "template-mcp-server-dcr-gated" in payload:
                tokens["template-mcp-server-dcr-gated"] = "tok-gated"
            return "continue"

        mw = McpRuntimeToolsMiddleware()
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch("deep_agent.aegra.mcp._current_user_id"),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value=servers,
            ),
            patch(
                "deep_agent.aegra.mcp._resolve_connection_token",
                new=resolve,
            ),
            patch(
                "deep_agent.aegra.mcp_auth.get_mcp_credential_resolver",
            ) as mock_resolver,
            patch(
                "deep_agent.aegra.mcp_runtime_tools.interrupt",
                side_effect=on_interrupt,
            ) as mock_interrupt,
        ):
            mock_resolver.return_value.connect_url.side_effect = lambda mcp_name: (
                f"/mcp/{mcp_name}/connect"
            )
            result = await mw.aafter_model(state, None)
        assert result is None
        assert mock_interrupt.call_count == 2
        first, second = (c[0][0] for c in mock_interrupt.call_args_list)
        assert "template-mcp-server-dcr-open" in first
        assert "template-mcp-server-dcr-gated" in second

    @pytest.mark.asyncio
    async def test_resource_call_skips_sso_and_fenced_out_dcr(self):
        state = _ai_state(
            {
                "name": "mcp_list_resources",
                "id": "sso",
                "args": {"mcp_name": "template-mcp-server"},
            },
            {
                "name": "mcp_list_resources",
                "id": "gated",
                "args": {"mcp_name": "template-mcp-server-dcr-gated"},
            },
        )
        servers = {
            "template-mcp-server": {"enabled": True, "auth_mode": "sso"},
            "template-mcp-server-dcr-gated": {
                "enabled": True,
                "auth_mode": "dcr",
            },
        }
        mw = McpRuntimeToolsMiddleware(
            mcp_names=frozenset({"template-mcp-server-dcr-open"}),
        )
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch("deep_agent.aegra.mcp._current_user_id"),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value=servers,
            ),
            patch(
                "deep_agent.aegra.mcp._resolve_connection_token",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "deep_agent.aegra.mcp_runtime_tools.interrupt",
            ) as mock_interrupt,
        ):
            result = await mw.aafter_model(state, None)
        assert result is None
        mock_interrupt.assert_not_called()

    @pytest.mark.asyncio
    async def test_hitl_for_live_tool_when_token_present(self):
        state = _ai_state(
            {
                "name": "template_validate_email",
                "id": "e1",
                "args": {"email": "pat@redhat.com"},
            },
            {
                "name": "jira_search",
                "id": "j1",
                "args": {"q": "bugs"},
            },
        )
        hitl = MagicMock()
        hitl.enabled = True
        hitl.mode = "all"
        hitl.exclude = []
        resolved = MagicMock()
        resolved.human_approval = hitl
        mw = McpRuntimeToolsMiddleware()
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch("deep_agent.aegra.mcp._current_user_id"),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value=_SERVERS,
            ),
            patch(
                "deep_agent.aegra.mcp._resolve_connection_token",
                new=AsyncMock(return_value="tok"),
            ),
            patch(
                "deep_agent.src.agent.config.agent_config.get_orchestrator_config",
                return_value={"model": "gemini-x"},
            ),
            patch(
                "deep_agent.src.agent.config.agent_config.resolve_agent_middleware",
                return_value=resolved,
            ),
            patch(
                "deep_agent.aegra.mcp_runtime_tools.interrupt",
                return_value={"decisions": [{"type": "approve"}]},
            ) as mock_interrupt,
        ):
            result = await mw.aafter_model(state, None)
        assert result is not None
        payload = mock_interrupt.call_args[0][0]
        assert payload["action_requests"][0]["name"] == "jira_search"
        names = [c["name"] for c in result["messages"][0].tool_calls]
        assert "template_validate_email" in names
        assert "jira_search" in names

    @pytest.mark.asyncio
    async def test_reject_drops_live_call(self):
        state = _ai_state(
            {
                "name": "jira_search",
                "id": "j1",
                "args": {"q": "bugs"},
            }
        )
        hitl = MagicMock()
        hitl.enabled = True
        hitl.mode = "all"
        hitl.exclude = []
        resolved = MagicMock()
        resolved.human_approval = hitl
        mw = McpRuntimeToolsMiddleware()
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch("deep_agent.aegra.mcp._current_user_id"),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value=_SERVERS,
            ),
            patch(
                "deep_agent.aegra.mcp._resolve_connection_token",
                new=AsyncMock(return_value="tok"),
            ),
            patch(
                "deep_agent.src.agent.config.agent_config.get_orchestrator_config",
                return_value={"model": "gemini-x"},
            ),
            patch(
                "deep_agent.src.agent.config.agent_config.resolve_agent_middleware",
                return_value=resolved,
            ),
            patch(
                "deep_agent.aegra.mcp_runtime_tools.interrupt",
                return_value={"decisions": [{"type": "reject", "message": "nope"}]},
            ),
        ):
            result = await mw.aafter_model(state, None)
        assert result is not None
        assert result["messages"][0].tool_calls == []
        rejected = result["messages"][1]
        assert isinstance(rejected, ToolMessage)
        assert rejected.status == "error"
        assert "nope" in rejected.content

    @pytest.mark.asyncio
    async def test_sso_only_does_not_interrupt(self):
        state = _ai_state(
            {
                "name": "template_validate_email",
                "id": "e1",
                "args": {"email": "a@b.com"},
            }
        )
        mw = McpRuntimeToolsMiddleware()
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch("deep_agent.aegra.mcp._current_user_id"),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value=_SERVERS,
            ),
            patch(
                "deep_agent.aegra.mcp_runtime_tools.interrupt",
            ) as mock_interrupt,
        ):
            result = await mw.aafter_model(state, None)
        assert result is None
        mock_interrupt.assert_not_called()

    @pytest.mark.asyncio
    async def test_hitl_for_unprefixed_live_name_via_catalog(self):
        from deep_agent.aegra.mcp import record_oauth_live_names

        state = _ai_state({"name": "search", "id": "s1", "args": {"q": "bugs"}})
        servers = {"acme-jira": {"enabled": True, "auth_mode": "dcr"}}
        mw = McpRuntimeToolsMiddleware()
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch("deep_agent.aegra.mcp._current_user_id"),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value=servers,
            ),
            patch("deep_agent.aegra.redis.cache_set_persistent", return_value=True),
            patch(
                "deep_agent.aegra.mcp._resolve_connection_token",
                new=AsyncMock(return_value="tok"),
            ),
            patch(
                "deep_agent.src.agent.config.agent_config.get_orchestrator_config",
                return_value={"model": "gemini-x"},
            ),
            patch(
                "deep_agent.src.agent.config.agent_config.resolve_agent_middleware",
                return_value=_hitl_all(),
            ),
            patch(
                "deep_agent.aegra.mcp_runtime_tools.interrupt",
                return_value={"decisions": [{"type": "approve"}]},
            ) as mock_interrupt,
        ):
            record_oauth_live_names("acme-jira", ["search"])
            result = await mw.aafter_model(state, None)
        assert result is not None
        payload = mock_interrupt.call_args[0][0]
        assert payload["action_requests"][0]["name"] == "search"
        assert "mcp_auth_required" not in payload

    @pytest.mark.asyncio
    async def test_auth_interrupt_for_unprefixed_live_name_via_catalog(self):
        from deep_agent.aegra.mcp import record_oauth_live_names

        state = _ai_state({"name": "search", "id": "s1", "args": {}})
        servers = {"acme-jira": {"enabled": True, "auth_mode": "dcr"}}
        mw = McpRuntimeToolsMiddleware()
        resolve = AsyncMock(side_effect=[None, "tok"])
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch("deep_agent.aegra.mcp._current_user_id"),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value=servers,
            ),
            patch("deep_agent.aegra.redis.cache_set_persistent", return_value=True),
            patch(
                "deep_agent.aegra.mcp._resolve_connection_token",
                new=resolve,
            ),
            patch(
                "deep_agent.aegra.mcp_auth.get_mcp_credential_resolver",
            ) as mock_resolver,
            patch(
                "deep_agent.src.agent.config.agent_config.get_orchestrator_config",
                return_value={"model": "gemini-x"},
            ),
            patch(
                "deep_agent.src.agent.config.agent_config.resolve_agent_middleware",
                return_value=_hitl_all(),
            ),
            patch(
                "deep_agent.aegra.mcp_runtime_tools.interrupt",
                side_effect=["continue", {"decisions": [{"type": "approve"}]}],
            ) as mock_interrupt,
        ):
            mock_resolver.return_value.connect_url.return_value = (
                "/mcp/acme-jira/connect"
            )
            record_oauth_live_names("acme-jira", ["search"])
            result = await mw.aafter_model(state, None)
        assert result is not None
        auth_payload = mock_interrupt.call_args_list[0][0][0]
        hitl_payload = mock_interrupt.call_args_list[1][0][0]
        assert "mcp_auth_required" in auth_payload
        assert "acme-jira" in auth_payload
        assert hitl_payload["action_requests"][0]["name"] == "search"

    @pytest.mark.asyncio
    async def test_hitl_skips_live_name_outside_allowlist(self):
        from deep_agent.aegra.mcp import record_oauth_live_names

        state = _ai_state({"name": "create_issue", "id": "c1", "args": {}})
        servers = {"acme-jira": {"enabled": True, "auth_mode": "dcr"}}
        mw = McpRuntimeToolsMiddleware(allowed_tool_names=frozenset({"search"}))
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch("deep_agent.aegra.mcp._current_user_id"),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value=servers,
            ),
            patch("deep_agent.aegra.redis.cache_set_persistent", return_value=True),
            patch(
                "deep_agent.aegra.mcp._resolve_connection_token",
                new=AsyncMock(return_value="tok"),
            ),
            patch(
                "deep_agent.src.agent.config.agent_config.get_orchestrator_config",
                return_value={"model": "gemini-x"},
            ),
            patch(
                "deep_agent.src.agent.config.agent_config.resolve_agent_middleware",
                return_value=_hitl_all(),
            ),
            patch(
                "deep_agent.aegra.mcp_runtime_tools.interrupt",
            ) as mock_interrupt,
        ):
            record_oauth_live_names("acme-jira", ["search", "create_issue"])
            result = await mw.aafter_model(state, None)
        assert result is None
        mock_interrupt.assert_not_called()

    @pytest.mark.asyncio
    async def test_hitl_skips_live_name_outside_mcp_fence(self):
        from deep_agent.aegra.mcp import record_oauth_live_names

        state = _ai_state({"name": "search", "id": "s1", "args": {}})
        servers = {
            "acme-jira": {"enabled": True, "auth_mode": "dcr"},
            "acme-vault": {"enabled": True, "auth_mode": "dcr"},
        }
        mw = McpRuntimeToolsMiddleware(mcp_names=frozenset({"acme-vault"}))
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch("deep_agent.aegra.mcp._current_user_id"),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value=servers,
            ),
            patch("deep_agent.aegra.redis.cache_set_persistent", return_value=True),
            patch(
                "deep_agent.aegra.mcp._resolve_connection_token",
                new=AsyncMock(return_value="tok"),
            ),
            patch(
                "deep_agent.src.agent.config.agent_config.get_orchestrator_config",
                return_value={"model": "gemini-x"},
            ),
            patch(
                "deep_agent.src.agent.config.agent_config.resolve_agent_middleware",
                return_value=_hitl_all(),
            ),
            patch(
                "deep_agent.aegra.mcp_runtime_tools.interrupt",
            ) as mock_interrupt,
        ):
            record_oauth_live_names("acme-jira", ["search"])
            result = await mw.aafter_model(state, None)
        assert result is None
        mock_interrupt.assert_not_called()

    @pytest.mark.asyncio
    async def test_hitl_fenced_server_pauses_unprefixed_live_name(self):
        from deep_agent.aegra.mcp import record_oauth_live_names

        state = _ai_state({"name": "search", "id": "s1", "args": {}})
        servers = {
            "acme-jira": {"enabled": True, "auth_mode": "dcr"},
            "acme-vault": {"enabled": True, "auth_mode": "dcr"},
        }
        mw = McpRuntimeToolsMiddleware(mcp_names=frozenset({"acme-jira"}))
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch("deep_agent.aegra.mcp._current_user_id"),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value=servers,
            ),
            patch("deep_agent.aegra.redis.cache_set_persistent", return_value=True),
            patch(
                "deep_agent.aegra.mcp._resolve_connection_token",
                new=AsyncMock(return_value="tok"),
            ),
            patch(
                "deep_agent.src.agent.config.agent_config.get_orchestrator_config",
                return_value={"model": "gemini-x"},
            ),
            patch(
                "deep_agent.src.agent.config.agent_config.resolve_agent_middleware",
                return_value=_hitl_all(),
            ),
            patch(
                "deep_agent.aegra.mcp_runtime_tools.interrupt",
                return_value={"decisions": [{"type": "approve"}]},
            ) as mock_interrupt,
        ):
            record_oauth_live_names("acme-jira", ["search"])
            result = await mw.aafter_model(state, None)
        assert result is not None
        payload = mock_interrupt.call_args[0][0]
        assert payload["action_requests"][0]["name"] == "search"


class TestNoPrefixRuntimeContract:
    _SERVERS = {
        "acme-jira": {"enabled": True, "auth_mode": "dcr"},
        "acme-vault": {"enabled": True, "auth_mode": "dcr"},
    }

    def setup_method(self):
        from deep_agent.aegra import mcp as mcp_mod

        mcp_mod._oauth_live_name_index.clear()

    def teardown_method(self):
        from deep_agent.aegra import mcp as mcp_mod

        mcp_mod._oauth_live_name_index.clear()

    @pytest.mark.asyncio
    async def test_two_placeholders_attach_only_listed_live_tools(self):
        ph_jira = _mcp_tool("mcp__acme_jira")
        ph_vault = _mcp_tool("mcp__acme_vault")
        search = _mcp_tool("search", "acme-jira")
        req = _model_request([ph_jira, ph_vault])
        handler = AsyncMock(return_value="ok")
        mw = McpRuntimeToolsMiddleware(
            mcp_names=frozenset({"acme-jira", "acme-vault"}),
        )

        async def listing(_user_id, server_names=None):
            assert set(server_names or []) == {"acme-jira", "acme-vault"}
            return [search]

        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch("deep_agent.aegra.mcp._current_user_id"),
            patch(
                "deep_agent.aegra.mcp.get_authenticated_oauth_mcp_tools",
                new=listing,
            ),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value=self._SERVERS,
            ),
        ):
            await mw.awrap_model_call(req, handler)
        names = [t.name for t in handler.call_args[0][0].tools]
        assert "mcp__acme_jira" not in names
        assert names.count("mcp__acme_vault") == 1
        assert "search" in names
        assert "read_secret" not in names

    @pytest.mark.asyncio
    async def test_allowlist_keeps_search_drops_create_without_prefix(self):
        from deep_agent.aegra.mcp_runtime_tools import (
            build_mcp_runtime_tools_middleware_from_declared,
        )

        ph = _mcp_tool("mcp__acme_jira")
        search = _mcp_tool("search", "acme-jira")
        create = _mcp_tool("create_issue", "acme-jira")
        req = _model_request([ph])
        handler = AsyncMock(return_value="ok")
        mw = build_mcp_runtime_tools_middleware_from_declared(
            ["search"],
            ["acme-jira"],
        )
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch("deep_agent.aegra.mcp._current_user_id"),
            patch(
                "deep_agent.aegra.mcp.get_authenticated_oauth_mcp_tools",
                new=AsyncMock(return_value=[search, create]),
            ),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value=self._SERVERS,
            ),
        ):
            await mw.awrap_model_call(req, handler)
        names = [t.name for t in handler.call_args[0][0].tools]
        assert "search" in names
        assert "create_issue" not in names
        assert "mcp__acme_jira" not in names

    @pytest.mark.asyncio
    async def test_reject_unprefixed_live_tool(self):
        from deep_agent.aegra.mcp import record_oauth_live_names

        state = _ai_state({"name": "search", "id": "s1", "args": {}})
        mw = McpRuntimeToolsMiddleware(mcp_names=frozenset({"acme-jira"}))
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch("deep_agent.aegra.mcp._current_user_id"),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value=self._SERVERS,
            ),
            patch("deep_agent.aegra.redis.cache_set_persistent", return_value=True),
            patch(
                "deep_agent.aegra.mcp._resolve_connection_token",
                new=AsyncMock(return_value="tok"),
            ),
            patch(
                "deep_agent.src.agent.config.agent_config.get_orchestrator_config",
                return_value={"model": "gemini-x"},
            ),
            patch(
                "deep_agent.src.agent.config.agent_config.resolve_agent_middleware",
                return_value=_hitl_all(),
            ),
            patch(
                "deep_agent.aegra.mcp_runtime_tools.interrupt",
                return_value={"decisions": [{"type": "reject", "message": "nope"}]},
            ),
        ):
            record_oauth_live_names("acme-jira", ["search"])
            result = await mw.aafter_model(state, None)
        assert result is not None
        assert result["messages"][0].tool_calls == []
        rejected = result["messages"][1]
        assert isinstance(rejected, ToolMessage)
        assert rejected.status == "error"
        assert "nope" in rejected.content


class TestApplyLiveHitlDecisions:
    def test_reject_returns_error_tool_message(self):
        calls = [{"name": "jira_search", "id": "j1", "args": {}}]
        revised, messages = _apply_live_hitl_decisions(
            calls, [0], {"decisions": [{"type": "reject", "message": "nope"}]}
        )
        assert revised == []
        assert messages[0].status == "error"
        assert "nope" in messages[0].content

    def test_continue_is_detected(self):
        assert _is_auth_continue("continue")
        assert _is_auth_continue({"type": "continue"})
        assert not _is_auth_continue(None)
        assert not _is_auth_continue({"decisions": [{"type": "approve"}]})

    def test_unknown_type_is_reject(self):
        calls = [{"name": "search", "id": "s1", "args": {}}]
        revised, messages = _apply_live_hitl_decisions(
            calls, [0], {"decisions": [{"type": "whatever"}]}
        )
        assert revised == []
        assert messages[0].status == "error"
        assert "Unknown HITL decision type." in messages[0].content

    def test_interrupt_hitl_retries_after_continue(self):
        resumes = iter(["continue", {"decisions": [{"type": "approve"}]}])
        with patch(
            "deep_agent.aegra.mcp_runtime_tools.interrupt",
            side_effect=lambda _payload: next(resumes),
        ):
            raw = _interrupt_hitl({"action_requests": []})
        assert raw == {"decisions": [{"type": "approve"}]}

    def test_compiled_hitl_drains_auth_continue(self):
        from langchain.agents.middleware.human_in_the_loop import (
            HumanInTheLoopMiddleware,
        )

        install_compiled_hitl_auth_resume()
        mw = HumanInTheLoopMiddleware(interrupt_on={"task": True})
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "id": "t1",
                            "args": {"description": "x"},
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }
        resumes = iter(["continue", {"decisions": [{"type": "approve"}]}])
        with patch(
            "deep_agent.aegra.mcp_runtime_tools.interrupt",
            side_effect=lambda _payload: next(resumes),
        ):
            result = mw.after_model(state, MagicMock())
        assert result is not None
        assert result["messages"][0].tool_calls[0]["name"] == "task"

    def test_compiled_hitl_empty_resume_rejects_without_typeerror(self):
        from langchain.agents.middleware.human_in_the_loop import (
            HumanInTheLoopMiddleware,
        )

        install_compiled_hitl_auth_resume()
        mw = HumanInTheLoopMiddleware(interrupt_on={"task": True})
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "id": "t1",
                            "args": {"description": "x"},
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }
        with patch(
            "deep_agent.aegra.mcp_runtime_tools.interrupt",
            return_value=None,
        ):
            result = mw.after_model(state, MagicMock())
        assert result is not None
        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert tool_msgs
        assert tool_msgs[0].status == "error"
        assert "Missing HITL decision." in tool_msgs[0].content

    def test_missing_decision_is_reject(self):
        assert _decision_at(None, 0) == {
            "type": "reject",
            "message": "Missing HITL decision.",
        }
        assert _decision_at({"decisions": []}, 0)["type"] == "reject"
        assert _decision_at({"decisions": [{"type": "approve"}]}, 1)["type"] == "reject"
        assert _decision_at({"decisions": ["approve"]}, 0)["type"] == "reject"
        assert _decision_at({"decisions": [{}]}, 0)["type"] == "reject"

    def test_missing_decisions_drop_live_call(self):
        calls = [{"name": "search", "id": "s1", "args": {}}]
        revised, messages = _apply_live_hitl_decisions(calls, [0], None)
        assert revised == []
        assert messages[0].status == "error"
        assert "Missing HITL decision." in messages[0].content
        revised, messages = _apply_live_hitl_decisions(
            calls, [0], {"decisions": [{"type": "approve"}]}
        )
        assert revised == calls
        assert messages == []
        revised, messages = _apply_live_hitl_decisions(calls, [0], {"decisions": []})
        assert revised == []
        assert messages[0].status == "error"


class TestRewriteThenRuntimeAttach:
    @pytest.mark.asyncio
    async def test_live_frontmatter_name_attaches_after_rewrite(self):
        from deep_agent.aegra.mcp import rewrite_oauth_dcr_tool_names
        from deep_agent.src.agent.config.resolver import resolve_tools

        placeholder = _mcp_tool("mcp__jira_mcp")
        live = _mcp_tool("jira_search", "jira-mcp")
        servers = {
            "jira-mcp": {
                "enabled": True,
                "auth_mode": "dcr",
                "tool_prefix": "jira",
            }
        }
        with patch(
            "deep_agent.aegra.mcp._get_server_configs",
            return_value=servers,
        ):
            names = rewrite_oauth_dcr_tool_names(["jira_search"])
            bound = resolve_tools(names, [placeholder], "jira-child")
            req = _model_request(bound)
            handler = AsyncMock(return_value="ok")
            mw = McpRuntimeToolsMiddleware()
            with (
                patch(
                    "deep_agent.aegra.mcp._resolve_mcp_user_id",
                    return_value="user-1",
                ),
                patch("deep_agent.aegra.mcp._current_user_id") as mock_ctx,
                patch(
                    "deep_agent.aegra.mcp.get_authenticated_oauth_mcp_tools",
                    new=AsyncMock(return_value=[live]),
                ),
            ):
                mock_ctx.set = MagicMock()
                result = await mw.awrap_model_call(req, handler)
        assert result == "ok"
        overridden = handler.call_args[0][0]
        names_out = [t.name for t in overridden.tools]
        assert "mcp__jira_mcp" not in names_out
        assert "jira_search" in names_out


class TestRuntimeMcpAttachFilters:
    def test_both_empty_attaches_nothing(self):
        assert runtime_mcp_attach_filters([], []) == (frozenset(), frozenset())
        assert runtime_mcp_attach_filters(None, None) == (frozenset(), frozenset())

    def test_tools_only_allowlists_names(self):
        assert runtime_mcp_attach_filters(["search", "create"], []) == (
            frozenset({"search", "create"}),
            None,
        )

    def test_mcps_only_allows_all_on_fence(self):
        assert runtime_mcp_attach_filters([], ["acme-jira"]) == (
            None,
            frozenset({"acme-jira"}),
        )

    def test_tools_and_mcps(self):
        assert runtime_mcp_attach_filters(["search"], ["acme-jira"]) == (
            frozenset({"search"}),
            frozenset({"acme-jira"}),
        )

    def test_placeholder_only_tools_attaches_all_live_on_fence(self):
        assert runtime_mcp_attach_filters(["mcp__acme_jira"], ["acme-jira"]) == (
            None,
            frozenset({"acme-jira"}),
        )

    def test_placeholder_plus_live_name_keep_live_allowlist(self):
        assert runtime_mcp_attach_filters(
            ["mcp__acme_jira", "search"], ["acme-jira"]
        ) == (
            frozenset({"search"}),
            frozenset({"acme-jira"}),
        )
