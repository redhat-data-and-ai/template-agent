"""Unit tests for host MCP resource tools (resources/list, templates, read)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from deep_agent.aegra.mcp import _current_access_token, _current_user_id
from deep_agent.aegra.mcp_auth import NeedsAuthorization
from deep_agent.aegra.mcp_resource_tools import (
    LIST_TOOL,
    READ_TOOL,
    TEMPLATES_TOOL,
    build_mcp_resource_tools,
)


def _tool(tools, name):
    return next(t for t in tools if t.name == name)


@pytest.fixture
def auth_ctx():
    uid = _current_user_id.set("user-1")
    tok = _current_access_token.set("sso-token")
    try:
        yield
    finally:
        _current_user_id.reset(uid)
        _current_access_token.reset(tok)


class TestGetMcpResourceTools:
    def test_all_enabled_when_server_names_omitted(self):
        with (
            patch(
                "deep_agent.aegra.mcp_resource_tools._get_server_configs",
                return_value={
                    "a": {"enabled": True},
                    "b": {"enabled": False},
                    "c": {"enabled": True},
                },
            ),
            patch(
                "deep_agent.aegra.mcp_resource_tools.build_mcp_resource_tools",
                return_value=[],
            ) as mock_build,
        ):
            from deep_agent.aegra.mcp_resource_tools import get_mcp_resource_tools

            get_mcp_resource_tools(server_names=None)
        mock_build.assert_called_once_with(
            allowed_servers=["a", "c"], allowed_uris=None
        )

    def test_intersects_declared_mcps_with_enabled(self):
        with (
            patch(
                "deep_agent.aegra.mcp_resource_tools._get_server_configs",
                return_value={
                    "a": {"enabled": True},
                    "b": {"enabled": False},
                    "c": {"enabled": True},
                },
            ),
            patch(
                "deep_agent.aegra.mcp_resource_tools.build_mcp_resource_tools",
                return_value=[],
            ) as mock_build,
        ):
            from deep_agent.aegra.mcp_resource_tools import get_mcp_resource_tools

            get_mcp_resource_tools(server_names=["b", "c", "missing"])
        mock_build.assert_called_once_with(allowed_servers=["c"], allowed_uris=None)

    def test_forwards_allowed_uris(self):
        with (
            patch(
                "deep_agent.aegra.mcp_resource_tools._get_server_configs",
                return_value={"a": {"enabled": True}},
            ),
            patch(
                "deep_agent.aegra.mcp_resource_tools.build_mcp_resource_tools",
                return_value=[],
            ) as mock_build,
        ):
            from deep_agent.aegra.mcp_resource_tools import get_mcp_resource_tools

            get_mcp_resource_tools(server_names=None, allowed_uris=["template://about"])
        mock_build.assert_called_once_with(
            allowed_servers=["a"], allowed_uris=["template://about"]
        )


class TestBuildMcpResourceTools:
    def test_empty_uri_allowlist_still_builds_tools(self):
        tools = build_mcp_resource_tools(
            allowed_servers=["template-mcp-server"],
            allowed_uris=[],
        )
        assert len(tools) == 3

    def test_no_servers_returns_no_tools(self):
        tools = build_mcp_resource_tools(allowed_servers=[], allowed_uris=None)
        assert tools == []

    def test_unrestricted_returns_three_tools(self):
        tools = build_mcp_resource_tools(
            allowed_servers=["template-mcp-server"],
            allowed_uris=None,
        )
        assert [t.name for t in tools] == [LIST_TOOL, TEMPLATES_TOOL, READ_TOOL]
        assert "omitted" in _tool(tools, READ_TOOL).description
        for t in tools:
            assert t.func is None
            assert t.coroutine is not None


class TestListResourcesTool:
    @pytest.mark.asyncio
    async def test_rejects_unknown_server(self, auth_ctx):
        tools = build_mcp_resource_tools(
            allowed_servers=["template-mcp-server"],
            allowed_uris=None,
        )
        result = await _tool(tools, LIST_TOOL).ainvoke({"mcp_name": "other"})
        assert "disallowed" in result
        assert "template-mcp-server" in result

    @pytest.mark.asyncio
    async def test_passes_cursor_and_auth(self, auth_ctx):
        tools = build_mcp_resource_tools(
            allowed_servers=["template-mcp-server"],
            allowed_uris=None,
        )
        payload = {
            "resources": [{"uri": "template://about", "name": "about"}],
            "nextCursor": "page-2",
        }
        with patch(
            "deep_agent.aegra.mcp_resource_tools.list_resources",
            new_callable=AsyncMock,
            return_value=payload,
        ) as mock_list:
            result = await _tool(tools, LIST_TOOL).ainvoke(
                {"mcp_name": "template-mcp-server", "cursor": "abc"}
            )
        mock_list.assert_awaited_once_with(
            "template-mcp-server",
            cursor="abc",
            user_id="user-1",
            sso_token="sso-token",
        )
        parsed = json.loads(result)
        assert parsed["resources"][0]["uri"] == "template://about"
        assert parsed["nextCursor"] == "page-2"

    @pytest.mark.asyncio
    async def test_falls_back_to_run_config_user_id(self):
        tools = build_mcp_resource_tools(
            allowed_servers=["template-mcp-server"],
            allowed_uris=None,
        )
        with (
            patch(
                "deep_agent.aegra.mcp_resource_tools.list_resources",
                new_callable=AsyncMock,
                return_value={"resources": []},
            ) as mock_list,
            patch(
                "langgraph.config.get_config",
                return_value={
                    "configurable": {"langgraph_auth_user_id": "jwt-sub-1"},
                },
            ),
        ):
            await _tool(tools, LIST_TOOL).ainvoke({"mcp_name": "template-mcp-server"})
        mock_list.assert_awaited_once_with(
            "template-mcp-server",
            cursor=None,
            user_id="jwt-sub-1",
            sso_token=None,
        )

    @pytest.mark.asyncio
    async def test_filters_list_by_allowlist(self, auth_ctx):
        tools = build_mcp_resource_tools(
            allowed_servers=["s"],
            allowed_uris=["template://about"],
        )
        with patch(
            "deep_agent.aegra.mcp_resource_tools.list_resources",
            new_callable=AsyncMock,
            return_value={
                "resources": [
                    {"uri": "template://about", "name": "about"},
                    {"uri": "template://secret", "name": "secret"},
                ]
            },
        ):
            result = await _tool(tools, LIST_TOOL).ainvoke({"mcp_name": "s"})
        uris = [r["uri"] for r in json.loads(result)["resources"]]
        assert uris == ["template://about"]

    @pytest.mark.asyncio
    async def test_authorization_required_raises(self, auth_ctx):
        tools = build_mcp_resource_tools(allowed_servers=["s"], allowed_uris=None)
        with patch(
            "deep_agent.aegra.mcp_resource_tools.list_resources",
            new_callable=AsyncMock,
            side_effect=HTTPException(
                status_code=401,
                detail={
                    "error": "authorization_required",
                    "mcp_name": "s",
                    "connect_url": "/mcp/s/connect",
                },
            ),
        ):
            with pytest.raises(NeedsAuthorization) as exc:
                await _tool(tools, LIST_TOOL).ainvoke({"mcp_name": "s"})
        assert exc.value.mcp_name == "s"
        assert exc.value.connect_url == "/mcp/s/connect"

    @pytest.mark.asyncio
    async def test_other_http_errors_are_strings(self, auth_ctx):
        tools = build_mcp_resource_tools(allowed_servers=["s"], allowed_uris=None)
        with patch(
            "deep_agent.aegra.mcp_resource_tools.list_resources",
            new_callable=AsyncMock,
            side_effect=HTTPException(status_code=404, detail="Unknown or disabled"),
        ):
            result = await _tool(tools, LIST_TOOL).ainvoke({"mcp_name": "s"})
        assert result.startswith("MCP resource request failed (404)")

    @pytest.mark.asyncio
    async def test_timeout_is_stable_string(self, auth_ctx):
        tools = build_mcp_resource_tools(allowed_servers=["s"], allowed_uris=None)
        with patch(
            "deep_agent.aegra.mcp_resource_tools.list_resources",
            new_callable=AsyncMock,
            side_effect=TimeoutError,
        ):
            result = await _tool(tools, LIST_TOOL).ainvoke({"mcp_name": "s"})
        assert result == "MCP resource request failed: timed out"

    @pytest.mark.asyncio
    async def test_generic_error_does_not_leak_exception_text(self, auth_ctx):
        tools = build_mcp_resource_tools(allowed_servers=["s"], allowed_uris=None)
        leak = "http://internal:443 tlsv1 alert"
        with patch(
            "deep_agent.aegra.mcp_resource_tools.list_resources",
            new_callable=AsyncMock,
            side_effect=RuntimeError(leak),
        ):
            result = await _tool(tools, LIST_TOOL).ainvoke({"mcp_name": "s"})
        assert result == "MCP resource request failed"
        assert leak not in result


class TestListTemplatesTool:
    @pytest.mark.asyncio
    async def test_filters_templates_by_exact_uri_template(self, auth_ctx):
        tools = build_mcp_resource_tools(
            allowed_servers=["s"],
            allowed_uris=["template://echo/{text}"],
        )
        with patch(
            "deep_agent.aegra.mcp_resource_tools.list_resource_templates",
            new_callable=AsyncMock,
            return_value={
                "resourceTemplates": [
                    {"uriTemplate": "template://echo/{text}", "name": "echo"},
                    {"uriTemplate": "template://other/{id}", "name": "other"},
                ]
            },
        ):
            result = await _tool(tools, TEMPLATES_TOOL).ainvoke({"mcp_name": "s"})
        templates = json.loads(result)["resourceTemplates"]
        assert len(templates) == 1
        assert templates[0]["uriTemplate"] == "template://echo/{text}"


class TestReadResourceTool:
    @pytest.mark.asyncio
    async def test_returns_text_and_stubs_blob(self, auth_ctx):
        tools = build_mcp_resource_tools(allowed_servers=["s"], allowed_uris=None)
        blob = "cG5nLWJ5dGVz"
        payload = {
            "contents": [
                {"uri": "template://about", "mimeType": "text/plain", "text": "hello"},
                {"uri": "template://logo", "mimeType": "image/png", "blob": blob},
            ]
        }
        with patch(
            "deep_agent.aegra.mcp_resource_tools.read_resource",
            new_callable=AsyncMock,
            return_value=payload,
        ) as mock_read:
            result = await _tool(tools, READ_TOOL).ainvoke(
                {"mcp_name": "s", "uri": "template://logo"}
            )
        mock_read.assert_awaited_once_with(
            "s", "template://logo", user_id="user-1", sso_token="sso-token"
        )
        assert "hello" in result
        assert blob not in result
        assert "Binary content omitted" in result
        assert "image/png" in result

    @pytest.mark.asyncio
    async def test_text_only_returns_plain_text(self, auth_ctx):
        tools = build_mcp_resource_tools(allowed_servers=["s"], allowed_uris=None)
        with patch(
            "deep_agent.aegra.mcp_resource_tools.read_resource",
            new_callable=AsyncMock,
            return_value={
                "contents": [
                    {
                        "uri": "template://about",
                        "mimeType": "text/plain",
                        "text": "hello",
                    }
                ]
            },
        ):
            result = await _tool(tools, READ_TOOL).ainvoke(
                {"mcp_name": "s", "uri": "template://about"}
            )
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_extracts_text_ignores_blob_on_same_item(self, auth_ctx):
        tools = build_mcp_resource_tools(allowed_servers=["s"], allowed_uris=None)
        with patch(
            "deep_agent.aegra.mcp_resource_tools.read_resource",
            new_callable=AsyncMock,
            return_value={
                "contents": [
                    {
                        "uri": "template://both",
                        "mimeType": "text/plain",
                        "text": "caption",
                        "blob": "eA==",
                    }
                ]
            },
        ):
            result = await _tool(tools, READ_TOOL).ainvoke(
                {"mcp_name": "s", "uri": "template://both"}
            )
        assert "caption" in result
        assert "eA==" not in result

    @pytest.mark.asyncio
    async def test_rejects_uri_not_on_allowlist(self, auth_ctx):
        tools = build_mcp_resource_tools(
            allowed_servers=["s"],
            allowed_uris=["template://about"],
        )
        with patch(
            "deep_agent.aegra.mcp_resource_tools.read_resource",
            new_callable=AsyncMock,
        ) as mock_read:
            result = await _tool(tools, READ_TOOL).ainvoke(
                {"mcp_name": "s", "uri": "template://secret"}
            )
        mock_read.assert_not_awaited()
        assert "not allowed" in result

    @pytest.mark.asyncio
    async def test_allows_uri_matching_listed_template(self, auth_ctx):
        tools = build_mcp_resource_tools(
            allowed_servers=["s"],
            allowed_uris=["template://echo/{text}"],
        )
        with patch(
            "deep_agent.aegra.mcp_resource_tools.read_resource",
            new_callable=AsyncMock,
            return_value={"contents": [{"uri": "template://echo/hi", "text": "hi"}]},
        ) as mock_read:
            result = await _tool(tools, READ_TOOL).ainvoke(
                {"mcp_name": "s", "uri": "template://echo/hi"}
            )
        mock_read.assert_awaited_once()
        assert "hi" in result

    @pytest.mark.asyncio
    async def test_template_does_not_match_extra_path_segment(self, auth_ctx):
        tools = build_mcp_resource_tools(
            allowed_servers=["s"],
            allowed_uris=["template://echo/{text}"],
        )
        with patch(
            "deep_agent.aegra.mcp_resource_tools.read_resource",
            new_callable=AsyncMock,
        ) as mock_read:
            result = await _tool(tools, READ_TOOL).ainvoke(
                {"mcp_name": "s", "uri": "template://echo/a/b"}
            )
        mock_read.assert_not_awaited()
        assert "not allowed" in result

    @pytest.mark.asyncio
    async def test_default_limit_paginates(self, auth_ctx):
        tools = build_mcp_resource_tools(allowed_servers=["s"], allowed_uris=None)
        body = "\n".join(f"line-{i}" for i in range(150))
        with patch(
            "deep_agent.aegra.mcp_resource_tools.read_resource",
            new_callable=AsyncMock,
            return_value={"contents": [{"text": body}]},
        ):
            result = await _tool(tools, READ_TOOL).ainvoke(
                {"mcp_name": "s", "uri": "template://big"}
            )
        assert "line-0" in result
        assert "line-99" in result
        assert "line-100" not in result
        assert "offset=100" in result

    @pytest.mark.asyncio
    async def test_offset_and_limit(self, auth_ctx):
        tools = build_mcp_resource_tools(allowed_servers=["s"], allowed_uris=None)
        body = "\n".join(f"line-{i}" for i in range(200))
        with patch(
            "deep_agent.aegra.mcp_resource_tools.read_resource",
            new_callable=AsyncMock,
            return_value={"contents": [{"text": body}]},
        ):
            result = await _tool(tools, READ_TOOL).ainvoke(
                {"mcp_name": "s", "uri": "template://big", "offset": 100, "limit": 50}
            )
        assert "line-100" in result
        assert "line-149" in result
        assert "line-150" not in result
        assert "offset=150" in result

    @pytest.mark.asyncio
    async def test_short_body_no_truncation(self, auth_ctx):
        tools = build_mcp_resource_tools(allowed_servers=["s"], allowed_uris=None)
        body = "\n".join(f"line-{i}" for i in range(10))
        with patch(
            "deep_agent.aegra.mcp_resource_tools.read_resource",
            new_callable=AsyncMock,
            return_value={"contents": [{"text": body}]},
        ):
            result = await _tool(tools, READ_TOOL).ainvoke(
                {"mcp_name": "s", "uri": "template://small"}
            )
        assert "line-0" in result
        assert "line-9" in result
        assert "truncated" not in result.lower()

    @pytest.mark.asyncio
    async def test_huge_line_is_split_and_pageable(self, auth_ctx):
        tools = build_mcp_resource_tools(allowed_servers=["s"], allowed_uris=None)
        huge = "x" * 500_000
        with patch(
            "deep_agent.aegra.mcp_resource_tools.read_resource",
            new_callable=AsyncMock,
            return_value={"contents": [{"text": huge}]},
        ):
            r1 = await _tool(tools, READ_TOOL).ainvoke(
                {"mcp_name": "s", "uri": "template://huge"}
            )
        assert len(r1) < 500_000
        assert "offset=1" in r1
        with patch(
            "deep_agent.aegra.mcp_resource_tools.read_resource",
            new_callable=AsyncMock,
            return_value={"contents": [{"text": huge}]},
        ):
            r2 = await _tool(tools, READ_TOOL).ainvoke(
                {"mcp_name": "s", "uri": "template://huge", "offset": 1}
            )
        assert "x" in r2
        assert "truncated" not in r2.lower()

    @pytest.mark.asyncio
    async def test_char_cap_cuts_at_line_boundary(self, auth_ctx):
        tools = build_mcp_resource_tools(allowed_servers=["s"], allowed_uris=None)
        short = [f"short-{i}" for i in range(50)]
        long = [f"long-{i}-" + "x" * 10000 for i in range(50)]
        body = "\n".join(short + long)
        with patch(
            "deep_agent.aegra.mcp_resource_tools.read_resource",
            new_callable=AsyncMock,
            return_value={"contents": [{"text": body}]},
        ):
            result = await _tool(tools, READ_TOOL).ainvoke(
                {"mcp_name": "s", "uri": "template://mixed"}
            )
        content_before_notice = result.split("\n\n[Output truncated")[0]
        assert content_before_notice.endswith("\n")
        assert "offset=" in result
        offset_val = int(result.split("offset=")[1].split(" ")[0].rstrip(","))
        assert offset_val < 100
        assert offset_val >= 50
        assert "short-0" in content_before_notice
        assert "short-49" in content_before_notice
        assert f"long-{offset_val - 50}-" not in content_before_notice

    @pytest.mark.asyncio
    async def test_negative_limit_clamped_to_1(self, auth_ctx):
        tools = build_mcp_resource_tools(allowed_servers=["s"], allowed_uris=None)
        body = "\n".join(f"line-{i}" for i in range(10))
        with patch(
            "deep_agent.aegra.mcp_resource_tools.read_resource",
            new_callable=AsyncMock,
            return_value={"contents": [{"text": body}]},
        ):
            result = await _tool(tools, READ_TOOL).ainvoke(
                {"mcp_name": "s", "uri": "template://x", "limit": -1}
            )
        assert "line-0" in result
        assert "offset=-1" not in result

    @pytest.mark.asyncio
    async def test_zero_limit_clamped_to_1(self, auth_ctx):
        tools = build_mcp_resource_tools(allowed_servers=["s"], allowed_uris=None)
        body = "\n".join(f"line-{i}" for i in range(10))
        with patch(
            "deep_agent.aegra.mcp_resource_tools.read_resource",
            new_callable=AsyncMock,
            return_value={"contents": [{"text": body}]},
        ):
            result = await _tool(tools, READ_TOOL).ainvoke(
                {"mcp_name": "s", "uri": "template://x", "limit": 0}
            )
        assert "line-0" in result
        assert "limit=0" not in result

    @pytest.mark.asyncio
    async def test_offset_past_end_returns_notice(self, auth_ctx):
        tools = build_mcp_resource_tools(allowed_servers=["s"], allowed_uris=None)
        body = "\n".join(f"line-{i}" for i in range(10))
        with patch(
            "deep_agent.aegra.mcp_resource_tools.read_resource",
            new_callable=AsyncMock,
            return_value={"contents": [{"text": body}]},
        ):
            result = await _tool(tools, READ_TOOL).ainvoke(
                {"mcp_name": "s", "uri": "template://x", "offset": 999}
            )
        assert "No content at offset=999" in result
        assert "10 lines" in result

    @pytest.mark.asyncio
    async def test_blank_line_then_long_line_paginates(self, auth_ctx):
        tools = build_mcp_resource_tools(allowed_servers=["s"], allowed_uris=None)
        body = "\n" + "x" * 500_000
        with patch(
            "deep_agent.aegra.mcp_resource_tools.read_resource",
            new_callable=AsyncMock,
            return_value={"contents": [{"text": body}]},
        ):
            result = await _tool(tools, READ_TOOL).ainvoke(
                {"mcp_name": "s", "uri": "template://blank-start"}
            )
        assert "offset=" in result
        assert "size limits" not in result


class TestPaginateTextDirect:
    """Direct _paginate_text tests with controlled max_chars."""

    def test_small_budget_truncates_and_paginates(self):
        from deep_agent.aegra.mcp_resource_tools import _paginate_text

        text = "\n".join(f"line-{i}: content" for i in range(10))
        result = _paginate_text(text, "u", offset=0, limit=10, max_chars=50)
        assert "line-0" in result
        assert "offset=" in result
        lines_before_notice = result.split("\n\n[")[0].count("\n")
        assert lines_before_notice < 10

    def test_budget_fits_all_no_truncation(self):
        from deep_agent.aegra.mcp_resource_tools import _paginate_text

        text = "short\ntext\n"
        result = _paginate_text(text, "u", offset=0, limit=10, max_chars=1000)
        assert result == "short\ntext\n"
        assert "truncated" not in result.lower()

    def test_line_boundary_respected(self):
        from deep_agent.aegra.mcp_resource_tools import _paginate_text

        text = "a" * 30 + "\n" + "b" * 30 + "\n"
        result = _paginate_text(text, "u", offset=0, limit=10, max_chars=40)
        content = result.split("\n\n[")[0]
        assert content.endswith("\n")
        assert "b" not in content

    def test_long_line_split_at_small_budget(self):
        from deep_agent.aegra.mcp_resource_tools import _paginate_text

        text = "x" * 200
        result = _paginate_text(text, "u", offset=0, limit=1, max_chars=50)
        assert "offset=1" in result
        assert len(result.split("\n\n[")[0]) <= 50


class TestEvictionSkipList:
    def test_adds_read_tool_to_skip_list(self):
        from deepagents.middleware import filesystem as _fs

        from deep_agent.aegra.graph import _exclude_resource_read_from_eviction

        original = _fs.TOOLS_EXCLUDED_FROM_EVICTION
        try:
            _fs.TOOLS_EXCLUDED_FROM_EVICTION = ("ls", "read_file")
            _exclude_resource_read_from_eviction()
            assert READ_TOOL in _fs.TOOLS_EXCLUDED_FROM_EVICTION
        finally:
            _fs.TOOLS_EXCLUDED_FROM_EVICTION = original

    def test_idempotent(self):
        from deepagents.middleware import filesystem as _fs

        from deep_agent.aegra.graph import _exclude_resource_read_from_eviction

        original = _fs.TOOLS_EXCLUDED_FROM_EVICTION
        try:
            _fs.TOOLS_EXCLUDED_FROM_EVICTION = ("ls", "read_file")
            _exclude_resource_read_from_eviction()
            _exclude_resource_read_from_eviction()
            count = _fs.TOOLS_EXCLUDED_FROM_EVICTION.count(READ_TOOL)
            assert count == 1
        finally:
            _fs.TOOLS_EXCLUDED_FROM_EVICTION = original
