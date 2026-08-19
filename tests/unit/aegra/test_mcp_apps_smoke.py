"""Compliance smoke tests for the MCP Apps host (agent side).

Locks the generic host contract: capability advertising, visibility filtering,
and request-scoped resources/list + resources/templates/list + resources/read + tools/call rules.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from langchain_core.tools import StructuredTool, ToolException
from mcp import types

from deep_agent.aegra.mcp_apps import (
    MCP_APPS_EXTENSION_ID,
    MCP_APPS_MIME_TYPE,
    annotate_mcp_tool,
    capture_call_tool_result_for_app,
    ensure_mcp_apps_capability_advertised,
    is_app_callable,
    is_model_visible,
    mcp_apps_extension_settings,
    prepare_tools_for_model,
)
from deep_agent.aegra.mcp_host import (
    call_app_tool,
    list_resource_templates,
    read_resource,
)


def _tool(name: str, meta: dict | None = None) -> StructuredTool:
    return StructuredTool.from_function(
        func=lambda: "ok",
        name=name,
        description=name,
        metadata={"_meta": meta} if meta is not None else {},
    )


def _server_cfg(**overrides):
    cfg = {
        "url": "http://localhost:5003/mcp",
        "transport": "streamable_http",
        "enabled": True,
        "auth": False,
        "auth_mode": "sso",
        "ssl_verify": False,
        "timeout": 30,
    }
    cfg.update(overrides)
    return cfg


@asynccontextmanager
async def _fake_session(session):
    yield session


class TestMcpAppsSmokeCapability:
    def test_extension_settings_match_sep_1865(self):
        settings = mcp_apps_extension_settings()
        assert settings["mimeTypes"] == [MCP_APPS_MIME_TYPE]
        assert MCP_APPS_EXTENSION_ID == "io.modelcontextprotocol/ui"
        assert MCP_APPS_MIME_TYPE == "text/html;profile=mcp-app"

    def test_capability_patch_is_installed(self):
        ensure_mcp_apps_capability_advertised()
        from mcp.client.session import ClientSession
        from deep_agent.aegra import mcp_apps

        assert mcp_apps._patch_installed is True
        assert ClientSession.initialize is mcp_apps._initialize_with_mcp_apps


class TestMcpAppsSmokeVisibility:
    def test_app_only_tools_hidden_from_model(self):
        model_tool = _tool(
            "show",
            {"ui": {"resourceUri": "ui://x", "visibility": ["model", "app"]}},
        )
        app_only = _tool("refresh", {"ui": {"visibility": ["app"]}})
        annotate_mcp_tool(model_tool, "charts")
        annotate_mcp_tool(app_only, "charts")

        assert is_model_visible(model_tool) is True
        assert is_model_visible(app_only) is False
        assert is_app_callable(app_only) is True

        prepared = prepare_tools_for_model([model_tool, app_only], "charts")
        assert [t.name for t in prepared] == ["show"]
        assert prepared[0].metadata.get("mcp_server") == "charts"


class TestMcpAppsSmokeHostProxy:
    @pytest.mark.asyncio
    async def test_resources_read_rejects_empty_uri(self):
        with pytest.raises(HTTPException) as exc:
            await read_resource(
                "charts",
                "",
                user_id="u1",
                sso_token=None,
            )
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_resource_templates_list_is_request_scoped(self):
        session = MagicMock()
        session.list_resource_templates = AsyncMock(
            return_value=types.ListResourceTemplatesResult(resourceTemplates=[])
        )
        with (
            patch(
                "deep_agent.aegra.mcp_host._get_server_configs",
                return_value={"charts": _server_cfg()},
            ),
            patch(
                "deep_agent.aegra.mcp_host._resolve_connection_token",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "deep_agent.aegra.mcp_host.MultiServerMCPClient",
            ) as mock_client_cls,
        ):
            client = MagicMock()
            client.session = lambda _name: _fake_session(session)
            mock_client_cls.return_value = client
            result = await list_resource_templates(
                "charts",
                user_id="u1",
                sso_token=None,
            )
        session.list_resource_templates.assert_awaited_once_with(cursor=None)
        assert (
            result.get("resourceTemplates") == []
            or result.get("resource_templates") == []
        )

    @pytest.mark.asyncio
    async def test_error_tool_result_still_embeds_mcp_app(self):
        async def _coro(**_kwargs):
            capture_call_tool_result_for_app(
                types.CallToolResult(
                    content=[types.TextContent(type="text", text="failed")],
                    isError=True,
                )
            )
            raise ToolException("failed")

        tool = _tool(
            "show",
            {
                "ui": {
                    "resourceUri": "ui://charts/app.html",
                    "visibility": ["model", "app"],
                }
            },
        )
        tool.coroutine = _coro
        prepared = prepare_tools_for_model([tool], "charts")
        _content, artifact = await prepared[0].coroutine()
        assert artifact["mcp_app"]["result"]["isError"] is True
        assert artifact["mcp_app"]["result"]["content"][0]["text"] == "failed"
        assert artifact["mcp_app"]["resourceUri"] == "ui://charts/app.html"

    @pytest.mark.asyncio
    async def test_tools_call_rejects_model_only(self):
        tool = types.Tool.model_validate(
            {
                "name": "model_only",
                "inputSchema": {"type": "object"},
                "_meta": {"ui": {"visibility": ["model"]}},
            }
        )
        session = MagicMock()
        session.list_tools = AsyncMock(return_value=types.ListToolsResult(tools=[tool]))
        session.call_tool = AsyncMock()

        with (
            patch(
                "deep_agent.aegra.mcp_host._get_server_configs",
                return_value={"charts": _server_cfg()},
            ),
            patch(
                "deep_agent.aegra.mcp_host._resolve_connection_token",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "deep_agent.aegra.mcp_host.MultiServerMCPClient",
            ) as mock_client_cls,
            pytest.raises(HTTPException) as exc,
        ):
            client = MagicMock()
            client.session = lambda _name: _fake_session(session)
            mock_client_cls.return_value = client

            await call_app_tool(
                "charts",
                "model_only",
                {},
                user_id="u1",
                sso_token=None,
            )

        assert exc.value.status_code == 403
        session.call_tool.assert_not_called()
