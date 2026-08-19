"""Unit tests for MCP Apps client capability advertising."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp import types
from mcp.client.session import ClientSession

from deep_agent.aegra import mcp_apps
from deep_agent.aegra.mcp_apps import (
    MCP_APPS_EXTENSION_ID,
    MCP_APPS_MIME_TYPE,
    McpAppCallToolResultInterceptor,
    annotate_mcp_tool,
    attach_mcp_app_to_tool_result,
    build_mcp_app_descriptor,
    capture_call_tool_result_for_app,
    ensure_mcp_apps_capability_advertised,
    extract_mcp_app_from_message,
    get_tool_ui_meta,
    get_tool_visibility,
    inject_ui_extension_into_request,
    is_app_callable,
    is_model_visible,
    mcp_apps_extension_settings,
    prepare_tools_for_model,
    serialize_call_tool_result_for_app,
    take_captured_call_tool_result,
)


def _make_initialize_request(
    *,
    extensions: dict | None = None,
) -> types.ClientRequest:
    caps_kwargs: dict = {"experimental": None}
    if extensions is not None:
        caps_kwargs["extensions"] = extensions
    params = types.InitializeRequestParams(
        protocolVersion=types.LATEST_PROTOCOL_VERSION,
        capabilities=types.ClientCapabilities(**caps_kwargs),
        clientInfo=types.Implementation(name="test-host", version="0.0.1"),
    )
    return types.ClientRequest(types.InitializeRequest(params=params))


class TestMcpAppsExtensionSettings:
    def test_mime_type_matches_spec(self):
        settings = mcp_apps_extension_settings()
        assert settings == {"mimeTypes": [MCP_APPS_MIME_TYPE]}
        assert MCP_APPS_MIME_TYPE == "text/html;profile=mcp-app"
        assert MCP_APPS_EXTENSION_ID == "io.modelcontextprotocol/ui"


class TestInjectUiExtensionIntoRequest:
    def test_injects_extensions_on_initialize(self):
        request = _make_initialize_request()
        updated = inject_ui_extension_into_request(request)

        caps = updated.root.params.capabilities
        extensions = getattr(caps, "extensions", None)
        assert extensions is not None
        assert extensions[MCP_APPS_EXTENSION_ID]["mimeTypes"] == [MCP_APPS_MIME_TYPE]

        dumped = updated.model_dump(exclude_none=True)
        assert dumped["params"]["capabilities"]["extensions"][MCP_APPS_EXTENSION_ID][
            "mimeTypes"
        ] == [MCP_APPS_MIME_TYPE]

    def test_idempotent_when_already_present(self):
        request = _make_initialize_request(
            extensions={
                MCP_APPS_EXTENSION_ID: mcp_apps_extension_settings(),
            }
        )
        updated = inject_ui_extension_into_request(request)
        assert updated is request

    def test_preserves_other_extensions(self):
        request = _make_initialize_request(
            extensions={"com.example/other": {"enabled": True}}
        )
        updated = inject_ui_extension_into_request(request)
        extensions = updated.root.params.capabilities.extensions
        assert extensions["com.example/other"] == {"enabled": True}
        assert extensions[MCP_APPS_EXTENSION_ID]["mimeTypes"] == [MCP_APPS_MIME_TYPE]

    def test_ignores_non_initialize_requests(self):
        request = types.ClientRequest(types.PingRequest())
        assert inject_ui_extension_into_request(request) is request


def _tool_with_meta(name: str, meta: dict | None = None) -> SimpleNamespace:
    metadata: dict = {}
    if meta is not None:
        metadata["_meta"] = meta
    return SimpleNamespace(name=name, metadata=metadata)


class TestToolUiMetaAndVisibility:
    def test_no_meta_defaults_visibility(self):
        tool = _tool_with_meta("plain")
        assert get_tool_ui_meta(tool) == {}
        assert get_tool_visibility(tool) == ["model", "app"]
        assert is_model_visible(tool) is True
        assert is_app_callable(tool) is True

    def test_nested_ui_meta(self):
        tool = _tool_with_meta(
            "chart",
            {
                "ui": {
                    "resourceUri": "ui://charts/app.html",
                    "visibility": ["model", "app"],
                }
            },
        )
        assert get_tool_ui_meta(tool)["resourceUri"] == "ui://charts/app.html"
        assert is_model_visible(tool) is True

    def test_app_only_hidden_from_model(self):
        tool = _tool_with_meta(
            "refresh",
            {"ui": {"resourceUri": "ui://x", "visibility": ["app"]}},
        )
        assert is_model_visible(tool) is False
        assert is_app_callable(tool) is True

    def test_explicit_empty_visibility_is_not_default(self):
        tool = _tool_with_meta(
            "hidden",
            {"ui": {"resourceUri": "ui://x", "visibility": []}},
        )
        assert get_tool_visibility(tool) == []
        assert is_model_visible(tool) is False
        assert is_app_callable(tool) is False

    def test_deprecated_flat_resource_uri(self):
        tool = _tool_with_meta(
            "legacy",
            {"ui/resourceUri": "ui://legacy/app.html"},
        )
        assert get_tool_ui_meta(tool) == {"resourceUri": "ui://legacy/app.html"}
        assert is_model_visible(tool) is True

    def test_annotate_and_prepare_for_model(self):
        model_tool = _tool_with_meta(
            "show",
            {"ui": {"resourceUri": "ui://show", "visibility": ["model", "app"]}},
        )
        app_only = _tool_with_meta(
            "refresh",
            {"ui": {"visibility": ["app"]}},
        )
        plain = _tool_with_meta("search")

        prepared = prepare_tools_for_model(
            [model_tool, app_only, plain], "chart-mcp-server"
        )
        names = [t.name for t in prepared]
        assert names == ["show", "search"]
        assert app_only.name not in names
        assert model_tool.metadata["mcp_server"] == "chart-mcp-server"
        assert plain.metadata["mcp_server"] == "chart-mcp-server"
        # App-only tool is still annotated even though filtered out of model list
        assert app_only.metadata["mcp_server"] == "chart-mcp-server"

    def test_annotate_preserves_existing_meta(self):
        tool = _tool_with_meta(
            "show",
            {"ui": {"resourceUri": "ui://x"}},
        )
        annotate_mcp_tool(tool, "my-server")
        assert tool.metadata["_meta"]["ui"]["resourceUri"] == "ui://x"
        assert tool.metadata["mcp_server"] == "my-server"

    def test_build_descriptor_and_attach_to_result(self):
        tool = _tool_with_meta(
            "show_chart",
            {"ui": {"resourceUri": "ui://charts/app.html"}},
        )
        annotate_mcp_tool(tool, "chart-mcp-server")
        descriptor = build_mcp_app_descriptor(tool)
        assert descriptor is not None
        assert descriptor["server"] == "chart-mcp-server"
        assert descriptor["resourceUri"] == "ui://charts/app.html"

        content, artifact = attach_mcp_app_to_tool_result(
            (
                [{"type": "text", "text": "ok"}],
                {"structured_content": {"rows": [1]}},
            ),
            descriptor,
        )
        assert content[0]["text"] == "ok"
        assert artifact["mcp_app"]["server"] == "chart-mcp-server"
        assert artifact["mcp_app"]["result"]["structuredContent"] == {"rows": [1]}
        assert artifact["mcp_app"]["result"]["isError"] is False

    def test_attach_prefers_raw_mcp_result_shape(self):
        tool = _tool_with_meta(
            "show_chart",
            {"ui": {"resourceUri": "ui://charts/app.html"}},
        )
        annotate_mcp_tool(tool, "chart-mcp-server")
        descriptor = build_mcp_app_descriptor(tool)
        assert descriptor is not None

        # LangChain-shaped content would use base64/mime_type; MCP uses data/mimeType.
        lc_content = [{"type": "image", "base64": "YWJj", "mime_type": "image/png"}]
        mcp_result = {
            "content": [
                {"type": "image", "data": "YWJj", "mimeType": "image/png"},
            ],
            "structuredContent": {"rows": [1]},
            "isError": False,
            "_meta": {"source": "charts"},
        }
        _content, artifact = attach_mcp_app_to_tool_result(
            (lc_content, None),
            descriptor,
            mcp_result=mcp_result,
        )
        app_result = artifact["mcp_app"]["result"]
        assert app_result["content"] == mcp_result["content"]
        assert app_result["structuredContent"] == {"rows": [1]}
        assert app_result["_meta"] == {"source": "charts"}
        assert app_result["isError"] is False

    @pytest.mark.asyncio
    async def test_interceptor_captures_call_tool_result(self):
        mcp_result = types.CallToolResult(
            content=[
                types.TextContent(type="text", text="ok"),
                types.ImageContent(type="image", data="YWJj", mimeType="image/png"),
            ],
            structuredContent={"k": 1},
            isError=False,
        )

        async def handler(_request):
            return mcp_result

        interceptor = McpAppCallToolResultInterceptor()
        out = await interceptor(SimpleNamespace(name="show"), handler)
        assert out is mcp_result
        captured = take_captured_call_tool_result()
        assert captured is not None
        assert captured["content"][0] == {"type": "text", "text": "ok"}
        assert captured["content"][1]["mimeType"] == "image/png"
        assert captured["content"][1]["data"] == "YWJj"
        assert captured["structuredContent"] == {"k": 1}
        assert take_captured_call_tool_result() is None

    @pytest.mark.asyncio
    async def test_prepare_wraps_ui_tool_coroutine(self):
        async def _coro(**kwargs):
            return ("plain", {"structured_content": {"v": 1}})

        tool = _tool_with_meta(
            "show",
            {"ui": {"resourceUri": "ui://x"}},
        )
        tool.coroutine = _coro
        prepared = prepare_tools_for_model([tool], "srv")
        assert len(prepared) == 1

        capture_call_tool_result_for_app(
            types.CallToolResult(
                content=[types.TextContent(type="text", text="mcp-ok")],
                structuredContent={"v": 1},
                isError=False,
            )
        )
        _content, artifact = await prepared[0].coroutine(topic="sales")
        assert artifact["mcp_app"]["server"] == "srv"
        assert artifact["mcp_app"]["arguments"] == {"topic": "sales"}
        assert artifact["mcp_app"]["result"]["content"] == [
            {"type": "text", "text": "mcp-ok"}
        ]
        assert artifact["mcp_app"]["result"]["structuredContent"] == {"v": 1}

    @pytest.mark.asyncio
    async def test_wrap_attaches_mcp_app_on_tool_exception(self):
        from langchain_core.tools import ToolException

        async def _coro(**kwargs):
            capture_call_tool_result_for_app(
                types.CallToolResult(
                    content=[types.TextContent(type="text", text="boom")],
                    structuredContent={"err": True},
                    isError=True,
                )
            )
            raise ToolException("boom")

        tool = _tool_with_meta(
            "show",
            {"ui": {"resourceUri": "ui://x"}},
        )
        tool.coroutine = _coro
        prepared = prepare_tools_for_model([tool], "srv")
        _content, artifact = await prepared[0].coroutine()
        assert artifact["mcp_app"]["result"]["isError"] is True
        assert artifact["mcp_app"]["result"]["content"] == [
            {"type": "text", "text": "boom"}
        ]
        assert artifact["mcp_app"]["result"]["structuredContent"] == {"err": True}
        assert _content[0]["text"] == "boom"

    @pytest.mark.asyncio
    async def test_wrap_attaches_mcp_app_on_audio_conversion_failure(self):
        """langchain-mcp-adapters cannot convert AudioContent; App must still mount."""

        async def _coro(**kwargs):
            capture_call_tool_result_for_app(
                types.CallToolResult(
                    content=[
                        types.TextContent(type="text", text="lab ready"),
                        types.AudioContent(
                            type="audio",
                            data="UklGRg==",
                            mimeType="audio/wav",
                        ),
                    ],
                    structuredContent={"contentType": "audio"},
                    isError=False,
                )
            )
            raise NotImplementedError(
                "AudioContent conversion to LangChain content blocks "
                "is not yet supported. Received audio with mime type: audio/wav"
            )

        tool = _tool_with_meta(
            "open_conformance_lab",
            {"ui": {"resourceUri": "ui://lab/app.html"}},
        )
        tool.coroutine = _coro
        prepared = prepare_tools_for_model([tool], "srv")
        content, artifact = await prepared[0].coroutine(contentType="audio")
        assert content[0]["text"] == "lab ready"
        app_result = artifact["mcp_app"]["result"]
        assert app_result["isError"] is False
        assert app_result["structuredContent"] == {"contentType": "audio"}
        assert app_result["content"][0] == {"type": "text", "text": "lab ready"}
        assert app_result["content"][1]["type"] == "audio"
        assert app_result["content"][1]["mimeType"] == "audio/wav"

    @pytest.mark.asyncio
    async def test_interceptor_ignores_non_call_tool_result(self):
        async def handler(_request):
            return {"not": "a CallToolResult"}

        interceptor = McpAppCallToolResultInterceptor()
        out = await interceptor(SimpleNamespace(name="x"), handler)
        assert out == {"not": "a CallToolResult"}
        assert take_captured_call_tool_result() is None

    def test_serialize_handles_dict_and_plain_blocks(self):
        class _Meta:
            def model_dump(self, **_kwargs):
                return {"k": 1}

        result = SimpleNamespace(
            content=[
                {"type": "text", "text": "dict-block"},
                42,
            ],
            isError=False,
            structuredContent={"ok": True},
            meta=_Meta(),
        )
        serialized = serialize_call_tool_result_for_app(result)
        assert serialized["content"][0] == {"type": "text", "text": "dict-block"}
        assert serialized["content"][1] == {"type": "text", "text": "42"}
        assert serialized["structuredContent"] == {"ok": True}
        assert serialized["_meta"] == {"k": 1}

    def test_attach_non_tuple_uses_content_blocks(self):
        descriptor = {
            "server": "srv",
            "resourceUri": "ui://x",
            "toolName": "show",
            "visibility": ["model", "app"],
        }
        _content, artifact = attach_mcp_app_to_tool_result(
            "hello",
            descriptor,
        )
        assert _content == "hello"
        assert artifact["mcp_app"]["result"]["content"] == [
            {"type": "text", "text": "hello"}
        ]

    def test_attach_list_content_without_mcp_result(self):
        descriptor = {
            "server": "srv",
            "resourceUri": "ui://x",
            "toolName": "show",
            "visibility": ["app"],
        }
        _content, artifact = attach_mcp_app_to_tool_result(
            ([{"type": "text", "text": "a"}], {"structuredContent": {"n": 2}}),
            descriptor,
        )
        assert artifact["mcp_app"]["result"]["structuredContent"] == {"n": 2}

    def test_extract_mcp_app_from_additional_kwargs(self):
        from langchain_core.messages import ToolMessage

        msg = ToolMessage(
            content="ok",
            tool_call_id="tc1",
            name="show",
            additional_kwargs={
                "mcpApp": {
                    "server": "srv",
                    "resourceUri": "ui://from-kwargs",
                    "result": {"content": [], "isError": False},
                }
            },
        )
        assert extract_mcp_app_from_message(msg)["resourceUri"] == "ui://from-kwargs"

    def test_extract_mcp_app_returns_none_without_payload(self):
        from langchain_core.messages import ToolMessage

        msg = ToolMessage(content="ok", tool_call_id="tc1", name="show")
        assert extract_mcp_app_from_message(msg) is None

    def test_serialize_call_tool_result_for_app(self):
        serialized = serialize_call_tool_result_for_app(
            types.CallToolResult(
                content=[types.TextContent(type="text", text="x")],
                isError=True,
            )
        )
        assert serialized["isError"] is True
        assert serialized["content"] == [{"type": "text", "text": "x"}]

    def test_extract_mcp_app_from_tool_message(self):
        from langchain_core.messages import ToolMessage

        msg = ToolMessage(
            content="ok",
            tool_call_id="tc1",
            name="show",
            artifact={
                "mcp_app": {
                    "server": "srv",
                    "resourceUri": "ui://x",
                    "result": {"content": [], "isError": False},
                }
            },
        )
        assert extract_mcp_app_from_message(msg)["resourceUri"] == "ui://x"


class TestEnsureMcpAppsCapabilityAdvertised:
    def test_install_is_idempotent(self):
        # Module import of deep_agent.aegra.mcp may already have installed the patch.
        ensure_mcp_apps_capability_advertised()
        second = ensure_mcp_apps_capability_advertised()
        assert second is False
        assert mcp_apps._patch_installed is True
        assert ClientSession.initialize is mcp_apps._initialize_with_mcp_apps

    @pytest.mark.asyncio
    async def test_patched_initialize_sends_extensions(self):
        ensure_mcp_apps_capability_advertised()

        captured: dict[str, types.ClientRequest] = {}

        async def fake_original_initialize(
            self: ClientSession,
        ) -> types.InitializeResult:
            # Simulate stock initialize issuing an InitializeRequest via send_request.
            req = _make_initialize_request()
            await self.send_request(req, types.InitializeResult)
            return types.InitializeResult(
                protocolVersion=types.LATEST_PROTOCOL_VERSION,
                capabilities=types.ServerCapabilities(),
                serverInfo=types.Implementation(name="mock", version="0"),
            )

        async def fake_send_request(request, result_type, **kwargs):
            captured["request"] = request
            return types.InitializeResult(
                protocolVersion=types.LATEST_PROTOCOL_VERSION,
                capabilities=types.ServerCapabilities(),
                serverInfo=types.Implementation(name="mock", version="0"),
            )

        session = object.__new__(ClientSession)
        original_send = AsyncMock(side_effect=fake_send_request)
        session.send_request = original_send

        previous = mcp_apps._original_initialize
        mcp_apps._original_initialize = fake_original_initialize
        try:
            result = await ClientSession.initialize(session)
            assert isinstance(result, types.InitializeResult)
            assert "request" in captured
            caps = captured["request"].root.params.capabilities
            assert caps.extensions[MCP_APPS_EXTENSION_ID]["mimeTypes"] == [
                MCP_APPS_MIME_TYPE
            ]
            # send_request must be restored after initialize
            assert session.send_request is original_send
        finally:
            mcp_apps._original_initialize = previous
