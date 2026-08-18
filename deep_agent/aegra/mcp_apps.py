"""MCP Apps (SEP-1865) client-side capability helpers.

Advertises ``io.modelcontextprotocol/ui`` during MCP ``initialize`` so
spec-compliant App servers can enable UI-bound tools. The Python MCP SDK
and langchain-mcp-adapters do not yet expose a first-class hook for
``capabilities.extensions``, so we wrap ``ClientSession.initialize`` once
at process start.

Also provides pure helpers to read tool ``_meta.ui``, stamp ``mcp_server``,
and filter app-only tools out of the model-facing tool list (no shared
in-memory Apps registry — safe for multi-pod).
"""

from __future__ import annotations

import contextvars
import inspect
from typing import Any

from mcp import types
from mcp.client.session import ClientSession

# SEP-1865 / ext-apps extension identifier and MVP MIME type.
MCP_APPS_EXTENSION_ID = "io.modelcontextprotocol/ui"
MCP_APPS_MIME_TYPE = "text/html;profile=mcp-app"

_DEFAULT_VISIBILITY: tuple[str, ...] = ("model", "app")

_original_initialize: Any | None = None
_patch_installed = False

# Request-scoped (same async task) capture of the raw MCP CallToolResult so we can
# embed MCP-shaped content on ToolMessage.artifact.mcp_app. Not a cross-pod cache —
# the snapshot is copied onto the message before the tool call returns.
_captured_call_tool_result: contextvars.ContextVar[dict[str, Any] | None] = (
    contextvars.ContextVar("_captured_call_tool_result", default=None)
)


def mcp_apps_extension_settings() -> dict[str, Any]:
    """Return the settings map advertised under capabilities.extensions."""
    return {"mimeTypes": [MCP_APPS_MIME_TYPE]}


def get_tool_ui_meta(tool: Any) -> dict[str, Any]:
    """Extract MCP Apps UI metadata from a LangChain tool.

    Reads ``metadata["_meta"]["ui"]`` (current) and falls back to the
    deprecated flat ``metadata["_meta"]["ui/resourceUri"]`` form.
    """
    metadata = getattr(tool, "metadata", None) or {}
    if not isinstance(metadata, dict):
        return {}

    meta = metadata.get("_meta")
    if not isinstance(meta, dict):
        return {}

    ui = meta.get("ui")
    if isinstance(ui, dict):
        return dict(ui)

    # Deprecated: _meta["ui/resourceUri"] = "ui://..."
    deprecated_uri = meta.get("ui/resourceUri")
    if isinstance(deprecated_uri, str) and deprecated_uri:
        return {"resourceUri": deprecated_uri}

    return {}


def get_tool_visibility(tool: Any) -> list[str]:
    """Return tool visibility list; defaults to ``["model", "app"]`` when omitted.

    An explicit empty list ``[]`` means visible to neither model nor app.
    """
    ui = get_tool_ui_meta(tool)
    visibility = ui.get("visibility")
    if isinstance(visibility, list):
        return [str(v) for v in visibility]
    return list(_DEFAULT_VISIBILITY)


def is_model_visible(tool: Any) -> bool:
    """True if the tool may be listed for / called by the model."""
    return "model" in get_tool_visibility(tool)


def is_app_callable(tool: Any) -> bool:
    """True if an MCP App may call this tool via the host proxy."""
    return "app" in get_tool_visibility(tool)


def annotate_mcp_tool(tool: Any, mcp_server: str) -> Any:
    """Stamp ``mcp_server`` onto tool.metadata (in place). Preserves ``_meta``."""
    metadata = getattr(tool, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)
    metadata["mcp_server"] = mcp_server
    tool.metadata = metadata
    return tool


def build_mcp_app_descriptor(tool: Any) -> dict[str, Any] | None:
    """Return a host-facing Apps descriptor if the tool declares a ui:// resource."""
    ui = get_tool_ui_meta(tool)
    resource_uri = ui.get("resourceUri")
    if not isinstance(resource_uri, str) or not resource_uri.startswith("ui://"):
        return None

    metadata = getattr(tool, "metadata", None) or {}
    server = metadata.get("mcp_server") if isinstance(metadata, dict) else None
    if not isinstance(server, str) or not server:
        return None

    return {
        "server": server,
        "resourceUri": resource_uri,
        "toolName": getattr(tool, "name", None),
        "visibility": get_tool_visibility(tool),
    }


def _as_content_blocks(content: Any) -> list[Any]:
    """Normalize LangChain tool content into MCP CallToolResult content blocks."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return list(content)
    return [{"type": "text", "text": str(content)}]


def serialize_call_tool_result_for_app(result: Any) -> dict[str, Any]:
    """Convert a raw MCP ``CallToolResult`` into a JSON-friendly Apps host payload."""
    content_out: list[Any] = []
    for block in getattr(result, "content", None) or []:
        if hasattr(block, "model_dump"):
            content_out.append(
                block.model_dump(by_alias=True, exclude_none=True, mode="json")
            )
        elif isinstance(block, dict):
            content_out.append(block)
        else:
            content_out.append({"type": "text", "text": str(block)})

    meta = getattr(result, "meta", None)
    if meta is not None and hasattr(meta, "model_dump"):
        meta = meta.model_dump(by_alias=True, exclude_none=True, mode="json")

    payload: dict[str, Any] = {
        "content": content_out,
        "isError": bool(getattr(result, "isError", False)),
    }
    # Omit nulls — hosts validate with CallToolResultSchema which rejects null.
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        payload["structuredContent"] = structured
    if meta is not None:
        payload["_meta"] = meta
    return payload


def capture_call_tool_result_for_app(result: Any) -> None:
    """Store a raw MCP result for the current tool invocation (same async task)."""
    _captured_call_tool_result.set(serialize_call_tool_result_for_app(result))


def take_captured_call_tool_result() -> dict[str, Any] | None:
    """Return and clear the captured MCP result for this tool invocation."""
    value = _captured_call_tool_result.get()
    _captured_call_tool_result.set(None)
    return value


class McpAppCallToolResultInterceptor:
    """Capture raw MCP ``CallToolResult`` before LangChain content conversion.

    Registered on ``MultiServerMCPClient`` so model-invoked UI tools can embed a
    spec-faithful ``mcp_app.result`` (including ``isError`` and ``_meta``) on the
    ToolMessage artifact. Capture is contextvar-scoped to the call — not a pod cache.
    """

    async def __call__(self, request: Any, handler: Any) -> Any:
        """Run the next handler and capture ``CallToolResult`` when present."""
        _captured_call_tool_result.set(None)
        result = await handler(request)
        if isinstance(result, types.CallToolResult):
            capture_call_tool_result_for_app(result)
        return result


def attach_mcp_app_to_tool_result(
    result: Any,
    descriptor: dict[str, Any],
    *,
    mcp_result: dict[str, Any] | None = None,
) -> Any:
    """Embed ``mcp_app`` on a content_and_artifact tool result for streaming hosts.

    Prefer ``mcp_result`` (raw MCP ``CallToolResult`` shape) when provided so the
    View receives MCP content blocks, ``structuredContent``, ``isError``, and ``_meta``.
    """
    if isinstance(result, tuple) and len(result) == 2:
        content, artifact = result
    else:
        content, artifact = result, None

    art: dict[str, Any] = dict(artifact) if isinstance(artifact, dict) else {}

    if mcp_result is not None:
        app_result: dict[str, Any] = {
            "content": list(mcp_result.get("content") or []),
            "isError": bool(mcp_result.get("isError")),
        }
        structured = mcp_result.get("structuredContent")
        if structured is not None:
            app_result["structuredContent"] = structured
        meta = mcp_result.get("_meta")
        if meta is not None:
            app_result["_meta"] = meta
    else:
        structured = art.get("structured_content")
        if structured is None:
            structured = art.get("structuredContent")
        app_result = {
            "content": _as_content_blocks(content),
            "isError": False,
        }
        if structured is not None:
            app_result["structuredContent"] = structured

    art["mcp_app"] = {**descriptor, "result": app_result}
    return content, art


def extract_mcp_app_from_message(msg: Any) -> dict[str, Any] | None:
    """Pull ``mcp_app`` from a ToolMessage (artifact / kwargs / response_metadata)."""
    artifact = getattr(msg, "artifact", None)
    if isinstance(artifact, dict):
        payload = artifact.get("mcp_app") or artifact.get("mcpApp")
        if isinstance(payload, dict):
            return payload

    for container_name in ("additional_kwargs", "response_metadata"):
        container = getattr(msg, container_name, None)
        if isinstance(container, dict):
            payload = container.get("mcpApp") or container.get("mcp_app")
            if isinstance(payload, dict):
                return payload
    return None


def wrap_tool_to_attach_mcp_app(tool: Any) -> Any:
    """Wrap a tool coroutine so UI-bound results carry ``artifact.mcp_app``."""
    descriptor = build_mcp_app_descriptor(tool)
    if not descriptor:
        return tool

    coroutine = getattr(tool, "coroutine", None)
    if not inspect.iscoroutinefunction(coroutine):
        return tool

    async def wrapped_coroutine(**kwargs: Any) -> Any:
        from langchain_core.tools import ToolException

        try:
            result = await coroutine(**kwargs)
        except Exception as exc:
            # ToolException: adapters raise when CallToolResult.isError is true.
            # NotImplementedError (etc.): adapters fail converting AudioContent /
            # other blocks to LangChain — interceptor already captured the raw MCP
            # result; keep the App mountable with a text stub for the model.
            captured = take_captured_call_tool_result()
            if captured is None:
                if not isinstance(exc, ToolException):
                    raise
                captured = {
                    "content": [{"type": "text", "text": str(exc)}],
                    "isError": True,
                }
            text_parts: list[str] = []
            for block in captured.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str) and text:
                        text_parts.append(text)
            if text_parts:
                fallback = "\n".join(text_parts)
            elif isinstance(exc, ToolException):
                fallback = str(exc)
            else:
                fallback = (
                    "Tool returned content the model path cannot convert "
                    f"({type(exc).__name__}); see the interactive UI."
                )
            return attach_mcp_app_to_tool_result(
                ([{"type": "text", "text": fallback}], None),
                {**descriptor, "arguments": kwargs},
                mcp_result=captured,
            )

        captured = take_captured_call_tool_result()
        return attach_mcp_app_to_tool_result(
            result,
            {**descriptor, "arguments": kwargs},
            mcp_result=captured,
        )

    try:
        return tool.model_copy(update={"coroutine": wrapped_coroutine})
    except Exception:
        tool.coroutine = wrapped_coroutine
        return tool


def prepare_tools_for_model(tools: list[Any], mcp_server: str) -> list[Any]:
    """Annotate tools with ``mcp_server`` and drop app-only tools for the LLM.

    UI-bound tools are wrapped so their results embed an ``mcp_app`` descriptor
    for the chat host. Does not store tools in a shared registry.
    """
    prepared: list[Any] = []
    for tool in tools:
        annotate_mcp_tool(tool, mcp_server)
        if is_model_visible(tool):
            prepared.append(wrap_tool_to_attach_mcp_app(tool))
    return prepared


def inject_ui_extension_into_request(
    request: types.ClientRequest,
) -> types.ClientRequest:
    """Return a copy of *request* with MCP Apps UI capability on initialize.

    Non-initialize requests are returned unchanged. Idempotent if extensions
    are already present.
    """
    root = request.root
    if not isinstance(root, types.InitializeRequest):
        return request

    caps = root.params.capabilities
    existing = getattr(caps, "extensions", None) or {}
    if (
        isinstance(existing, dict)
        and MCP_APPS_EXTENSION_ID in existing
        and isinstance(existing[MCP_APPS_EXTENSION_ID], dict)
        and MCP_APPS_MIME_TYPE
        in (existing[MCP_APPS_EXTENSION_ID].get("mimeTypes") or [])
    ):
        return request

    extensions = {
        **(existing if isinstance(existing, dict) else {}),
        MCP_APPS_EXTENSION_ID: mcp_apps_extension_settings(),
    }
    new_caps = caps.model_copy(update={"extensions": extensions})
    new_params = root.params.model_copy(update={"capabilities": new_caps})
    new_root = root.model_copy(update={"params": new_params})
    return types.ClientRequest(new_root)


async def _initialize_with_mcp_apps(self: ClientSession) -> types.InitializeResult:
    """Wrap stock initialize so the handshake advertises MCP Apps support."""
    if _original_initialize is None:
        raise RuntimeError("MCP Apps initialize patch was not installed")

    original_send_request = self.send_request

    async def send_request_with_extensions(
        request: Any,
        result_type: Any,
        **kwargs: Any,
    ) -> Any:
        if isinstance(request, types.ClientRequest):
            request = inject_ui_extension_into_request(request)
        return await original_send_request(request, result_type, **kwargs)

    self.send_request = send_request_with_extensions
    try:
        return await _original_initialize(self)
    finally:
        self.send_request = original_send_request


def ensure_mcp_apps_capability_advertised() -> bool:
    """Patch ``ClientSession.initialize`` once (idempotent).

    Returns:
        True if the patch was newly installed, False if already installed.
    """
    global _original_initialize, _patch_installed  # noqa: PLW0603

    if _patch_installed:
        return False

    _original_initialize = ClientSession.initialize
    ClientSession.initialize = _initialize_with_mcp_apps
    _patch_installed = True
    return True
