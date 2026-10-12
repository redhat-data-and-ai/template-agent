"""Unit tests for deep_agent.src.capability.tool_proxy."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import ToolMessage

from deep_agent.src.capability.manifest import EXPLICIT, CapabilityManifest
from deep_agent.src.capability.tool_proxy import (
    CAPABILITY_DENIED_RESULT,
    CapabilityToolProxy,
    _get_tool_call_id,
    _make_denied_result,
    enforce_capability,
)


def _make_inner_tool(name="my_tool", description="does stuff"):
    """Create a MagicMock tool with the given name and description."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.args_schema = None
    return tool


def _manifest(allowed: set[str], agent_name="orchestrator", source=EXPLICIT):
    """Build a CapabilityManifest from a set of allowed tool names."""
    return CapabilityManifest(
        agent_name=agent_name, allowed_tool_names=frozenset(allowed), source=source
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class TestGetToolCallId:
    """Tests for _get_tool_call_id helper."""

    def test_returns_id_from_dict(self):
        """Extracts id from a LangGraph ToolCall dict."""
        assert _get_tool_call_id({"id": "abc-123"}) == "abc-123"

    def test_returns_empty_for_non_dict(self):
        """Non-dict input returns empty string."""
        assert _get_tool_call_id("nope") == ""


class TestMakeDeniedResult:
    """Tests for _make_denied_result helper."""

    def test_returns_error_tool_message(self):
        """Returns a ToolMessage with error status and CAPABILITY_DENIED content."""
        result = _make_denied_result("dangerous_tool", {"id": "call-1"})
        assert isinstance(result, ToolMessage)
        assert result.content == CAPABILITY_DENIED_RESULT
        assert result.name == "dangerous_tool"
        assert result.tool_call_id == "call-1"
        assert result.status == "error"


# ---------------------------------------------------------------------------
# CapabilityToolProxy
# ---------------------------------------------------------------------------


class TestCapabilityToolProxyInit:
    """Tests for CapabilityToolProxy construction."""

    def test_copies_name_and_description(self):
        """Proxy copies name and description from the inner tool."""
        inner = _make_inner_tool(name="searcher", description="searches stuff")
        proxy = CapabilityToolProxy(inner, _manifest({"searcher"}))
        assert proxy.name == "searcher"
        assert proxy.description == "searches stuff"

    def test_stores_inner_tool_and_manifest(self):
        """Proxy holds references to the inner tool and manifest."""
        inner = _make_inner_tool()
        manifest = _manifest({"my_tool"})
        proxy = CapabilityToolProxy(inner, manifest)
        assert proxy._inner is inner
        assert proxy._manifest is manifest


class TestCapabilityToolProxyAinvoke:
    """Tests for the async ainvoke hot path."""

    @pytest.mark.asyncio
    async def test_allowed_tool_delegates_to_inner(self):
        """Allowed tool forwards to inner.ainvoke and returns its result."""
        inner = _make_inner_tool(name="search")
        safe_result = ToolMessage(content="ok", name="search", tool_call_id="id-1")
        inner.ainvoke = AsyncMock(return_value=safe_result)
        proxy = CapabilityToolProxy(inner, _manifest({"search"}))

        result = await proxy.ainvoke({"id": "id-1"})

        assert result is safe_result
        inner.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_disallowed_tool_is_denied_without_calling_inner(self):
        """Denied tool returns CAPABILITY_DENIED and never calls inner."""
        inner = _make_inner_tool(name="delete_everything")
        inner.ainvoke = AsyncMock(return_value="should never be returned")
        proxy = CapabilityToolProxy(inner, _manifest({"search"}))

        with patch("deep_agent.src.audit.emitter.emit_audit_event") as emit:
            result = await proxy.ainvoke({"id": "call-9"})

        assert isinstance(result, ToolMessage)
        assert result.content == CAPABILITY_DENIED_RESULT
        assert result.status == "error"
        assert result.tool_call_id == "call-9"
        inner.ainvoke.assert_not_called()
        emit.assert_called_once()
        assert emit.call_args.args[0] == "capability_denied"
        assert emit.call_args.kwargs["tool"] == "delete_everything"

    @pytest.mark.asyncio
    async def test_denial_survives_audit_emit_failure(self):
        """Denial still works even when the audit sink is broken."""
        inner = _make_inner_tool(name="delete_everything")
        inner.ainvoke = AsyncMock()
        proxy = CapabilityToolProxy(inner, _manifest({"search"}))

        with patch(
            "deep_agent.src.audit.emitter.emit_audit_event",
            side_effect=RuntimeError("sink down"),
        ):
            result = await proxy.ainvoke({"id": "call-9"})

        assert isinstance(result, ToolMessage)
        assert result.content == CAPABILITY_DENIED_RESULT
        inner.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_manifest_denies_every_tool(self):
        """An empty manifest denies all tools."""
        inner = _make_inner_tool(name="anything")
        inner.ainvoke = AsyncMock()
        proxy = CapabilityToolProxy(inner, _manifest(set()))

        result = await proxy.ainvoke({"id": "call-1"})

        assert result.content == CAPABILITY_DENIED_RESULT
        inner.ainvoke.assert_not_called()


class TestCapabilityToolProxyRun:
    """Tests for the sync _run fallback path."""

    def test_allowed_structured_tool_forwards_kwargs_as_single_input_dict(self):
        """`BaseTool.run()` delivers a structured (multi-field) tool's parsed
        arguments to a generic `_run(*args, **kwargs)` as kwargs only, never
        mixed with positional args. `invoke()` expects that same input back
        as a single dict, not re-spread as separate kwargs (which would miss
        its own required `input` argument)."""
        inner = _make_inner_tool(name="search")
        inner.invoke = MagicMock(return_value="sync result")
        proxy = CapabilityToolProxy(inner, _manifest({"search"}))

        result = proxy._run(query="val", limit=5)

        inner.invoke.assert_called_once_with({"query": "val", "limit": 5})
        assert result == "sync result"

    def test_allowed_unstructured_tool_forwards_single_positional_input(self):
        """A single-string-arg tool delivers its value as one positional arg;
        that must be passed through as `invoke`'s `input`, not spread."""
        inner = _make_inner_tool(name="search")
        inner.invoke = MagicMock(return_value="sync result")
        proxy = CapabilityToolProxy(inner, _manifest({"search"}))

        result = proxy._run("plain text input")

        inner.invoke.assert_called_once_with("plain text input")
        assert result == "sync result"

    def test_disallowed_tool_denied_without_calling_inner(self):
        """Denied tool returns CAPABILITY_DENIED without calling inner."""
        inner = _make_inner_tool(name="delete_everything")
        inner.invoke = MagicMock(return_value="should never be returned")
        proxy = CapabilityToolProxy(inner, _manifest({"search"}))

        result = proxy._run()

        assert result == CAPABILITY_DENIED_RESULT
        inner.invoke.assert_not_called()


# ---------------------------------------------------------------------------
# enforce_capability
# ---------------------------------------------------------------------------


class TestEnforceCapability:
    """Tests for the enforce_capability wrapping function."""

    def test_returns_unchanged_when_tools_empty(self):
        """Empty tool list is returned as-is."""
        manifest = _manifest({"search"})
        assert enforce_capability([], manifest) == []

    def test_wraps_every_tool_with_the_same_manifest(self):
        """Every tool gets wrapped with a CapabilityToolProxy sharing one manifest."""
        t1 = _make_inner_tool(name="tool_a")
        t2 = _make_inner_tool(name="tool_b")
        manifest = _manifest({"tool_a", "tool_b"})

        result = enforce_capability([t1, t2], manifest)

        assert len(result) == 2
        assert all(isinstance(r, CapabilityToolProxy) for r in result)
        assert result[0].name == "tool_a"
        assert result[1].name == "tool_b"
        assert all(r._manifest is manifest for r in result)

    @pytest.mark.asyncio
    async def test_wrapped_tool_outside_manifest_is_denied_end_to_end(self):
        """Defense-in-depth: even if a tool sneaks into the list without being
        in the manifest, dispatch is blocked."""
        rogue = _make_inner_tool(name="not_in_manifest")
        rogue.ainvoke = AsyncMock(return_value="leaked!")
        manifest = _manifest({"search"})  # rogue tool is not a member

        wrapped = enforce_capability([rogue], manifest)

        result = await wrapped[0].ainvoke({"id": "x"})

        assert result.content == CAPABILITY_DENIED_RESULT
        rogue.ainvoke.assert_not_called()
