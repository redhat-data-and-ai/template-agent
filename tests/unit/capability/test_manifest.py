"""Unit tests for deep_agent.src.capability.manifest."""

from unittest.mock import MagicMock, patch

import pytest

from deep_agent.src.capability.manifest import (
    EXPLICIT,
    IMPLICIT_ALL_MCP,
    NO_MANIFEST,
    CapabilityManifest,
    resolve_capability_manifest,
)


def _tool(name: str):
    """Create a MagicMock tool with the given name."""
    tool = MagicMock()
    tool.name = name
    return tool


# ---------------------------------------------------------------------------
# CapabilityManifest
# ---------------------------------------------------------------------------


class TestCapabilityManifest:
    """Tests for the CapabilityManifest frozen dataclass."""

    def test_allows_returns_true_for_member(self):
        """Member tool names are authorized."""
        manifest = CapabilityManifest(
            agent_name="orchestrator",
            allowed_tool_names=frozenset({"search", "read_file"}),
            source=EXPLICIT,
        )
        assert manifest.allows("search") is True

    def test_allows_returns_false_for_non_member(self):
        """Non-member tool names are denied."""
        manifest = CapabilityManifest(
            agent_name="orchestrator",
            allowed_tool_names=frozenset({"search"}),
            source=EXPLICIT,
        )
        assert manifest.allows("delete_everything") is False

    def test_allows_returns_false_for_empty_manifest(self):
        """Empty manifest denies everything."""
        manifest = CapabilityManifest(
            agent_name="orchestrator",
            allowed_tool_names=frozenset(),
            source=NO_MANIFEST,
        )
        assert manifest.allows("anything") is False

    def test_is_frozen(self):
        """Frozen dataclass rejects attribute assignment."""
        manifest = CapabilityManifest(
            agent_name="a", allowed_tool_names=frozenset(), source=NO_MANIFEST
        )
        with pytest.raises(Exception):
            manifest.agent_name = "b"  # type: ignore[misc]

    def test_merged_with_adds_names_without_mutating_original(self):
        """merged_with returns a new manifest; original stays unchanged."""
        original = CapabilityManifest(
            agent_name="orchestrator",
            allowed_tool_names=frozenset({"search"}),
            source=EXPLICIT,
        )

        merged = original.merged_with(["mcp_read_resource", "mcp_list_resources"])

        assert merged.allows("search") is True
        assert merged.allows("mcp_read_resource") is True
        assert merged.allows("mcp_list_resources") is True
        assert merged.agent_name == "orchestrator"
        assert merged.source == EXPLICIT
        # Original manifest must be unaffected (frozen dataclass semantics).
        assert original.allowed_tool_names == frozenset({"search"})
        assert original.allows("mcp_read_resource") is False


# ---------------------------------------------------------------------------
# resolve_capability_manifest
# ---------------------------------------------------------------------------


class TestResolveCapabilityManifestExplicit:
    """Tests for explicit tools: list resolution."""

    def test_explicit_tools_resolve_to_named_subset(self):
        """Only named tools are included; undeclared MCP tools are excluded."""
        available = [_tool("search"), _tool("write_file"), _tool("delete_repo")]

        tools, manifest = resolve_capability_manifest(
            ["search", "write_file"],
            available,
            mcp_server_names=["some-mcp"],
            agent_name="orchestrator",
        )

        assert {t.name for t in tools} == {"search", "write_file"}
        assert manifest.source == EXPLICIT
        assert manifest.allowed_tool_names == frozenset({"search", "write_file"})
        # The undeclared tool must not be authorized even though it came from
        # a declared MCP server.
        assert manifest.allows("delete_repo") is False

    def test_explicit_list_with_no_matches_does_not_fall_back_to_implicit(self):
        """An explicit (but unmatched) tools: list must not silently expand to
        every MCP tool -- that would defeat the author's own allow-list.
        Source is EXPLICIT (not NO_MANIFEST) because the builder intended a restriction."""
        available = [_tool("search"), _tool("delete_repo")]

        tools, manifest = resolve_capability_manifest(
            ["nonexistent_tool"],
            available,
            mcp_server_names=["some-mcp"],
            agent_name="orchestrator",
        )

        assert tools == []
        assert manifest.source == EXPLICIT
        assert manifest.allowed_tool_names == frozenset()

    def test_explicit_empty_resolution_emits_warning(self, caplog):
        """When tool_names are declared but none resolve, a capability_explicit_empty
        warning is logged with the declared tool names."""
        available = [_tool("search")]

        with caplog.at_level("WARNING"):
            tools, manifest = resolve_capability_manifest(
                ["fake_tool_a", "fake_tool_b"],
                available,
                mcp_server_names=["some-mcp"],
                agent_name="test_agent",
            )

        assert tools == []
        assert manifest.source == EXPLICIT
        assert manifest.allowed_tool_names == frozenset()
        assert any("capability_explicit_empty" in r.message for r in caplog.records)


class TestResolveCapabilityManifestImplicit:
    """Tests for the implicit-all-mcp grant when tools: is omitted."""

    def test_declared_mcp_servers_without_explicit_tools_grants_all(self):
        """`tools:` genuinely omitted (None) -- not merely an empty list."""
        available = [_tool("search"), _tool("write_file")]

        with patch("deep_agent.src.audit.emitter.emit_audit_event") as emit:
            tools, manifest = resolve_capability_manifest(
                None,
                available,
                mcp_server_names=["some-mcp"],
                agent_name="orchestrator",
            )

        assert tools == available
        assert manifest.source == IMPLICIT_ALL_MCP
        assert manifest.allowed_tool_names == frozenset({"search", "write_file"})
        emit.assert_called_once()
        assert emit.call_args.args[0] == "capability_implicit_grant"
        assert emit.call_args.kwargs["agent"] == "orchestrator"
        assert emit.call_args.kwargs["mcp_servers"] == ["some-mcp"]

    def test_audit_emit_failure_does_not_raise(self):
        """Broken audit sink must not crash manifest resolution."""
        available = [_tool("search")]

        with patch(
            "deep_agent.src.audit.emitter.emit_audit_event",
            side_effect=RuntimeError("sink unavailable"),
        ):
            tools, manifest = resolve_capability_manifest(
                None,
                available,
                mcp_server_names=["some-mcp"],
                agent_name="orchestrator",
            )

        assert tools == available
        assert manifest.source == IMPLICIT_ALL_MCP

    def test_explicit_empty_tools_list_does_not_grant_all(self):
        """Explicit `tools: []` must not fall back to granting every
        tool on declared MCP server(s)."""
        available = [_tool("search"), _tool("write_file")]

        with patch("deep_agent.src.audit.emitter.emit_audit_event") as emit:
            tools, manifest = resolve_capability_manifest(
                [],
                available,
                mcp_server_names=["some-mcp"],
                agent_name="orchestrator",
            )

        assert tools == []
        assert manifest.source == NO_MANIFEST
        assert manifest.allowed_tool_names == frozenset()
        emit.assert_not_called()


class TestResolveCapabilityManifestNone:
    """Tests for the empty/none manifest path."""

    def test_no_tools_and_no_mcp_servers_grants_nothing(self):
        """Neither tools nor MCP servers yields an empty manifest."""
        tools, manifest = resolve_capability_manifest(
            [], [], mcp_server_names=[], agent_name="orchestrator"
        )
        assert tools == []
        assert manifest.source == NO_MANIFEST
        assert manifest.allowed_tool_names == frozenset()

    def test_mcp_servers_declared_but_no_tools_available_grants_nothing(self):
        """MCP declared but no tools loaded still yields empty manifest."""
        tools, manifest = resolve_capability_manifest(
            [], [], mcp_server_names=["some-mcp"], agent_name="orchestrator"
        )
        assert tools == []
        assert manifest.source == NO_MANIFEST

    def test_none_mcp_server_names_is_handled(self):
        """None mcp_server_names does not raise."""
        tools, manifest = resolve_capability_manifest(
            [], [_tool("search")], mcp_server_names=None, agent_name="orchestrator"
        )
        assert tools == []
        assert manifest.source == NO_MANIFEST
