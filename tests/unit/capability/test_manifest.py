"""Unit tests for deep_agent.src.capability.manifest."""

from unittest.mock import MagicMock, patch

import pytest

from deep_agent.src.capability.manifest import (
    EXPLICIT,
    IMPLICIT_ALL_MCP,
    NONE,
    CapabilityManifest,
    resolve_capability_manifest,
)


def _tool(name: str):
    """Build a MagicMock tool stub with the given name."""
    tool = MagicMock()
    tool.name = name
    return tool


# ---------------------------------------------------------------------------
# CapabilityManifest
# ---------------------------------------------------------------------------


class TestCapabilityManifest:
    def test_allows_returns_true_for_member(self):
        """Test that allows() returns True for a tool name in the manifest."""
        manifest = CapabilityManifest(
            agent_name="orchestrator",
            allowed_tool_names=frozenset({"search", "read_file"}),
            source=EXPLICIT,
        )
        assert manifest.allows("search") is True

    def test_allows_returns_false_for_non_member(self):
        """Test that allows() returns False for a tool name outside the manifest."""
        manifest = CapabilityManifest(
            agent_name="orchestrator",
            allowed_tool_names=frozenset({"search"}),
            source=EXPLICIT,
        )
        assert manifest.allows("delete_everything") is False

    def test_allows_returns_false_for_empty_manifest(self):
        """Test that allows() returns False for every tool name on an empty manifest."""
        manifest = CapabilityManifest(
            agent_name="orchestrator", allowed_tool_names=frozenset(), source=NONE
        )
        assert manifest.allows("anything") is False

    def test_is_frozen(self):
        """Test that CapabilityManifest is frozen and rejects attribute mutation."""
        manifest = CapabilityManifest(
            agent_name="a", allowed_tool_names=frozenset(), source=NONE
        )
        with pytest.raises(Exception):
            manifest.agent_name = "b"  # type: ignore[misc]

    def test_merged_with_adds_names_without_mutating_original(self):
        """Test that merged_with() adds names on a new instance without mutating the original."""
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
    def test_explicit_tools_resolve_to_named_subset(self):
        """Test that an explicit 'tools:' list resolves to exactly the named subset."""
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
        every MCP tool -- that would defeat the author's own allow-list."""
        available = [_tool("search"), _tool("delete_repo")]

        tools, manifest = resolve_capability_manifest(
            ["nonexistent_tool"],
            available,
            mcp_server_names=["some-mcp"],
            agent_name="orchestrator",
        )

        assert tools == []
        assert manifest.source == NONE
        assert manifest.allowed_tool_names == frozenset()


class TestResolveCapabilityManifestImplicit:
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
        """Test that a failure emitting the implicit-grant audit event does not raise."""
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
        """Regression guard (OFFSEC-384 review): an author writing `tools: []`
        on purpose is the most restrictive possible declaration and must not
        be treated the same as omitting `tools:` altogether -- it must NOT
        fall back to granting every tool on the declared MCP server(s)."""
        available = [_tool("search"), _tool("write_file")]

        with patch("deep_agent.src.audit.emitter.emit_audit_event") as emit:
            tools, manifest = resolve_capability_manifest(
                [],
                available,
                mcp_server_names=["some-mcp"],
                agent_name="orchestrator",
            )

        assert tools == []
        assert manifest.source == NONE
        assert manifest.allowed_tool_names == frozenset()
        emit.assert_not_called()


class TestResolveCapabilityManifestNone:
    def test_no_tools_and_no_mcp_servers_grants_nothing(self):
        """Test that no 'tools:' and no 'mcps:' grants nothing."""
        tools, manifest = resolve_capability_manifest(
            [], [], mcp_server_names=[], agent_name="orchestrator"
        )
        assert tools == []
        assert manifest.source == NONE
        assert manifest.allowed_tool_names == frozenset()

    def test_mcp_servers_declared_but_no_tools_available_grants_nothing(self):
        """Test that declared 'mcps:' with no available tools still grants nothing."""
        tools, manifest = resolve_capability_manifest(
            [], [], mcp_server_names=["some-mcp"], agent_name="orchestrator"
        )
        assert tools == []
        assert manifest.source == NONE

    def test_none_mcp_server_names_is_handled(self):
        """Test that a None mcp_server_names is handled the same as an empty list."""
        tools, manifest = resolve_capability_manifest(
            [], [_tool("search")], mcp_server_names=None, agent_name="orchestrator"
        )
        assert tools == []
        assert manifest.source == NONE
