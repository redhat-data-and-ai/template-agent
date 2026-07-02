"""Unit tests for the HITL interrupt_on builder."""

from unittest.mock import MagicMock

import pytest

from deep_agent.src.agent.config.hitl import (
    _DEEPAGENTS_BUILTIN_TOOLS,
    build_interrupt_on,
)
from deep_agent.src.agent.config.middleware import HumanApprovalConfig


def _tool(name: str) -> MagicMock:
    t = MagicMock()
    t.name = name
    return t


class TestBuildInterruptOn:
    def test_disabled_returns_empty(self):
        config = HumanApprovalConfig(enabled=False, mode="all")
        result = build_interrupt_on(config, [_tool("send_email"), _tool("delete_record")])
        assert result == {}

    def test_mode_none_returns_empty(self):
        config = HumanApprovalConfig(enabled=True, mode="none")
        result = build_interrupt_on(config, [_tool("send_email")])
        assert result == {}

    def test_mode_all_includes_explicit_tools(self):
        tools = [_tool("send_email"), _tool("search_web"), _tool("delete_record")]
        config = HumanApprovalConfig(enabled=True, mode="all")
        result = build_interrupt_on(config, tools)
        assert result["send_email"] is True
        assert result["search_web"] is True
        assert result["delete_record"] is True

    def test_mode_all_always_includes_builtins(self):
        """Built-in deepagents tools must be interrupted even with no explicit tools."""
        config = HumanApprovalConfig(enabled=True, mode="all")
        result = build_interrupt_on(config, [])
        for builtin in _DEEPAGENTS_BUILTIN_TOOLS:
            assert builtin in result, f"built-in tool '{builtin}' missing from interrupt_on"

    def test_empty_tool_list_still_covers_builtins(self):
        """An agent with no MCP tools still gets HITL for built-in filesystem tools."""
        config = HumanApprovalConfig(enabled=True, mode="all")
        result = build_interrupt_on(config, [])
        assert len(result) == len(_DEEPAGENTS_BUILTIN_TOOLS)
        assert result == {name: True for name in _DEEPAGENTS_BUILTIN_TOOLS}

    def test_exclude_removes_listed_tools(self):
        tools = [_tool("send_email"), _tool("search_web"), _tool("health_check")]
        config = HumanApprovalConfig(
            enabled=True, mode="all", exclude=["health_check", "search_web", "ls", "read_file"]
        )
        result = build_interrupt_on(config, tools)
        assert result.get("send_email") is True
        assert "health_check" not in result
        assert "search_web" not in result
        assert "ls" not in result
        assert "read_file" not in result

    def test_exclude_nonexistent_tool_is_harmless(self):
        tools = [_tool("send_email")]
        config = HumanApprovalConfig(enabled=True, mode="all", exclude=["nonexistent"])
        result = build_interrupt_on(config, tools)
        assert result["send_email"] is True
        # builtins still present
        assert "ls" in result

    def test_all_tools_excluded_returns_empty(self):
        all_names = list(_DEEPAGENTS_BUILTIN_TOOLS) + ["send_email"]
        tools = [_tool("send_email")]
        config = HumanApprovalConfig(enabled=True, mode="all", exclude=all_names)
        result = build_interrupt_on(config, tools)
        assert result == {}

    def test_default_config_disabled(self):
        """Default HumanApprovalConfig should produce no interrupts."""
        config = HumanApprovalConfig()
        result = build_interrupt_on(config, [_tool("send_email")])
        assert result == {}
