"""Unit tests for tool access control."""

from unittest.mock import MagicMock, patch

import pytest

from deep_agent.src.exceptions import AppException, ErrorCodes
from deep_agent.src.infrastructure.tool_access import (
    _wrap_tool_with_approval,
    apply_tool_approval,
    filter_denied_tools,
    migrate_tools_field,
)


def _make_tool(name: str, *, with_func: bool = False) -> MagicMock:
    """Create a mock tool with the given name.

    Args:
        name: Tool name to set on the mock.
        with_func: If True, configure the mock so that wrapping
            falls back to in-place attribute assignment (matching the
            fallback path in _wrap_tool_with_approval when model_copy
            is not available).
    """
    tool = MagicMock()
    tool.name = name
    if with_func:
        # Force fallback path: model_copy raises so the wrapper
        # assigns .func / .coroutine directly on the mock.
        tool.model_copy.side_effect = AttributeError("not a pydantic model")
        tool.coroutine = None
    return tool


class TestFilterDeniedTools:
    """Tests for filter_denied_tools."""

    def test_removes_denied_tools(self):
        """Denying one tool out of three leaves two."""
        t1, t2, t3 = _make_tool("a"), _make_tool("b"), _make_tool("c")
        result = filter_denied_tools([t1, t2, t3], ["b"], agent_name="test")
        assert result == [t1, t3]

    def test_empty_denied_list_returns_unchanged(self):
        """No denied names means the original list is returned."""
        tools = [_make_tool("x"), _make_tool("y")]
        result = filter_denied_tools(tools, [], agent_name="test")
        assert result is tools

    def test_denied_tool_not_in_list_is_noop(self):
        """Denying a name that does not exist causes no error."""
        t1, t2 = _make_tool("a"), _make_tool("b")
        result = filter_denied_tools([t1, t2], ["nonexistent"], agent_name="test")
        assert result == [t1, t2]

    def test_all_tools_denied(self):
        """Denying every tool returns an empty list."""
        t1, t2 = _make_tool("a"), _make_tool("b")
        result = filter_denied_tools([t1, t2], ["a", "b"], agent_name="test")
        assert result == []

    def test_deny_wins_over_presence(self):
        """A tool present in both the list and the deny set is removed."""
        t1 = _make_tool("search")
        result = filter_denied_tools([t1], ["search"], agent_name="test")
        assert t1 not in result

    def test_preserves_tool_order(self):
        """Remaining tools maintain their original order."""
        t1, t2, t3, t4 = (
            _make_tool("a"),
            _make_tool("b"),
            _make_tool("c"),
            _make_tool("d"),
        )
        result = filter_denied_tools([t1, t2, t3, t4], ["b", "d"], agent_name="test")
        assert result == [t1, t3]


class TestApplyToolApproval:
    """Tests for apply_tool_approval."""

    def test_wraps_named_tools_only(self):
        """Only the tool whose name is in approval_names gets wrapped."""
        t1, t2, t3 = _make_tool("a"), _make_tool("b", with_func=True), _make_tool("c")
        original_func = lambda **kw: "original"
        t2.func = original_func

        result = apply_tool_approval([t1, t2, t3], ["b"], agent_name="test")

        # t1 and t3 are passed through unchanged
        assert result[0] is t1
        assert result[2] is t3
        # t2 is the same object but its func has been replaced
        assert result[1] is t2
        assert result[1].func is not original_func

    def test_empty_approval_list_returns_unchanged(self):
        """No approval names means the original list is returned."""
        tools = [_make_tool("x")]
        result = apply_tool_approval(tools, [], agent_name="test")
        assert result is tools

    def test_unknown_approval_tool_logs_warning(self):
        """Approving a name not in the tool list logs a warning."""
        t1 = _make_tool("a")
        with patch("deep_agent.src.infrastructure.tool_access.logger") as mock_logger:
            result = apply_tool_approval([t1], ["nonexistent"], agent_name="test")
            mock_logger.warning.assert_called_once()
            assert mock_logger.warning.call_args[1]["tool"] == "nonexistent"
        # t1 is still in the result unchanged
        assert result == [t1]

    def test_all_tools_wrapped(self):
        """When all tool names are in approval_names, all get wrapped."""
        t1, t2 = _make_tool("a", with_func=True), _make_tool("b", with_func=True)
        orig_a = lambda **kw: "orig_a"
        orig_b = lambda **kw: "orig_b"
        t1.func = orig_a
        t2.func = orig_b

        result = apply_tool_approval([t1, t2], ["a", "b"], agent_name="test")

        assert result[0].func is not orig_a
        assert result[1].func is not orig_b

    def test_wrapped_tool_preserves_name(self):
        """The wrapped tool retains the original .name."""
        t1 = _make_tool("search", with_func=True)
        t1.func = lambda **kw: "original"

        result = apply_tool_approval([t1], ["search"], agent_name="test")

        assert result[0].name == "search"


class TestWrapToolWithApproval:
    """Tests for _wrap_tool_with_approval."""

    def test_interrupt_called_on_invocation(self):
        """Invoking the wrapped sync tool triggers interrupt with HITL payload."""
        tool = _make_tool("run_query", with_func=True)
        tool.func = lambda **kw: "result"

        with patch(
            "deep_agent.src.infrastructure.tool_access.interrupt",
            return_value=[{"type": "approve"}],
        ) as mock_interrupt:
            wrapped = _wrap_tool_with_approval(tool, agent_name="analyst")
            wrapped.func()
            mock_interrupt.assert_called_once()
            payload = mock_interrupt.call_args[0][0]
            assert isinstance(payload, dict)
            assert "action_requests" in payload
            assert payload["action_requests"][0]["name"] == "run_query"
            assert "analyst" in payload["action_requests"][0]["args"]["agent"]

    def test_approved_executes_original(self):
        """When frontend sends approve decision, the original function runs."""
        original_func = MagicMock(return_value="query_result")
        tool = _make_tool("run_query", with_func=True)
        tool.func = original_func

        with patch(
            "deep_agent.src.infrastructure.tool_access.interrupt",
            return_value=[{"type": "approve"}],
        ):
            wrapped = _wrap_tool_with_approval(tool, agent_name="analyst")
            result = wrapped.func(sql="SELECT 1")
            original_func.assert_called_once_with(sql="SELECT 1")
            assert result == "query_result"

    def test_rejected_returns_message(self):
        """When frontend sends reject decision, a rejection message is returned."""
        tool = _make_tool("dangerous_op", with_func=True)
        tool.func = lambda **kw: "should not run"

        with patch(
            "deep_agent.src.infrastructure.tool_access.interrupt",
            return_value=[{"type": "reject", "message": "No"}],
        ):
            wrapped = _wrap_tool_with_approval(tool, agent_name="analyst")
            result = wrapped.func()
            assert result == "Tool 'dangerous_op' was rejected by the user."

    def test_case_insensitive_approval(self):
        """String 'approved' still works for backward compat / testing."""
        original_func = MagicMock(return_value="ok")
        tool = _make_tool("action", with_func=True)
        tool.func = original_func

        with patch(
            "deep_agent.src.infrastructure.tool_access.interrupt",
            return_value="Approved",
        ):
            wrapped = _wrap_tool_with_approval(tool, agent_name="analyst")
            result = wrapped.func()
            original_func.assert_called_once()
            assert result == "ok"


class TestMigrateToolsField:
    """Tests for migrate_tools_field."""

    def test_tools_migrated_to_allowed_tools(self):
        """Config with 'tools' and no 'allowed_tools' gets migrated."""
        config = {"tools": ["a", "b"], "name": "analyst"}
        result = migrate_tools_field(config, agent_name="analyst")

        assert result["allowed_tools"] == ["a", "b"]
        assert "tools" not in result

    def test_both_present_raises_error(self):
        """Config with both 'tools' and 'allowed_tools' raises AppException."""
        config = {"tools": ["a"], "allowed_tools": ["b"]}
        with pytest.raises(AppException) as exc_info:
            migrate_tools_field(config, agent_name="analyst")
        assert exc_info.value.error_code == ErrorCodes.CONFIGURATION_VALIDATION_ERROR

    def test_neither_present_is_noop(self):
        """Config with neither key is unchanged."""
        config = {"name": "analyst", "description": "Test"}
        result = migrate_tools_field(config, agent_name="analyst")
        assert result == {"name": "analyst", "description": "Test"}

    def test_allowed_tools_only_is_noop(self):
        """Config with only 'allowed_tools' is unchanged."""
        config = {"allowed_tools": ["a", "b"]}
        result = migrate_tools_field(config, agent_name="analyst")
        assert result == {"allowed_tools": ["a", "b"]}
        assert "tools" not in result
