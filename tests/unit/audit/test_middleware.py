"""Unit tests for AuditMiddleware classification."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_agent.src.audit.middleware import (
    AuditMiddleware,
    classify_tool_call,
    _is_memory_write,
)


class TestMemoryWriteDetection:
    @pytest.mark.parametrize(
        ("tool", "args", "expected"),
        [
            ("edit_file", {"path": "/memories/notes.md"}, True),
            ("write_file", {"file_path": "memories/foo.txt"}, True),
            ("edit_file", {"path": "/reports/out.md"}, False),
            ("search_web", {"query": "test"}, False),
        ],
    )
    def test_is_memory_write(self, tool, args, expected):
        assert _is_memory_write(tool, args) is expected


class TestClassifyToolCallParity:
    """Orchestrator and subagent use the same classification rules."""

    @pytest.mark.parametrize(
        ("tool", "args", "mcp_names", "expected"),
        [
            ("task", {"subagent": "researcher"}, frozenset(), "subagent_delegation"),
            (
                "gitlab_search",
                {"q": "x"},
                frozenset({"gitlab_search"}),
                "mcp_tool_call",
            ),
            ("edit_file", {"path": "/memories/x.md"}, frozenset(), "memory_write"),
            ("calculate_bmi", {}, frozenset(), ""),
        ],
    )
    def test_shared_rules(self, tool, args, mcp_names, expected):
        assert classify_tool_call(tool, args, mcp_tool_names=mcp_names) == expected


class TestAuditMiddlewareClassification:
    def test_sync_llm_call_with_subagent(self):
        mw = AuditMiddleware(agent="researcher")
        request = MagicMock()
        request.model = "gemini-2.5-flash"
        request.messages = []
        handler = MagicMock(return_value=MagicMock())

        with patch(
            "deep_agent.src.audit.middleware.is_audit_enabled", return_value=True
        ):
            with patch("deep_agent.src.audit.middleware.emit_audit_event") as emit:
                mw.wrap_model_call(request, handler)
                assert emit.call_count == 2
                assert emit.call_args_list[0].args[0] == "llm_call"
                assert emit.call_args_list[0].kwargs["agent"] == "researcher"
                assert emit.call_args_list[0].kwargs["phase"] == "start"

    def test_orchestrator_llm_includes_agent(self):
        mw = AuditMiddleware()
        request = MagicMock()
        request.model = "gemini-2.5-flash"
        request.messages = []
        handler = MagicMock(return_value=MagicMock())

        with patch(
            "deep_agent.src.audit.middleware.is_audit_enabled", return_value=True
        ):
            with patch("deep_agent.src.audit.middleware.emit_audit_event") as emit:
                mw.wrap_model_call(request, handler)
                assert emit.call_args_list[0].kwargs["agent"] == "orchestrator"

    @pytest.mark.asyncio
    async def test_subagent_delegation(self):
        mw = AuditMiddleware(mcp_tool_names=frozenset())
        request = MagicMock()
        request.tool_call = {
            "name": "task",
            "args": {"subagent": "researcher"},
            "id": "1",
        }
        handler = AsyncMock(return_value=MagicMock())

        with patch(
            "deep_agent.src.audit.middleware.is_audit_enabled", return_value=True
        ):
            with patch("deep_agent.src.audit.middleware.emit_audit_event") as emit:
                await mw.awrap_tool_call(request, handler)
                emit.assert_called_once()
                assert emit.call_args.args[0] == "subagent_delegation"
                assert emit.call_args.kwargs["delegated_subagent"] == "researcher"
                assert emit.call_args.kwargs["agent"] == "orchestrator"

    @pytest.mark.asyncio
    async def test_mcp_tool_call_on_subagent(self):
        mw = AuditMiddleware(
            mcp_tool_names=frozenset({"gitlab_search"}),
            agent="researcher",
        )
        request = MagicMock()
        request.tool_call = {"name": "gitlab_search", "args": {"q": "x"}, "id": "1"}
        handler = AsyncMock(return_value=MagicMock())

        with patch(
            "deep_agent.src.audit.middleware.is_audit_enabled", return_value=True
        ):
            with patch("deep_agent.src.audit.middleware.emit_audit_event") as emit:
                await mw.awrap_tool_call(request, handler)
                emit.assert_called_once()
                assert emit.call_args.args[0] == "mcp_tool_call"
                assert emit.call_args.kwargs["agent"] == "researcher"

    @pytest.mark.asyncio
    async def test_skips_unclassified_tools_on_orchestrator(self):
        mw = AuditMiddleware(mcp_tool_names=frozenset())
        request = MagicMock()
        request.tool_call = {"name": "calculate_bmi", "args": {}, "id": "1"}
        handler = AsyncMock(return_value=MagicMock())

        with patch(
            "deep_agent.src.audit.middleware.is_audit_enabled", return_value=True
        ):
            with patch("deep_agent.src.audit.middleware.emit_audit_event") as emit:
                await mw.awrap_tool_call(request, handler)
                emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_unclassified_tools_on_subagent(self):
        mw = AuditMiddleware(mcp_tool_names=frozenset(), agent="researcher")
        request = MagicMock()
        request.tool_call = {"name": "calculate_bmi", "args": {}, "id": "1"}
        handler = AsyncMock(return_value=MagicMock())

        with patch(
            "deep_agent.src.audit.middleware.is_audit_enabled", return_value=True
        ):
            with patch("deep_agent.src.audit.middleware.emit_audit_event") as emit:
                await mw.awrap_tool_call(request, handler)
                emit.assert_not_called()
