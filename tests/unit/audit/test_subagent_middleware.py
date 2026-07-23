"""Unit tests for audit middleware on subagents."""

from unittest.mock import MagicMock, patch

from deep_agent.src.infrastructure.subagents import _subagent_middleware
from deep_agent.src.audit.middleware import AuditMiddleware


class TestSubagentMiddleware:
    def test_includes_audit_when_enabled(self):
        tool = MagicMock()
        tool.name = "mcp_search"
        audit_mw = AuditMiddleware(
            mcp_tool_names=frozenset({"mcp_search"}),
            agent="researcher",
        )
        with (
            patch(
                "deep_agent.src.infrastructure.subagents.build_audit_middleware",
                return_value=audit_mw,
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.build_opa_middleware",
                return_value=None,
            ),
        ):
            result = _subagent_middleware("researcher", [tool], [])
        assert result is not None
        assert isinstance(result[0], AuditMiddleware)
        assert result[0]._agent == "researcher"
        assert "mcp_search" in result[0]._mcp_tool_names

    def test_returns_none_when_audit_and_opa_disabled_and_no_fallback(self):
        with (
            patch(
                "deep_agent.src.infrastructure.subagents.build_audit_middleware",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.build_opa_middleware",
                return_value=None,
            ),
        ):
            assert _subagent_middleware("researcher", [], []) is None

    def test_includes_opa_when_enabled(self):
        opa_mw = MagicMock(name="OPAMiddleware")
        with (
            patch(
                "deep_agent.src.infrastructure.subagents.build_audit_middleware",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.build_opa_middleware",
                return_value=opa_mw,
            ),
        ):
            result = _subagent_middleware("researcher", [], [])
        assert result == [opa_mw]

    def test_fallback_only_when_audit_disabled(self):
        fallback = MagicMock()
        with (
            patch(
                "deep_agent.src.infrastructure.subagents.build_audit_middleware",
                return_value=None,
            ),
            patch(
                "deep_agent.src.infrastructure.subagents.build_opa_middleware",
                return_value=None,
            ),
        ):
            result = _subagent_middleware("researcher", [], [fallback])
        assert result == [fallback]
