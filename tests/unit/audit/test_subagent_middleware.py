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
        from deep_agent.aegra.mcp_runtime_tools import McpRuntimeToolsMiddleware

        assert result is not None
        assert isinstance(result[0], AuditMiddleware)
        assert result[0]._agent == "researcher"
        assert "mcp_search" in result[0]._mcp_tool_names
        assert any(isinstance(m, McpRuntimeToolsMiddleware) for m in result)

    def test_includes_runtime_tools_when_audit_and_opa_disabled(self):
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
            from deep_agent.aegra.mcp_runtime_tools import McpRuntimeToolsMiddleware

            result = _subagent_middleware("researcher", [], [])
            assert len(result) == 1
            assert isinstance(result[0], McpRuntimeToolsMiddleware)

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
        from deep_agent.aegra.mcp_runtime_tools import McpRuntimeToolsMiddleware

        assert result[0] is opa_mw
        assert isinstance(result[-1], McpRuntimeToolsMiddleware)

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
        from deep_agent.aegra.mcp_runtime_tools import McpRuntimeToolsMiddleware

        assert result[0] is fallback
        assert isinstance(result[-1], McpRuntimeToolsMiddleware)

    def test_runtime_middleware_uses_declared_yaml(self):
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
            result = _subagent_middleware(
                "researcher",
                [],
                [],
                declared_tools=["search"],
                declared_mcps=["acme-jira"],
            )
        from deep_agent.aegra.mcp_runtime_tools import McpRuntimeToolsMiddleware

        mw = result[-1]
        assert isinstance(mw, McpRuntimeToolsMiddleware)
        assert mw._allowlist == frozenset({"search"})
        assert mw._mcp_names == frozenset({"acme-jira"})
