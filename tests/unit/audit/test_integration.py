"""Unit tests for platform audit middleware builder integration."""

from unittest.mock import MagicMock, patch

from deep_agent.src.agent.config.middleware import ResolvedMiddlewareConfig
from deep_agent.src.infrastructure.middleware import build_middleware_list
from deep_agent.src.audit.middleware import AuditMiddleware


class TestBuildMiddlewareListAudit:
    def test_includes_audit_middleware_when_enabled(self):
        resolved = ResolvedMiddlewareConfig(summarization_tool_enabled=False)
        with (
            patch("deep_agent.src.infrastructure.middleware.settings") as mock_settings,
            patch(
                "deep_agent.src.audit.config.is_audit_enabled",
                return_value=True,
            ),
        ):
            mock_settings.MIDDLEWARE_ENABLED = True
            result = build_middleware_list(
                resolved, mcp_tool_names=frozenset({"tool_a"})
            )
        assert isinstance(result[0], AuditMiddleware)
        assert result[0]._mcp_tool_names == frozenset({"tool_a"})

    def test_no_audit_middleware_when_disabled(self):
        resolved = ResolvedMiddlewareConfig(summarization_tool_enabled=False)
        with (
            patch("deep_agent.src.infrastructure.middleware.settings") as mock_settings,
            patch(
                "deep_agent.src.audit.config.is_audit_enabled",
                return_value=False,
            ),
        ):
            mock_settings.MIDDLEWARE_ENABLED = True
            result = build_middleware_list(resolved)
        assert not any(isinstance(m, AuditMiddleware) for m in result)

    def test_audit_middleware_when_master_middleware_disabled(self):
        resolved = ResolvedMiddlewareConfig(summarization_tool_enabled=True)
        mock_mw = MagicMock()
        with (
            patch("deep_agent.src.infrastructure.middleware.settings") as mock_settings,
            patch(
                "deep_agent.src.audit.config.is_audit_enabled",
                return_value=True,
            ),
            patch(
                "deep_agent.src.infrastructure.middleware._build_summarization_tool_middleware",
                return_value=mock_mw,
            ),
        ):
            mock_settings.MIDDLEWARE_ENABLED = False
            result = build_middleware_list(resolved)
        assert isinstance(result[0], AuditMiddleware)
        assert result == [result[0]]
