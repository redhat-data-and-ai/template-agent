"""Unit tests for the middleware builder module."""

from unittest.mock import MagicMock, patch

import pytest

from deep_agent.src.agent.config.middleware import ResolvedMiddlewareConfig
from deep_agent.src.infrastructure.middleware import (
    _build_model_fallback,
    _build_summarization_tool_middleware,
    _import_middleware,
    build_excluded_middleware,
    build_middleware_list,
    resolve_memory_param,
)


class TestBuildMiddlewareList:
    """Test middleware instance construction from resolved config."""

    @pytest.fixture(autouse=True)
    def _disable_audit(self):
        with patch(
            "deep_agent.src.audit.config.is_audit_enabled",
            return_value=False,
        ):
            yield

    def test_returns_empty_when_master_switch_off(self):
        resolved = ResolvedMiddlewareConfig(summarization_tool_enabled=True)
        with patch(
            "deep_agent.src.infrastructure.middleware.settings"
        ) as mock_settings:
            mock_settings.MIDDLEWARE_ENABLED = False
            result = build_middleware_list(resolved)
        assert result == []

    def test_includes_summarization_tool_when_enabled(self):
        resolved = ResolvedMiddlewareConfig(summarization_tool_enabled=True)
        mock_mw = MagicMock()
        with (
            patch("deep_agent.src.infrastructure.middleware.settings") as mock_settings,
            patch(
                "deep_agent.src.infrastructure.middleware._build_summarization_tool_middleware",
                return_value=mock_mw,
            ),
        ):
            mock_settings.MIDDLEWARE_ENABLED = True
            result = build_middleware_list(resolved)
        assert mock_mw in result

    def test_excludes_summarization_tool_when_disabled(self):
        resolved = ResolvedMiddlewareConfig(
            summarization_tool_enabled=False, extra_middleware=[]
        )
        with (
            patch("deep_agent.src.infrastructure.middleware.settings") as mock_settings,
            patch(
                "deep_agent.src.infrastructure.middleware._build_summarization_tool_middleware",
            ) as build_sum,
        ):
            mock_settings.MIDDLEWARE_ENABLED = True
            result = build_middleware_list(resolved)
            build_sum.assert_not_called()
        # Default guardrails (model/tool limits + model retry) still apply.
        assert len(result) == 3

    def test_includes_extra_middleware(self):
        resolved = ResolvedMiddlewareConfig(
            summarization_tool_enabled=False,
            extra_middleware=[
                "tests.unit.test_infrastructure_middleware:_DummyMiddleware"
            ],
        )
        with patch(
            "deep_agent.src.infrastructure.middleware.settings"
        ) as mock_settings:
            mock_settings.MIDDLEWARE_ENABLED = True
            result = build_middleware_list(resolved)
        assert len(result) == 4
        assert any(isinstance(m, _DummyMiddleware) for m in result)


class TestBuildExcludedMiddleware:
    """Test excluded middleware list generation."""

    def test_empty_when_all_enabled(self):
        resolved = ResolvedMiddlewareConfig(
            patch_tool_calls_enabled=True, excluded_middleware=[]
        )
        result = build_excluded_middleware(resolved)
        assert result == []

    def test_includes_patch_tool_calls_when_disabled(self):
        resolved = ResolvedMiddlewareConfig(
            patch_tool_calls_enabled=False, excluded_middleware=[]
        )
        result = build_excluded_middleware(resolved)
        assert "PatchToolCallsMiddleware" in result

    def test_preserves_profile_exclusions(self):
        resolved = ResolvedMiddlewareConfig(
            patch_tool_calls_enabled=True,
            excluded_middleware=["SomeCustomMiddleware"],
        )
        result = build_excluded_middleware(resolved)
        assert "SomeCustomMiddleware" in result


class TestResolveMemoryParam:
    """Test memory parameter resolution for create_deep_agent()."""

    def test_returns_none_when_master_disabled(self):
        resolved = ResolvedMiddlewareConfig(memory_enabled=True)
        with patch(
            "deep_agent.src.infrastructure.middleware.settings"
        ) as mock_settings:
            mock_settings.MIDDLEWARE_ENABLED = False
            result = resolve_memory_param(resolved)
        assert result is None

    def test_returns_none_when_memory_disabled(self):
        resolved = ResolvedMiddlewareConfig(memory_enabled=False)
        with patch(
            "deep_agent.src.infrastructure.middleware.settings"
        ) as mock_settings:
            mock_settings.MIDDLEWARE_ENABLED = True
            result = resolve_memory_param(resolved)
        assert result is None

    def test_returns_namespaces_when_enabled(self):
        resolved = ResolvedMiddlewareConfig(
            memory_enabled=True, memory_namespaces=["user_mem", "shared"]
        )
        with patch(
            "deep_agent.src.infrastructure.middleware.settings"
        ) as mock_settings:
            mock_settings.MIDDLEWARE_ENABLED = True
            result = resolve_memory_param(resolved)
        assert result == ["user_mem", "shared"]


class TestImportMiddleware:
    """Test dynamic middleware importing."""

    def test_invalid_path_without_colon(self):
        result = _import_middleware("no_colon_here")
        assert result is None

    def test_nonexistent_module(self):
        result = _import_middleware("nonexistent.module:Class")
        assert result is None

    def test_valid_import(self):
        result = _import_middleware(
            "tests.unit.test_infrastructure_middleware:_DummyMiddleware"
        )
        assert result is not None


class _DummyMiddleware:
    """Test fixture — a no-op middleware class."""

    pass


class TestBuildModelFallbackEdgeCases:
    """Test edge cases for _build_model_fallback."""

    def test_exception_in_init_returns_none(self):
        with patch(
            "langchain.agents.middleware.ModelFallbackMiddleware",
            side_effect=Exception("model init failed"),
        ):
            result = _build_model_fallback("some-model")
        assert result is None


class TestBuildSummarizationToolMiddlewareEdgeCases:
    """Test edge cases for _build_summarization_tool_middleware."""

    def test_none_model_returns_none(self):
        result = _build_summarization_tool_middleware(model=None, backend=MagicMock())
        assert result is None

    def test_none_backend_returns_none(self):
        result = _build_summarization_tool_middleware(model=MagicMock(), backend=None)
        assert result is None

    def test_exception_during_creation_returns_none(self):
        with patch(
            "deepagents.middleware.summarization.create_summarization_tool_middleware",
            side_effect=Exception("creation error"),
        ):
            result = _build_summarization_tool_middleware(
                model=MagicMock(), backend=MagicMock()
            )
        assert result is None
