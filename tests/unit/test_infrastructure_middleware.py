"""Unit tests for the middleware builder module."""

from unittest.mock import MagicMock, patch

import pytest

from deep_agent.src.agent.config.middleware import ResolvedMiddlewareConfig
from deep_agent.src.infrastructure.middleware import (
    _import_middleware,
    build_excluded_middleware,
    build_middleware_list,
    resolve_memory_param,
)


class TestBuildMiddlewareList:
    """Test middleware instance construction from resolved config."""

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
        with patch(
            "deep_agent.src.infrastructure.middleware.settings"
        ) as mock_settings:
            mock_settings.MIDDLEWARE_ENABLED = True
            result = build_middleware_list(resolved)
        assert result == []

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
        assert len(result) == 1


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
