"""Unit tests for backend module."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deep_agent.src.infrastructure.backend import (
    _base_python,
    _build_env,
    _get_assistant_id_from_config,
    _safe_namespace_assistant,
    _safe_namespace_org,
    _safe_namespace_user,
    _STORE_NAMESPACE_FACTORIES,
)


class TestBasePython:
    def test_returns_string(self):
        result = _base_python()
        assert isinstance(result, str)
        assert "python" in result.lower()


class TestBuildEnv:
    def test_contains_virtual_env(self, tmp_path):
        env = _build_env(tmp_path)
        assert env["VIRTUAL_ENV"] == str(tmp_path)

    def test_contains_path(self, tmp_path):
        env = _build_env(tmp_path)
        assert str(tmp_path) in env["PATH"]

    def test_extra_env_overrides(self, tmp_path):
        env = _build_env(tmp_path, extra={"MY_VAR": "my_val"})
        assert env["MY_VAR"] == "my_val"

    def test_passthrough_vars(self, tmp_path):
        with patch.dict(os.environ, {"HOME": "/test/home", "USER": "tester"}):
            env = _build_env(tmp_path)
            assert env.get("HOME") == "/test/home"
            assert env.get("USER") == "tester"


def _make_ctx(server_info=None, config=None, context=None):
    """Build a minimal ctx mock for namespace tests."""
    ctx = MagicMock()
    ctx.runtime.server_info = server_info
    ctx.runtime.config = config
    if context is not None:
        ctx.runtime.context = context
    return ctx


def _make_server_info(assistant_id="", user_identity=""):
    """Build a server_info mock with optional assistant_id and user."""
    si = MagicMock()
    si.assistant_id = assistant_id
    if user_identity:
        si.user = MagicMock()
        si.user.identity = user_identity
    else:
        si.user = None
    return si


class TestGetAssistantIdFromConfig:
    """Tests for _get_assistant_id_from_config."""

    def test_returns_assistant_id_from_metadata(self):
        ctx = _make_ctx(config={"metadata": {"assistant_id": "agent-42"}})
        assert _get_assistant_id_from_config(ctx) == "agent-42"

    def test_returns_default_when_config_is_none(self):
        ctx = _make_ctx(config=None)
        assert _get_assistant_id_from_config(ctx) == "default"

    def test_returns_default_when_config_missing_metadata(self):
        ctx = _make_ctx(config={"other_key": "val"})
        assert _get_assistant_id_from_config(ctx) == "default"

    def test_returns_default_when_metadata_has_no_assistant_id(self):
        ctx = _make_ctx(config={"metadata": {}})
        assert _get_assistant_id_from_config(ctx) == "default"

    def test_returns_default_when_assistant_id_is_empty_string(self):
        ctx = _make_ctx(config={"metadata": {"assistant_id": ""}})
        assert _get_assistant_id_from_config(ctx) == "default"

    def test_returns_default_when_config_is_not_dict(self):
        ctx = _make_ctx(config="not-a-dict")
        assert _get_assistant_id_from_config(ctx) == "default"

    def test_returns_default_when_runtime_has_no_config_attr(self):
        ctx = MagicMock()
        del ctx.runtime.config
        assert _get_assistant_id_from_config(ctx) == "default"

    def test_returns_default_when_metadata_is_none(self):
        ctx = _make_ctx(config={"metadata": None})
        assert _get_assistant_id_from_config(ctx) == "default"

    def test_returns_default_when_metadata_is_not_dict(self):
        ctx = _make_ctx(config={"metadata": ["not", "a", "dict"]})
        assert _get_assistant_id_from_config(ctx) == "default"


class TestSafeNamespaceUser:
    """Tests for _safe_namespace_user."""

    def test_returns_assistant_id_and_user_identity_from_server_info(self):
        si = _make_server_info(assistant_id="asst-1", user_identity="user@example.com")
        ctx = _make_ctx(server_info=si)
        assert _safe_namespace_user(ctx) == ("asst-1", "user@example.com")

    def test_returns_only_assistant_id_when_user_is_none(self):
        si = _make_server_info(assistant_id="asst-1", user_identity="")
        ctx = _make_ctx(server_info=si)
        assert _safe_namespace_user(ctx) == ("asst-1",)

    def test_returns_only_assistant_id_when_user_identity_empty(self):
        si = MagicMock()
        si.assistant_id = "asst-1"
        si.user = MagicMock()
        si.user.identity = ""
        ctx = _make_ctx(server_info=si)
        assert _safe_namespace_user(ctx) == ("asst-1",)

    def test_returns_only_assistant_id_when_server_info_has_no_user_attr(self):
        si = MagicMock(spec=[])
        si.assistant_id = "asst-1"
        ctx = _make_ctx(server_info=si)
        assert _safe_namespace_user(ctx) == ("asst-1",)

    def test_falls_back_to_config_when_server_info_is_none(self):
        ctx = _make_ctx(
            server_info=None,
            config={"metadata": {"assistant_id": "cfg-agent"}},
        )
        assert _safe_namespace_user(ctx) == ("cfg-agent",)

    def test_falls_back_to_config_when_assistant_id_empty(self):
        si = _make_server_info(assistant_id="", user_identity="user@x.com")
        ctx = _make_ctx(
            server_info=si,
            config={"metadata": {"assistant_id": "cfg-agent"}},
        )
        assert _safe_namespace_user(ctx) == ("cfg-agent",)

    def test_falls_back_to_default_when_no_server_info_and_no_config(self):
        ctx = _make_ctx(server_info=None, config=None)
        assert _safe_namespace_user(ctx) == ("default",)


class TestSafeNamespaceAssistant:
    """Tests for _safe_namespace_assistant."""

    def test_returns_assistant_id_from_server_info(self):
        si = _make_server_info(assistant_id="asst-2", user_identity="ignored")
        ctx = _make_ctx(server_info=si)
        assert _safe_namespace_assistant(ctx) == ("asst-2",)

    def test_falls_back_to_config_when_server_info_is_none(self):
        ctx = _make_ctx(
            server_info=None,
            config={"metadata": {"assistant_id": "cfg-asst"}},
        )
        assert _safe_namespace_assistant(ctx) == ("cfg-asst",)

    def test_falls_back_to_config_when_assistant_id_empty(self):
        si = _make_server_info(assistant_id="")
        ctx = _make_ctx(
            server_info=si,
            config={"metadata": {"assistant_id": "cfg-asst"}},
        )
        assert _safe_namespace_assistant(ctx) == ("cfg-asst",)

    def test_falls_back_to_default_when_no_config(self):
        ctx = _make_ctx(server_info=None, config=None)
        assert _safe_namespace_assistant(ctx) == ("default",)


class TestSafeNamespaceOrg:
    """Tests for _safe_namespace_org."""

    def test_returns_org_id(self):
        context = MagicMock()
        context.org_id = "org-123"
        ctx = _make_ctx(context=context)
        assert _safe_namespace_org(ctx) == ("org-123",)


class TestStoreNamespaceFactories:
    """Tests for the _STORE_NAMESPACE_FACTORIES mapping."""

    def test_contains_all_expected_keys(self):
        assert set(_STORE_NAMESPACE_FACTORIES.keys()) == {"user", "assistant", "org"}

    def test_user_maps_to_safe_namespace_user(self):
        assert _STORE_NAMESPACE_FACTORIES["user"] is _safe_namespace_user

    def test_assistant_maps_to_safe_namespace_assistant(self):
        assert _STORE_NAMESPACE_FACTORIES["assistant"] is _safe_namespace_assistant

    def test_org_maps_to_safe_namespace_org(self):
        assert _STORE_NAMESPACE_FACTORIES["org"] is _safe_namespace_org
