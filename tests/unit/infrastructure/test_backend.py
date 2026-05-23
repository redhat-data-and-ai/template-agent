"""Unit tests for backend module."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deep_agent.src.infrastructure.backend import (
    _base_python,
    _build_env,
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
