"""Tests for Langfuse mock graceful degradation."""

import os
import subprocess
import sys

from template_agent.utils.langfuse_mock import (
    NoOpCallbackHandler,
    NoOpLangfuse,
    install_langfuse_mock,
    is_langfuse_configured,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


class TestIsLangfuseConfigured:
    """Tests for Langfuse credential detection."""

    def test_returns_false_when_all_missing(self):
        assert not is_langfuse_configured(None, None, None)

    def test_returns_false_when_any_empty(self):
        assert not is_langfuse_configured("pk-test", "sk-test", "")
        assert not is_langfuse_configured("pk-test", "", "https://cloud.langfuse.com")
        assert not is_langfuse_configured("", "sk-test", "https://cloud.langfuse.com")

    def test_returns_false_when_partially_configured(self):
        assert not is_langfuse_configured("pk-test", None, None)
        assert not is_langfuse_configured("pk-test", "sk-test", None)

    def test_returns_true_when_all_configured(self):
        assert is_langfuse_configured(
            "pk-test", "sk-test", "https://cloud.langfuse.com"
        )


class TestNoOpImplementations:
    """Tests for no-op Langfuse stand-ins."""

    def test_noop_langfuse_score_is_safe(self):
        client = NoOpLangfuse(environment="development")
        assert client.score(trace_id="run-1", name="quality", value=4.5) is None

    def test_noop_callback_handler_is_base_callback_handler(self):
        handler = NoOpCallbackHandler(trace_name="template-agent")
        assert isinstance(handler, NoOpCallbackHandler)


class TestInstallLangfuseMock:
    """Tests for sys.modules injection."""

    def test_install_registers_mock_modules(self):
        modules_snapshot = {
            name: sys.modules.get(name) for name in ("langfuse", "langfuse.callback")
        }
        for name in ("langfuse", "langfuse.callback"):
            sys.modules.pop(name, None)

        try:
            install_langfuse_mock()

            from langfuse import Langfuse
            from langfuse.callback import CallbackHandler

            assert Langfuse is NoOpLangfuse
            assert CallbackHandler is NoOpCallbackHandler
        finally:
            for name, module in modules_snapshot.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module


class TestPackageBootstrap:
    """Integration tests for package-level mock installation."""

    def test_feedback_import_uses_mock_without_credentials(self):
        env = os.environ.copy()
        env.update(
            {
                "LANGFUSE_PUBLIC_KEY": "",
                "LANGFUSE_SECRET_KEY": "",
                "LANGFUSE_BASE_URL": "",
            }
        )
        script = """
import template_agent
from langfuse import Langfuse
from template_agent.src.routes.feedback import client
from template_agent.src.core.manager import langfuse_handler
assert isinstance(client, Langfuse)
assert client.score(trace_id="run-1", name="quality", value=1.0) is None
print("mock-ok")
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0, result.stderr
        assert "mock-ok" in result.stdout

    def test_package_does_not_install_mock_when_configured(self):
        env = os.environ.copy()
        env.update(
            {
                "LANGFUSE_PUBLIC_KEY": "pk-test",
                "LANGFUSE_SECRET_KEY": "sk-test",
                "LANGFUSE_BASE_URL": "https://cloud.langfuse.com",
            }
        )
        script = """
import importlib
import sys

for module_name in list(sys.modules):
    if module_name == "template_agent" or module_name.startswith("template_agent."):
        del sys.modules[module_name]
for module_name in ("langfuse", "langfuse.callback"):
    sys.modules.pop(module_name, None)

import template_agent
from langfuse import Langfuse
from template_agent.utils.langfuse_mock import NoOpLangfuse
assert Langfuse is not NoOpLangfuse
print("real-ok")
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0, result.stderr
        assert "real-ok" in result.stdout
