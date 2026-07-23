"""Unit tests for deep_agent.src.guardrails __init__ public API."""

import pytest

import deep_agent.src.guardrails as guardrails_mod
from deep_agent.src.guardrails import (
    ContentSafetyError,
    InputContentSafetyError,
    ToolContentSafetyError,
    get_guardrails_config,
    init_guardrails,
)


@pytest.fixture(autouse=True)
def reset_guardrails_config():
    """Reset global _config between tests."""
    original = guardrails_mod._config
    guardrails_mod._config = None
    yield
    guardrails_mod._config = original


class TestErrorHierarchy:
    def test_content_safety_error_is_value_error(self):
        assert issubclass(ContentSafetyError, ValueError)

    def test_input_content_safety_error_inherits(self):
        assert issubclass(InputContentSafetyError, ContentSafetyError)

    def test_tool_content_safety_error_inherits(self):
        assert issubclass(ToolContentSafetyError, ContentSafetyError)

    def test_errors_are_raiseable(self):
        with pytest.raises(InputContentSafetyError):
            raise InputContentSafetyError("blocked")
        with pytest.raises(ToolContentSafetyError):
            raise ToolContentSafetyError("blocked")


class TestInitGuardrails:
    def test_get_guardrails_config_returns_none_before_init(self):
        assert get_guardrails_config() is None

    def test_init_guardrails_stores_config(self):
        from unittest.mock import MagicMock

        cfg = MagicMock()
        init_guardrails(cfg)
        assert get_guardrails_config() is cfg

    def test_init_guardrails_overwrites_previous(self):
        from unittest.mock import MagicMock

        cfg1, cfg2 = MagicMock(), MagicMock()
        init_guardrails(cfg1)
        init_guardrails(cfg2)
        assert get_guardrails_config() is cfg2
