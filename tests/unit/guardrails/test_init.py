"""Unit tests for deep_agent.src.guardrails __init__ public API."""

from unittest.mock import MagicMock, patch

import pytest

import deep_agent.src.guardrails as guardrails_mod
from deep_agent.src.guardrails import (
    ContentSafetyError,
    InputContentSafetyError,
    ToolContentSafetyError,
    close_guardrails_circuit,
    disable_guardrails_runtime,
    get_guardrails_config,
    guardrails_circuit_state,
    init_guardrails,
)


@pytest.fixture(autouse=True)
def reset_guardrails_state():
    """Reset global state between tests."""
    original_config = guardrails_mod._config
    original_state = guardrails_mod._circuit_state
    original_at = guardrails_mod._circuit_opened_at
    original_reason = guardrails_mod._circuit_reason
    guardrails_mod._config = None
    guardrails_mod._circuit_state = "closed"
    guardrails_mod._circuit_opened_at = 0.0
    guardrails_mod._circuit_reason = ""
    yield
    guardrails_mod._config = original_config
    guardrails_mod._circuit_state = original_state
    guardrails_mod._circuit_opened_at = original_at
    guardrails_mod._circuit_reason = original_reason


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
        cfg = MagicMock()
        init_guardrails(cfg)
        assert get_guardrails_config() is cfg

    def test_init_guardrails_overwrites_previous(self):
        cfg1, cfg2 = MagicMock(), MagicMock()
        init_guardrails(cfg1)
        init_guardrails(cfg2)
        assert get_guardrails_config() is cfg2


class TestCircuitBreaker:
    def test_disable_opens_circuit(self):
        cfg = MagicMock()
        init_guardrails(cfg)
        assert get_guardrails_config() is cfg
        disable_guardrails_runtime(reason="test")
        assert get_guardrails_config() is None
        assert guardrails_circuit_state() == "open"

    def test_disable_is_idempotent(self):
        disable_guardrails_runtime(reason="first")
        disable_guardrails_runtime(reason="second")
        assert guardrails_mod._circuit_state == "open"

    def test_config_not_returned_when_circuit_open(self):
        cfg = MagicMock(enabled=True)
        init_guardrails(cfg)
        guardrails_mod._circuit_state = "open"
        guardrails_mod._circuit_opened_at = guardrails_mod._time()
        assert get_guardrails_config() is None

    def test_circuit_transitions_to_half_open_after_cooldown(self):
        cfg = MagicMock(enabled=True)
        init_guardrails(cfg)
        disable_guardrails_runtime(reason="test")
        assert get_guardrails_config() is None

        with patch.object(
            guardrails_mod,
            "_time",
            return_value=guardrails_mod._circuit_opened_at
            + guardrails_mod._CIRCUIT_COOLDOWN_SECONDS,
        ):
            assert guardrails_circuit_state() == "half-open"
            result = get_guardrails_config()
            assert result is cfg
            assert guardrails_mod._circuit_state == "half-open"

    def test_close_circuit_re_enables_guardrails(self):
        cfg = MagicMock(enabled=True)
        init_guardrails(cfg)
        disable_guardrails_runtime(reason="test")
        assert get_guardrails_config() is None

        close_guardrails_circuit()
        assert guardrails_circuit_state() == "closed"
        assert get_guardrails_config() is cfg

    def test_close_is_idempotent(self):
        close_guardrails_circuit()
        assert guardrails_circuit_state() == "closed"

    def test_circuit_reopens_on_second_failure(self):
        cfg = MagicMock(enabled=True)
        init_guardrails(cfg)

        disable_guardrails_runtime(reason="first failure")
        assert guardrails_circuit_state() == "open"

        close_guardrails_circuit()
        assert guardrails_circuit_state() == "closed"

        disable_guardrails_runtime(reason="second failure")
        assert guardrails_circuit_state() == "open"
        assert get_guardrails_config() is None

    def test_enabled_false_does_not_run_guardian(self):
        assert get_guardrails_config() is None

    def test_enabled_true_runs_guardian(self):
        cfg = MagicMock(enabled=True)
        init_guardrails(cfg)
        assert get_guardrails_config() is cfg
