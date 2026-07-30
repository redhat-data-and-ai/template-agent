"""Unit tests for deep_agent.src.guardrails __init__ public API."""

import pytest

import deep_agent.src.guardrails as guardrails_mod
from deep_agent.src.guardrails import (
    ContentSafetyError,
    InputContentSafetyError,
    ToolContentSafetyError,
    disable_guardrails_runtime,
    get_guardrails_config,
    init_guardrails,
)


@pytest.fixture(autouse=True)
def reset_guardrails_state():
    """Reset global _config and _runtime_disabled between tests."""
    original_config = guardrails_mod._config
    original_disabled = guardrails_mod._runtime_disabled
    guardrails_mod._config = None
    guardrails_mod._runtime_disabled = False
    yield
    guardrails_mod._config = original_config
    guardrails_mod._runtime_disabled = original_disabled


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


class TestDisableGuardrailsRuntime:
    def test_get_guardrails_config_returns_none_after_disable(self):
        from unittest.mock import MagicMock

        init_guardrails(MagicMock())
        assert get_guardrails_config() is not None
        disable_guardrails_runtime(reason="test")
        assert get_guardrails_config() is None

    def test_disable_is_idempotent(self):
        """Second call must not raise and must not log again."""
        disable_guardrails_runtime(reason="first")
        disable_guardrails_runtime(reason="second")  # must not raise
        assert guardrails_mod._runtime_disabled is True

    def test_config_not_returned_when_runtime_disabled_even_if_set(self):
        from unittest.mock import MagicMock

        cfg = MagicMock(enabled=True)
        init_guardrails(cfg)
        guardrails_mod._runtime_disabled = True
        assert get_guardrails_config() is None

    def test_enabled_false_does_not_run_guardian(self):
        """enabled=False: guardrails never initialised, get_guardrails_config stays None."""
        # Simulate setup_guardian_guardrails() short-circuit: init_guardrails never called.
        assert get_guardrails_config() is None

    def test_enabled_true_runs_guardian(self):
        """enabled=True: after init_guardrails the config is returned."""
        from unittest.mock import MagicMock

        cfg = MagicMock(enabled=True)
        init_guardrails(cfg)
        assert get_guardrails_config() is cfg
