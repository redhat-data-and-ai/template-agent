"""Unit tests for setup_guardian_guardrails and setup_token_budget_tracking."""

import sys
from unittest.mock import MagicMock, patch

from deep_agent.aegra import telemetry
from deep_agent.aegra.telemetry import (
    setup_guardian_guardrails,
    setup_token_budget_tracking,
)


class TestSetupGuardianGuardrails:
    def setup_method(self):
        telemetry._guardian_initialized = False

    def test_idempotent_skips_all_calls(self):
        telemetry._guardian_initialized = True
        setup_guardian_guardrails()
        assert telemetry._guardian_initialized is True

    def test_disabled_in_config_returns_early(self):
        mock_cfg = MagicMock()
        mock_cfg.enabled = False
        mock_ac = MagicMock()
        mock_ac.get_guardrails_config.return_value = mock_cfg
        mock_settings = MagicMock()
        mock_settings.GUARDIAN_API_BASE = "http://guardian.example.com"

        with (
            patch("deep_agent.src.agent.config.agent_config", mock_ac),
            patch("deep_agent.src.guardrails.init_guardrails") as mock_init,
            patch("deep_agent.src.settings.settings", mock_settings),
        ):
            setup_guardian_guardrails()

        mock_init.assert_not_called()

    def test_no_api_base_returns_early(self):
        mock_cfg = MagicMock()
        mock_cfg.enabled = True
        mock_ac = MagicMock()
        mock_ac.get_guardrails_config.return_value = mock_cfg
        mock_settings = MagicMock()
        mock_settings.GUARDIAN_API_BASE = ""

        with (
            patch("deep_agent.src.agent.config.agent_config", mock_ac),
            patch("deep_agent.src.guardrails.init_guardrails") as mock_init,
            patch("deep_agent.src.settings.settings", mock_settings),
        ):
            setup_guardian_guardrails()

        mock_init.assert_not_called()

    def test_full_path_calls_init_and_registers_callback(self):
        mock_cfg = MagicMock()
        mock_cfg.enabled = True
        mock_ac = MagicMock()
        mock_ac.get_guardrails_config.return_value = mock_cfg
        mock_settings = MagicMock()
        mock_settings.GUARDIAN_API_BASE = "http://guardian.example.com"
        mock_init = MagicMock()
        mock_lc_context = MagicMock()
        mock_guardrails_callback = MagicMock()

        with (
            patch("deep_agent.src.agent.config.agent_config", mock_ac),
            patch("deep_agent.src.guardrails.init_guardrails", mock_init),
            patch("deep_agent.src.settings.settings", mock_settings),
            patch.dict(
                sys.modules,
                {
                    "langchain_core.tracers.context": mock_lc_context,
                    "deep_agent.src.guardrails.callback": mock_guardrails_callback,
                },
            ),
        ):
            setup_guardian_guardrails()

        mock_init.assert_called_once_with(mock_cfg)
        mock_lc_context.register_configure_hook.assert_called_once()

    def test_import_error_logs_warning_no_crash(self):
        mock_cfg = MagicMock()
        mock_cfg.enabled = True
        mock_ac = MagicMock()
        mock_ac.get_guardrails_config.return_value = mock_cfg
        mock_settings = MagicMock()
        mock_settings.GUARDIAN_API_BASE = "http://guardian.example.com"
        mock_init = MagicMock()

        with (
            patch("deep_agent.src.agent.config.agent_config", mock_ac),
            patch("deep_agent.src.guardrails.init_guardrails", mock_init),
            patch("deep_agent.src.settings.settings", mock_settings),
            patch.dict(sys.modules, {"langchain_core.tracers.context": None}),
        ):
            setup_guardian_guardrails()

        mock_init.assert_called_once_with(mock_cfg)


class TestSetupTokenBudgetTracking:
    def setup_method(self):
        telemetry._token_budget_tracing_initialized = False

    def test_idempotent_skips_all_calls(self):
        telemetry._token_budget_tracing_initialized = True
        setup_token_budget_tracking()
        assert telemetry._token_budget_tracing_initialized is True

    def test_not_active_returns_early(self):
        mock_budget_cfg = MagicMock()
        mock_budget_cfg.is_active = False
        mock_ac = MagicMock()
        mock_ac.get_token_budget_config.return_value = mock_budget_cfg
        mock_lc_context = MagicMock()

        with (
            patch("deep_agent.src.agent.config.agent_config", mock_ac),
            patch.dict(
                sys.modules, {"langchain_core.tracers.context": mock_lc_context}
            ),
        ):
            setup_token_budget_tracking()

        mock_lc_context.register_configure_hook.assert_not_called()
