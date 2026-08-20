"""Unit tests for OPA service — ControlResult, _apply_modes, _parse_result."""

from unittest.mock import patch

import pytest

from deep_agent.src.opa.service import (
    ControlResult,
    OpaResult,
    _apply_modes,
    _parse_result,
)


class TestControlResult:
    def test_creation_with_defaults(self):
        cr = ControlResult(id="BW-01", status="fail", mode="ENFORCE")
        assert cr.id == "BW-01"
        assert cr.status == "fail"
        assert cr.mode == "ENFORCE"
        assert cr.reason == ""

    def test_creation_with_reason(self):
        cr = ControlResult(id="TG-06", status="pass", mode="OFF", reason="ok")
        assert cr.reason == "ok"


class TestApplyModes:
    def test_empty_controls(self):
        result = _apply_modes([])
        assert result.allowed is True
        assert result.denial_reasons == []
        assert result.warnings == []
        assert result.controls == []

    def test_all_off_skips(self):
        controls = [
            {"id": "BW-01", "status": "fail", "mode": "OFF", "reason": "banned word"},
            {"id": "BW-02", "status": "fail", "mode": "off", "reason": "another"},
        ]
        result = _apply_modes(controls)
        assert result.allowed is True
        assert result.denial_reasons == []
        assert result.warnings == []
        assert len(result.controls) == 2

    def test_all_pass_allowed(self):
        controls = [
            {"id": "BW-01", "status": "pass", "mode": "ENFORCE"},
            {"id": "TG-06", "status": "pass", "mode": "WARN"},
        ]
        result = _apply_modes(controls)
        assert result.allowed is True
        assert result.denial_reasons == []
        assert result.warnings == []

    def test_warn_allows_but_logs(self):
        controls = [
            {
                "id": "BW-01",
                "status": "fail",
                "mode": "WARN",
                "reason": "banned word found",
            },
        ]
        result = _apply_modes(controls)
        assert result.allowed is True
        assert result.denial_reasons == []
        assert len(result.warnings) == 1
        assert "BW-01" in result.warnings[0]
        assert "banned word found" in result.warnings[0]

    def test_enforce_denies(self):
        controls = [
            {
                "id": "BW-01",
                "status": "fail",
                "mode": "ENFORCE",
                "reason": "banned word",
            },
        ]
        result = _apply_modes(controls)
        assert result.allowed is False
        assert len(result.denial_reasons) == 1
        assert "BW-01" in result.denial_reasons[0]

    def test_mixed_modes(self):
        controls = [
            {"id": "BW-01", "status": "fail", "mode": "OFF", "reason": "skipped"},
            {"id": "BW-02", "status": "fail", "mode": "WARN", "reason": "warned"},
            {"id": "BW-03", "status": "fail", "mode": "ENFORCE", "reason": "blocked"},
            {"id": "BW-04", "status": "pass", "mode": "ENFORCE"},
        ]
        result = _apply_modes(controls)
        assert result.allowed is False
        assert len(result.denial_reasons) == 1
        assert "BW-03" in result.denial_reasons[0]
        assert len(result.warnings) == 1
        assert "BW-02" in result.warnings[0]
        assert len(result.controls) == 4

    def test_missing_fields_use_defaults(self):
        controls = [{"id": "X-01"}]
        result = _apply_modes(controls)
        assert result.allowed is True
        assert result.controls[0].status == "pass"
        assert result.controls[0].mode == "OFF"

    def test_no_reason_produces_default_message(self):
        controls = [{"id": "BW-01", "status": "fail", "mode": "ENFORCE"}]
        result = _apply_modes(controls)
        assert "policy violation" in result.denial_reasons[0]


class TestParseResult:
    def test_missing_result_key(self):
        result = _parse_result({})
        assert result.allowed is False
        assert len(result.denial_reasons) == 1

    def test_result_not_dict(self):
        result = _parse_result({"result": "string"})
        assert result.allowed is False

    @patch("deep_agent.src.opa.service.settings")
    def test_legacy_deny_reasons_empty(self, mock_settings):
        mock_settings.OPA_MODES_ENABLED = False
        result = _parse_result({"result": {"deny_reasons": []}})
        assert result.allowed is True
        assert result.denial_reasons == []

    @patch("deep_agent.src.opa.service.settings")
    def test_legacy_deny_reasons_with_entries(self, mock_settings):
        mock_settings.OPA_MODES_ENABLED = False
        result = _parse_result({"result": {"deny_reasons": ["banned word"]}})
        assert result.allowed is False
        assert result.denial_reasons == ["banned word"]

    @patch("deep_agent.src.opa.service.settings")
    def test_modes_enabled_with_controls(self, mock_settings):
        mock_settings.OPA_MODES_ENABLED = True
        data = {
            "result": {
                "controls": [
                    {
                        "id": "BW-01",
                        "status": "fail",
                        "mode": "ENFORCE",
                        "reason": "bad",
                    },
                    {"id": "BW-02", "status": "pass", "mode": "ENFORCE"},
                ],
            },
        }
        result = _parse_result(data)
        assert result.allowed is False
        assert len(result.denial_reasons) == 1
        assert len(result.controls) == 2

    @patch("deep_agent.src.opa.service.settings")
    def test_modes_enabled_falls_back_to_legacy(self, mock_settings):
        mock_settings.OPA_MODES_ENABLED = True
        data = {"result": {"deny_reasons": ["legacy reason"]}}
        result = _parse_result(data)
        assert result.allowed is False
        assert result.denial_reasons == ["legacy reason"]

    @patch("deep_agent.src.opa.service.settings")
    def test_modes_disabled_ignores_controls(self, mock_settings):
        mock_settings.OPA_MODES_ENABLED = False
        data = {
            "result": {
                "controls": [
                    {
                        "id": "BW-01",
                        "status": "fail",
                        "mode": "ENFORCE",
                        "reason": "bad",
                    },
                ],
                "deny_reasons": [],
            },
        }
        result = _parse_result(data)
        assert result.allowed is True
        assert result.controls == []

    def test_deny_reasons_missing_key(self):
        result = _parse_result({"result": {"other": "stuff"}})
        assert result.allowed is False

    def test_deny_reasons_not_list(self):
        result = _parse_result({"result": {"deny_reasons": "not a list"}})
        assert result.allowed is False

    def test_deny_reasons_none(self):
        result = _parse_result({"result": {"deny_reasons": None}})
        assert result.allowed is False
