"""Unit tests for OPA compliance state caching."""

from deep_agent.src.opa.service import (
    ComplianceState,
    ControlResult,
    OpaResult,
    get_compliance_state,
    update_compliance_state,
)


class TestUpdateComplianceState:
    """Tests for update_compliance_state and get_compliance_state."""

    def test_compliant_when_all_pass(self):
        opa = OpaResult(
            allowed=True,
            controls=[
                ControlResult(id="BW-01", status="pass", mode="ENFORCE"),
                ControlResult(id="BW-02", status="pass", mode="WARN"),
            ],
        )
        update_compliance_state(opa)
        state = get_compliance_state()
        assert state["status"] == "compliant"
        assert len(state["controls"]) == 2
        assert state["evaluated_at"]

    def test_non_compliant_when_enforce_fails(self):
        opa = OpaResult(
            allowed=False,
            denial_reasons=["[BW-01] banned word"],
            controls=[
                ControlResult(
                    id="BW-01", status="fail", mode="ENFORCE", reason="banned word"
                ),
            ],
        )
        update_compliance_state(opa)
        state = get_compliance_state()
        assert state["status"] == "non_compliant"
        assert state["controls"][0]["id"] == "BW-01"
        assert state["controls"][0]["reason"] == "banned word"

    def test_warning_when_warn_mode_fails(self):
        opa = OpaResult(
            allowed=True,
            warnings=["[BW-01] banned word"],
            controls=[
                ControlResult(
                    id="BW-01", status="fail", mode="WARN", reason="banned word"
                ),
            ],
        )
        update_compliance_state(opa)
        state = get_compliance_state()
        assert state["status"] == "warning"

    def test_compliant_when_off_mode_fails(self):
        opa = OpaResult(
            allowed=True,
            controls=[
                ControlResult(id="BW-01", status="fail", mode="OFF"),
            ],
        )
        update_compliance_state(opa)
        state = get_compliance_state()
        assert state["status"] == "compliant"

    def test_empty_controls(self):
        opa = OpaResult(allowed=True, controls=[])
        update_compliance_state(opa)
        state = get_compliance_state()
        assert state["status"] == "compliant"
        assert state["controls"] == []

    def test_no_controls_legacy_result(self):
        opa = OpaResult(allowed=True, denial_reasons=[])
        update_compliance_state(opa)
        state = get_compliance_state()
        assert state["status"] == "compliant"

    def test_reason_omitted_when_empty(self):
        opa = OpaResult(
            allowed=True,
            controls=[ControlResult(id="BW-01", status="pass", mode="ENFORCE")],
        )
        update_compliance_state(opa)
        state = get_compliance_state()
        assert "reason" not in state["controls"][0]
