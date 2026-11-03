"""Offline evaluation for Phase 1. This module never prompts."""

from __future__ import annotations

import json
import sys
from typing import Any

from agent_guardrails.checks.builtin import (
    BENIGN_TEXTS,
    CONTENT_EXAMPLES,
    INJECTION_EXAMPLES,
    SAFETY_EXAMPLES,
)
from agent_guardrails.checks.intent import LogisticIntentClassifier
from agent_guardrails.config import InputGateConfig
from agent_guardrails.gate import InputGate
from agent_guardrails.types import InputDecision

_INTENT_TRAIN: tuple[tuple[str, str], ...] = (
    ("show revenue by region", "data_query"),
    ("query the sales table", "data_query"),
    ("select total orders", "data_query"),
    ("how many customers last quarter", "data_query"),
    ("count orders by month", "data_query"),
    ("what is total revenue", "data_query"),
    ("write a file report", "tool_action"),
    ("send the email now", "tool_action"),
    ("delete the old records", "tool_action"),
    ("update the dashboard config", "tool_action"),
    ("please delete those rows", "tool_action"),
    ("create a new file", "tool_action"),
    ("hello there", "chitchat"),
    ("thanks that is all", "chitchat"),
    ("good morning", "chitchat"),
    ("how are you", "chitchat"),
    ("hello friend", "chitchat"),
    ("thank you so much", "chitchat"),
)

_READ_ONLY = ["search", "get_sql"]


class EvalReport:
    """Machine-readable result of one automatic evaluation pass."""

    def __init__(self) -> None:
        """Start an empty report."""
        self.passed = False
        self.cases = 0
        self.failures: list[str] = []
        self.latency_ms: dict[str, float] = {}

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable report."""
        return {
            "passed": self.passed,
            "cases": self.cases,
            "failures": self.failures,
            "latency_ms": self.latency_ms,
            "phase": "input",
            "hitl": False,
        }


def build_eval_gate() -> InputGate:
    """Gate used by the offline eval, with a fitted intent head."""
    config = InputGateConfig(
        schema_check={"allowed_fields": ["question", "table"]},
        pii={"backend": "pattern"},
        intent={
            "tools_for_label": {
                "data_query": ["search", "get_sql"],
                "tool_action": ["search", "get_sql", "write_file", "send_email"],
                "chitchat": ["search"],
            },
            "read_only_tools": _READ_ONLY,
        },
    )
    classifier = LogisticIntentClassifier()
    classifier.fit(
        [text for text, _label in _INTENT_TRAIN],
        [label for _text, label in _INTENT_TRAIN],
    )
    return InputGate(config, intent=classifier)


def run_eval(gate: InputGate | None = None) -> EvalReport:
    """Score the input gate. Never reads stdin and never waits for approval."""
    gate = gate or build_eval_gate()
    report = EvalReport()
    samples: list[float] = []

    def observe(name: str, decision: InputDecision, **expects: Any) -> None:
        report.cases += 1
        if decision.requires_approval:
            report.failures.append(f"{name}: waited for approval")
        action = expects.get("action")
        if action is not None and decision.action.value != action:
            report.failures.append(
                f"{name}: action {decision.action.value} != {action}"
            )
        blocked_by = expects.get("blocked_by")
        if blocked_by is not None and blocked_by not in decision.blocked_by:
            report.failures.append(
                f"{name}: blocked_by {decision.blocked_by} missing {blocked_by}"
            )
        for snippet in expects.get("absent", ()):
            if snippet in decision.text:
                report.failures.append(f"{name}: still contains {snippet!r}")
        for snippet in expects.get("present", ()):
            if snippet not in decision.text:
                report.failures.append(f"{name}: missing {snippet!r}")
        if "tools" in expects and decision.allowed_tools != expects["tools"]:
            report.failures.append(
                f"{name}: tools {decision.allowed_tools} != {expects['tools']}"
            )
        if expects.get("read_only") is True and (
            decision.intent is None or not decision.intent.read_only
        ):
            report.failures.append(f"{name}: expected read-only intent")
        if expects.get("tool_blocked"):
            tool_name = expects["tool_blocked"]
            if decision.tool_allowed(tool_name):
                report.failures.append(f"{name}: {tool_name} was allowed")
        if expects.get("tool_ok"):
            tool_name = expects["tool_ok"]
            if not decision.tool_allowed(tool_name):
                report.failures.append(f"{name}: {tool_name} was blocked")

    for text in BENIGN_TEXTS:
        decision = gate.evaluate(text)
        samples.append(decision.latency_ms)
        observe(f"allow {text[:32]}", decision, action="allow")

    for text in (*CONTENT_EXAMPLES, *INJECTION_EXAMPLES, *SAFETY_EXAMPLES):
        decision = gate.evaluate(text)
        samples.append(decision.latency_ms)
        observe(f"block {text[:32]}", decision, action="block")

    pii = gate.evaluate(
        "Mail ada@example.com or call +1 415-555-2671 about card 4111 1111 1111 1111."
    )
    samples.append(pii.latency_ms)
    observe(
        "pii redact",
        pii,
        action="redact",
        absent=("ada@example.com", "4111 1111 1111 1111", "415-555-2671"),
        present=("[EMAIL]", "[PHONE]", "[CREDIT_CARD]"),
    )
    observe(
        "max length",
        gate.evaluate("x" * 8001),
        action="block",
        blocked_by="rules",
    )
    observe(
        "unknown field",
        gate.evaluate('{"question": "revenue", "drop": "yes"}'),
        action="block",
        blocked_by="schema",
    )
    observe(
        "known fields",
        gate.evaluate('{"question": "revenue", "table": "sales"}'),
        action="allow",
    )

    high = gate.evaluate("send the email now")
    observe(
        "intent tool action",
        high,
        action="allow",
        tools=["search", "get_sql", "write_file", "send_email"],
        tool_ok="send_email",
        tool_blocked="delete_rows",
    )
    low = gate.evaluate("asdf qqqq zzzz")
    observe(
        "intent low confidence",
        low,
        action="allow",
        read_only=True,
        tools=_READ_ONLY,
        tool_ok="search",
        tool_blocked="write_file",
    )

    report.latency_ms = {"input_p95": _percentile(samples, 95)}
    if report.latency_ms["input_p95"] > gate.config.budgets.input_ms:
        report.failures.append(
            f"input p95 {report.latency_ms['input_p95']:.2f}ms exceeded "
            f"{gate.config.budgets.input_ms:.0f}ms"
        )
    report.passed = not report.failures
    return report


def main() -> None:
    """Print the report as JSON and exit 0 only when every gate passed."""
    report = run_eval()
    json.dump(report.as_dict(), sys.stdout, indent=2)
    sys.stdout.write("\n")
    raise SystemExit(0 if report.passed else 1)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (percentile / 100)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


if __name__ == "__main__":
    main()
