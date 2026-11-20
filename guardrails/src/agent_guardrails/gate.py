"""Phase 1 input gate.

Order matches the architecture: rules, structured fields, prompt injection,
HAP, the safety vote, PII redaction, secret masking, then intent. A safety
score at or above its threshold blocks the turn. The host otherwise receives
masked text, the intent, and the permitted tool list.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from agent_guardrails.budget import score_text
from agent_guardrails.checks.builtin import (
    content_scorer,
    injection_scorer,
    safety_scorer,
)
from agent_guardrails.checks.intent import LogisticIntentClassifier, build_encoder
from agent_guardrails.checks.pii import redact_pii
from agent_guardrails.checks.rules import text_rule_violation
from agent_guardrails.checks.schema import structured_field_violation
from agent_guardrails.checks.secrets import mask_secrets
from agent_guardrails.config import ClassifierConfig, InputGateConfig
from agent_guardrails.models import HuggingFaceTextScorer, UnavailablePolicyScorer
from agent_guardrails.types import (
    Action,
    CheckResult,
    InputDecision,
    IntentPrediction,
)

logger = logging.getLogger("agent_guardrails.gate")


class InputGate:
    """Stateless Phase 1 runner. The host keeps the decision for the turn."""

    def __init__(
        self,
        config: InputGateConfig | None = None,
        *,
        content: Any | None = None,
        injection: Any | None = None,
        safety: Any | None = None,
        intent: LogisticIntentClassifier | None = None,
    ) -> None:
        """Build scorers from ``config``, using injected ones in tests."""
        self.config = config or InputGateConfig()
        self.content = (
            content
            if content is not None
            else _build_text_scorer(self.config.content_filter, content_scorer())
        )
        self.injection = (
            injection
            if injection is not None
            else _build_text_scorer(self.config.injection, injection_scorer())
        )
        self.safety = (
            safety
            if safety is not None
            else _build_text_scorer(self.config.safety, safety_scorer())
        )
        self.intent = intent if intent is not None else _build_intent(self.config)

    def evaluate(self, text: str) -> InputDecision:
        """Screen one user message."""
        started = time.perf_counter()
        timeout_s = self.config.budgets.input_ms / 1000
        checks: list[CheckResult] = []
        blocked = False
        error: str | None = None
        model_calls = 0

        if self.config.rules.enabled:
            reason = text_rule_violation(
                text,
                max_chars=self.config.rules.max_input_chars,
                permitted_characters=self.config.rules.permitted_characters,
            )
            blocked, error = _record_code(checks, "rules", reason, blocked, error)

        if self.config.schema_check.enabled and not blocked:
            reason = structured_field_violation(
                text,
                allowed_fields=self.config.schema_check.allowed_fields,
                required_fields=self.config.schema_check.required_fields,
            )
            blocked, error = _record_code(checks, "schema", reason, blocked, error)

        if not blocked:
            for name, scorer, enabled, threshold in (
                (
                    "injection",
                    self.injection,
                    self.config.injection.enabled,
                    self.config.injection.threshold,
                ),
                (
                    "content_filter",
                    self.content,
                    self.config.content_filter.enabled,
                    self.config.content_filter.threshold,
                ),
                (
                    "safety",
                    self.safety,
                    self.config.safety.enabled,
                    self.config.safety.threshold,
                ),
            ):
                if not enabled:
                    continue
                remaining = timeout_s - (time.perf_counter() - started)
                outcome, calls = _unsafe_score(name, scorer, text, remaining, threshold)
                checks.append(outcome)
                model_calls += calls
                if outcome.action == Action.BLOCK:
                    blocked = True
                    error = _block_error(name, outcome, self.config.budgets.input_ms)
                    break

        masked, redacted, mask_error = self._mask(text, checks)
        if mask_error and not blocked:
            blocked = True
            error = mask_error

        prediction: IntentPrediction | None = None
        allowed: list[str] | None = None
        if self.config.intent.enabled and not blocked:
            prediction, allowed = self._classify_intent(masked)
            checks.append(
                CheckResult(
                    name="intent",
                    action=Action.ALLOW,
                    score=None if prediction is None else prediction.confidence,
                    detail="" if prediction is None else prediction.label,
                )
            )

        blocked_by = [check.name for check in checks if check.action == Action.BLOCK]
        if blocked or blocked_by:
            action = Action.BLOCK
        elif redacted:
            action = Action.REDACT
        else:
            action = Action.ALLOW
        if action == Action.BLOCK:
            logger.info("input_guardrail_blocked", extra={"blocked_by": blocked_by})
        return InputDecision(
            action=action,
            text=masked,
            checks=checks,
            blocked_by=blocked_by,
            intent=prediction,
            allowed_tools=allowed,
            error=error if action == Action.BLOCK else None,
            redacted=redacted,
            latency_ms=(time.perf_counter() - started) * 1000,
            model_calls=model_calls,
        )

    def _mask(
        self, text: str, checks: list[CheckResult]
    ) -> tuple[str, bool, str | None]:
        masked = text
        redacted = False
        if self.config.pii.enabled:
            try:
                masked, kinds = redact_pii(masked, backend=self.config.pii.backend)
            except Exception as exc:
                checks.append(
                    CheckResult(
                        name="pii",
                        action=Action.BLOCK,
                        detail=f"unavailable:{type(exc).__name__}",
                    )
                )
                return text, False, "Denied because PII redaction failed closed."
            found = bool(kinds)
            redacted = redacted or found
            checks.append(
                CheckResult(
                    name="pii",
                    action=Action.REDACT if found else Action.ALLOW,
                    detail=",".join(kinds),
                )
            )
        if self.config.secrets.enabled:
            masked, kinds = mask_secrets(masked, self.config.secrets.mask)
            found = bool(kinds)
            redacted = redacted or found
            checks.append(
                CheckResult(
                    name="secrets",
                    action=Action.REDACT if found else Action.ALLOW,
                    detail=",".join(kinds),
                )
            )
        return masked, redacted, None

    def _classify_intent(
        self, text: str
    ) -> tuple[IntentPrediction | None, list[str] | None]:
        if not self.intent.is_fitted:
            return (
                IntentPrediction(label="unknown", confidence=0.0),
                None,
            )
        prediction = self.intent.predict(text)
        if prediction.confidence < self.config.intent.min_confidence:
            prediction = prediction.model_copy(update={"read_only": True})
            return prediction, list(self.config.intent.read_only_tools)
        tools = self.config.intent.tools_for_label.get(prediction.label)
        if tools is None:
            prediction = prediction.model_copy(update={"read_only": True})
            return prediction, list(self.config.intent.read_only_tools)
        return prediction, list(tools)


def _build_text_scorer(config: ClassifierConfig, builtin: Any) -> Any:
    if not config.enabled or config.backend == "builtin":
        return builtin
    primary = HuggingFaceTextScorer(config.model_id)
    return UnavailablePolicyScorer(primary, builtin, config.on_unavailable)


def _build_intent(config: InputGateConfig) -> LogisticIntentClassifier:
    encoder = build_encoder(config.intent.embedding, config.intent.model_id)
    return LogisticIntentClassifier(encoder)


def _unsafe_score(
    name: str,
    scorer: Any,
    text: str,
    timeout_s: float,
    threshold: float,
) -> tuple[CheckResult, int]:
    score, timed_out, failed, latency = score_text(scorer, text, timeout_s)
    blocked = timed_out or failed or score >= threshold
    if timed_out:
        detail = "timeout"
    elif failed:
        detail = "scorer_error"
    else:
        detail = ""
    return (
        CheckResult(
            name=name,
            action=Action.BLOCK if blocked else Action.ALLOW,
            score=score,
            detail=detail,
            latency_ms=latency,
            timed_out=timed_out,
        ),
        1 if getattr(scorer, "uses_model", False) else 0,
    )


def _record_code(
    checks: list[CheckResult],
    name: str,
    reason: str | None,
    blocked: bool,
    error: str | None,
) -> tuple[bool, str | None]:
    if reason:
        checks.append(CheckResult(name=name, action=Action.BLOCK, detail=reason))
        return True, error or reason
    checks.append(CheckResult(name=name, action=Action.ALLOW))
    return blocked, error


def _block_error(name: str, outcome: CheckResult, budget_ms: float) -> str:
    if outcome.timed_out:
        return f"Denied because {name} exceeded the {budget_ms:.0f}ms input budget."
    if outcome.detail == "scorer_error":
        return f"Denied because {name} failed closed."
    return f"Blocked by {name}."
