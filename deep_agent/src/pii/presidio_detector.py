"""Presidio-backed PII detection engine.

Wraps Microsoft Presidio's AnalyzerEngine to produce list[PIIMatch]
using the same interface as PIIDetector.find_all().

Presidio imports are deferred to __init__ so that systems using
detector="regex" (or without presidio-analyzer installed) can import
this module without error.
"""

from __future__ import annotations

from deep_agent.src.pii.config import PIIRule
from deep_agent.src.pii.detector import PIIMatch

# Explicit mappings where rule.name doesn't directly match the Presidio entity name.
# For anything not listed here, rule.name.upper() is tried automatically —
# e.g. "us_passport" → "US_PASSPORT", "uk_nhs" → "UK_NHS".
_RULE_TO_ENTITY: dict[str, str] = {
    "email": "EMAIL_ADDRESS",
    "phone": "PHONE_NUMBER",
    "credit_card": "CREDIT_CARD",
    "ssn": "US_SSN",
    "ip_address": "IP_ADDRESS",
    "iban": "IBAN_CODE",
    "address": "LOCATION",
}


class PresidioDetector:
    """Detects PII using Presidio's AnalyzerEngine.

    Builtin rules are mapped to the corresponding Presidio entity type.
    Custom rules register a PatternRecognizer using the provided regex.
    AnalyzerEngine is constructed once at startup to avoid reloading the
    spaCy model on every request.

    Raises:
        ImportError: If presidio-analyzer is not installed.
        ValueError: If a builtin rule name has no Presidio entity mapping,
            or a custom rule is missing its regex field.
    """

    def __init__(self, rules: list[PIIRule]) -> None:
        """Build analyzer with entity mappings for the given rules."""
        from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer

        self._entities: list[str] = []
        self._entity_to_rule: dict[str, PIIRule] = {}

        analyzer = AnalyzerEngine()

        supported = set(analyzer.get_supported_entities())

        for rule in rules:
            if rule.pattern_type == "builtin":
                # 1. Explicit mapping table
                entity_type = _RULE_TO_ENTITY.get(rule.name)
                # 2. Dynamic fallback: rule.name.upper() (e.g. "us_passport" → "US_PASSPORT")
                if entity_type is None:
                    candidate = rule.name.upper()
                    if candidate in supported:
                        entity_type = candidate
                if entity_type is None:
                    raise ValueError(
                        f"Presidio: no entity mapping for '{rule.name}'. "
                        f"Add it to _RULE_TO_ENTITY or use a name that matches a "
                        f"Presidio entity (e.g. 'us_passport' → 'US_PASSPORT'). "
                        f"Supported: {sorted(supported)}"
                    )
            else:
                if not rule.regex:
                    raise ValueError(
                        f"Custom rule '{rule.name}' must specify a 'regex' field."
                    )
                entity_type = rule.name.upper()
                recognizer = PatternRecognizer(
                    supported_entity=entity_type,
                    patterns=[Pattern(name=rule.name, regex=rule.regex, score=0.9)],
                )
                analyzer.registry.add_recognizer(recognizer)

            self._entities.append(entity_type)
            self._entity_to_rule[entity_type] = rule

        self._analyzer = analyzer

    def find_all(self, text: str) -> list[PIIMatch]:
        """Return all non-overlapping PII matches ordered by position."""
        if not text or not self._entities:
            return []

        results = self._analyzer.analyze(
            text=text, entities=self._entities, language="en"
        )
        results.sort(key=lambda r: (r.start, r.end))

        deduped: list[PIIMatch] = []
        last_end = -1
        for result in results:
            if result.start < last_end:
                continue
            rule = self._entity_to_rule.get(result.entity_type)
            if rule is None:
                continue
            deduped.append(
                PIIMatch(
                    start=result.start,
                    end=result.end,
                    value=text[result.start : result.end],
                    rule_name=rule.name,
                    label=rule.effective_label(),
                    action=rule.action.value,
                )
            )
            last_end = result.end

        return deduped
