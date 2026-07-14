"""PII pattern detection engine.

Compiles regex patterns per rule and finds all non-overlapping matches
in a given text, ordered by position.
"""

import re
from dataclasses import dataclass

from deep_agent.src.pii.config import PIIRule

# Built-in compiled patterns — order matters for readability; credit card before phone
# to avoid partial digit matches being claimed by phone first.
BUILTIN_PATTERNS: dict[str, str] = {
    "credit_card": (
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|"           # Visa
        r"5[1-5][0-9]{14}|"                          # Mastercard
        r"3[47][0-9]{13}|"                           # Amex
        r"3(?:0[0-5]|[68][0-9])[0-9]{11}|"          # Diners
        r"6(?:011|5[0-9]{2})[0-9]{12}|"             # Discover
        r"(?:2131|1800|35\d{3})\d{11})\b"           # JCB
    ),
    "ssn": r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b",
    "email": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
    "phone": (
        r"(?<!\d)"
        r"(?:\+?1[-.\s]?)?"                          # optional US country code
        r"(?:\(?\d{3}\)?[-.\s]?)?"                   # optional area code
        r"\d{3}[-.\s]?\d{4}"
        r"(?!\d)"
    ),
    "ip_address": r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
    "url": r"https?://[^\s<>\"'{}|\\^`\[\]]+",
    "iban": r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b",
    "address": (
        r"(?:"
        # Western style: 123 Main Street / 123 Oak Ave Blvd
        r"\b\d{1,5}\s+(?:[A-Za-z]+[\s,]+){1,4}"
        r"(?:St(?:reet)?|Ave(?:nue)?|Blvd|Boulevard|Dr(?:ive)?|Rd|Road"
        r"|Ln|Lane|Way|Ct|Court|Pl(?:ace)?|Sq(?:uare)?|Terr(?:ace)?"
        r"|Pkwy|Parkway|Hwy|Highway)\.?\b"
        r"|"
        # Comma-separated locality style: 343, HSR Avenue, Blr / 1848, HSR Layout, Bangalore
        r"\b(?:No\.?\s*)?\d{1,5}[A-Za-z]?[,\s]+(?:[A-Za-z][A-Za-z\s]{2,}[,\s]\s*){1,3}[A-Za-z]{3,}"
        r")"
    ),
}


@dataclass
class PIIMatch:
    start: int
    end: int
    value: str
    rule_name: str
    label: str
    action: str


class PIIDetector:
    """Detects PII in text using compiled regex patterns.

    Args:
        rules: List of PIIRule objects defining what to detect.

    Raises:
        ValueError: If a builtin pattern name is unknown or a custom rule
            is missing its regex.
    """

    def __init__(self, rules: list[PIIRule]) -> None:
        self._patterns: list[tuple[re.Pattern[str], PIIRule, str]] = []
        for rule in rules:
            if rule.pattern_type == "builtin":
                raw = BUILTIN_PATTERNS.get(rule.name)
                if raw is None:
                    raise ValueError(
                        f"Unknown builtin PII pattern '{rule.name}'. "
                        f"Available: {list(BUILTIN_PATTERNS)}"
                    )
            else:
                if not rule.regex:
                    raise ValueError(
                        f"Custom rule '{rule.name}' must specify a 'regex' field."
                    )
                raw = rule.regex
            self._patterns.append((re.compile(raw), rule, rule.effective_label()))

    def find_all(self, text: str) -> list[PIIMatch]:
        """Return all non-overlapping PII matches ordered by position.

        When two patterns match overlapping ranges, the earlier-starting
        match wins; ties are broken by rule registration order.
        """
        if not text:
            return []

        candidates: list[PIIMatch] = []
        for regex, rule, label in self._patterns:
            for m in regex.finditer(text):
                candidates.append(
                    PIIMatch(
                        start=m.start(),
                        end=m.end(),
                        value=m.group(),
                        rule_name=rule.name,
                        label=label,
                        action=rule.action.value,
                    )
                )

        # Sort by position, then deduplicate overlapping spans (first wins).
        candidates.sort(key=lambda x: (x.start, x.end))
        deduped: list[PIIMatch] = []
        last_end = -1
        for match in candidates:
            if match.start >= last_end:
                deduped.append(match)
                last_end = match.end
        return deduped
