"""Unit tests for PIIDetector — regex pattern matching and span deduplication."""

import pytest

from deep_agent.src.pii.config import ActionType, PIIRule
from deep_agent.src.pii.detector import PIIDetector


def _rule(name: str, strategy: str = "redact", regex: str | None = None) -> PIIRule:
    provider = "custom" if regex else "regex"
    return PIIRule(
        name=name, strategy=ActionType(strategy), provider=provider, regex=regex
    )


class TestBuiltinPatterns:
    """Test detection of built-in PII pattern types."""

    def test_detects_email(self):
        detector = PIIDetector([_rule("email")])
        matches = detector.find_all("Contact us at user@example.com for help.")
        assert len(matches) == 1
        assert matches[0].value == "user@example.com"
        assert matches[0].rule_name == "email"

    def test_detects_credit_card(self):
        detector = PIIDetector([_rule("credit_card", "mask")])
        matches = detector.find_all("Card: 4111111111111111 was declined.")
        assert len(matches) == 1
        assert matches[0].value == "4111111111111111"

    def test_detects_ssn(self):
        detector = PIIDetector([_rule("ssn", "redact")])
        matches = detector.find_all("SSN is 123-45-6789.")
        assert len(matches) == 1
        assert matches[0].value == "123-45-6789"

    def test_detects_ip_address(self):
        detector = PIIDetector([_rule("ip_address", "redact")])
        matches = detector.find_all("Server at 192.168.1.100 is down.")
        assert len(matches) == 1
        assert matches[0].value == "192.168.1.100"

    def test_detects_url(self):
        detector = PIIDetector([_rule("url", "redact")])
        matches = detector.find_all("Visit https://example.com/path?q=1 for details.")
        assert len(matches) == 1
        assert matches[0].value == "https://example.com/path?q=1"

    def test_multiple_emails_in_text(self):
        detector = PIIDetector([_rule("email")])
        matches = detector.find_all("Send to a@x.com and b@y.com.")
        assert len(matches) == 2
        assert {m.value for m in matches} == {"a@x.com", "b@y.com"}

    def test_no_match_returns_empty(self):
        detector = PIIDetector([_rule("email")])
        matches = detector.find_all("No email here, just plain text.")
        assert matches == []

    def test_empty_text_returns_empty(self):
        detector = PIIDetector([_rule("email")])
        assert detector.find_all("") == []


class TestCustomRegexRule:
    """Test custom regex rules."""

    def test_custom_pan_card_pattern(self):
        rule = _rule("pan_card", "block", regex=r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
        detector = PIIDetector([rule])
        matches = detector.find_all("PAN: ABCDE1234F is invalid.")
        assert len(matches) == 1
        assert matches[0].value == "ABCDE1234F"

    def test_custom_employee_id(self):
        rule = _rule("employee_id", "scrub", regex=r"\bEMP-\d{6}\b")
        detector = PIIDetector([rule])
        matches = detector.find_all("Employee EMP-123456 submitted a ticket.")
        assert len(matches) == 1
        assert matches[0].value == "EMP-123456"

    def test_custom_rule_no_match(self):
        rule = _rule("employee_id", "scrub", regex=r"\bEMP-\d{6}\b")
        detector = PIIDetector([rule])
        assert detector.find_all("No employee ID here.") == []


class TestSpanDeduplication:
    """Test overlapping span handling — earlier match wins."""

    def test_overlapping_patterns_first_wins(self):
        email_rule = _rule("email")
        url_rule = _rule("url", "redact")
        detector = PIIDetector([email_rule, url_rule])
        # URL pattern could consume the email inside a URL; email should win (registered first)
        matches = detector.find_all("user@example.com")
        assert len(matches) == 1

    def test_non_overlapping_matches_both_returned(self):
        detector = PIIDetector([_rule("email"), _rule("ip_address")])
        matches = detector.find_all("Email user@x.com from IP 10.0.0.1.")
        assert len(matches) == 2


class TestValidationErrors:
    """Test that invalid rule configurations raise errors at construction time."""

    def test_unknown_builtin_raises_value_error(self):
        rule = _rule("nonexistent_pii_type")
        with pytest.raises(ValueError, match="Unknown builtin PII pattern"):
            PIIDetector([rule])

    def test_custom_rule_without_regex_raises_value_error(self):
        rule = PIIRule(name="my_rule", provider="custom", strategy=ActionType.redact)
        with pytest.raises(ValueError, match="Unknown builtin PII pattern"):
            PIIDetector([rule])


class TestMatchMetadata:
    """Test that match objects carry correct metadata."""

    def test_match_label_defaults_to_uppercase_name(self):
        detector = PIIDetector([_rule("email")])
        matches = detector.find_all("user@example.com")
        assert matches[0].label == "EMAIL"

    def test_match_label_uses_custom_label(self):
        rule = PIIRule(
            name="email", provider="regex", strategy=ActionType.scrub, label="MAIL"
        )
        detector = PIIDetector([rule])
        matches = detector.find_all("user@example.com")
        assert matches[0].label == "MAIL"

    def test_match_action_reflects_strategy(self):
        detector = PIIDetector([_rule("email", "mask")])
        matches = detector.find_all("user@example.com")
        assert matches[0].action == "mask"

    def test_match_positions_are_correct(self):
        text = "Email: user@example.com end"
        detector = PIIDetector([_rule("email")])
        matches = detector.find_all(text)
        assert matches[0].start == text.index("user@example.com")
        assert matches[0].end == matches[0].start + len("user@example.com")
