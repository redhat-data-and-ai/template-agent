"""Unit tests for PIIScrubber — tokenization, masking, redaction, and restoration."""

import pytest

from deep_agent.src.pii.config import ActionType, PIIConfig, PIIRule
from deep_agent.src.pii.scrubber import (
    PIIScrubber,
    _label_counters,
    _token_map,
    _value_map,
)


@pytest.fixture(autouse=True)
def _reset_context_vars():
    """Reset per-request ContextVars before each test to prevent state leakage."""
    _token_map.set(None)
    _value_map.set(None)
    _label_counters.set(None)
    yield
    _token_map.set(None)
    _value_map.set(None)
    _label_counters.set(None)


def _make_rule(
    name: str, strategy: str, regex: str | None = None, label: str | None = None
) -> PIIRule:
    provider = "custom" if regex else "regex"
    return PIIRule(
        name=name,
        strategy=ActionType(strategy),
        provider=provider,
        regex=regex,
        label=label,
    )


def _scrubber(*rules: PIIRule, hash_key: bytes = b"test-key") -> PIIScrubber:
    config = PIIConfig(enabled=True, rules=list(rules))
    return PIIScrubber(config, hash_key=hash_key)


class TestScrubStrategy:
    """Test scrub (reversible tokenization) strategy."""

    def test_scrub_replaces_email_with_token(self):
        s = _scrubber(_make_rule("email", "scrub"))
        result = s.scrub("Contact user@example.com for help.")
        assert "user@example.com" not in result
        assert "[EMAIL_1]" in result

    def test_scrub_restore_round_trip(self):
        s = _scrubber(_make_rule("email", "scrub"))
        original = "Email user@example.com please."
        scrubbed = s.scrub(original)
        restored = s.restore(scrubbed)
        assert restored == original

    def test_same_value_gets_same_token(self):
        s = _scrubber(_make_rule("email", "scrub"))
        s.scrub("First: user@example.com")
        result = s.scrub("Again: user@example.com")
        assert result.count("[EMAIL_1]") == 1
        assert "[EMAIL_2]" not in result

    def test_different_values_get_different_tokens(self):
        s = _scrubber(_make_rule("email", "scrub"))
        r1 = s.scrub("a@x.com")
        r2 = s.scrub("b@y.com")
        assert "[EMAIL_1]" in r1
        assert "[EMAIL_2]" in r2

    def test_custom_label_used_in_token(self):
        s = _scrubber(_make_rule("email", "scrub", label="MAIL"))
        result = s.scrub("user@example.com")
        assert "[MAIL_1]" in result

    def test_text_without_pii_unchanged(self):
        s = _scrubber(_make_rule("email", "scrub"))
        text = "No PII in this message."
        assert s.scrub(text) == text

    def test_empty_string_unchanged(self):
        s = _scrubber(_make_rule("email", "scrub"))
        assert s.scrub("") == ""


class TestMaskStrategy:
    """Test mask (partial, one-way) strategy."""

    def test_mask_preserves_last_four_chars(self):
        s = _scrubber(_make_rule("credit_card", "mask"))
        result = s.scrub("Card: 4111111111111111")
        assert "****" in result
        assert "1111" in result
        assert "4111111111111111" not in result

    def test_mask_short_value_fully_masked(self):
        s = _scrubber(_make_rule("ssn", "mask"))
        result = s.scrub("SSN: 123-45-6789")
        assert "123-45-6789" not in result

    def test_mask_is_not_reversible(self):
        s = _scrubber(_make_rule("credit_card", "mask"))
        scrubbed = s.scrub("4111111111111111")
        restored = s.restore(scrubbed)
        assert "4111111111111111" not in restored


class TestRedactStrategy:
    """Test redact (one-way ***REDACTED***) strategy."""

    def test_redact_replaces_with_placeholder(self):
        s = _scrubber(_make_rule("email", "redact"))
        result = s.scrub("Send to user@example.com now.")
        assert "user@example.com" not in result
        assert "***REDACTED***" in result

    def test_redact_is_not_reversible(self):
        s = _scrubber(_make_rule("email", "redact"))
        scrubbed = s.scrub("user@example.com")
        restored = s.restore(scrubbed)
        assert "user@example.com" not in restored


class TestBlockStrategy:
    """Test block strategy — block_detector is pre-built for fast input checking."""

    def test_block_rule_builds_block_detector(self):
        s = _scrubber(
            _make_rule("pan_card", "block", regex=r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
        )
        assert s._block_detector is not None

    def test_no_block_rules_leaves_block_detector_none(self):
        s = _scrubber(_make_rule("email", "redact"))
        assert s._block_detector is None

    def test_block_detector_finds_blocked_value(self):
        s = _scrubber(
            _make_rule("pan_card", "block", regex=r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
        )
        matches = s._block_detector.find_all("PAN: ABCDE1234F here")
        assert len(matches) == 1
        assert matches[0].value == "ABCDE1234F"


class TestScrubOneWay:
    """Test scrub_one_way — stateless sanitization that does not modify the token map."""

    def test_scrub_one_way_does_not_populate_token_map(self):
        s = _scrubber(_make_rule("email", "scrub"))
        s.scrub_one_way("user@example.com")
        assert s.snapshot_token_map() == {}

    def test_scrub_one_way_still_removes_pii(self):
        s = _scrubber(_make_rule("email", "scrub"))
        result = s.scrub_one_way("user@example.com")
        assert "user@example.com" not in result

    def test_scrub_one_way_redacts_instead_of_tokenizing(self):
        s = _scrubber(_make_rule("email", "scrub"))
        result = s.scrub_one_way("user@example.com")
        assert "[EMAIL_" not in result
        assert "***REDACTED***" in result


class TestScrubForTraceHash:
    """Test scrub_for_trace_hash — deterministic HMAC hashing for log correlation."""

    def test_produces_hash_placeholder(self):
        s = _scrubber(_make_rule("email", "scrub"), hash_key=b"fixed-key")
        result = s.scrub_for_trace_hash("user@example.com")
        assert "[HASH:" in result
        assert "user@example.com" not in result

    def test_same_value_same_key_produces_same_hash(self):
        s = _scrubber(_make_rule("email", "scrub"), hash_key=b"fixed-key")
        r1 = s.scrub_for_trace_hash("user@example.com")
        r2 = s.scrub_for_trace_hash("user@example.com")
        assert r1 == r2

    def test_different_values_produce_different_hashes(self):
        s = _scrubber(_make_rule("email", "scrub"), hash_key=b"fixed-key")
        r1 = s.scrub_for_trace_hash("a@x.com")
        r2 = s.scrub_for_trace_hash("b@y.com")
        assert r1 != r2

    def test_does_not_modify_token_map(self):
        s = _scrubber(_make_rule("email", "scrub"), hash_key=b"fixed-key")
        s.scrub_for_trace_hash("user@example.com")
        assert s.snapshot_token_map() == {}


class TestTokenMapPersistence:
    """Test load/save of the token map for cross-request continuity."""

    def test_snapshot_returns_current_map(self):
        s = _scrubber(_make_rule("email", "scrub"))
        s.scrub("user@example.com")
        snapshot = s.snapshot_token_map()
        assert any("user@example.com" in v for v in snapshot.values())

    def test_load_token_map_restores_tokens(self):
        s = _scrubber(_make_rule("email", "scrub"))
        saved = {"[EMAIL_1]": "user@example.com"}
        s.load_token_map(saved)
        restored = s.restore("Reply to [EMAIL_1] soon.")
        assert "user@example.com" in restored

    def test_loaded_map_reuses_existing_token_for_same_value(self):
        s = _scrubber(_make_rule("email", "scrub"))
        s.load_token_map({"[EMAIL_1]": "user@example.com"})
        result = s.scrub("user@example.com again")
        assert "[EMAIL_1]" in result
        assert "[EMAIL_2]" not in result

    def test_snapshot_to_container_copies_map(self):
        s = _scrubber(_make_rule("email", "scrub"))
        container: dict = {}
        s.set_shared_container(container)
        s.scrub("user@example.com")
        s.snapshot_to_container()
        assert any("user@example.com" in v for v in container.values())

    def test_snapshot_to_container_noop_without_registration(self):
        s = _scrubber(_make_rule("email", "scrub"))
        s.scrub("user@example.com")
        s.snapshot_to_container()  # should not raise


class TestMultipleRules:
    """Test scrubber behaviour with multiple rules active simultaneously."""

    def test_multiple_rules_each_scrub_their_type(self):
        s = _scrubber(
            _make_rule("email", "scrub"),
            _make_rule("ip_address", "redact"),
        )
        result = s.scrub("Email user@example.com from IP 10.0.0.1.")
        assert "user@example.com" not in result
        assert "10.0.0.1" not in result
        assert "[EMAIL_1]" in result
        assert "***REDACTED***" in result

    def test_restore_only_restores_scrub_tokens(self):
        s = _scrubber(
            _make_rule("email", "scrub"),
            _make_rule("ip_address", "redact"),
        )
        scrubbed = s.scrub("Email user@example.com from IP 10.0.0.1.")
        restored = s.restore(scrubbed)
        assert "user@example.com" in restored
        assert "10.0.0.1" not in restored
