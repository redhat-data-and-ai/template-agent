"""Unit tests for PresidioDetector — Presidio-backed PII detection engine."""

import pytest
from unittest.mock import patch

pytest.importorskip("presidio_analyzer")

from deep_agent.src.pii.config import ActionType, PIIRule
from deep_agent.src.pii.detector import PIIMatch
from deep_agent.src.pii.presidio_detector import PresidioDetector, _RULE_TO_ENTITY


def _rule(
    name: str,
    strategy: str = "scrub",
    regex: str | None = None,
    provider: str | None = None,
    label: str | None = None,
) -> PIIRule:
    if provider is None:
        provider = "custom" if regex else "presidio"
    return PIIRule(
        name=name,
        strategy=ActionType(strategy),
        provider=provider,
        regex=regex,
        label=label,
    )


# ---------------------------------------------------------------------------
# TestPresidioDetectorInit — construction, entity mapping, and validation
# ---------------------------------------------------------------------------


class TestPresidioDetectorInit:
    """Test PresidioDetector construction, entity mapping, and error handling."""

    def test_builtin_email_maps_to_email_address(self):
        detector = PresidioDetector([_rule("email")])
        assert "EMAIL_ADDRESS" in detector._entities
        assert "EMAIL_ADDRESS" in detector._entity_to_rule

    def test_builtin_phone_maps_to_phone_number(self):
        detector = PresidioDetector([_rule("phone")])
        assert "PHONE_NUMBER" in detector._entities
        assert "PHONE_NUMBER" in detector._entity_to_rule

    def test_builtin_credit_card_maps_to_credit_card(self):
        detector = PresidioDetector([_rule("credit_card")])
        assert "CREDIT_CARD" in detector._entities

    def test_builtin_ssn_maps_to_us_ssn(self):
        detector = PresidioDetector([_rule("ssn")])
        assert "US_SSN" in detector._entities

    def test_builtin_ip_address_maps_to_ip_address(self):
        detector = PresidioDetector([_rule("ip_address")])
        assert "IP_ADDRESS" in detector._entities

    def test_builtin_iban_maps_to_iban_code(self):
        detector = PresidioDetector([_rule("iban")])
        assert "IBAN_CODE" in detector._entities

    def test_builtin_address_maps_to_location(self):
        detector = PresidioDetector([_rule("address")])
        assert "LOCATION" in detector._entities

    def test_custom_regex_rule_registers_entity(self):
        detector = PresidioDetector([_rule("cust_id", regex=r"\bCUST-\d+\b")])
        assert "CUST_ID" in detector._entities
        assert "CUST_ID" in detector._entity_to_rule

    def test_custom_regex_rule_entity_is_name_uppercased(self):
        detector = PresidioDetector([_rule("order_ref", regex=r"\bORD-\d{6}\b")])
        assert "ORDER_REF" in detector._entities

    def test_unknown_builtin_raises_value_error(self):
        with pytest.raises(ValueError, match="no entity mapping"):
            PresidioDetector([_rule("nonexistent_xyz_type")])

    def test_custom_rule_without_regex_raises_value_error(self):
        rule = PIIRule(
            name="my_custom",
            provider="custom",
            strategy=ActionType.scrub,
        )
        with patch.object(
            type(rule),
            "pattern_type",
            new_callable=lambda: property(lambda self: "custom"),
        ):
            with pytest.raises(ValueError, match="must specify a 'regex' field"):
                PresidioDetector([rule])

    def test_dynamic_fallback_for_unmapped_entity(self):
        """rule.name.upper() is tried when not in _RULE_TO_ENTITY."""
        from presidio_analyzer import AnalyzerEngine

        supported = set(AnalyzerEngine().get_supported_entities())
        # Find a supported entity that is NOT in _RULE_TO_ENTITY
        mapped_entities = set(_RULE_TO_ENTITY.values())
        unmapped = supported - mapped_entities
        if not unmapped:
            pytest.skip("All supported entities already mapped")

        entity = next(iter(unmapped))
        rule_name = entity.lower()
        detector = PresidioDetector([_rule(rule_name)])
        assert entity in detector._entities

    def test_multiple_rules_all_registered(self):
        rules = [_rule("email"), _rule("phone"), _rule("credit_card")]
        detector = PresidioDetector(rules)
        assert len(detector._entities) == 3
        assert set(detector._entities) == {
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "CREDIT_CARD",
        }

    def test_mixed_builtin_and_custom_rules(self):
        rules = [
            _rule("email"),
            _rule("employee_id", regex=r"\bEMP-\d{6}\b"),
        ]
        detector = PresidioDetector(rules)
        assert "EMAIL_ADDRESS" in detector._entities
        assert "EMPLOYEE_ID" in detector._entities


# ---------------------------------------------------------------------------
# TestFindAll — detection, ordering, deduplication
# ---------------------------------------------------------------------------


class TestFindAll:
    """Test find_all detection, match ordering, and deduplication."""

    def test_empty_text_returns_empty(self):
        detector = PresidioDetector([_rule("email")])
        assert detector.find_all("") == []

    def test_no_entities_configured_returns_empty(self):
        detector = PresidioDetector([])
        assert detector.find_all("user@example.com") == []

    def test_detects_email_address(self):
        detector = PresidioDetector([_rule("email")])
        matches = detector.find_all("Contact user@example.com for help.")
        assert len(matches) >= 1
        values = [m.value for m in matches]
        assert any("user@example.com" in v for v in values)

    def test_detects_phone_number(self):
        detector = PresidioDetector([_rule("phone")])
        matches = detector.find_all("Call me at 212-555-1234 today.")
        assert len(matches) >= 1
        assert any(m.value == "212-555-1234" for m in matches)

    def test_detects_credit_card(self):
        detector = PresidioDetector([_rule("credit_card")])
        matches = detector.find_all("Card number is 4111111111111111.")
        assert len(matches) >= 1
        assert any("4111111111111111" in m.value for m in matches)

    def test_returns_correct_pii_match_fields(self):
        detector = PresidioDetector([_rule("email", strategy="redact")])
        text = "Reach out to admin@corp.io please."
        matches = detector.find_all(text)
        assert len(matches) >= 1

        match = next(m for m in matches if "admin@corp.io" in m.value)
        assert isinstance(match, PIIMatch)
        assert match.start == text.index("admin@corp.io")
        assert match.end == match.start + len("admin@corp.io")
        assert match.value == "admin@corp.io"
        assert match.rule_name == "email"
        assert match.label == "EMAIL"
        assert match.action == "redact"

    def test_match_label_uses_custom_label(self):
        rule = _rule("email", label="MAIL")
        detector = PresidioDetector([rule])
        matches = detector.find_all("user@example.com")
        assert len(matches) >= 1
        assert matches[0].label == "MAIL"

    def test_match_action_reflects_strategy(self):
        detector = PresidioDetector([_rule("email", strategy="mask")])
        matches = detector.find_all("user@example.com")
        assert len(matches) >= 1
        assert matches[0].action == "mask"

    def test_multiple_matches_ordered_by_position(self):
        detector = PresidioDetector([_rule("email")])
        text = "First a@x.com then b@y.com end."
        matches = detector.find_all(text)
        assert len(matches) >= 2
        positions = [m.start for m in matches]
        assert positions == sorted(positions)

    def test_overlapping_matches_deduplicated(self):
        """When Presidio returns overlapping spans, first-by-position wins."""
        detector = PresidioDetector([_rule("email"), _rule("phone")])
        text = "Contact user@example.com for info."
        matches = detector.find_all(text)
        email_match = next((m for m in matches if m.rule_name == "email"), None)
        assert email_match is not None
        assert email_match.value == "user@example.com"
        starts = [m.start for m in matches]
        assert len(starts) == len(set(starts)), "Duplicate start positions found"
        for i in range(1, len(matches)):
            assert matches[i].start >= matches[i - 1].end, (
                f"Overlap: match[{i - 1}] ends at {matches[i - 1].end}, "
                f"match[{i}] starts at {matches[i].start}"
            )

    def test_custom_regex_pattern_detected(self):
        detector = PresidioDetector([_rule("cust_id", regex=r"\bCUST-\d{4,8}\b")])
        matches = detector.find_all("Customer CUST-12345 placed an order.")
        assert len(matches) == 1
        assert matches[0].value == "CUST-12345"
        assert matches[0].rule_name == "cust_id"
        assert matches[0].label == "CUST_ID"

    def test_custom_regex_no_match_returns_empty(self):
        detector = PresidioDetector([_rule("cust_id", regex=r"\bCUST-\d{4,8}\b")])
        assert detector.find_all("No customer ID here.") == []

    def test_multiple_entity_types_detected(self):
        detector = PresidioDetector([_rule("email"), _rule("credit_card")])
        text = "Email user@example.com, card 4111111111111111."
        matches = detector.find_all(text)
        rule_names = {m.rule_name for m in matches}
        assert "email" in rule_names
        assert "credit_card" in rule_names


# ---------------------------------------------------------------------------
# TestEdgeCases — boundary conditions and unusual inputs
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_text_with_no_pii_returns_empty(self):
        detector = PresidioDetector([_rule("email"), _rule("phone")])
        matches = detector.find_all("This is perfectly clean text with no PII.")
        assert matches == []

    def test_single_character_text_returns_empty(self):
        detector = PresidioDetector([_rule("email")])
        assert detector.find_all("x") == []

    def test_whitespace_only_text_returns_empty(self):
        detector = PresidioDetector([_rule("email")])
        assert detector.find_all("   \n\t  ") == []

    def test_none_like_empty_returns_empty(self):
        """Empty string is falsy, should short-circuit."""
        detector = PresidioDetector([_rule("email")])
        assert detector.find_all("") == []

    def test_long_text_without_pii(self):
        detector = PresidioDetector([_rule("email")])
        text = "lorem ipsum dolor sit amet " * 200
        assert detector.find_all(text) == []

    def test_pii_at_start_of_text(self):
        detector = PresidioDetector([_rule("email")])
        matches = detector.find_all("user@example.com is the contact.")
        assert len(matches) >= 1
        assert matches[0].start == 0

    def test_pii_at_end_of_text(self):
        detector = PresidioDetector([_rule("email")])
        matches = detector.find_all("Contact: user@example.com")
        assert len(matches) >= 1

    def test_custom_regex_multiple_matches(self):
        detector = PresidioDetector([_rule("ticket", regex=r"\bTKT-\d{5}\b")])
        matches = detector.find_all("Tickets TKT-00001 and TKT-00002 are open.")
        assert len(matches) == 2
        values = {m.value for m in matches}
        assert values == {"TKT-00001", "TKT-00002"}


# ---------------------------------------------------------------------------
# TestRuleToEntityMapping — validate the constant dict itself
# ---------------------------------------------------------------------------


class TestRuleToEntityMapping:
    """Validate the _RULE_TO_ENTITY constant dict."""

    def test_all_expected_keys_present(self):
        expected = {
            "email",
            "phone",
            "credit_card",
            "ssn",
            "ip_address",
            "iban",
            "address",
        }
        assert set(_RULE_TO_ENTITY.keys()) == expected

    def test_email_maps_correctly(self):
        assert _RULE_TO_ENTITY["email"] == "EMAIL_ADDRESS"

    def test_phone_maps_correctly(self):
        assert _RULE_TO_ENTITY["phone"] == "PHONE_NUMBER"

    def test_credit_card_maps_correctly(self):
        assert _RULE_TO_ENTITY["credit_card"] == "CREDIT_CARD"

    def test_ssn_maps_correctly(self):
        assert _RULE_TO_ENTITY["ssn"] == "US_SSN"

    def test_ip_address_maps_correctly(self):
        assert _RULE_TO_ENTITY["ip_address"] == "IP_ADDRESS"

    def test_iban_maps_correctly(self):
        assert _RULE_TO_ENTITY["iban"] == "IBAN_CODE"

    def test_address_maps_correctly(self):
        assert _RULE_TO_ENTITY["address"] == "LOCATION"
