"""Checksum-aware PII patterns and secret regexes."""

from __future__ import annotations

import pytest
from agent_guardrails.checks.pii import redact_pii
from agent_guardrails.checks.secrets import mask_secrets


def test_pattern_masks_checksummed_identifiers() -> None:
    text = (
        "ada@example.com 4111 1111 1111 1111 123-45-6789 "
        "GB82WEST12345698765432 12345678Z"
    )
    masked, kinds = redact_pii(text, backend="pattern")
    assert "ada@example.com" not in masked
    assert "4111 1111 1111 1111" not in masked
    assert "123-45-6789" not in masked
    assert "GB82WEST12345698765432" not in masked
    assert "12345678Z" not in masked
    assert set(kinds) >= {"EMAIL", "CREDIT_CARD", "SSN", "IBAN", "NIF"}


def test_invalid_card_and_nif_stay() -> None:
    text = "card 4111 1111 1111 1112 and id 12345678A"
    masked, kinds = redact_pii(text, backend="pattern")
    assert "4111 1111 1111 1112" in masked
    assert "12345678A" in masked
    assert kinds == []


def test_auto_falls_back_when_presidio_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(text: str, analyzer: object) -> tuple[str, list[str]]:
        del text, analyzer
        raise RuntimeError("no spacy model")

    monkeypatch.setattr("agent_guardrails.checks.pii._redact_presidio", _boom)
    masked, kinds = redact_pii("ada@example.com", backend="auto")
    assert masked == "[EMAIL]"
    assert kinds == ["EMAIL"]


def test_presidio_backend_does_not_swallow_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(text: str, analyzer: object) -> tuple[str, list[str]]:
        del text, analyzer
        raise RuntimeError("no spacy model")

    monkeypatch.setattr("agent_guardrails.checks.pii._redact_presidio", _boom)
    with pytest.raises(RuntimeError, match="no spacy model"):
        redact_pii("ada@example.com", backend="presidio")


def test_assigned_secret_and_private_key() -> None:
    begin = "-----BEGIN " + "PRIVATE KEY-----"
    end = "-----END " + "PRIVATE KEY-----"
    text = f"token=abcdEFGH1234\n{begin}\nabc\n{end}"  # gitleaks:allow
    masked, kinds = mask_secrets(text)
    assert "abcdEFGH1234" not in masked
    assert begin not in masked
    assert end not in masked
    assert masked.count("[SECRET]") == 2
    assert kinds == ["secret", "secret"]
