"""PII masking. The agent continues with the masked text.

``pattern`` needs no spaCy model. It covers email, phone, Luhn card numbers,
SSN structure, IBAN mod-97, and Spanish NIF. ``presidio`` uses Microsoft
Presidio. ``auto`` uses Presidio when that stack imports, and patterns otherwise.
"""

from __future__ import annotations

import re
from typing import Any

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(
    r"(?<!\w)(?:\+\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4}(?!\w)"
)
_CARD = re.compile(r"(?<!\d)(?:\d{13,19}|\d{4}(?:[ -]\d{4}){3})(?!\d)")
_SSN = re.compile(r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
_NIF = re.compile(r"\b\d{8}[A-Z]\b")
_NIF_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"

_PRESIDIO_LABELS = {
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "CREDIT_CARD": "CREDIT_CARD",
    "US_SSN": "SSN",
    "IBAN_CODE": "IBAN",
    "ES_NIF": "NIF",
    "NIF": "NIF",
}


def redact_pii(
    text: str,
    *,
    backend: str = "auto",
    analyzer: Any | None = None,
) -> tuple[str, list[str]]:
    """Mask PII. Raises when ``backend`` is ``presidio`` and it cannot run."""
    if backend in ("presidio", "auto"):
        try:
            return _redact_presidio(text, analyzer)
        except Exception:
            if backend == "presidio":
                raise
    return _redact_patterns(text)


def _redact_patterns(text: str) -> tuple[str, list[str]]:
    spans: list[tuple[int, int, str]] = []
    _collect(spans, text, _CARD, "CREDIT_CARD", _luhn_ok)
    _collect(spans, text, _IBAN, "IBAN", _iban_ok)
    _collect(spans, text, _SSN, "SSN", None)
    _collect(spans, text, _NIF, "NIF", _nif_ok)
    _collect(spans, text, _EMAIL, "EMAIL", None)
    _collect(spans, text, _PHONE, "PHONE", None)
    return _apply(text, _resolve(spans))


def _collect(
    spans: list[tuple[int, int, str]],
    text: str,
    pattern: re.Pattern[str],
    kind: str,
    accept: Any,
) -> None:
    for match in pattern.finditer(text):
        raw = match.group(0)
        if accept is not None and not accept(raw):
            continue
        spans.append((match.start(), match.end(), kind))


def _resolve(spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    priority = {
        "CREDIT_CARD": 0,
        "IBAN": 1,
        "SSN": 2,
        "NIF": 3,
        "EMAIL": 4,
        "PHONE": 5,
    }
    chosen: list[tuple[int, int, str]] = []
    for span in sorted(spans, key=lambda item: (priority[item[2]], item[0])):
        if all(span[1] <= kept[0] or span[0] >= kept[1] for kept in chosen):
            chosen.append(span)
    chosen.sort()
    return chosen


def _apply(text: str, spans: list[tuple[int, int, str]]) -> tuple[str, list[str]]:
    kinds: list[str] = []
    cursor = 0
    parts: list[str] = []
    for start, end, kind in spans:
        parts.append(text[cursor:start])
        parts.append(f"[{kind}]")
        kinds.append(kind)
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts), kinds


def _digits(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


def _luhn_ok(value: str) -> bool:
    digits = _digits(value)
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    reverse = digits[::-1]
    for index, char in enumerate(reverse):
        number = int(char)
        if index % 2 == 1:
            number *= 2
            if number > 9:
                number -= 9
        total += number
    return total % 10 == 0


def _iban_ok(value: str) -> bool:
    compact = re.sub(r"\s+", "", value).upper()
    if not 15 <= len(compact) <= 34:
        return False
    if re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]+", compact) is None:
        return False
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(int(char, 36)) for char in rearranged)
    return int(numeric) % 97 == 1


def _nif_ok(value: str) -> bool:
    compact = value.strip().upper()
    if re.fullmatch(r"\d{8}[A-Z]", compact) is None:
        return False
    expected = _NIF_LETTERS[int(compact[:8]) % 23]
    return compact[-1] == expected


def _redact_presidio(text: str, analyzer: Any | None) -> tuple[str, list[str]]:
    engine = analyzer if analyzer is not None else _default_analyzer()
    results = engine.analyze(text=text, language="en")
    spans: list[tuple[int, int, str]] = []
    for item in results:
        label = _PRESIDIO_LABELS.get(getattr(item, "entity_type", ""), "")
        if not label:
            continue
        spans.append((int(item.start), int(item.end), label))
    return _apply(text, _resolve(spans))


_analyzer: Any = None


def _default_analyzer() -> Any:
    global _analyzer
    if _analyzer is None:
        from presidio_analyzer import AnalyzerEngine

        _analyzer = AnalyzerEngine()
    return _analyzer
