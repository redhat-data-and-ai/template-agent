"""Regex masking for keys, tokens, and private-key blocks. No model."""

from __future__ import annotations

import re

_ASSIGNED = re.compile(
    r"(?i)\b(?:api[_-]?key|access[_-]?token|secret|token)\b[\s:=]+['\"]?"
    r"([A-Za-z0-9_\-]{12,})"
)
_AWS = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_SK = re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")
_PRIVATE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.DOTALL,
)


def mask_secrets(text: str, mask: str = "[SECRET]") -> tuple[str, list[str]]:
    """Replace secret spans. The kind list is empty when nothing matched."""
    kinds: list[str] = []

    def _mask_match(match: re.Match[str]) -> str:
        kinds.append("secret")
        if match.lastindex:
            full = match.group(0)
            rel_start = match.start(1) - match.start(0)
            rel_end = match.end(1) - match.start(0)
            return full[:rel_start] + mask + full[rel_end:]
        return mask

    masked = _PRIVATE.sub(_mask_match, text)
    masked = _AWS.sub(_mask_match, masked)
    masked = _SK.sub(_mask_match, masked)
    masked = _ASSIGNED.sub(_mask_match, masked)
    return masked, kinds
