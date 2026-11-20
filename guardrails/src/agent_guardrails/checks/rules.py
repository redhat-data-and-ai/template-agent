"""Length and character rules. No model."""

from __future__ import annotations

import re


def text_rule_violation(
    text: str,
    *,
    max_chars: int,
    permitted_characters: str | None,
) -> str | None:
    """Return a reason to block, or ``None`` when the text passes."""
    if max_chars >= 0 and len(text) > max_chars:
        return f"Input exceeds {max_chars} characters."
    if permitted_characters:
        try:
            pattern = re.compile(permitted_characters)
        except re.error as exc:
            return f"Permitted-character pattern is invalid: {exc}"
        if pattern.fullmatch(text) is None:
            return "Input contains characters outside the permitted set."
    return None
