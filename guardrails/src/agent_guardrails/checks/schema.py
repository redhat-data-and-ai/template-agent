"""Structured-query field check. No model.

Free text is not a structured query. A JSON object is checked when the host
configured allowed or required fields.
"""

from __future__ import annotations

import json


def structured_field_violation(
    text: str,
    *,
    allowed_fields: list[str],
    required_fields: list[str],
) -> str | None:
    """Return a reason to block, or ``None`` when the query is valid."""
    if not allowed_fields and not required_fields:
        return None
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return "Structured query is not valid JSON."
    if not isinstance(payload, dict):
        return "Structured query must be a JSON object."
    allowed = set(allowed_fields)
    if allowed:
        unknown = sorted(key for key in payload if key not in allowed)
        if unknown:
            return f"Unknown field {unknown[0]!r}."
    missing = [name for name in required_fields if name not in payload]
    if missing:
        return f"Missing field {missing[0]!r}."
    return None
