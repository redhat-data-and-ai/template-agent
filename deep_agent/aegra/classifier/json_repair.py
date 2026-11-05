"""JSON repair utilities for DiffusionGemma output quirks.

Handles: markdown code fences, missing outer braces, truncated JSON,
and nested JSON extraction.
"""

from __future__ import annotations

import json
import re

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*\n?([\s\S]*?)\n?\s*```")


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences wrapping JSON output."""
    match = _CODE_FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def fix_missing_outer_brace(text: str) -> str:
    """Fix DiffusionGemma quirk: JSON missing the opening '{'."""
    stripped = text.strip()
    if stripped.startswith('"') and not stripped.startswith("{"):
        stripped = "{" + stripped
    if stripped.startswith("{") and not stripped.endswith("}"):
        stripped = stripped.rstrip(", \n") + "}"
    return stripped


def repair_truncated_json(text: str) -> str | None:
    """Close unclosed braces/brackets in truncated JSON."""
    in_string = False
    escape_next = False
    stack: list[str] = []

    for c in text:
        if escape_next:
            escape_next = False
            continue
        if c == "\\":
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
        if in_string:
            continue
        if c in ("{", "["):
            stack.append("}" if c == "{" else "]")
        elif c in ("}", "]") and stack:
            stack.pop()

    if not stack:
        return None

    last_quote = text.rfind('"')
    if last_quote > 0 and in_string:
        text = text[: last_quote + 1]

    trimmed = text.rstrip(", \n\t")
    return trimmed + "".join(reversed(stack))


def _parsed_json(text: str) -> dict | list | None:
    """Return a dict or list, or None when the text is not that JSON type."""
    parsed = json.loads(text)
    if isinstance(parsed, (dict, list)):
        return parsed
    return None


def safe_parse_json(text: str) -> dict | list | None:
    """Parse JSON from LLM text, handling common quirks."""
    cleaned = strip_code_fences(text)

    try:
        return _parsed_json(cleaned)
    except json.JSONDecodeError:
        pass

    fixed = fix_missing_outer_brace(cleaned)
    try:
        return _parsed_json(fixed)
    except json.JSONDecodeError:
        pass

    repaired = repair_truncated_json(fixed)
    if repaired:
        try:
            return _parsed_json(repaired)
        except json.JSONDecodeError:
            pass

    repaired_raw = repair_truncated_json(fix_missing_outer_brace(text))
    if repaired_raw:
        try:
            return _parsed_json(repaired_raw)
        except json.JSONDecodeError:
            pass

    return None
