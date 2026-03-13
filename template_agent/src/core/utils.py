"""Shared utility functions for the template agent."""

from __future__ import annotations

import json
import re
from typing import Any


def safe_json_parse(
    text: str,
    pattern: str = r"\{[\s\S]*\}",
    default: Any = None,
) -> Any:
    """Extract and parse the first JSON object/array from text.

    Handles markdown-wrapped LLM output by searching for JSON patterns
    and parsing them safely.

    Args:
        text: Raw text that may contain embedded JSON.
        pattern: Regex pattern to locate JSON in the text.
            Defaults to matching a top-level JSON object.
        default: Value to return when no JSON is found or parsing fails.

    Returns:
        Parsed JSON value, or *default* on failure.
    """
    if not text:
        return default
    try:
        match = re.search(pattern, text)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass
    return default


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate *text* to *max_length* characters, appending *suffix* if truncated."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def strip_annotation_tags(text: str) -> str:
    """Remove annotation tags from text output.

    Strips [UNVERIFIED], [CORRECTED: ...], [NOTE: ...], [FLAGGED: ...] and
    similar tags that may be added during processing but shouldn't appear
    in final output.
    """
    text = re.sub(r" ?\[UNVERIFIED[^\]]{0,200}\]", "", text)
    text = re.sub(r" ?\[CORRECTED[^\]]{0,200}\]", "", text)
    text = re.sub(r" ?\[NOTE[^\]]{0,200}\]", "", text)
    text = re.sub(r" ?\[FLAGGED[^\]]{0,200}\]", "", text)
    return text.strip()


# Patterns for error simplification (technical -> user-friendly)
_ERROR_SIMPLIFICATION_PATTERNS = [
    (["timeout", "timed out"], "Request timed out"),
    (["rate limit", "429"], "Rate limit exceeded"),
    (["permission denied", "access denied", "forbidden", "403"], "Access denied"),
    (["not found", "404"], "Resource not found"),
    (["connection", "network", "unreachable"], "Connection error"),
]


def _simplify_error_default(error: str) -> str:
    """Default error simplification for unknown patterns."""
    simplified = str(error).strip()
    if "File " in simplified:
        simplified = simplified.split("File ")[0].strip()
    if len(simplified) > 100:
        simplified = simplified[:97] + "..."
    return simplified or "Unknown error"


def simplify_error_for_display(error: str) -> str:
    """Simplify technical error messages for user-friendly display."""
    error_lower = (error or "").lower()
    for patterns, message in _ERROR_SIMPLIFICATION_PATTERNS:
        if any(p in error_lower for p in patterns):
            return message
    return _simplify_error_default(error)


def is_model_config_error(e: Exception) -> bool:
    """Check if an exception indicates a non-retryable model configuration error."""
    error_str = str(e).lower()
    return (
        "max_tokens" in error_str
        and ("maximum allowed" in error_str or "which is the maximum" in error_str)
    ) or ("invalid_request_error" in error_str and "max_tokens" in error_str)
