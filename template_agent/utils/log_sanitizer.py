"""Utilities for redacting sensitive data before logging."""

import json
import re
from typing import Any

EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

TOKEN_PATTERN = re.compile(
    r"(Bearer\s+\S+|"
    r"sk-[A-Za-z0-9_-]+|"
    r"pk-[A-Za-z0-9_-]+|"
    r"AIza[A-Za-z0-9_-]+|"
    r"AQ\.[A-Za-z0-9_-]+)"
)

SENSITIVE_HEADERS = {
    "authorization",
    "x-token",
    "cookie",
    "set-cookie",
    "x-api-key",
    "api-key",
}

SENSITIVE_KEYS = {
    "message",
    "password",
    "token",
    "api_key",
    "secret",
    "authorization",
    "cookie",
    "credentials",
    "google_application_credentials_content",
}

REDACTED = "[REDACTED]"


def sanitize_string(
    text: str, redact_pii: bool = True, redact_tokens: bool = True
) -> str:
    """Redact PII and tokens from a plain string."""
    if not text:
        return text

    result = text
    if redact_tokens:
        result = TOKEN_PATTERN.sub(REDACTED, result)
    if redact_pii:
        result = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", result)
    return result


def sanitize_headers(headers: dict[str, str]) -> dict[str, str]:
    """Redact sensitive HTTP headers."""
    safe: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in SENSITIVE_HEADERS:
            safe[key] = REDACTED
        else:
            safe[key] = value
    return safe


def sanitize_dict(
    data: dict[str, Any],
    redact_pii: bool = True,
    redact_tokens: bool = True,
) -> dict[str, Any]:
    """Recursively sanitize a dictionary for logging."""
    safe: dict[str, Any] = {}
    for key, value in data.items():
        if key.lower() in SENSITIVE_KEYS:
            safe[key] = REDACTED
        elif isinstance(value, dict):
            safe[key] = sanitize_dict(value, redact_pii, redact_tokens)
        elif isinstance(value, list):
            safe[key] = [
                sanitize_dict(item, redact_pii, redact_tokens)
                if isinstance(item, dict)
                else sanitize_string(item, redact_pii, redact_tokens)
                if isinstance(item, str)
                else item
                for item in value
            ]
        elif isinstance(value, str):
            safe[key] = sanitize_string(value, redact_pii, redact_tokens)
        else:
            safe[key] = value
    return safe


def sanitize_log_data(
    data: Any,
    *,
    enabled: bool = True,
    redact_pii: bool = True,
    redact_tokens: bool = True,
) -> Any:
    """Entry point: sanitize any log payload when enabled."""
    if not enabled:
        return data
    if isinstance(data, str):
        return sanitize_string(data, redact_pii, redact_tokens)
    if isinstance(data, dict):
        return sanitize_dict(data, redact_pii, redact_tokens)
    return data


def sanitize_request_body(
    body: str,
    *,
    enabled: bool = True,
    redact_pii: bool = True,
    redact_tokens: bool = True,
) -> Any:
    """Sanitize a request body string, parsing JSON when possible."""
    if not enabled:
        return body
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            return sanitize_dict(parsed, redact_pii, redact_tokens)
    except json.JSONDecodeError:
        pass
    return sanitize_string(body, redact_pii, redact_tokens)


def message_log_metadata(content: Any) -> dict[str, Any]:
    """Return safe metadata for logging message content without exposing it."""
    text = "" if content is None else str(content)
    return {"content_length": len(text)}
