"""PII scrubbing utilities for production error responses.

Removes personally identifiable information from error messages, stack traces,
and other sensitive data before sending to clients.
"""

import re
from typing import Any

from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

# Patterns to redact from error messages
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
FILEPATH_PATTERN = re.compile(r"(/[a-zA-Z0-9_\-./]+)|([A-Z]:\\[a-zA-Z0-9_\-\\./]+)")
UUID_PATTERN = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I
)
IP_ADDRESS_PATTERN = re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")
JWT_PATTERN = re.compile(r"\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")

# Keywords that indicate sensitive data in field names
SENSITIVE_KEYWORDS = {
    "password",
    "secret",
    "token",
    "key",
    "credential",
    "auth",
    "ssn",
    "social_security",
    "credit_card",
    "api_key",
}


def scrub_pii(text: str) -> str:
    """Remove PII patterns from text.

    Redacts:
    - Email addresses
    - File paths
    - UUIDs (partial redaction)
    - IP addresses
    - JWT tokens

    Args:
        text: Input text potentially containing PII

    Returns:
        Text with PII replaced by [REDACTED]
    """
    if not settings.is_production:
        return text

    # Redact emails
    text = EMAIL_PATTERN.sub("[EMAIL_REDACTED]", text)

    # Redact file paths (keep just filename)
    def redact_path(match: re.Match) -> str:
        path = match.group(0)
        # Keep just the filename
        parts = path.replace("\\", "/").split("/")
        return f"[PATH]/{parts[-1]}" if parts else "[PATH_REDACTED]"

    text = FILEPATH_PATTERN.sub(redact_path, text)

    # Redact UUIDs (keep first 8 chars for tracing)
    def redact_uuid(match: re.Match) -> str:
        uuid = match.group(0)
        return f"{uuid[:8]}-[REDACTED]"

    text = UUID_PATTERN.sub(redact_uuid, text)

    # Redact IP addresses
    text = IP_ADDRESS_PATTERN.sub("[IP_REDACTED]", text)

    # Redact JWT tokens
    text = JWT_PATTERN.sub("[TOKEN_REDACTED]", text)

    return text


def scrub_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively scrub PII from dictionary values.

    Redacts values for keys containing sensitive keywords.
    Also scrubs string values for PII patterns.

    Args:
        data: Dictionary potentially containing PII

    Returns:
        Dictionary with PII scrubbed
    """
    if not settings.is_production:
        return data

    scrubbed: dict[str, Any] = {}
    for key, value in data.items():
        key_lower = key.lower()

        # Check if key name suggests sensitive data
        if any(kw in key_lower for kw in SENSITIVE_KEYWORDS):
            scrubbed[key] = "[REDACTED]"
        elif isinstance(value, str):
            scrubbed[key] = scrub_pii(value)
        elif isinstance(value, dict):
            scrubbed[key] = scrub_dict(value)
        elif isinstance(value, list):
            scrubbed[key] = [
                scrub_dict(item)
                if isinstance(item, dict)
                else scrub_pii(item)
                if isinstance(item, str)
                else item
                for item in value
            ]
        else:
            scrubbed[key] = value

    return scrubbed


def scrub_error_response(detail: str, exc: Exception | None = None) -> dict[str, Any]:
    """Create a scrubbed error response for production.

    In production:
    - Scrubs PII from detail message
    - Omits stack traces
    - Removes internal paths

    In development:
    - Returns full error details for debugging

    Args:
        detail: Error message detail
        exc: Optional exception for additional context

    Returns:
        Scrubbed error response dictionary
    """
    if not settings.is_production:
        # Development: return full details
        response = {"detail": detail}
        if exc:
            response["exception_type"] = type(exc).__name__
            response["exception_message"] = str(exc)
        return response

    # Production: scrub PII and limit information
    scrubbed_detail = scrub_pii(detail)

    response = {
        "detail": scrubbed_detail,
        "error_type": "internal_error",
    }

    # Only include exception type, not the message (may contain PII)
    if exc:
        response["exception_type"] = type(exc).__name__

    return response
