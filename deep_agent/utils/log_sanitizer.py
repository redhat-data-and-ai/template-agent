"""Log sanitization for redacting credentials and sensitive content.

Protects the logging pipeline. Agent-level PII middleware
(:mod:`deep_agent.src.pii`) covers personal data in prompts/responses;
this module covers secrets and free-text fields that should never appear
in logs.

Division of responsibility:
- Credentials/secrets: regex redaction here (Bearer, JWT, API keys, etc.).
- Personal PII: delegated to the global ``PIIScrubber`` via ``get_scrubber()``
  when available; otherwise credentials-only.
- User-authored content (message/prompt/output): length-only placeholders so
  free text is never emitted even when no regex match exists.
"""

from __future__ import annotations

import re
from typing import Any

from structlog.typing import EventDict, Processor, WrappedLogger

REDACTED = "***REDACTED***"

CREDENTIAL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE),
        "Bearer ***TOKEN***",
    ),
    (re.compile(r"Basic\s+[A-Za-z0-9+/]+=*", re.IGNORECASE), "Basic ***TOKEN***"),
    (
        re.compile(r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
        "***JWT***",
    ),
    (
        re.compile(
            r"(?i)(?:api[_-]?key|apikey)[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9\-._~+/]{16,}[\"']?"
        ),
        "***API_KEY***",
    ),
    (
        re.compile(
            r"(?i)(?:password|passwd|pwd)[\"']?\s*[:=]\s*[\"']?[^\s\"',}{]+[\"']?"
        ),
        "***PASSWORD***",
    ),
    (
        re.compile(
            r"(?i)(?:secret[_-]?key|client[_-]?secret)[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9\-._~+/]{8,}[\"']?"
        ),
        "***SECRET***",
    ),
    (re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}"), "***AWS_KEY***"),
    (
        re.compile(r"(?i)(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,}"),
        "***GITHUB_TOKEN***",
    ),
    (re.compile(r"AIza[0-9A-Za-z\-_]{20,}"), "***GOOGLE_API_KEY***"),
    (re.compile(r"AQ\.[A-Za-z0-9_\-]{20,}"), "***API_KEY***"),
]

SENSITIVE_HEADER_KEYS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "x-token",
        "x-auth-token",
    }
)

SENSITIVE_DICT_KEYS = frozenset(
    {
        "password",
        "passwd",
        "pwd",
        "secret",
        "secret_key",
        "secretkey",
        "client_secret",
        "token",
        "access_token",
        "refresh_token",
        "id_token",
        "api_key",
        "apikey",
        "private_key",
        "privatekey",
        "credential",
        "credentials",
        "authorization",
        "session_key",
        "google_application_credentials_content",
    }
)

USER_CONTENT_KEYS = frozenset(
    {
        "message",
        "content",
        "input",
        "output",
        "prompt",
        "query",
        "question",
        "answer",
        "completion",
        "text",
        "user_input",
    }
)

ID_LIKE_KEYS = frozenset(
    {
        "id",
        "run_id",
        "parent_run_id",
        "tool_call_id",
        "thread_id",
        "checkpoint_id",
        "checkpoint_ns",
        "trace_id",
        "span_id",
        "session_id",
        "request_id",
        "correlation_id",
        "call_id",
        "org_id",
        "agent_id",
        "user_id",
    }
)


def content_placeholder(value: Any) -> str:
    """Return a length-only stand-in for user-authored content."""
    if value is None:
        return REDACTED
    text = value if isinstance(value, str) else str(value)
    return f"<redacted: {len(text)} chars>"


class LogSanitizer:
    """Redact credentials, sensitive keys, and optional PII from log payloads."""

    def __init__(
        self,
        enabled: bool = True,
        custom_patterns: list[tuple[re.Pattern[str], str]] | None = None,
        scrub_pii: bool = True,
        redact_user_content: bool = True,
    ) -> None:
        """Configure redaction toggles and optional custom credential patterns."""
        self.enabled = enabled
        self.scrub_pii = scrub_pii
        self.redact_user_content = redact_user_content
        self._patterns: list[tuple[re.Pattern[str], str]] = []
        if enabled:
            self._patterns = list(CREDENTIAL_PATTERNS)
            if custom_patterns:
                self._patterns.extend(custom_patterns)

    def _scrub_pii_text(self, value: str) -> str:
        if not self.scrub_pii:
            return value
        try:
            from deep_agent.src.pii import get_scrubber

            scrubber = get_scrubber()
            if scrubber is None:
                return value
            return scrubber.scrub_one_way(value)
        except Exception:
            return value

    def sanitize_string(self, value: str, scrub_pii: bool = True) -> str:
        """Redact credentials and optionally PII from a string."""
        if not self.enabled or not value:
            return value
        for pattern, replacement in self._patterns:
            value = pattern.sub(replacement, value)
        if scrub_pii:
            value = self._scrub_pii_text(value)
        return value

    def sanitize_value(self, value: Any, scrub_pii: bool = True) -> Any:
        """Recursively sanitize a string, mapping, or sequence."""
        if not self.enabled:
            return value
        if isinstance(value, str):
            return self.sanitize_string(value, scrub_pii=scrub_pii)
        if isinstance(value, dict):
            return self._sanitize_dict(value)
        if isinstance(value, list):
            return [self.sanitize_value(item, scrub_pii=scrub_pii) for item in value]
        if isinstance(value, tuple):
            return tuple(
                self.sanitize_value(item, scrub_pii=scrub_pii) for item in value
            )
        return value

    def _sanitize_dict(self, data: dict[Any, Any]) -> dict[Any, Any]:
        result: dict[Any, Any] = {}
        for key, val in data.items():
            lowered = str(key).lower()
            normalised = lowered.replace("-", "_")
            if lowered in SENSITIVE_HEADER_KEYS or normalised in SENSITIVE_DICT_KEYS:
                result[key] = REDACTED
            elif self.redact_user_content and normalised in USER_CONTENT_KEYS:
                result[key] = content_placeholder(val)
            else:
                result[key] = self.sanitize_value(
                    val, scrub_pii=normalised not in ID_LIKE_KEYS
                )
        return result


_default_sanitizer: LogSanitizer | None = None


def parse_custom_patterns(raw: str) -> list[tuple[re.Pattern[str], str]]:
    """Compile a comma-separated list of regexes, skipping invalid entries."""
    if not raw:
        return []
    patterns: list[tuple[re.Pattern[str], str]] = []
    for entry in raw.split(","):
        stripped = entry.strip()
        if not stripped:
            continue
        try:
            patterns.append((re.compile(stripped), REDACTED))
        except re.error:
            continue
    return patterns


def get_default_sanitizer() -> LogSanitizer:
    """Return the cached sanitizer built from settings (enabled on failure)."""
    global _default_sanitizer  # noqa: PLW0603
    if _default_sanitizer is None:
        try:
            from deep_agent.src.settings import settings

            _default_sanitizer = LogSanitizer(
                enabled=settings.LOG_SANITIZATION_ENABLED,
                custom_patterns=parse_custom_patterns(
                    settings.LOG_SANITIZATION_CUSTOM_PATTERNS
                ),
                redact_user_content=settings.LOG_REDACT_USER_CONTENT,
            )
        except Exception:
            _default_sanitizer = LogSanitizer(enabled=True)
    return _default_sanitizer


def reset_default_sanitizer() -> None:
    """Drop the cached sanitizer so the next call rereads settings."""
    global _default_sanitizer  # noqa: PLW0603
    _default_sanitizer = None


def sanitize_headers(headers: dict[str, str]) -> dict[str, str]:
    """Redact sensitive HTTP header values before logging."""
    sanitized: dict[str, str] = get_default_sanitizer().sanitize_value(headers)
    return sanitized


def create_sanitize_processor() -> Processor:
    """Build a structlog processor that sanitizes every event-dict value."""

    def sanitize_processor(
        logger: WrappedLogger, method_name: str, event_dict: EventDict
    ) -> EventDict:
        del logger, method_name
        sanitizer = get_default_sanitizer()
        if not sanitizer.enabled:
            return event_dict
        sanitized: EventDict = sanitizer.sanitize_value(event_dict)
        return sanitized

    return sanitize_processor
