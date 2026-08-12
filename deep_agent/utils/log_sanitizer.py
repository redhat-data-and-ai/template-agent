"""Log sanitization for redacting credentials and PII from log output.

This module protects the *logging* pipeline, which is not covered by the
agent-level PII middleware in :mod:`deep_agent.src.pii`.

Division of responsibility
--------------------------
* **Credentials / secrets** (bearer tokens, JWTs, API keys, passwords, AWS and
  GitHub tokens) are matched here with regexes.  The PII subsystem deliberately
  does not model secrets — its ``BUILTIN_PATTERNS`` and the Presidio entity set
  both cover *personal* data only — so these patterns are new.
* **Personal PII** (emails, phone numbers, SSNs, credit cards, addresses, ...)
  is delegated to the already-configured global ``PIIScrubber`` via
  :func:`deep_agent.src.pii.get_scrubber`, using its stateless
  ``scrub_one_way()`` entry point.  PII detection is therefore never
  reimplemented here, and log redaction automatically follows whatever rules
  are declared in ``agent.yaml``.
* **User-authored content** (prompts, messages, model output) is replaced with
  a length-only placeholder rather than pattern-matched, because a prompt can
  disclose sensitive information without containing any token a regex or PII
  detector would recognise.  Pattern matching cannot make free text safe, so
  the content is simply never emitted.

When the scrubber has not been initialised (``get_scrubber()`` returns
``None``) sanitization degrades to credentials-only rather than falling back to
a second, divergent set of PII regexes.  Duplicating PII detection would apply
a policy the operator never configured and would reintroduce the false-positive
problems the scrubber already solves.  Credentials are always redacted because
they are never legitimately loggable.

Environment variables:
    LOG_SANITIZATION_ENABLED: enable/disable sanitization (default: true)
    LOG_SANITIZATION_CUSTOM_PATTERNS: comma-separated extra regexes to redact
    LOG_REDACT_USER_CONTENT: replace prompt/message/output values with a
        length-only placeholder (default: true)
"""

from __future__ import annotations

import re
from typing import Any

from structlog.typing import EventDict, Processor, WrappedLogger

REDACTED = "***REDACTED***"

# ---------------------------------------------------------------------------
# Credential patterns
# ---------------------------------------------------------------------------

# (compiled_regex, replacement).  Order matters: more specific patterns come
# first so that a generic pattern cannot claim part of a longer secret.
#
# Personal PII is intentionally absent — see the module docstring.
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
]

# HTTP headers whose value is redacted wholesale, regardless of content.
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

# Mapping/event-dict keys whose value is redacted wholesale.  Keys are
# normalised to lowercase with ``-`` replaced by ``_`` before lookup.
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
    }
)

# Keys whose values are user- or model-authored free text.  Pattern matching
# alone is not enough here: a prompt can disclose sensitive information without
# containing anything a regex or PII detector would flag.  The value is replaced
# with a length-only placeholder, which keeps the field useful for debugging
# (empty vs truncated vs oversized input) while never emitting what was said.
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

# Keys holding correlation identifiers rather than free text.  PII scrubbing is
# skipped for these because the detectors produce false positives on opaque IDs
# (a UUID substring can match the phone-number pattern), which would corrupt
# values that must stay byte-for-byte stable across log lines.  This mirrors
# ``_ID_LIKE_KEYS`` in :mod:`deep_agent.src.pii.scrubber`; ``org_id`` and
# ``agent_id`` are added because pylogger injects them into every event.
# Credential patterns still apply — they are keyword-anchored and cannot match
# an opaque identifier.
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
    }
)


# ---------------------------------------------------------------------------
# Sanitizer
# ---------------------------------------------------------------------------


def content_placeholder(value: Any) -> str:
    """Return a length-only stand-in for user-authored content.

    Reporting the length keeps the common debugging questions answerable —
    was the input empty, unexpectedly short, or oversized — without
    disclosing the text itself.
    """
    if value is None:
        return REDACTED
    text = value if isinstance(value, str) else str(value)
    return f"<redacted: {len(text)} chars>"


class LogSanitizer:
    """Redacts credentials, sensitive keys, and PII from log payloads.

    Credential redaction is regex-based and always available.  PII redaction is
    delegated to the global ``PIIScrubber`` and is therefore active only once
    the PII middleware has been initialised.  User-authored content is replaced
    with a length-only placeholder rather than pattern-matched.
    """

    def __init__(
        self,
        enabled: bool = True,
        custom_patterns: list[tuple[re.Pattern[str], str]] | None = None,
        scrub_pii: bool = True,
        redact_user_content: bool = True,
    ) -> None:
        """Initialise the sanitizer.

        Args:
            enabled: Master toggle. When false every method is a passthrough.
            custom_patterns: Extra ``(compiled_regex, replacement)`` pairs
                applied after the built-in credential patterns.
            scrub_pii: Whether to delegate personal-PII redaction to the
                global ``PIIScrubber``.
            redact_user_content: Whether to replace values under
                ``USER_CONTENT_KEYS`` with a length-only placeholder.
        """
        self.enabled = enabled
        self.scrub_pii = scrub_pii
        self.redact_user_content = redact_user_content
        self._patterns: list[tuple[re.Pattern[str], str]] = []
        if enabled:
            self._patterns = list(CREDENTIAL_PATTERNS)
            if custom_patterns:
                self._patterns.extend(custom_patterns)

    def _scrub_pii_text(self, value: str) -> str:
        """Delegate PII redaction to the global scrubber, if one is active.

        Returns *value* unchanged when the PII middleware was never
        initialised, when the import fails, or when the scrubber raises.
        Emitting a log line must never fail because sanitization could not
        run, so the import is inside the guarded block: callers log from
        inside ``except ImportError`` handlers, where a lazy import can
        otherwise raise a second error that escapes the handler.
        """
        if not self.scrub_pii:
            return value
        try:
            # Imported lazily: deep_agent.src.pii.scrubber imports pylogger,
            # which imports this module.
            from deep_agent.src.pii import get_scrubber

            scrubber = get_scrubber()
            if scrubber is None:
                return value
            return scrubber.scrub_one_way(value)
        except Exception:
            return value

    def sanitize_string(self, value: str, scrub_pii: bool = True) -> str:
        """Redact credentials and, optionally, PII from a string.

        Args:
            value: The text to sanitize.
            scrub_pii: Set false for values held under an ID-like key, where
                PII detection would corrupt a correlation identifier.
        """
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
        """Redact sensitive keys outright and recurse into everything else."""
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


# ---------------------------------------------------------------------------
# Module-level default sanitizer
# ---------------------------------------------------------------------------

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
            # A malformed operator-supplied pattern must not stop logging.
            continue
    return patterns


def get_default_sanitizer() -> LogSanitizer:
    """Return the cached sanitizer, building it from settings on first use.

    Settings are imported lazily to break the import cycle
    ``settings -> pylogger -> log_sanitizer -> settings``.  If settings cannot
    be loaded the sanitizer still defaults to *enabled*, so a configuration
    failure can never silently disable redaction.
    """
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
    """Redact sensitive HTTP header values.

    No middleware on this branch logs headers today, so this has no production
    caller yet; it is the entry point for any that starts to. Header keys are
    already covered by the structlog processor via ``SENSITIVE_HEADER_KEYS``, so
    this is defence in depth for callers that want to scrub a header mapping
    before it ever reaches a log event.
    """
    sanitized: dict[str, str] = get_default_sanitizer().sanitize_value(headers)
    return sanitized


# ---------------------------------------------------------------------------
# structlog processor
# ---------------------------------------------------------------------------


def create_sanitize_processor() -> Processor:
    """Build a structlog processor that sanitizes every event-dict value.

    Returned as a closure so that settings and the global PII scrubber are
    resolved on the first log call rather than at import time.
    """

    def sanitize_processor(
        logger: WrappedLogger, method_name: str, event_dict: EventDict
    ) -> EventDict:
        sanitizer = get_default_sanitizer()
        if not sanitizer.enabled:
            return event_dict
        sanitized: EventDict = sanitizer.sanitize_value(event_dict)
        return sanitized

    return sanitize_processor
