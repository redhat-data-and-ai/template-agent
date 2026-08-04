"""Unit tests for log sanitization — credential, header, and PII redaction."""

import re
import sys
from unittest.mock import MagicMock, patch

import pytest

from deep_agent.utils.log_sanitizer import (
    REDACTED,
    LogSanitizer,
    create_sanitize_processor,
    get_default_sanitizer,
    parse_custom_patterns,
    reset_default_sanitizer,
    sanitize_headers,
)


@pytest.fixture(autouse=True)
def _reset_sanitizer():
    """Reset the cached module-level sanitizer around every test."""
    reset_default_sanitizer()
    yield
    reset_default_sanitizer()


@pytest.fixture()
def no_scrubber():
    """Patch the global PII scrubber to be uninitialised."""
    with patch("deep_agent.src.pii.get_scrubber", return_value=None):
        yield


def _pii_scrubber(*names: str):
    """Build a real regex-backed PIIScrubber for the given builtin rules."""
    from deep_agent.src.pii.config import ActionType, PIIConfig, PIIRule
    from deep_agent.src.pii.scrubber import PIIScrubber

    rules = [
        PIIRule(name=n, strategy=ActionType.redact, provider="regex") for n in names
    ]
    return PIIScrubber(PIIConfig(enabled=True, rules=rules), hash_key=b"test-key")


class TestCredentialRedaction:
    """Credentials must be redacted regardless of PII scrubber state."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Authorization: Bearer abc123XYZ", "Bearer ***TOKEN***"),
            ("Authorization: Basic dXNlcjpwYXNz", "Basic ***TOKEN***"),
            ("password=hunter2", "***PASSWORD***"),
            ("passwd: s3cr3t", "***PASSWORD***"),
            ("api_key=abcdefghijklmnop1234", "***API_KEY***"),
            ("secret_key=abcd1234efgh", "***SECRET***"),
            ("client_secret=abcd1234efgh", "***SECRET***"),
            ("key AKIAIOSFODNN7EXAMPLE here", "***AWS_KEY***"),
            ("ghp_" + "a" * 36, "***GITHUB_TOKEN***"),
        ],
    )
    def test_credentials_are_redacted(self, raw, expected, no_scrubber):
        result = LogSanitizer().sanitize_string(raw)
        assert expected in result

    def test_jwt_is_redacted(self, no_scrubber):
        token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NSJ9.abcDEF123_-x"
        result = LogSanitizer().sanitize_string(f"token={token}")
        assert "***JWT***" in result
        assert token not in result

    def test_secret_value_never_survives(self, no_scrubber):
        result = LogSanitizer().sanitize_string("Bearer supersecrettokenvalue")
        assert "supersecrettokenvalue" not in result


class TestNonSensitivePassthrough:
    """Ordinary log content must be left byte-for-byte intact."""

    def test_plain_message_unchanged(self, no_scrubber):
        msg = "agent started on port 5002 with 3 tools"
        assert LogSanitizer().sanitize_string(msg) == msg

    def test_empty_string_unchanged(self, no_scrubber):
        assert LogSanitizer().sanitize_string("") == ""

    def test_non_string_scalars_unchanged(self, no_scrubber):
        s = LogSanitizer()
        assert s.sanitize_value(42) == 42
        assert s.sanitize_value(None) is None
        assert s.sanitize_value(True) is True

    def test_nested_collections_preserved(self, no_scrubber):
        s = LogSanitizer()
        result = s.sanitize_value({"items": ["a", ("b", 1)], "count": 2})
        assert result == {"items": ["a", ("b", 1)], "count": 2}
        assert isinstance(result["items"][1], tuple)

    def test_nested_credential_inside_list_is_redacted(self, no_scrubber):
        result = LogSanitizer().sanitize_value(["Bearer abc123", "safe"])
        assert result[0] == "Bearer ***TOKEN***"
        assert result[1] == "safe"


class TestHeaderRedaction:
    """Sensitive header and mapping keys are redacted wholesale."""

    @pytest.mark.parametrize(
        "key",
        [
            "authorization",
            "Authorization",
            "proxy-authorization",
            "cookie",
            "set-cookie",
            "x-api-key",
            "x-token",
            "x-auth-token",
        ],
    )
    def test_sensitive_headers_redacted(self, key, no_scrubber):
        assert sanitize_headers({key: "anything"})[key] == REDACTED

    @pytest.mark.parametrize(
        "key",
        ["password", "api_key", "access_token", "private_key", "credentials"],
    )
    def test_sensitive_dict_keys_redacted(self, key, no_scrubber):
        assert LogSanitizer().sanitize_value({key: "value"})[key] == REDACTED

    def test_hyphenated_key_normalised(self, no_scrubber):
        result = LogSanitizer().sanitize_value({"Access-Token": "abc"})
        assert result["Access-Token"] == REDACTED

    def test_benign_headers_untouched(self, no_scrubber):
        headers = {"user-agent": "curl/8.0", "content-type": "application/json"}
        assert sanitize_headers(headers) == headers

    def test_non_string_key_does_not_raise(self, no_scrubber):
        assert LogSanitizer().sanitize_value({1: "plain"}) == {1: "plain"}


class TestIdLikeKeys:
    """Correlation identifiers must survive PII scrubbing unchanged."""

    def test_id_like_values_not_pii_scrubbed(self):
        scrubber = _pii_scrubber("phone")
        trace = "550e8400-e29b-41d4-a716-446655440000"
        with patch("deep_agent.src.pii.get_scrubber", return_value=scrubber):
            result = LogSanitizer().sanitize_value(
                {"trace_id": trace, "request_id": trace}
            )
        assert result["trace_id"] == trace
        assert result["request_id"] == trace

    def test_free_text_still_pii_scrubbed(self):
        scrubber = _pii_scrubber("email")
        with patch("deep_agent.src.pii.get_scrubber", return_value=scrubber):
            result = LogSanitizer().sanitize_value({"message": "mail a@b.com now"})
        assert "a@b.com" not in result["message"]

    def test_credentials_still_redacted_under_id_key(self, no_scrubber):
        result = LogSanitizer().sanitize_value({"run_id": "Bearer abc123"})
        assert result["run_id"] == "Bearer ***TOKEN***"


class TestPiiDelegation:
    """PII handling is delegated to the global scrubber, never reimplemented."""

    def test_uses_scrubber_one_way(self):
        scrubber = MagicMock()
        scrubber.scrub_one_way.return_value = "clean"
        with patch("deep_agent.src.pii.get_scrubber", return_value=scrubber):
            assert LogSanitizer().sanitize_string("dirty") == "clean"
        scrubber.scrub_one_way.assert_called_once_with("dirty")

    def test_none_scrubber_falls_back_to_credentials_only(self, no_scrubber):
        result = LogSanitizer().sanitize_string("a@b.com used Bearer abc123")
        assert "Bearer ***TOKEN***" in result
        assert "a@b.com" in result

    def test_scrubber_failure_is_swallowed(self):
        scrubber = MagicMock()
        scrubber.scrub_one_way.side_effect = RuntimeError("boom")
        with patch("deep_agent.src.pii.get_scrubber", return_value=scrubber):
            assert LogSanitizer().sanitize_string("text") == "text"

    def test_survives_broken_import_machinery(self):
        """Logging from inside an ``except ImportError`` block must not re-raise.

        Callers such as deep_agent.aegra.redis log a warning from their
        ImportError handler; if the lazy scrubber import propagated, that
        second error would escape the caller's handler.
        """
        import builtins

        real_import = builtins.__import__

        def _explode(name, *args, **kwargs):
            if name == "deep_agent.src.pii":
                raise ImportError("no pii")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_explode):
            assert LogSanitizer().sanitize_string("plain text") == "plain text"

    def test_scrub_pii_disabled_skips_scrubber(self):
        scrubber = MagicMock()
        with patch("deep_agent.src.pii.get_scrubber", return_value=scrubber):
            assert LogSanitizer(scrub_pii=False).sanitize_string("text") == "text"
        scrubber.scrub_one_way.assert_not_called()


class TestDisabled:
    """A disabled sanitizer is a strict passthrough."""

    def test_string_untouched(self):
        s = LogSanitizer(enabled=False)
        assert s.sanitize_string("Bearer abc123") == "Bearer abc123"

    def test_dict_untouched(self):
        s = LogSanitizer(enabled=False)
        payload = {"authorization": "Bearer abc123"}
        assert s.sanitize_value(payload) == payload

    def test_processor_returns_event_unchanged(self):
        with patch(
            "deep_agent.utils.log_sanitizer.get_default_sanitizer",
            return_value=LogSanitizer(enabled=False),
        ):
            event = {"authorization": "Bearer abc123"}
            assert create_sanitize_processor()(None, "info", event) == event


class TestCustomPatterns:
    """Operator-supplied regexes extend the built-in credential set."""

    def test_empty_string_yields_no_patterns(self):
        assert parse_custom_patterns("") == []

    def test_valid_patterns_compiled(self):
        patterns = parse_custom_patterns(r"INTERNAL-\d+, ACCT\d+")
        assert len(patterns) == 2
        assert all(isinstance(p, re.Pattern) for p, _ in patterns)

    def test_blank_entries_skipped(self):
        assert len(parse_custom_patterns("abc, ,,def")) == 2

    def test_invalid_regex_skipped(self):
        patterns = parse_custom_patterns(r"valid\d+,[unclosed")
        assert len(patterns) == 1

    def test_custom_pattern_applied(self, no_scrubber):
        s = LogSanitizer(custom_patterns=parse_custom_patterns(r"INTERNAL-\d+"))
        assert s.sanitize_string("id INTERNAL-42") == f"id {REDACTED}"

    def test_custom_patterns_ignored_when_disabled(self):
        s = LogSanitizer(enabled=False, custom_patterns=parse_custom_patterns(r"X\d+"))
        assert s.sanitize_string("X1") == "X1"


class TestDefaultSanitizer:
    """The module-level sanitizer is cached and settings-driven."""

    def test_reads_settings(self):
        with patch("deep_agent.src.settings.settings") as mock_settings:
            mock_settings.LOG_SANITIZATION_ENABLED = False
            mock_settings.LOG_SANITIZATION_CUSTOM_PATTERNS = ""
            assert get_default_sanitizer().enabled is False

    def test_result_is_cached(self):
        first = get_default_sanitizer()
        assert get_default_sanitizer() is first

    def test_reset_rebuilds(self):
        first = get_default_sanitizer()
        reset_default_sanitizer()
        assert get_default_sanitizer() is not first

    def test_defaults_to_enabled_when_settings_unavailable(self):
        with patch.dict(sys.modules, {"deep_agent.src.settings": None}):
            assert get_default_sanitizer().enabled is True


class TestSanitizeProcessor:
    """The structlog processor sanitizes whole event dicts."""

    def test_redacts_credentials_and_headers(self, no_scrubber):
        processor = create_sanitize_processor()
        event = {
            "event": "incoming_request",
            "headers": {"Authorization": "Bearer abc123", "User-Agent": "curl"},
            "note": "password=hunter2",
        }
        result = processor(None, "info", event)
        assert result["headers"]["Authorization"] == REDACTED
        assert result["headers"]["User-Agent"] == "curl"
        assert result["note"] == "***PASSWORD***"
        assert result["event"] == "incoming_request"

    def test_returns_a_dict(self, no_scrubber):
        result = create_sanitize_processor()(None, "info", {"event": "ok"})
        assert isinstance(result, dict)


class TestPyloggerWiring:
    """The processor must be installed in both structlog chains."""

    def test_uvicorn_foreign_pre_chain_includes_processor(self):
        from deep_agent.utils.pylogger import get_uvicorn_log_config

        chain = get_uvicorn_log_config("INFO")["formatters"]["default"][
            "foreign_pre_chain"
        ]
        assert any(getattr(p, "__name__", "") == "sanitize_processor" for p in chain), (
            "sanitize_processor missing from Uvicorn foreign_pre_chain"
        )

    def test_processor_precedes_renderer_in_structlog_chain(self):
        import structlog

        from deep_agent.utils.pylogger import force_reconfigure_all_loggers

        force_reconfigure_all_loggers("INFO")
        processors = structlog.get_config()["processors"]
        names = [getattr(p, "__name__", type(p).__name__) for p in processors]
        assert "sanitize_processor" in names
        assert names.index("sanitize_processor") == len(names) - 2
