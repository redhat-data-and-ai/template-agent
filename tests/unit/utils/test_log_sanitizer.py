"""Unit tests for log sanitization — credentials, headers, and user content."""

from unittest.mock import patch

import pytest

from deep_agent.utils.log_sanitizer import (
    REDACTED,
    LogSanitizer,
    content_placeholder,
    create_sanitize_processor,
    get_default_sanitizer,
    parse_custom_patterns,
    reset_default_sanitizer,
    sanitize_headers,
)


@pytest.fixture(autouse=True)
def _reset_sanitizer():
    reset_default_sanitizer()
    yield
    reset_default_sanitizer()


@pytest.fixture()
def no_scrubber():
    with patch("deep_agent.src.pii.get_scrubber", return_value=None):
        yield


class TestCredentialRedaction:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Authorization: Bearer abc123XYZ", "Bearer ***TOKEN***"),
            ("Authorization: Basic dXNlcjpwYXNz", "Basic ***TOKEN***"),
            ("password=hunter2", "***PASSWORD***"),
            ("api_key=abcdefghijklmnop1234", "***API_KEY***"),
            ("secret_key=abcd1234efgh", "***SECRET***"),
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


class TestSensitiveKeys:
    def test_dict_keys_redacted(self, no_scrubber):
        result = LogSanitizer().sanitize_value(
            {"password": "secret", "api_key": "abcd", "host": "localhost"}
        )
        assert result["password"] == REDACTED
        assert result["api_key"] == REDACTED
        assert result["host"] == "localhost"

    def test_user_content_placeholder(self, no_scrubber):
        result = LogSanitizer().sanitize_value({"message": "hello world"})
        assert result["message"] == "<redacted: 11 chars>"

    def test_id_like_keys_skip_pii(self, no_scrubber):
        tid = "847c6285-8fc9-4560-a83f-4e6285809254"
        result = LogSanitizer().sanitize_value({"thread_id": tid})
        assert result["thread_id"] == tid


class TestHeaders:
    def test_sanitize_headers(self, no_scrubber):
        headers = {
            "Authorization": "Bearer secret",
            "X-Token": "abc123",
            "cookie": "session=xyz",
            "host": "localhost",
        }
        safe = sanitize_headers(headers)
        assert safe["Authorization"] == REDACTED
        assert safe["X-Token"] == REDACTED
        assert safe["cookie"] == REDACTED
        assert safe["host"] == "localhost"


class TestDisabled:
    def test_disabled_passthrough(self, no_scrubber):
        text = "Bearer secretpasswordvalue"
        assert LogSanitizer(enabled=False).sanitize_string(text) == text


class TestCustomPatterns:
    def test_parse_and_apply(self, no_scrubber):
        patterns = parse_custom_patterns(r"INTERNAL-\d{4}")
        result = LogSanitizer(custom_patterns=patterns).sanitize_string(
            "code INTERNAL-1234 ok"
        )
        assert REDACTED in result
        assert "INTERNAL-1234" not in result

    def test_invalid_pattern_skipped(self):
        assert parse_custom_patterns("(unclosed") == []


class TestContentPlaceholder:
    def test_length(self):
        assert content_placeholder("abc") == "<redacted: 3 chars>"

    def test_none(self):
        assert content_placeholder(None) == REDACTED


class TestStructlogProcessor:
    def test_processor_redacts_event(self, no_scrubber):
        processor = create_sanitize_processor()
        event = {
            "event": "stream",
            "message": "secret user text",
            "Authorization": "Bearer abc",
        }
        result = processor(None, "info", event)
        assert result["message"] == "<redacted: 16 chars>"
        assert result["Authorization"] == REDACTED


class TestDefaultSanitizer:
    def test_reads_settings(self, no_scrubber):
        s = get_default_sanitizer()
        assert s.enabled is True
        assert s.redact_user_content is True
