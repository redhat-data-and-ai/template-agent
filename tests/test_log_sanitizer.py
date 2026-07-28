"""Tests for log sanitization utilities."""

from template_agent.utils.log_sanitizer import (
    REDACTED,
    message_log_metadata,
    sanitize_dict,
    sanitize_headers,
    sanitize_log_data,
    sanitize_request_body,
    sanitize_string,
)


class TestSanitizeString:
    def test_sanitize_email(self):
        text = "Contact john@example.com please"
        result = sanitize_string(text)
        assert "[REDACTED_EMAIL]" in result
        assert "john@example.com" not in result

    def test_sanitize_bearer_token(self):
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.abc"
        result = sanitize_string(text)
        assert REDACTED in result
        assert "eyJhbGciOiJIUzI1NiJ9" not in result

    def test_sanitize_api_key_prefixes(self):
        assert REDACTED in sanitize_string("key=AIzaSyAbCdEfGhIjKlMn")
        assert REDACTED in sanitize_string("sk-lf-abc123")
        assert REDACTED in sanitize_string("pk-lf-abc123")

    def test_sanitize_disabled_returns_original(self):
        text = "john@example.com"
        assert sanitize_log_data(text, enabled=False) == text


class TestSanitizeHeaders:
    def test_redacts_sensitive_headers(self):
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


class TestSanitizeDict:
    def test_redacts_sensitive_keys(self):
        data = {
            "message": "My secret question",
            "user_id": "user_123",
            "thread_id": "thread_456",
        }
        safe = sanitize_dict(data)
        assert safe["message"] == REDACTED
        assert safe["user_id"] == "user_123"

    def test_redacts_nested_message(self):
        data = {"content": {"message": "hidden", "type": "human"}}
        safe = sanitize_dict(data)
        assert safe["content"]["message"] == REDACTED


class TestSanitizeRequestBody:
    def test_sanitizes_json_stream_request(self):
        body = (
            '{"message":"Email me at alice@example.com",'
            '"user_id":"user_1","thread_id":"t1"}'
        )
        safe = sanitize_request_body(body)
        assert safe["message"] == REDACTED
        assert safe["user_id"] == "user_1"
        assert "alice@example.com" not in str(safe)

    def test_sanitizes_plain_text_body(self):
        body = "password=secret123"
        safe = sanitize_request_body(body)
        assert isinstance(safe, str)


class TestMessageLogMetadata:
    def test_returns_length_not_content(self):
        metadata = message_log_metadata("hello world")
        assert metadata == {"content_length": 11}
        assert "hello" not in str(metadata)
