"""Unit tests for emitter sensitive key scrubbing."""

from unittest.mock import patch

from deep_agent.src.audit.emitter import _scrub_details


class TestScrubDetails:
    def test_redacts_sensitive_keys(self):
        scrubbed = _scrub_details({"access_token": "secret", "model": "gemini"})
        assert scrubbed["access_token"] == "[REDACTED]"
        assert scrubbed["model"] == "gemini"

    def test_does_not_redact_author_field(self):
        scrubbed = _scrub_details({"author": "alice", "authorization": "Bearer x"})
        assert scrubbed["author"] == "alice"
        assert scrubbed["authorization"] == "[REDACTED]"

    def test_redacts_nested_sensitive_keys(self):
        scrubbed = _scrub_details({"meta": {"api_key": "k", "count": 1}})
        assert scrubbed["meta"]["api_key"] == "[REDACTED]"
        assert scrubbed["meta"]["count"] == 1
