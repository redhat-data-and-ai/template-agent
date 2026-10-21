"""A missing token-encryption key is a configuration fault, not a bug.

It used to reach the catch-all handler and come back as "An
unexpected error occurred. Please try again later." — advice to wait
for a condition that never changes on its own, for a fault the
server had already identified precisely.
"""

from __future__ import annotations

import pytest

from deep_agent.aegra.mcp_crypto import McpKeyMissingError, _parse_fernet_key


class TestParseFernetKey:
    def test_returns_the_key_when_set(self, monkeypatch):
        monkeypatch.setenv("A_KEY", "  some-key  ")
        assert _parse_fernet_key("A_KEY") == "some-key"

    def test_raises_its_own_type_when_absent(self, monkeypatch):
        monkeypatch.delenv("A_KEY", raising=False)
        with pytest.raises(McpKeyMissingError):
            _parse_fernet_key("A_KEY")

    def test_raises_when_blank(self, monkeypatch):
        monkeypatch.setenv("A_KEY", "   ")
        with pytest.raises(McpKeyMissingError):
            _parse_fernet_key("A_KEY")

    def test_is_still_a_runtimeerror_for_existing_handlers(self, monkeypatch):
        monkeypatch.delenv("A_KEY", raising=False)
        with pytest.raises(RuntimeError):
            _parse_fernet_key("A_KEY")

    def test_says_which_variable_and_how_to_generate_one(self, monkeypatch):
        monkeypatch.delenv("MCP_TOKEN_ENCRYPTION_KEY", raising=False)
        with pytest.raises(McpKeyMissingError) as exc:
            _parse_fernet_key("MCP_TOKEN_ENCRYPTION_KEY")

        message = str(exc.value)
        assert "MCP_TOKEN_ENCRYPTION_KEY" in message
        assert "Fernet.generate_key()" in message
        # The message is shown to whoever hit Connect, so it must
        # carry no key material of its own.
        assert "=" not in message.split("Generate one with")[0]


class TestKeyMissingHandler:
    """The handler that turns the raise into a usable response."""

    @pytest.mark.asyncio
    async def test_answers_503_with_what_is_missing(self):
        from unittest.mock import MagicMock

        from deep_agent.aegra.http_app import mcp_key_missing_handler

        exc = McpKeyMissingError(
            "MCP_TOKEN_ENCRYPTION_KEY is required for MCP OAuth token storage."
        )
        response = await mcp_key_missing_handler(MagicMock(), exc)

        import json

        body = json.loads(response.body)
        # 503, not 500: the dependency is not configured, and
        # retrying changes nothing until it is.
        assert response.status_code == 503
        assert body["error_type"] == "configuration_error"
        assert "MCP_TOKEN_ENCRYPTION_KEY" in body["detail"]

    def test_the_handler_is_registered_before_the_catch_all(self):
        # Registration order decides which one wins; behind the
        # Exception handler it would never run and the response would
        # go back to "an unexpected error occurred".
        from deep_agent.aegra.http_app import app

        handlers = list(app.exception_handlers)
        assert McpKeyMissingError in handlers
        assert handlers.index(McpKeyMissingError) < handlers.index(Exception)
