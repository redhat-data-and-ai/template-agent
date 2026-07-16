"""Unit tests for OAuth scope validation."""

from __future__ import annotations

from deep_agent.aegra.mcp_oauth_scopes import (
    parse_token_scopes,
    requested_scopes,
    validate_granted_scopes,
)


class TestRequestedScopes:
    def test_parses_list(self):
        assert requested_scopes({"scopes": ["read", "write"]}) == ["read", "write"]

    def test_parses_string(self):
        assert requested_scopes({"scopes": "read write"}) == ["read", "write"]

    def test_empty_when_not_configured(self):
        assert requested_scopes({}) == []


class TestParseTokenScopes:
    def test_parses_space_delimited_string(self):
        assert parse_token_scopes({"scope": "read write"}) == ["read", "write"]

    def test_parses_list(self):
        assert parse_token_scopes({"scope": ["read", "write"]}) == ["read", "write"]

    def test_returns_none_when_missing(self):
        assert parse_token_scopes({}) is None


class TestValidateGrantedScopes:
    def test_accepts_when_all_requested_granted(self):
        assert validate_granted_scopes(
            ["read", "write", "openid"],
            ["read", "write"],
            "oauth-mcp",
        ) == ["read", "write", "openid"]

    def test_skips_validation_when_none_requested(self):
        assert validate_granted_scopes(["read"], [], "oauth-mcp") == ["read"]

    def test_rejects_missing_scopes(self, caplog):
        with caplog.at_level("ERROR"):
            assert (
                validate_granted_scopes(["read"], ["read", "write"], "oauth-mcp")
                is None
            )
        assert any("missing requested scopes" in r.message for r in caplog.records)

    def test_rejects_empty_granted_when_scopes_requested(self, caplog):
        with caplog.at_level("ERROR"):
            assert validate_granted_scopes(None, ["read"], "oauth-mcp") is None
        assert any("returned no scopes" in r.message for r in caplog.records)
