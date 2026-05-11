"""Tests for a2a/auth.py -- Bearer token authentication for A2A requests."""

from __future__ import annotations

from unittest.mock import MagicMock

import jwt as pyjwt
import pytest

from template_agent.src.a2a.auth import (
    A2AServerCallContextBuilder,
    AuthenticatedUser,
    _decode_jwt,
    _extract_bearer_token,
    _is_jwe,
    _validate_token,
)
from template_agent.src.a2a.executor import ACCESS_TOKEN_STATE_KEY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_jwt(claims: dict) -> str:
    """Create an unsigned JWT for testing."""
    return pyjwt.encode(claims, "secret", algorithm="HS256")


def _make_jwe_token() -> str:
    """Return a fake JWE token (5 dot-separated segments)."""
    return "header.encrypted_key.iv.ciphertext.tag"


class _FakeHeaders(dict):
    """Dict subclass that also supports Starlette's getlist()."""

    def getlist(self, key: str) -> list[str]:
        v = self.get(key)
        return [v] if v is not None else []


def _make_request(headers: dict | None = None) -> MagicMock:
    """Build a Starlette-like Request mock with the given headers."""
    request = MagicMock()
    request.headers = _FakeHeaders(headers or {})
    return request


# ---------------------------------------------------------------------------
# AuthenticatedUser
# ---------------------------------------------------------------------------
class TestAuthenticatedUser:
    def test_is_authenticated(self):
        user = AuthenticatedUser(claims={"sub": "u1"}, name="alice")
        assert user.is_authenticated is True

    def test_user_name(self):
        user = AuthenticatedUser(claims={}, name="bob")
        assert user.user_name == "bob"

    def test_claims_exposed(self):
        claims = {"sub": "u1", "email": "a@b.com"}
        user = AuthenticatedUser(claims=claims, name="alice")
        assert user.claims == claims


# ---------------------------------------------------------------------------
# _extract_bearer_token
# ---------------------------------------------------------------------------
class TestExtractBearerToken:
    def test_extracts_bearer_token(self):
        req = _make_request({"authorization": "Bearer tok123"})
        assert _extract_bearer_token(req) == "tok123"

    def test_case_insensitive_prefix(self):
        req = _make_request({"authorization": "bearer tok123"})
        assert _extract_bearer_token(req) == "tok123"

    def test_returns_none_without_header(self):
        req = _make_request({})
        assert _extract_bearer_token(req) is None

    def test_returns_none_for_non_bearer(self):
        req = _make_request({"authorization": "Basic abc"})
        assert _extract_bearer_token(req) is None

    def test_strips_whitespace(self):
        req = _make_request({"authorization": "Bearer   tok123  "})
        assert _extract_bearer_token(req) == "tok123"


# ---------------------------------------------------------------------------
# _is_jwe
# ---------------------------------------------------------------------------
class TestIsJwe:
    def test_jwe_five_segments(self):
        assert _is_jwe(_make_jwe_token()) is True

    def test_jwt_three_segments(self):
        assert _is_jwe(_make_jwt({"sub": "x"})) is False

    def test_arbitrary_string(self):
        assert _is_jwe("not-a-token") is False


# ---------------------------------------------------------------------------
# _decode_jwt
# ---------------------------------------------------------------------------
class TestDecodeJwt:
    def test_decodes_valid_jwt(self):
        token = _make_jwt({"sub": "user1", "email": "a@b.com"})
        claims = _decode_jwt(token)
        assert claims["sub"] == "user1"
        assert claims["email"] == "a@b.com"

    def test_does_not_verify_signature(self):
        token = _make_jwt({"sub": "u"})
        # Tamper with the signature portion
        parts = token.split(".")
        parts[2] = "invalidsignature"
        tampered = ".".join(parts)
        claims = _decode_jwt(tampered)
        assert claims["sub"] == "u"


# ---------------------------------------------------------------------------
# _validate_token
# ---------------------------------------------------------------------------
class TestValidateToken:
    def test_jwt_returns_claims_and_format(self):
        token = _make_jwt({"sub": "user1"})
        claims, fmt = _validate_token(token)
        assert fmt == "jwt"
        assert claims["sub"] == "user1"

    def test_jwe_returns_placeholder_claims(self):
        token = _make_jwe_token()
        claims, fmt = _validate_token(token)
        assert fmt == "jwe"
        assert claims == {"token_format": "jwe"}


# ---------------------------------------------------------------------------
# A2AServerCallContextBuilder
# ---------------------------------------------------------------------------
class TestA2AServerCallContextBuilder:
    def _builder(self) -> A2AServerCallContextBuilder:
        return A2AServerCallContextBuilder()

    def test_raises_on_missing_token(self):
        req = _make_request({})
        with pytest.raises(PermissionError, match="Missing Authorization"):
            self._builder().build(req)

    def test_jwt_happy_path(self):
        token = _make_jwt({"preferred_username": "alice", "sub": "u1"})
        req = _make_request({"authorization": f"Bearer {token}"})
        ctx = self._builder().build(req)

        assert ctx.user.is_authenticated is True
        assert ctx.user.user_name == "alice"
        assert ctx.state[ACCESS_TOKEN_STATE_KEY] == token
        assert ctx.state["token_format"] == "jwt"

    def test_jwt_falls_back_to_sub(self):
        token = _make_jwt({"sub": "user-sub-id"})
        req = _make_request({"authorization": f"Bearer {token}"})
        ctx = self._builder().build(req)
        assert ctx.user.user_name == "user-sub-id"

    def test_jwt_falls_back_to_email(self):
        token = _make_jwt({"email": "a@b.com"})
        req = _make_request({"authorization": f"Bearer {token}"})
        ctx = self._builder().build(req)
        assert ctx.user.user_name == "a@b.com"

    def test_jwt_falls_back_to_unknown(self):
        token = _make_jwt({"custom_claim": "x"})
        req = _make_request({"authorization": f"Bearer {token}"})
        ctx = self._builder().build(req)
        assert ctx.user.user_name == "unknown"

    def test_jwe_token_stored_and_user_is_jwe_authenticated(self):
        token = _make_jwe_token()
        req = _make_request({"authorization": f"Bearer {token}"})
        ctx = self._builder().build(req)

        assert ctx.user.user_name == "jwe-authenticated"
        assert ctx.state[ACCESS_TOKEN_STATE_KEY] == token
        assert ctx.state["token_format"] == "jwe"

    def test_state_contains_headers(self):
        token = _make_jwt({"sub": "u"})
        req = _make_request({
            "authorization": f"Bearer {token}",
            "x-custom": "val",
        })
        ctx = self._builder().build(req)
        assert "headers" in ctx.state
        assert ctx.state["headers"]["x-custom"] == "val"
