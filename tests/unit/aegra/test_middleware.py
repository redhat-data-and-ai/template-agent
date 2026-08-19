"""Unit tests for aegra middleware module."""

import base64
import hashlib
import hmac
import json
from unittest.mock import patch

import pytest

from deep_agent.aegra.middleware import (
    AuthError,
    authenticate,
    validate_api_key,
    validate_jwt_token,
)


def _make_hs256_token(secret: str, payload: dict | None = None) -> str:
    """Build a structurally valid HS256 JWT signed with the given secret."""
    header = (
        base64.urlsafe_b64encode(
            json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode()
        )
        .rstrip(b"=")
        .decode()
    )
    body = (
        base64.urlsafe_b64encode(
            json.dumps(payload or {"sub": "test-user"}, separators=(",", ":")).encode()
        )
        .rstrip(b"=")
        .decode()
    )
    signing_input = f"{header}.{body}".encode()
    sig = (
        base64.urlsafe_b64encode(
            hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
        )
        .rstrip(b"=")
        .decode()
    )
    return f"{header}.{body}.{sig}"


class TestAuthError:
    def test_default_status(self):
        err = AuthError("fail")
        assert err.status_code == 401
        assert err.message == "fail"

    def test_custom_status(self):
        err = AuthError("server error", status_code=500)
        assert err.status_code == 500


class TestValidateApiKey:
    def test_accepts_when_no_key_configured(self):
        with patch("deep_agent.aegra.middleware.API_KEY", ""):
            assert validate_api_key("anything") is True

    def test_accepts_correct_key(self):
        with patch("deep_agent.aegra.middleware.API_KEY", "secret123"):
            assert validate_api_key("secret123") is True

    def test_rejects_wrong_key(self):
        with patch("deep_agent.aegra.middleware.API_KEY", "secret123"):
            assert validate_api_key("wrong") is False


class TestValidateJwtToken:
    def test_malformed_token_raises(self):
        with pytest.raises(AuthError, match="JWT validation failed"):
            validate_jwt_token("not-a-jwt")

    def test_valid_token_accepted(self):
        token = _make_hs256_token("test-secret")
        with patch("deep_agent.aegra.middleware.JWT_SECRET", "test-secret"):
            claims = validate_jwt_token(token)
            assert claims["sub"] == "test-user"

    def test_invalid_signature_raises(self):
        token = _make_hs256_token("wrong-secret")
        with patch("deep_agent.aegra.middleware.JWT_SECRET", "correct-secret"):
            with pytest.raises(AuthError, match="JWT validation failed"):
                validate_jwt_token(token)


class TestAuthenticate:
    def test_noop_returns_empty(self):
        with patch("deep_agent.aegra.middleware.AUTH_TYPE", "noop"):
            result = authenticate({})
            assert result == {}

    def test_api_key_missing_header(self):
        with patch("deep_agent.aegra.middleware.AUTH_TYPE", "api_key"):
            with pytest.raises(AuthError, match="Missing X-API-Key"):
                authenticate({})

    def test_api_key_invalid(self):
        with patch("deep_agent.aegra.middleware.AUTH_TYPE", "api_key"):
            with patch("deep_agent.aegra.middleware.API_KEY", "correct"):
                with pytest.raises(AuthError, match="Invalid API key"):
                    authenticate({"x-api-key": "wrong"})

    def test_api_key_valid(self):
        with patch("deep_agent.aegra.middleware.AUTH_TYPE", "api_key"):
            with patch("deep_agent.aegra.middleware.API_KEY", "correct"):
                result = authenticate({"x-api-key": "correct"})
                assert result["auth_type"] == "api_key"

    def test_jwt_missing_header(self):
        with patch("deep_agent.aegra.middleware.AUTH_TYPE", "jwt"):
            with pytest.raises(AuthError, match="Missing or malformed"):
                authenticate({})

    def test_unknown_auth_type(self):
        with patch("deep_agent.aegra.middleware.AUTH_TYPE", "custom_nonsense"):
            with pytest.raises(AuthError, match="Unknown auth type"):
                authenticate({})
