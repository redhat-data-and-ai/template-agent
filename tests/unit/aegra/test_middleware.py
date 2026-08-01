"""Unit tests for aegra middleware module."""

from unittest.mock import patch

import pytest

from deep_agent.aegra.middleware import (
    AuthError,
    _hmac_validate,
    authenticate,
    validate_api_key,
)


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


class TestHmacValidate:
    def test_malformed_token_raises(self):
        with pytest.raises(AuthError, match="Malformed"):
            _hmac_validate("not-a-jwt")

    def test_invalid_signature_raises(self):
        with patch("deep_agent.aegra.middleware.JWT_SECRET", "secret"):
            with pytest.raises(AuthError, match="Invalid token signature"):
                _hmac_validate("header.payload.badsig")


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
