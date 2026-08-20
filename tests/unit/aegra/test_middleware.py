"""Unit tests for aegra middleware module."""

from unittest.mock import patch

import pytest

from deep_agent.aegra.middleware import (
    AuthError,
    _MIN_JWT_SECRET_BYTES,
    _hmac_validate,
    authenticate,
    validate_api_key,
    validate_auth_config,
    validate_jwt_token,
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
    def test_raises_when_no_key_configured(self):
        with patch("deep_agent.aegra.middleware.API_KEY", ""):
            with pytest.raises(AuthError, match="not configured") as exc_info:
                validate_api_key("anything")
            assert exc_info.value.status_code == 500

    def test_accepts_correct_key(self):
        with patch("deep_agent.aegra.middleware.API_KEY", "secret123"):
            assert validate_api_key("secret123") is True

    def test_rejects_wrong_key(self):
        with patch("deep_agent.aegra.middleware.API_KEY", "secret123"):
            assert validate_api_key("wrong") is False


class TestHmacValidate:
    def test_malformed_token_raises(self):
        with patch(
            "deep_agent.aegra.middleware.JWT_SECRET", "s" * _MIN_JWT_SECRET_BYTES
        ):
            with pytest.raises(AuthError, match="Malformed"):
                _hmac_validate("not-a-jwt")

    def test_invalid_signature_raises(self):
        with patch(
            "deep_agent.aegra.middleware.JWT_SECRET", "s" * _MIN_JWT_SECRET_BYTES
        ):
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


class TestValidateAuthConfig:
    def test_noop_passes(self):
        with patch("deep_agent.aegra.middleware.AUTH_TYPE", "noop"):
            validate_auth_config()

    def test_jwt_with_empty_secret_raises(self):
        with (
            patch("deep_agent.aegra.middleware.AUTH_TYPE", "jwt"),
            patch("deep_agent.aegra.middleware.JWT_SECRET", ""),
        ):
            with pytest.raises(ValueError, match="must be set"):
                validate_auth_config()

    def test_jwt_with_short_secret_raises(self):
        with (
            patch("deep_agent.aegra.middleware.AUTH_TYPE", "jwt"),
            patch("deep_agent.aegra.middleware.JWT_SECRET", "tooshort"),
        ):
            with pytest.raises(ValueError, match="at least"):
                validate_auth_config()

    def test_jwt_with_valid_secret_passes(self):
        secret = "a" * _MIN_JWT_SECRET_BYTES
        with (
            patch("deep_agent.aegra.middleware.AUTH_TYPE", "jwt"),
            patch("deep_agent.aegra.middleware.JWT_SECRET", secret),
        ):
            validate_auth_config()


class TestValidateJwtTokenRejectsWeakSecret:
    def test_empty_secret_raises_auth_error(self):
        with patch("deep_agent.aegra.middleware.JWT_SECRET", ""):
            with pytest.raises(
                AuthError, match="not configured or too short"
            ) as exc_info:
                validate_jwt_token("header.payload.sig")
            assert exc_info.value.status_code == 500

    def test_short_secret_raises_auth_error(self):
        with patch("deep_agent.aegra.middleware.JWT_SECRET", "short"):
            with pytest.raises(
                AuthError, match="not configured or too short"
            ) as exc_info:
                validate_jwt_token("header.payload.sig")
            assert exc_info.value.status_code == 500


class TestHmacValidateRejectsWeakSecret:
    def test_empty_secret_raises_auth_error(self):
        with patch("deep_agent.aegra.middleware.JWT_SECRET", ""):
            with pytest.raises(
                AuthError, match="not configured or too short"
            ) as exc_info:
                _hmac_validate("header.payload.sig")
            assert exc_info.value.status_code == 500

    def test_short_secret_raises_auth_error(self):
        with patch("deep_agent.aegra.middleware.JWT_SECRET", "short"):
            with pytest.raises(
                AuthError, match="not configured or too short"
            ) as exc_info:
                _hmac_validate("header.payload.sig")
            assert exc_info.value.status_code == 500
