"""Unit tests for production security hardening (RHITAIF-220)."""

import os
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from deep_agent.src.exceptions import AppException


class TestEnvironmentEnforcement:
    """Tests for ENVIRONMENT-based security enforcement."""

    def test_production_rejects_auth_bypass_at_startup(self):
        """Test that ENVIRONMENT=production rejects ENABLE_AUTH=false at import."""
        with patch.dict(
            os.environ, {"ENVIRONMENT": "production", "ENABLE_AUTH": "false"}
        ):
            with pytest.raises(
                RuntimeError, match="ENABLE_AUTH=false is not permitted"
            ):
                # Re-import auth module to trigger validation
                import importlib

                from deep_agent.aegra import auth

                importlib.reload(auth)

    def test_development_allows_auth_bypass(self):
        """Test that ENVIRONMENT=development allows ENABLE_AUTH=false."""
        with patch.dict(
            os.environ, {"ENVIRONMENT": "development", "ENABLE_AUTH": "false"}
        ):
            import importlib

            from deep_agent.aegra import auth

            importlib.reload(auth)
            assert auth.ENVIRONMENT == "development"
            assert auth.ENABLE_AUTH is False

    def test_production_flag_detection(self):
        """Test settings.is_production property."""
        from deep_agent.src.settings import Settings

        prod_settings = Settings(ENVIRONMENT="production")
        assert prod_settings.is_production is True

        dev_settings = Settings(ENVIRONMENT="development")
        assert dev_settings.is_production is False


class TestMCPSSLVerificationEnforcement:
    """Tests for MCP SSL verification in production."""

    def test_production_enforces_ssl_verify_true(self):
        """Test that ssl_verify=false is overridden in production."""
        from deep_agent.src.settings import Settings

        # Mock settings at module level before importing function
        prod_settings = Settings(ENVIRONMENT="production")
        with patch("deep_agent.src.settings.settings", prod_settings):
            # Import after patching
            from deep_agent.aegra.mcp import mcp_httpx_verify

            # Should return True even when config says False
            assert mcp_httpx_verify({"ssl_verify": False, "name": "test"}) is True

    def test_development_allows_ssl_verify_false(self):
        """Test that ssl_verify=false is allowed in development."""
        from deep_agent.src.settings import Settings

        dev_settings = Settings(ENVIRONMENT="development")
        with patch("deep_agent.src.settings.settings", dev_settings):
            from deep_agent.aegra.mcp import mcp_httpx_verify

            assert mcp_httpx_verify({"ssl_verify": False}) is False

    def test_ssl_verify_defaults_to_true(self):
        """Test that ssl_verify defaults to True when not specified."""
        from deep_agent.aegra.mcp import mcp_httpx_verify

        assert mcp_httpx_verify({}) is True


class TestSecurityHeaders:
    """Tests for HTTP security headers middleware."""

    def test_security_headers_present(self):
        """Test that all OWASP security headers are set."""
        from deep_agent.aegra.http_app import app

        client = TestClient(app)

        # Use a safe endpoint that doesn't require auth
        with patch.dict(os.environ, {"ENABLE_AUTH": "false"}):
            response = client.get("/")

            assert response.headers["X-Content-Type-Options"] == "nosniff"
            assert response.headers["X-Frame-Options"] == "DENY"
            assert response.headers["X-XSS-Protection"] == "1; mode=block"
            assert (
                response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
            )
            assert "Permissions-Policy" in response.headers
            assert "Content-Security-Policy" in response.headers

    def test_hsts_header_in_production(self):
        """Test that HSTS header is set in production."""
        from deep_agent.aegra.http_app import app
        from deep_agent.src.settings import Settings

        with patch(
            "deep_agent.aegra.security_middleware.settings",
            Settings(ENVIRONMENT="production"),
        ):
            client = TestClient(app)

            with patch.dict(os.environ, {"ENABLE_AUTH": "false"}):
                response = client.get("/")
                assert "Strict-Transport-Security" in response.headers


class TestRequestSizeLimit:
    """Tests for request body size validation."""

    def test_rejects_oversized_request(self):
        """Test that requests exceeding max size are rejected."""
        from deep_agent.aegra.http_app import app

        client = TestClient(app)

        # Simulate oversized request via Content-Length header
        with patch.dict(os.environ, {"ENABLE_AUTH": "false"}):
            response = client.post(
                "/feedback",
                json={"trace_id": "a" * 32},
                headers={"Content-Length": str(11 * 1024 * 1024)},  # 11MB
            )

            assert response.status_code == 413
            assert "exceeds maximum size" in response.json()["detail"]

    def test_accepts_normal_sized_request(self):
        """Test that normal-sized requests are accepted."""
        from deep_agent.aegra.http_app import app

        client = TestClient(app)

        normal_payload = {
            "trace_id": "a" * 32,
            "name": "test",
            "value": 1.0,
        }

        with patch.dict(os.environ, {"ENABLE_AUTH": "false"}):
            with patch(
                "deep_agent.aegra.feedback.get_langfuse_client", return_value=None
            ):
                response = client.post("/feedback", json=normal_payload)

            # Should not be rejected for size
            assert response.status_code != 413


class TestPIIScrubbing:
    """Tests for PII scrubbing in error responses."""

    def test_scrubs_email_addresses(self):
        """Test that email addresses are redacted."""
        from deep_agent.src.pii_scrubber import scrub_pii
        from deep_agent.src.settings import Settings

        with patch(
            "deep_agent.src.pii_scrubber.settings", Settings(ENVIRONMENT="production")
        ):
            text = "Error: user john.doe@example.com not found"
            scrubbed = scrub_pii(text)
            assert "john.doe@example.com" not in scrubbed
            assert "[EMAIL_REDACTED]" in scrubbed

    def test_scrubs_file_paths(self):
        """Test that file paths are redacted."""
        from deep_agent.src.pii_scrubber import scrub_pii
        from deep_agent.src.settings import Settings

        with patch(
            "deep_agent.src.pii_scrubber.settings", Settings(ENVIRONMENT="production")
        ):
            text = "File not found: /home/user/secrets/config.yaml"
            scrubbed = scrub_pii(text)
            assert "/home/user/secrets" not in scrubbed
            assert "[PATH]" in scrubbed

    def test_scrubs_jwt_tokens(self):
        """Test that JWT tokens are redacted."""
        from deep_agent.src.pii_scrubber import scrub_pii
        from deep_agent.src.settings import Settings

        with patch(
            "deep_agent.src.pii_scrubber.settings", Settings(ENVIRONMENT="production")
        ):
            text = "Token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
            scrubbed = scrub_pii(text)
            assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in scrubbed
            assert "[TOKEN_REDACTED]" in scrubbed

    def test_scrubs_sensitive_dict_keys(self):
        """Test that sensitive dictionary keys are redacted."""
        from deep_agent.src.pii_scrubber import scrub_dict
        from deep_agent.src.settings import Settings

        with patch(
            "deep_agent.src.pii_scrubber.settings", Settings(ENVIRONMENT="production")
        ):
            data = {
                "username": "alice",
                "password": "secret123",
                "api_key": "sk-1234567890",
                "message": "hello",
            }
            scrubbed = scrub_dict(data)
            assert scrubbed["password"] == "[REDACTED]"
            assert scrubbed["api_key"] == "[REDACTED]"
            assert scrubbed["username"] == "alice"  # not sensitive
            assert scrubbed["message"] == "hello"

    def test_development_mode_preserves_pii(self):
        """Test that PII is preserved in development mode."""
        from deep_agent.src.pii_scrubber import scrub_pii
        from deep_agent.src.settings import Settings

        with patch(
            "deep_agent.src.pii_scrubber.settings", Settings(ENVIRONMENT="development")
        ):
            text = "Error: user john@example.com at /home/user/file.txt"
            scrubbed = scrub_pii(text)
            # In development, nothing should be scrubbed
            assert scrubbed == text


class TestConfigValidation:
    """Tests for production configuration validation."""

    def test_validate_config_enforces_auth_in_production(self):
        """Test that validate_config rejects ENABLE_AUTH=false in production."""
        from deep_agent.src.settings import Settings, validate_config

        settings = Settings(ENVIRONMENT="production", ENABLE_AUTH=False)

        with pytest.raises(AppException, match="ENABLE_AUTH must be true"):
            validate_config(settings)

    def test_validate_config_allows_dev_mode(self):
        """Test that validate_config allows auth bypass in development."""
        from deep_agent.src.settings import Settings, validate_config

        settings = Settings(ENVIRONMENT="development", ENABLE_AUTH=False)

        # Should not raise
        validate_config(settings)


class TestErrorResponseScrubbing:
    """Tests for global exception handler PII scrubbing."""

    def test_error_response_scrubbed_in_production(self):
        """Test that unhandled exceptions are scrubbed in production."""
        from deep_agent.src.pii_scrubber import scrub_error_response
        from deep_agent.src.settings import Settings

        with patch(
            "deep_agent.src.pii_scrubber.settings", Settings(ENVIRONMENT="production")
        ):
            exc = ValueError("Invalid email: user@example.com")
            response = scrub_error_response("Error occurred", exc)

            # Should not contain PII
            assert "user@example.com" not in str(response)
            # Should contain exception type but not message
            assert response["exception_type"] == "ValueError"
            assert "exception_message" not in response

    def test_error_response_verbose_in_development(self):
        """Test that full error details are shown in development."""
        from deep_agent.src.pii_scrubber import scrub_error_response
        from deep_agent.src.settings import Settings

        with patch(
            "deep_agent.src.pii_scrubber.settings", Settings(ENVIRONMENT="development")
        ):
            exc = ValueError("Invalid email: user@example.com")
            response = scrub_error_response("Error occurred", exc)

            # Should contain full details in dev mode
            assert response["detail"] == "Error occurred"
            assert response["exception_type"] == "ValueError"
            assert "user@example.com" in response["exception_message"]
