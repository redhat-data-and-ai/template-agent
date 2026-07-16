"""Integration tests for production security hardening."""

from unittest.mock import patch

import pytest
from starlette.testclient import TestClient


@pytest.fixture
def prod_client():
    """Create a test client with production environment."""
    with patch.dict("os.environ", {"ENVIRONMENT": "production", "ENABLE_AUTH": "true"}):
        from deep_agent.aegra.http_app import app

        return TestClient(app)


def test_all_security_headers_present_in_production(prod_client):
    """Test that all security headers are present in production responses."""
    # Try to access root endpoint (may return 404 but should have headers)
    response = prod_client.get("/")

    required_headers = [
        "X-Content-Type-Options",
        "X-Frame-Options",
        "X-XSS-Protection",
        "Strict-Transport-Security",
        "Content-Security-Policy",
        "Referrer-Policy",
        "Permissions-Policy",
    ]

    for header in required_headers:
        assert header in response.headers, f"Missing security header: {header}"


def test_production_mode_config_validation():
    """Test that production mode validates configuration at startup."""
    from deep_agent.src.settings import Settings, validate_config

    # Production with auth disabled should fail validation
    prod_settings = Settings(ENVIRONMENT="production", ENABLE_AUTH=False)

    with pytest.raises(Exception, match="ENABLE_AUTH must be true"):
        validate_config(prod_settings)

    # Production with auth enabled should pass
    prod_settings_valid = Settings(ENVIRONMENT="production", ENABLE_AUTH=True)
    validate_config(prod_settings_valid)  # Should not raise
