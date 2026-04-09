"""Tests for the settings module."""

from unittest.mock import patch

import pytest

from template_agent.src.settings import Settings, validate_config
from template_agent.src.core.exceptions.exceptions import AppException


class TestSettings:
    """Test cases for Settings class."""

    def test_database_uri_with_custom_values(self):
        """Test database_uri with custom database settings."""
        with patch.dict(
            "os.environ",
            {
                "POSTGRES_USER": "testuser",
                "POSTGRES_PASSWORD": "testpass",
                "POSTGRES_HOST": "testhost",
                "POSTGRES_PORT": "5433",
                "POSTGRES_DB": "testdb",
            },
        ):
            settings = Settings()
            expected_uri = "postgresql://testuser:testpass@testhost:5433/testdb"
            assert settings.database_uri == expected_uri


class TestValidateConfig:
    """Test cases for validate_config function."""

    def test_validate_config_valid_settings(self):
        """Test validate_config with valid settings."""
        settings = Settings()
        # Should not raise any exceptions
        validate_config(settings)

    def test_validate_config_invalid_log_level(self):
        """Test validate_config with invalid log level."""
        settings = Settings()
        settings.PYTHON_LOG_LEVEL = "INVALID"

        with pytest.raises(AppException) as exc_info:
            validate_config(settings)

        assert "PYTHON_LOG_LEVEL must be one of" in exc_info.value.detail_message
        assert exc_info.value.error_code == "E_009"

    # Note: MCP_PORT and MCP_TRANSPORT_PROTOCOL were removed from settings
    # so these tests are no longer applicable
