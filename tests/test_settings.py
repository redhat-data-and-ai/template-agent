"""Tests for the settings module."""

from unittest.mock import patch

import pytest

from template_agent.src.settings import Settings, validate_config
from template_agent.src.core.exceptions.exceptions import AppException


class TestSettings:
    """Test cases for Settings class."""

    @patch.dict("os.environ", {}, clear=True)
    def test_settings_default_values(self):
        """Test Settings has correct default values."""
        settings = Settings()
        assert settings.AGENT_HOST == "0.0.0.0"
        assert settings.AGENT_PORT == 8081
        assert settings.PYTHON_LOG_LEVEL == "INFO"
        assert not settings.USE_INMEMORY_SAVER
        assert settings.POSTGRES_USER == "pgvector"
        assert settings.POSTGRES_PASSWORD == "pgvector"
        assert settings.POSTGRES_DB == "pgvector"
        assert settings.POSTGRES_HOST == "pgvector"
        assert settings.POSTGRES_PORT == 5432
        assert settings.LANGFUSE_TRACING_ENVIRONMENT == "development"

    @patch.dict("os.environ", {}, clear=True)
    def test_database_uri_property(self):
        """Test database_uri property generates correct URI."""
        settings = Settings()
        expected_uri = "postgresql://pgvector:pgvector@pgvector:5432/pgvector"
        assert settings.database_uri == expected_uri

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

    @patch.dict("os.environ", {}, clear=True)
    def test_optional_fields_default_to_none(self):
        """Test that optional fields default to None when no env vars are set."""
        settings = Settings()
        assert settings.AGENT_SSL_KEYFILE is None
        assert settings.AGENT_SSL_CERTFILE is None
        assert settings.GOOGLE_SERVICE_ACCOUNT_FILE is None
        assert settings.LANGFUSE_PUBLIC_KEY is None
        assert settings.LANGFUSE_SECRET_KEY is None
        assert settings.LANGFUSE_HOST is None
        assert settings.GOOGLE_APPLICATION_CREDENTIALS_CONTENT is None


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
