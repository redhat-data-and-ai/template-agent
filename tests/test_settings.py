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
        assert not settings.USE_OPENAI_COMPAT_LLM
        assert not settings.use_openai_compatible_llm
        assert settings.OPENAI_COMPAT_BASE_URL is None
        assert settings.OPENAI_COMPAT_API_KEY == "not-needed"
        assert settings.OPENAI_COMPAT_MODEL == "local"
        assert settings.LOG_SANITIZATION_ENABLED
        assert settings.LOG_SANITIZE_PII
        assert settings.LOG_SANITIZE_TOKENS

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
        assert settings.LANGFUSE_BASE_URL is None
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

    def test_use_openai_compatible_llm_when_flag_and_url_set(self):
        """OpenAI-compatible stack only when USE_OPENAI_COMPAT_LLM=true and URL set."""
        with patch.dict(
            "os.environ",
            {
                "USE_OPENAI_COMPAT_LLM": "true",
                "OPENAI_COMPAT_BASE_URL": "http://127.0.0.1:8080/v1",
            },
            clear=True,
        ):
            s = Settings()
            assert s.use_openai_compatible_llm

    def test_openai_compat_flag_false_ignores_base_url(self):
        """Stale OPENAI_COMPAT_BASE_URL does not enable OpenAI when flag is false."""
        with patch.dict(
            "os.environ",
            {
                "USE_OPENAI_COMPAT_LLM": "false",
                "OPENAI_COMPAT_BASE_URL": "http://127.0.0.1:8080/v1",
            },
            clear=True,
        ):
            s = Settings()
            assert not s.USE_OPENAI_COMPAT_LLM
            assert not s.use_openai_compatible_llm

    def test_validate_config_openai_flag_requires_url(self):
        s = Settings()
        s.USE_OPENAI_COMPAT_LLM = True
        s.OPENAI_COMPAT_BASE_URL = None
        with pytest.raises(AppException) as exc_info:
            validate_config(s)
        assert "OPENAI_COMPAT_BASE_URL" in exc_info.value.detail_message
        assert exc_info.value.error_code == "E_009"

    # Note: MCP_PORT and MCP_TRANSPORT_PROTOCOL were removed from settings
    # so these tests are no longer applicable
