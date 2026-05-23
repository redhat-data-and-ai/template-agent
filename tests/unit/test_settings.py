"""Unit tests for settings module."""

import pytest

from deep_agent.src.exceptions import AppException
from deep_agent.src.settings import Settings, validate_config


class TestSettings:
    """Tests for Settings Pydantic model."""

    def test_default_values(self):
        s = Settings()
        assert s.AGENT_HOST == "0.0.0.0"
        assert s.AGENT_PORT == 5002
        assert s.PYTHON_LOG_LEVEL == "INFO"
        assert s.POSTGRES_USER == "pgvector"
        assert s.POSTGRES_PORT == 5432
        assert s.MAX_OUTPUT_TOKENS == 8192

    def test_database_uri(self):
        s = Settings(
            POSTGRES_USER="u",
            POSTGRES_PASSWORD="p",
            POSTGRES_HOST="h",
            POSTGRES_PORT=1234,
            POSTGRES_DB="d",
        )
        assert s.database_uri == "postgresql://u:p@h:1234/d"

    def test_ssl_keyfile_none_when_empty(self):
        s = Settings(SSL_KEYFILE="")
        assert s.get_ssl_keyfile_path is None

    def test_ssl_keyfile_returns_path(self):
        s = Settings(SSL_KEYFILE="/path/to/key")
        assert s.get_ssl_keyfile_path == "/path/to/key"

    def test_ssl_certfile_none_when_empty(self):
        s = Settings(SSL_CERTFILE="")
        assert s.get_ssl_certfile_path is None

    def test_ssl_certfile_returns_path(self):
        s = Settings(SSL_CERTFILE="/path/to/cert")
        assert s.get_ssl_certfile_path == "/path/to/cert"

    def test_optional_fields_accept_none(self):
        s = Settings(
            LANGFUSE_PUBLIC_KEY=None,
            LANGFUSE_SECRET_KEY=None,
            LANGFUSE_BASE_URL=None,
            GOOGLE_APPLICATION_CREDENTIALS_CONTENT=None,
        )
        assert s.LANGFUSE_PUBLIC_KEY is None
        assert s.LANGFUSE_SECRET_KEY is None
        assert s.LANGFUSE_BASE_URL is None
        assert s.GOOGLE_APPLICATION_CREDENTIALS_CONTENT is None

    def test_request_logging_defaults(self):
        s = Settings()
        assert s.REQUEST_LOGGING_ENABLED is True
        assert s.REQUEST_LOG_HEADERS is True
        assert s.REQUEST_LOG_BODY is False
        assert s.REQUEST_LOG_BODY_MAX_SIZE == 10240


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_valid_config(self):
        s = Settings(AGENT_PORT=5002, PYTHON_LOG_LEVEL="INFO")
        validate_config(s)

    def test_port_too_low(self):
        s = Settings(AGENT_PORT=80)
        with pytest.raises(AppException, match="AGENT_PORT must be between"):
            validate_config(s)

    def test_port_too_high(self):
        s = Settings(AGENT_PORT=70000)
        with pytest.raises(AppException, match="AGENT_PORT must be between"):
            validate_config(s)

    def test_invalid_log_level(self):
        s = Settings(PYTHON_LOG_LEVEL="VERBOSE")
        with pytest.raises(AppException, match="PYTHON_LOG_LEVEL must be one of"):
            validate_config(s)

    def test_all_valid_log_levels(self):
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            s = Settings(PYTHON_LOG_LEVEL=level)
            validate_config(s)

    def test_port_boundary_low(self):
        s = Settings(AGENT_PORT=1024)
        validate_config(s)

    def test_port_boundary_high(self):
        s = Settings(AGENT_PORT=65535)
        validate_config(s)
