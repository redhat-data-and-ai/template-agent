"""Unit tests for exception hierarchy and error codes."""

import pytest
from starlette.status import (
    HTTP_401_UNAUTHORIZED,
    HTTP_429_TOO_MANY_REQUESTS,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_502_BAD_GATEWAY,
    HTTP_503_SERVICE_UNAVAILABLE,
    HTTP_504_GATEWAY_TIMEOUT,
)

from deep_agent.src.exceptions import (
    AppException,
    AuthenticationError,
    ConfigurationError,
    ErrorCode,
    ErrorCodes,
    LLMError,
    LLMTimeoutError,
    MCPError,
    MCPTimeoutError,
    RateLimitError,
    SubAgentError,
    TransientError,
)


class TestErrorCode:
    """Tests for ErrorCode dataclass."""

    def test_create_error_code(self):
        """Test creating an ErrorCode instance."""
        code = ErrorCode(status=404, message="Not Found", code="E_404")

        assert code.status == 404
        assert code.message == "Not Found"
        assert code.code == "E_404"

    def test_error_code_is_frozen(self):
        """Test that ErrorCode instances are immutable."""
        code = ErrorCode(status=500, message="Server Error", code="E_500")

        with pytest.raises(Exception):
            code.status = 400


class TestErrorCodes:
    """Tests for ErrorCodes constants."""

    def test_internal_server_error(self):
        error = ErrorCodes.INTERNAL_SERVER_ERROR
        assert error.status == HTTP_500_INTERNAL_SERVER_ERROR
        assert error.message == "Internal Server Error"
        assert error.code == "E_001"

    def test_llm_error(self):
        error = ErrorCodes.LLM_ERROR
        assert error.status == HTTP_502_BAD_GATEWAY
        assert error.code == "E_002"

    def test_llm_timeout(self):
        error = ErrorCodes.LLM_TIMEOUT
        assert error.status == HTTP_504_GATEWAY_TIMEOUT
        assert error.code == "E_003"

    def test_mcp_connection_error(self):
        error = ErrorCodes.MCP_CONNECTION_ERROR
        assert error.status == HTTP_502_BAD_GATEWAY
        assert error.message == "MCP Connection Failed"
        assert error.code == "E_004"

    def test_mcp_timeout(self):
        error = ErrorCodes.MCP_TIMEOUT
        assert error.status == HTTP_504_GATEWAY_TIMEOUT
        assert error.code == "E_005"

    def test_subagent_error(self):
        error = ErrorCodes.SUBAGENT_ERROR
        assert error.status == HTTP_500_INTERNAL_SERVER_ERROR
        assert error.code == "E_006"

    def test_configuration_initialization_error(self):
        error = ErrorCodes.CONFIGURATION_INITIALIZATION_ERROR
        assert error.status == HTTP_500_INTERNAL_SERVER_ERROR
        assert error.message == "Configuration Initialization Failed"
        assert error.code == "E_007"

    def test_configuration_validation_error(self):
        error = ErrorCodes.CONFIGURATION_VALIDATION_ERROR
        assert error.status == HTTP_500_INTERNAL_SERVER_ERROR
        assert error.message == "Configuration Validation Failed"
        assert error.code == "E_008"

    def test_rate_limit_error(self):
        error = ErrorCodes.RATE_LIMIT_ERROR
        assert error.status == HTTP_429_TOO_MANY_REQUESTS
        assert error.code == "E_009"

    def test_authentication_error(self):
        error = ErrorCodes.AUTHENTICATION_ERROR
        assert error.status == HTTP_401_UNAUTHORIZED
        assert error.code == "E_010"

    def test_service_unavailable(self):
        error = ErrorCodes.SERVICE_UNAVAILABLE
        assert error.status == HTTP_503_SERVICE_UNAVAILABLE
        assert error.code == "E_011"

    def test_legacy_alias_mcp(self):
        """Legacy PRODUCTION_MCP_CONNECTION_ERROR aliases MCP_CONNECTION_ERROR."""
        assert (
            ErrorCodes.PRODUCTION_MCP_CONNECTION_ERROR
            is ErrorCodes.MCP_CONNECTION_ERROR
        )

    def test_error_codes_are_frozen(self):
        with pytest.raises(Exception):
            ErrorCodes.INTERNAL_SERVER_ERROR.status = 400


class TestAppException:
    """Tests for AppException class."""

    def test_create_with_default_error_code(self):
        exc = AppException("Something went wrong")
        assert str(exc) == "Something went wrong"
        assert exc.detail == "Something went wrong"
        assert exc.status == HTTP_500_INTERNAL_SERVER_ERROR
        assert exc.message == "Internal Server Error"
        assert exc.code == "E_001"

    def test_create_with_custom_error_code(self):
        exc = AppException("MCP unreachable", ErrorCodes.MCP_CONNECTION_ERROR)
        assert exc.status == HTTP_502_BAD_GATEWAY
        assert exc.message == "MCP Connection Failed"
        assert exc.code == "E_004"

    def test_is_retryable_default_false(self):
        exc = AppException("error")
        assert exc.is_retryable is False

    def test_exception_is_raisable(self):
        with pytest.raises(AppException) as exc_info:
            raise AppException("Test error", ErrorCodes.INTERNAL_SERVER_ERROR)
        assert exc_info.value.detail == "Test error"
        assert exc_info.value.code == "E_001"

    def test_exception_preserves_traceback(self):
        try:
            raise AppException("Error with traceback")
        except AppException as exc:
            assert exc.detail == "Error with traceback"
            import traceback

            tb = traceback.format_exc()
            assert "AppException" in tb
            assert "Error with traceback" in tb


class TestTransientError:
    """Tests for TransientError and retryable subclasses."""

    def test_transient_is_retryable(self):
        exc = TransientError("transient failure")
        assert exc.is_retryable is True

    def test_llm_error(self):
        exc = LLMError("model creation failed")
        assert isinstance(exc, TransientError)
        assert isinstance(exc, AppException)
        assert exc.is_retryable is True
        assert exc.code == "E_002"

    def test_llm_timeout_error(self):
        exc = LLMTimeoutError("request timed out")
        assert exc.is_retryable is True
        assert exc.code == "E_003"

    def test_mcp_error(self):
        exc = MCPError("connection refused")
        assert isinstance(exc, TransientError)
        assert exc.is_retryable is True
        assert exc.code == "E_004"

    def test_mcp_timeout_error(self):
        exc = MCPTimeoutError("timeout")
        assert exc.is_retryable is True
        assert exc.code == "E_005"

    def test_rate_limit_error(self):
        exc = RateLimitError("too many requests")
        assert isinstance(exc, TransientError)
        assert exc.is_retryable is True
        assert exc.code == "E_009"


class TestNonRetryableErrors:
    """Tests for non-retryable exception subclasses."""

    def test_subagent_error(self):
        exc = SubAgentError("failed to build")
        assert isinstance(exc, AppException)
        assert not isinstance(exc, TransientError)
        assert exc.is_retryable is False
        assert exc.code == "E_006"

    def test_configuration_error(self):
        exc = ConfigurationError("missing config")
        assert isinstance(exc, AppException)
        assert exc.is_retryable is False
        assert exc.code == "E_007"

    def test_authentication_error(self):
        exc = AuthenticationError("invalid token")
        assert isinstance(exc, AppException)
        assert exc.is_retryable is False
        assert exc.code == "E_010"
