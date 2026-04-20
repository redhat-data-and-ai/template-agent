"""Unit tests for exception handling."""

import pytest
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

from template_agent.src.exceptions import AppException, ErrorCode, ErrorCodes


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

        with pytest.raises(Exception):  # FrozenInstanceError in Python 3.11+
            code.status = 400


class TestErrorCodes:
    """Tests for ErrorCodes constants."""

    def test_internal_server_error(self):
        """Test INTERNAL_SERVER_ERROR constant."""
        error = ErrorCodes.INTERNAL_SERVER_ERROR

        assert error.status == HTTP_500_INTERNAL_SERVER_ERROR
        assert error.message == "Internal Server Error"
        assert error.code == "E_003"

    def test_production_mcp_connection_error(self):
        """Test PRODUCTION_MCP_CONNECTION_ERROR constant."""
        error = ErrorCodes.PRODUCTION_MCP_CONNECTION_ERROR

        assert error.status == HTTP_500_INTERNAL_SERVER_ERROR
        assert error.message == "MCP Connection Failed"
        assert error.code == "E_007"

    def test_configuration_initialization_error(self):
        """Test CONFIGURATION_INITIALIZATION_ERROR constant."""
        error = ErrorCodes.CONFIGURATION_INITIALIZATION_ERROR

        assert error.status == HTTP_500_INTERNAL_SERVER_ERROR
        assert error.message == "Configuration Initialization Failed"
        assert error.code == "E_008"

    def test_configuration_validation_error(self):
        """Test CONFIGURATION_VALIDATION_ERROR constant."""
        error = ErrorCodes.CONFIGURATION_VALIDATION_ERROR

        assert error.status == HTTP_500_INTERNAL_SERVER_ERROR
        assert error.message == "Configuration Validation Failed"
        assert error.code == "E_009"

    def test_error_codes_are_frozen(self):
        """Test that ErrorCode instances in ErrorCodes are immutable."""
        with pytest.raises(Exception):  # FrozenInstanceError
            ErrorCodes.INTERNAL_SERVER_ERROR.status = 400


class TestAppException:
    """Tests for AppException class."""

    def test_create_with_default_error_code(self):
        """Test creating an exception with default error code."""
        exc = AppException("Something went wrong")

        assert str(exc) == "Something went wrong"
        assert exc.detail == "Something went wrong"
        assert exc.status == HTTP_500_INTERNAL_SERVER_ERROR
        assert exc.message == "Internal Server Error"
        assert exc.code == "E_003"

    def test_create_with_custom_error_code(self):
        """Test creating an exception with custom error code."""
        exc = AppException(
            "MCP server unreachable",
            ErrorCodes.PRODUCTION_MCP_CONNECTION_ERROR,
        )

        assert str(exc) == "MCP server unreachable"
        assert exc.detail == "MCP server unreachable"
        assert exc.status == HTTP_500_INTERNAL_SERVER_ERROR
        assert exc.message == "MCP Connection Failed"
        assert exc.code == "E_007"

    def test_status_property(self):
        """Test that status property delegates to error_code."""
        exc = AppException("Config error", ErrorCodes.CONFIGURATION_VALIDATION_ERROR)

        assert exc.status == ErrorCodes.CONFIGURATION_VALIDATION_ERROR.status
        assert exc.status == HTTP_500_INTERNAL_SERVER_ERROR

    def test_message_property(self):
        """Test that message property delegates to error_code."""
        exc = AppException("Init failed", ErrorCodes.CONFIGURATION_INITIALIZATION_ERROR)

        assert exc.message == ErrorCodes.CONFIGURATION_INITIALIZATION_ERROR.message
        assert exc.message == "Configuration Initialization Failed"

    def test_code_property(self):
        """Test that code property delegates to error_code."""
        exc = AppException("MCP error", ErrorCodes.PRODUCTION_MCP_CONNECTION_ERROR)

        assert exc.code == ErrorCodes.PRODUCTION_MCP_CONNECTION_ERROR.code
        assert exc.code == "E_007"

    def test_exception_is_raisable(self):
        """Test that AppException can be raised and caught."""
        with pytest.raises(AppException) as exc_info:
            raise AppException("Test error", ErrorCodes.INTERNAL_SERVER_ERROR)

        assert exc_info.value.detail == "Test error"
        assert exc_info.value.code == "E_003"

    def test_exception_preserves_traceback(self):
        """Test that AppException preserves traceback information."""
        try:
            raise AppException("Error with traceback")
        except AppException as exc:
            assert exc.detail == "Error with traceback"
            # Verify exception is properly raised with traceback
            import traceback

            tb = traceback.format_exc()
            assert "AppException" in tb
            assert "Error with traceback" in tb
