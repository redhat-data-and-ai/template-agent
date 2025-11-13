"""Tests for the exceptions module."""

import pytest
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
    HTTP_403_FORBIDDEN,
    HTTP_404_NOT_FOUND,
    HTTP_500_INTERNAL_SERVER_ERROR,
)

from template_agent.src.core.exceptions.exceptions import (
    AppException,
    AppExceptionCode,
    ForbiddenException,
    ToolCallException,
    UnauthorizedException,
)


class TestAppExceptionCode:
    """Test cases for AppExceptionCode enum."""

    def test_bad_request_error(self):
        """Test BAD_REQUEST_ERROR enum value."""
        code = AppExceptionCode.BAD_REQUEST_ERROR
        assert code.response_code == HTTP_400_BAD_REQUEST
        assert code.message == "Bad Request"
        assert code.error_code == "E_001"

    def test_not_found_error(self):
        """Test NOT_FOUND_ERROR enum value."""
        code = AppExceptionCode.NOT_FOUND_ERROR
        assert code.response_code == HTTP_404_NOT_FOUND
        assert code.message == "Not Found"
        assert code.error_code == "E_002"

    def test_internal_server_error(self):
        """Test INTERNAL_SERVER_ERROR enum value."""
        code = AppExceptionCode.INTERNAL_SERVER_ERROR
        assert code.response_code == HTTP_500_INTERNAL_SERVER_ERROR
        assert code.message == "Internal Server Error"
        assert code.error_code == "E_003"

    def test_unauthorised_access_error(self):
        """Test UNAUTHORISED_ACCESS_ERROR enum value."""
        code = AppExceptionCode.UNAUTHORISED_ACCESS_ERROR
        assert code.response_code == HTTP_401_UNAUTHORIZED
        assert code.message == "Unauthorized"
        assert code.error_code == "E_004"

    def test_forbidden_access_error(self):
        """Test FORBIDDEN_ACCESS_ERROR enum value."""
        code = AppExceptionCode.FORBIDDEN_ACCESS_ERROR
        assert code.response_code == HTTP_403_FORBIDDEN
        assert code.message == "Forbidden"
        assert code.error_code == "E_005"

    def test_tool_call_error(self):
        """Test TOOL_CALL_ERROR enum value."""
        code = AppExceptionCode.TOOL_CALL_ERROR
        assert code.response_code == HTTP_500_INTERNAL_SERVER_ERROR
        assert code.message == "Internal Server Error"
        assert code.error_code == "E_006"

    def test_production_mcp_connection_error(self):
        """Test PRODUCTION_MCP_CONNECTION_ERROR enum value."""
        code = AppExceptionCode.PRODUCTION_MCP_CONNECTION_ERROR
        assert code.response_code == HTTP_500_INTERNAL_SERVER_ERROR
        assert code.message == "Internal Server Error"
        assert code.error_code == "E_007"

    def test_configuration_initialization_error(self):
        """Test CONFIGURATION_INITIALIZATION_ERROR enum value."""
        code = AppExceptionCode.CONFIGURATION_INITIALIZATION_ERROR
        assert code.response_code == HTTP_500_INTERNAL_SERVER_ERROR
        assert code.message == "Internal Server Error"
        assert code.error_code == "E_008"

    def test_configuration_validation_error(self):
        """Test CONFIGURATION_VALIDATION_ERROR enum value."""
        code = AppExceptionCode.CONFIGURATION_VALIDATION_ERROR
        assert code.response_code == HTTP_500_INTERNAL_SERVER_ERROR
        assert code.message == "Internal Server Error"
        assert code.error_code == "E_009"

    def test_str_representation(self):
        """Test string representation of AppExceptionCode."""
        code = AppExceptionCode.BAD_REQUEST_ERROR
        expected = "response_code=400, message=Bad Request, error_code=E_001"
        assert str(code) == expected


class TestAppException:
    """Test cases for AppException class."""

    def test_app_exception_creation_with_default_code(self):
        """Test creating AppException with default exception code."""
        exception = AppException("Something went wrong")
        assert exception.detail_message == "Something went wrong"
        assert exception.response_code == HTTP_500_INTERNAL_SERVER_ERROR
        assert exception.message == "Internal Server Error"
        assert exception.error_code == "E_003"

    def test_app_exception_creation_with_custom_code(self):
        """Test creating AppException with custom exception code."""
        exception = AppException("Invalid request", AppExceptionCode.BAD_REQUEST_ERROR)
        assert exception.detail_message == "Invalid request"
        assert exception.response_code == HTTP_400_BAD_REQUEST
        assert exception.message == "Bad Request"
        assert exception.error_code == "E_001"

    def test_app_exception_str_representation(self):
        """Test string representation of AppException."""
        exception = AppException("Invalid request", AppExceptionCode.BAD_REQUEST_ERROR)
        expected = "response_code=400, message=Bad Request, detail_message=Invalid request, error_code=E_001"
        assert str(exception) == expected

    def test_app_exception_inheritance(self):
        """Test that AppException inherits from Exception."""
        exception = AppException("Test message")
        assert isinstance(exception, Exception)

    def test_app_exception_args(self):
        """Test that AppException properly passes args to parent Exception."""
        detail_message = "Test error message"
        exception = AppException(detail_message)
        assert exception.args == (detail_message,)


class TestToolCallException:
    """Test cases for ToolCallException class."""

    def test_tool_call_exception_creation(self):
        """Test creating ToolCallException."""
        exception = ToolCallException("Tool execution failed")
        assert exception.detail_message == "Tool execution failed"
        assert exception.response_code == HTTP_500_INTERNAL_SERVER_ERROR
        assert exception.message == "Internal Server Error"
        assert exception.error_code == "E_006"

    def test_tool_call_exception_inheritance(self):
        """Test that ToolCallException inherits from AppException."""
        exception = ToolCallException("Tool failed")
        assert isinstance(exception, AppException)
        assert isinstance(exception, Exception)

    def test_tool_call_exception_str_representation(self):
        """Test string representation of ToolCallException."""
        exception = ToolCallException("Tool execution failed")
        expected = "response_code=500, message=Internal Server Error, detail_message=Tool execution failed, error_code=E_006"
        assert str(exception) == expected


class TestUnauthorizedException:
    """Test cases for UnauthorizedException class."""

    def test_unauthorized_exception_creation(self):
        """Test creating UnauthorizedException."""
        exception = UnauthorizedException("Invalid credentials")
        assert exception.detail_message == "Invalid credentials"
        assert exception.response_code == HTTP_401_UNAUTHORIZED
        assert exception.message == "Unauthorized"
        assert exception.error_code == "E_004"

    def test_unauthorized_exception_inheritance(self):
        """Test that UnauthorizedException inherits from AppException."""
        exception = UnauthorizedException("Auth failed")
        assert isinstance(exception, AppException)
        assert isinstance(exception, Exception)

    def test_unauthorized_exception_str_representation(self):
        """Test string representation of UnauthorizedException."""
        exception = UnauthorizedException("Invalid credentials")
        expected = "response_code=401, message=Unauthorized, detail_message=Invalid credentials, error_code=E_004"
        assert str(exception) == expected


class TestForbiddenException:
    """Test cases for ForbiddenException class."""

    def test_forbidden_exception_creation(self):
        """Test creating ForbiddenException."""
        exception = ForbiddenException("Access denied")
        assert exception.detail_message == "Access denied"
        assert exception.response_code == HTTP_403_FORBIDDEN
        assert exception.message == "Forbidden"
        assert exception.error_code == "E_005"

    def test_forbidden_exception_inheritance(self):
        """Test that ForbiddenException inherits from AppException."""
        exception = ForbiddenException("Access denied")
        assert isinstance(exception, AppException)
        assert isinstance(exception, Exception)

    def test_forbidden_exception_str_representation(self):
        """Test string representation of ForbiddenException."""
        exception = ForbiddenException("Access denied")
        expected = "response_code=403, message=Forbidden, detail_message=Access denied, error_code=E_005"
        assert str(exception) == expected


class TestExceptionRaising:
    """Test cases for actually raising and catching exceptions."""

    def test_raise_app_exception(self):
        """Test raising and catching AppException."""
        with pytest.raises(AppException) as exc_info:
            raise AppException("Test error")

        assert exc_info.value.detail_message == "Test error"
        assert exc_info.value.response_code == HTTP_500_INTERNAL_SERVER_ERROR

    def test_raise_tool_call_exception(self):
        """Test raising and catching ToolCallException."""
        with pytest.raises(ToolCallException) as exc_info:
            raise ToolCallException("Tool failed")

        assert exc_info.value.detail_message == "Tool failed"
        assert exc_info.value.error_code == "E_006"

    def test_raise_unauthorized_exception(self):
        """Test raising and catching UnauthorizedException."""
        with pytest.raises(UnauthorizedException) as exc_info:
            raise UnauthorizedException("Auth failed")

        assert exc_info.value.detail_message == "Auth failed"
        assert exc_info.value.response_code == HTTP_401_UNAUTHORIZED

    def test_raise_forbidden_exception(self):
        """Test raising and catching ForbiddenException."""
        with pytest.raises(ForbiddenException) as exc_info:
            raise ForbiddenException("Access denied")

        assert exc_info.value.detail_message == "Access denied"
        assert exc_info.value.response_code == HTTP_403_FORBIDDEN

    def test_catch_base_exception(self):
        """Test catching derived exceptions as base AppException."""
        with pytest.raises(AppException) as exc_info:
            raise ToolCallException("Tool failed")

        assert isinstance(exc_info.value, ToolCallException)
        assert exc_info.value.detail_message == "Tool failed"


class TestExceptionChaining:
    """Test cases for exception chaining and context."""

    def test_exception_from_another_exception(self):
        """Test raising AppException from another exception."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise AppException("Wrapped error") from e
        except AppException as app_exc:
            assert app_exc.detail_message == "Wrapped error"
            assert isinstance(app_exc.__cause__, ValueError)
            assert str(app_exc.__cause__) == "Original error"

    def test_exception_context_preservation(self):
        """Test that exception context is preserved."""
        try:
            try:
                1 / 0  # ZeroDivisionError
            except ZeroDivisionError:
                raise ToolCallException("Division failed")
        except ToolCallException as tool_exc:
            assert tool_exc.detail_message == "Division failed"
            assert isinstance(tool_exc.__context__, ZeroDivisionError)
