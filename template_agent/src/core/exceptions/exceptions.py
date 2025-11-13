"""Exception handling for the Template MCP server."""

from __future__ import annotations

from enum import Enum

from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
    HTTP_403_FORBIDDEN,
    HTTP_404_NOT_FOUND,
    HTTP_500_INTERNAL_SERVER_ERROR,
)


class AppExceptionCode(Enum):
    """Defines custom App Exception codes for this service, associated with HTTP Status codes."""

    BAD_REQUEST_ERROR = (HTTP_400_BAD_REQUEST, "Bad Request", "E_001")
    NOT_FOUND_ERROR = (HTTP_404_NOT_FOUND, "Not Found", "E_002")
    INTERNAL_SERVER_ERROR = (
        HTTP_500_INTERNAL_SERVER_ERROR,
        "Internal Server Error",
        "E_003",
    )
    UNAUTHORISED_ACCESS_ERROR = (HTTP_401_UNAUTHORIZED, "Unauthorized", "E_004")
    FORBIDDEN_ACCESS_ERROR = (HTTP_403_FORBIDDEN, "Forbidden", "E_005")
    TOOL_CALL_ERROR = (HTTP_500_INTERNAL_SERVER_ERROR, "Internal Server Error", "E_006")
    PRODUCTION_MCP_CONNECTION_ERROR = (
        HTTP_500_INTERNAL_SERVER_ERROR,
        "Internal Server Error",
        "E_007",
    )
    CONFIGURATION_INITIALIZATION_ERROR = (
        HTTP_500_INTERNAL_SERVER_ERROR,
        "Internal Server Error",
        "E_008",
    )
    CONFIGURATION_VALIDATION_ERROR = (
        HTTP_500_INTERNAL_SERVER_ERROR,
        "Internal Server Error",
        "E_009",
    )

    def __init__(self, response_code: str, message: str, error_code: str):
        """Constructor to initialize the exception code with response_code, message, and error_code."""
        self._response_code = response_code
        self._message = message
        self._error_code = error_code

    @property
    def response_code(self):
        """HTTP status code for exception code."""
        return self._response_code

    @property
    def message(self):
        """HTTP status message for exception code."""
        return self._message

    @property
    def error_code(self):
        """HTTP error_code for exception code."""
        return self._error_code

    def __str__(self):
        """Str method for logging the exception code."""
        return f"response_code={self.response_code}, message={self.message}, error_code={self.error_code}"


class AppException(Exception):
    """Base exception for application."""

    def __init__(
        self,
        detail_message: str,
        app_exception_code: AppExceptionCode = AppExceptionCode.INTERNAL_SERVER_ERROR,
    ):
        """Constructor to initialize the exception."""
        self._detail_message = detail_message
        self._app_exception_code = app_exception_code
        super().__init__(detail_message)

    @property
    def detail_message(self):
        """Detail error message for exception."""
        return self._detail_message

    @property
    def response_code(self):
        """HTTP response code for exception."""
        return self._app_exception_code.response_code

    @property
    def message(self):
        """HTTP message for exception."""
        return self._app_exception_code.message

    @property
    def error_code(self):
        """Error code for exception."""
        return self._app_exception_code.error_code

    def __str__(self):
        """Str method for logging the exception."""
        return f"response_code={self.response_code}, message={self.message}, detail_message={self.detail_message}, error_code={self.error_code}"


class ToolCallException(AppException):
    """Raised when Tool call fails."""

    def __init__(self, detail_message: str):
        """Constructor to initialize the ToolCallException."""
        super().__init__(detail_message, AppExceptionCode.TOOL_CALL_ERROR)

    def __str__(self):
        """Str method for logging the ToolCallException."""
        return super().__str__()


class UnauthorizedException(AppException):
    """Raised when user Authentication fails."""

    def __init__(self, detail_message: str):
        """Constructor to initialize the UnauthorizedException."""
        super().__init__(detail_message, AppExceptionCode.UNAUTHORISED_ACCESS_ERROR)

    def __str__(self):
        """Str method for logging the UnauthorizedException."""
        return super().__str__()


class ForbiddenException(AppException):
    """Raised when user is forbidden."""

    def __init__(self, detail_message: str):
        """Constructor to initialize the ForbiddenException."""
        super().__init__(detail_message, AppExceptionCode.FORBIDDEN_ACCESS_ERROR)

    def __str__(self):
        """Str method for logging the ForbiddenException."""
        return super().__str__()
