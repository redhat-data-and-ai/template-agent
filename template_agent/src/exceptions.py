"""Application-wide exception handling.

This module defines custom exceptions and error codes used throughout the
application. All error codes include HTTP status codes for consistent API
error responses.

Classes:
    ErrorCode: Immutable error code with status, message, and code
    ErrorCodes: Collection of predefined error codes
    AppException: Base exception for all application errors
"""

from dataclasses import dataclass

from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR


@dataclass(frozen=True)
class ErrorCode:
    """Error code with HTTP status and message."""

    status: int
    message: str
    code: str


class ErrorCodes:
    """Error codes for the template agent."""

    INTERNAL_SERVER_ERROR = ErrorCode(
        HTTP_500_INTERNAL_SERVER_ERROR,
        "Internal Server Error",
        "E_003",
    )
    PRODUCTION_MCP_CONNECTION_ERROR = ErrorCode(
        HTTP_500_INTERNAL_SERVER_ERROR,
        "MCP Connection Failed",
        "E_007",
    )
    CONFIGURATION_INITIALIZATION_ERROR = ErrorCode(
        HTTP_500_INTERNAL_SERVER_ERROR,
        "Configuration Initialization Failed",
        "E_008",
    )
    CONFIGURATION_VALIDATION_ERROR = ErrorCode(
        HTTP_500_INTERNAL_SERVER_ERROR,
        "Configuration Validation Failed",
        "E_009",
    )


class AppException(Exception):
    """Base exception for application errors."""

    def __init__(
        self,
        detail: str,
        error_code: ErrorCode = ErrorCodes.INTERNAL_SERVER_ERROR,
    ):
        """Initialize exception with detail message and error code."""
        self.detail = detail
        self.error_code = error_code
        super().__init__(detail)

    @property
    def status(self) -> int:
        """HTTP status code."""
        return self.error_code.status

    @property
    def message(self) -> str:
        """Error message."""
        return self.error_code.message

    @property
    def code(self) -> str:
        """Error code."""
        return self.error_code.code
