"""Application-wide exception hierarchy and error codes.

This module defines the exception hierarchy and error codes used throughout
the application. Exceptions are organized by subsystem (LLM, MCP, subagent,
configuration) with a shared base class for consistent error handling.

Classes:
    ErrorCode: Immutable error code with status, message, and code
    ErrorCodes: Collection of predefined error codes
    AppException: Base exception for all application errors
    TransientError: Base for retryable errors
    LLMError: LLM/model creation failures
    MCPError: MCP server connection failures
    SubAgentError: Subagent loading/execution failures
    ConfigurationError: Configuration loading/validation failures
    RateLimitError: Rate limit exceeded (retryable)
    AuthenticationError: Authentication/authorization failures
"""

from dataclasses import dataclass

from starlette.status import (
    HTTP_401_UNAUTHORIZED,
    HTTP_429_TOO_MANY_REQUESTS,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_502_BAD_GATEWAY,
    HTTP_503_SERVICE_UNAVAILABLE,
    HTTP_504_GATEWAY_TIMEOUT,
)


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
        "E_001",
    )
    LLM_ERROR = ErrorCode(
        HTTP_502_BAD_GATEWAY,
        "LLM Service Error",
        "E_002",
    )
    LLM_TIMEOUT = ErrorCode(
        HTTP_504_GATEWAY_TIMEOUT,
        "LLM Request Timeout",
        "E_003",
    )
    MCP_CONNECTION_ERROR = ErrorCode(
        HTTP_502_BAD_GATEWAY,
        "MCP Connection Failed",
        "E_004",
    )
    MCP_TIMEOUT = ErrorCode(
        HTTP_504_GATEWAY_TIMEOUT,
        "MCP Request Timeout",
        "E_005",
    )
    SUBAGENT_ERROR = ErrorCode(
        HTTP_500_INTERNAL_SERVER_ERROR,
        "Subagent Execution Failed",
        "E_006",
    )
    CONFIGURATION_INITIALIZATION_ERROR = ErrorCode(
        HTTP_500_INTERNAL_SERVER_ERROR,
        "Configuration Initialization Failed",
        "E_007",
    )
    CONFIGURATION_VALIDATION_ERROR = ErrorCode(
        HTTP_500_INTERNAL_SERVER_ERROR,
        "Configuration Validation Failed",
        "E_008",
    )
    RATE_LIMIT_ERROR = ErrorCode(
        HTTP_429_TOO_MANY_REQUESTS,
        "Rate Limit Exceeded",
        "E_009",
    )
    AUTHENTICATION_ERROR = ErrorCode(
        HTTP_401_UNAUTHORIZED,
        "Authentication Failed",
        "E_010",
    )
    SERVICE_UNAVAILABLE = ErrorCode(
        HTTP_503_SERVICE_UNAVAILABLE,
        "Service Temporarily Unavailable",
        "E_011",
    )

    # Legacy aliases (kept for backward compatibility)
    PRODUCTION_MCP_CONNECTION_ERROR = MCP_CONNECTION_ERROR


class AppException(Exception):
    """Base exception for application errors."""

    def __init__(
        self,
        detail: str,
        error_code: ErrorCode = ErrorCodes.INTERNAL_SERVER_ERROR,
    ) -> None:
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

    @property
    def is_retryable(self) -> bool:
        """Whether this error is safe to retry."""
        return False


class TransientError(AppException):
    """Base for errors that are safe to retry.

    Subclasses represent failures from external services (LLM, MCP, network)
    that may succeed on a subsequent attempt.
    """

    @property
    def is_retryable(self) -> bool:
        """Return True; transient errors are retryable by definition."""
        return True


class LLMError(TransientError):
    """LLM model creation or invocation failure."""

    def __init__(self, detail: str) -> None:  # noqa: D107
        super().__init__(detail, ErrorCodes.LLM_ERROR)


class LLMTimeoutError(TransientError):
    """LLM request timed out."""

    def __init__(self, detail: str) -> None:  # noqa: D107
        super().__init__(detail, ErrorCodes.LLM_TIMEOUT)


class MCPError(TransientError):
    """MCP server connection or tool invocation failure."""

    def __init__(self, detail: str) -> None:  # noqa: D107
        super().__init__(detail, ErrorCodes.MCP_CONNECTION_ERROR)


class MCPTimeoutError(TransientError):
    """MCP server request timed out."""

    def __init__(self, detail: str) -> None:  # noqa: D107
        super().__init__(detail, ErrorCodes.MCP_TIMEOUT)


class SubAgentError(AppException):
    """Subagent loading or execution failure."""

    def __init__(self, detail: str) -> None:  # noqa: D107
        super().__init__(detail, ErrorCodes.SUBAGENT_ERROR)


class ConfigurationError(AppException):
    """Configuration loading or validation failure."""

    def __init__(self, detail: str) -> None:  # noqa: D107
        super().__init__(detail, ErrorCodes.CONFIGURATION_INITIALIZATION_ERROR)


class RateLimitError(TransientError):
    """Rate limit exceeded — should back off and retry."""

    def __init__(self, detail: str) -> None:  # noqa: D107
        super().__init__(detail, ErrorCodes.RATE_LIMIT_ERROR)


class AuthenticationError(AppException):
    """Authentication or authorization failure — do NOT retry."""

    def __init__(self, detail: str) -> None:  # noqa: D107
        super().__init__(detail, ErrorCodes.AUTHENTICATION_ERROR)
