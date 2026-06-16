"""Custom ModelFallbackMiddleware for handling model failures.

Provides fallback model support when the primary model fails with connection
or API errors. Works with deepagents' middleware system.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ContextT, ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel

from deep_agent.utils.pylogger import get_python_logger
from deep_agent.src.settings import settings

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)


class ModelFallbackMiddleware(AgentMiddleware[ModelRequest, ContextT, ModelResponse]):
    """Middleware that provides model fallback on connection/API errors.

    When the primary model fails with connection errors or API errors,
    this middleware automatically retries with the fallback model.

    Attributes:
        fallbacks: List of fallback BaseChatModel instances to try in order.
    """

    def __init__(self, fallbacks: list[BaseChatModel]) -> None:
        """Initialize the fallback middleware.

        Args:
            fallbacks: List of fallback models to try when primary fails.
                       Tried in order until one succeeds.
        """
        self.fallbacks = fallbacks
        logger.info(
            "ModelFallbackMiddleware initialized with %d fallback model(s)",
            len(fallbacks)
        )

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Wrap model calls with fallback logic.

        Tries the primary model first. On connection/API errors, tries each
        fallback model in sequence until one succeeds or all fail.

        Args:
            request: The model call request.
            handler: The next handler in the middleware chain.

        Returns:
            Response from the primary model or first successful fallback.

        Raises:
            The last exception if all models (primary + fallbacks) fail.
        """
        try:
            # Try primary model
            return await handler(request)
        except Exception as primary_error:
            # Only fallback on connection/API errors, not on validation/parsing errors
            error_type = type(primary_error).__name__
            if not _should_fallback(error_type, primary_error):
                logger.debug(
                    "Not falling back for error type '%s': %s",
                    error_type,
                    str(primary_error)[:100]
                )
                raise

            logger.warning(
                "Primary model failed with %s: %s - trying fallback(s)",
                error_type,
                str(primary_error)[:100]
            )

            # Try each fallback in sequence
            last_error = primary_error
            for i, fallback_model in enumerate(self.fallbacks, 1):
                try:
                    logger.info(
                        "Attempting fallback %d/%d: %s",
                        i,
                        len(self.fallbacks),
                        fallback_model.__class__.__name__
                    )
                    # Override the model in the request with the fallback
                    fallback_request = request.override(model=fallback_model)
                    result = await handler(fallback_request)
                    logger.info(
                        "Fallback %d succeeded with %s",
                        i,
                        fallback_model.__class__.__name__
                    )
                    return result
                except Exception as fallback_error:
                    logger.warning(
                        "Fallback %d failed with %s: %s",
                        i,
                        type(fallback_error).__name__,
                        str(fallback_error)[:100]
                    )
                    last_error = fallback_error
                    continue

            # All fallbacks failed - raise the last error
            logger.error(
                "All fallbacks exhausted - primary + %d fallback(s) failed",
                len(self.fallbacks)
            )
            raise last_error


def _should_fallback(error_type: str, error: Exception) -> bool:
    """Determine if we should fallback based on error type.

    Args:
        error_type: The exception class name.
        error: The exception instance.

    Returns:
        True if this error should trigger fallback, False otherwise.
    """
    # Connection and API errors should trigger fallback
    connection_errors = {
        'APIConnectionError',
        'ConnectError',
        'ConnectionError',
        'Timeout',
        'ReadTimeout',
        'ConnectTimeout',
        'HTTPError',
        'APIError',
        'RateLimitError',
        'ServiceUnavailableError',
    }

    if error_type in connection_errors:
        return True

    # Check for SSL/certificate errors
    error_str = str(error).lower()
    if any(phrase in error_str for phrase in [
        'ssl',
        'certificate',
        'connection error',
        'connect error',
        'timeout',
        'unavailable',
    ]):
        return True

    # Don't fallback on validation/parsing errors
    validation_errors = {
        'ValidationError',
        'ValueError',
        'TypeError',
        'OutputParserException',
        'JSONDecodeError',
    }

    if error_type in validation_errors:
        return False

    # Default: don't fallback for unknown errors
    return False
