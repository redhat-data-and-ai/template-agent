"""Main entry point for the template agent server.

This module provides the main application entry point, including
configuration validation, server startup, and graceful shutdown
handling for the template agent service.
"""

import sys
from typing import NoReturn

import uvicorn

from template_agent.src.api import app
from template_agent.src.core.exceptions.exceptions import AppException, AppExceptionCode
from template_agent.src.settings import settings
from template_agent.src.settings import validate_config as validate_config_func
from template_agent.utils.google_creds import initialize_google_genai
from template_agent.utils.pylogger import get_python_logger, get_uvicorn_log_config

# Initialize logger
logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


def validate_and_initialize_config() -> None:
    """Validate configuration settings and initialize external services.

    Performs additional runtime validation of configuration values
    beyond what's done in the Settings class initialization. This
    includes validating host configurations and initializing external
    services like Google Generative AI.

    Raises:
        ValueError: If required configuration values are missing or invalid.
        RuntimeError: If configuration is in an inconsistent state.
    """
    try:
        # Use the validate_config function from settings.py
        validate_config_func(settings)
        initialize_google_genai()

        logger.info("Configuration validation and initialization passed")

    except AttributeError:
        # Handle case where config object is not properly initialized
        raise AppException(
            "Failed to properly initialize configurations",
            AppExceptionCode.CONFIGURATION_INITIALIZATION_ERROR,
        )
    except Exception:
        # Re-raise as ValueError for consistent error handling
        raise AppException(
            "Configuration validation failed",
            AppExceptionCode.CONFIGURATION_VALIDATION_ERROR,
        )


def handle_startup_error(error: Exception, context: str = "server startup") -> NoReturn:
    """Handle startup errors with proper logging and exit codes.

    This function provides centralized error handling for different
    types of startup errors, ensuring appropriate logging and exit
    codes for different error scenarios.

    Args:
        error: The exception that occurred during startup.
        context: Context where the error occurred for better logging.

    Raises:
        SystemExit: Always raises SystemExit with appropriate exit code
            based on the error type.
    """
    if isinstance(error, ValueError):
        # Configuration or validation errors
        logger.critical(f"Configuration error during {context}: {error}")
        sys.exit(1)
    elif isinstance(error, KeyboardInterrupt):
        # User interrupted the startup
        logger.info("Server startup interrupted by user")
        sys.exit(0)
    elif isinstance(error, PermissionError):
        # Permission issues (e.g., port binding)
        logger.critical(f"Permission error during {context}: {error}")
        sys.exit(1)
    elif isinstance(error, ConnectionError):
        # Network-related errors
        logger.critical(f"Connection error during {context}: {error}")
        sys.exit(1)
    else:
        # Unexpected errors
        logger.critical(f"Unexpected error during {context}: {error}", exc_info=True)
        sys.exit(1)


def main() -> None:
    """Main entry point for the template agent server.

    Initializes logging, loads configuration, and starts the template
    agent server. Handles graceful shutdown on keyboard interrupt and
    logs any startup errors.

    The function performs the following steps:
    1. Validates configuration settings
    2. Initializes external services
    3. Configures uvicorn server settings
    4. Starts the server with appropriate error handling

    Raises:
        SystemExit: If the server fails to start due to configuration
            or other errors.
    """
    try:
        validate_and_initialize_config()

        logger.info(
            f"Starting template agent server on {settings.AGENT_HOST}:{settings.AGENT_PORT}"
        )

        # Configure uvicorn server settings
        uvicorn_config = {
            "app": app,
            "host": settings.AGENT_HOST,
            "port": settings.AGENT_PORT,
            "log_config": get_uvicorn_log_config(settings.PYTHON_LOG_LEVEL),
        }

        # Add SSL configuration if certificates are provided
        if settings.AGENT_SSL_KEYFILE and settings.AGENT_SSL_CERTFILE:
            uvicorn_config["ssl_keyfile"] = settings.AGENT_SSL_KEYFILE
            uvicorn_config["ssl_certfile"] = settings.AGENT_SSL_CERTFILE
            logger.info(
                "Starting server with SSL",
                ssl_keyfile=settings.AGENT_SSL_KEYFILE,
                ssl_certfile=settings.AGENT_SSL_CERTFILE,
            )

        uvicorn.run(**uvicorn_config)

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down")
    except Exception as e:
        handle_startup_error(e, "server startup")
    finally:
        logger.info("Template agent server shutting down")


def run() -> None:
    """Run the server with comprehensive error handling.

    Wraps the main function with additional error handling for graceful
    shutdown and proper exit codes. Provides a safety net for any
    unhandled exceptions that might occur during server startup or
    operation.

    Raises:
        SystemExit: If the server fails to start or encounters critical errors.
    """
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        # This should rarely be reached due to handle_startup_error
        logger.error("Server failed to start", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run()
