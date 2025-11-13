"""Google credentials management utilities.

This module provides functions for initializing Google Generative AI with various
credential formats including base64-encoded, file paths, and direct JSON content.
"""

import base64
import os
import tempfile

from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


def initialize_google_genai():
    """Initialize Google Generative AI with service account credentials."""
    credentials_file = None

    if not settings.GOOGLE_APPLICATION_CREDENTIALS_CONTENT:
        logger.warning("No Google service account credentials configured")
        return

    # Check if credentials are provided as base64-encoded environment variable
    if settings.GOOGLE_APPLICATION_CREDENTIALS_CONTENT.startswith("ewog"):
        # Validate that it's valid JSON
        import json

        try:
            # Decode base64 credentials
            credentials_json = base64.b64decode(
                settings.GOOGLE_APPLICATION_CREDENTIALS_CONTENT
            ).decode("utf-8")

            json.loads(credentials_json)  # This will raise an exception if invalid JSON

            # Create temporary file with credentials
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as temp_file:
                temp_file.write(credentials_json)
                credentials_file = temp_file.name

            logger.info(
                "Initialized Google Generative AI with base64-encoded service account credentials"
            )

        except (base64.binascii.Error, UnicodeDecodeError) as e:
            logger.error(f"Failed to decode base64 credentials: {e}")
            return
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in base64 credentials: {e}")
            return
        except Exception as e:
            logger.error(f"Unexpected error processing base64 credentials: {e}")
            return

    # Check if credentials are provided as file path
    elif os.path.exists(settings.GOOGLE_APPLICATION_CREDENTIALS_CONTENT):
        credentials_file = settings.GOOGLE_APPLICATION_CREDENTIALS_CONTENT
        logger.info(
            f"Initialized Google Generative AI with service account file: {settings.GOOGLE_SERVICE_ACCOUNT_FILE}"
        )

    # Check if credentials are provided as direct JSON content
    elif settings.GOOGLE_APPLICATION_CREDENTIALS_CONTENT.strip().startswith("{"):
        # Validate that it's valid JSON
        import json

        try:
            credentials_json = settings.GOOGLE_APPLICATION_CREDENTIALS_CONTENT.strip()
            json.loads(credentials_json)  # This will raise an exception if invalid JSON

            # Create temporary file with credentials
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as temp_file:
                temp_file.write(credentials_json)
                credentials_file = temp_file.name

            logger.info(
                "Initialized Google Generative AI with direct JSON service account credentials"
            )

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in direct credentials: {e}")
            return
        except Exception as e:
            logger.error(f"Unexpected error processing direct JSON credentials: {e}")
            return

    else:
        logger.warning(
            f"Google service account credentials not found or invalid format: {settings.GOOGLE_SERVICE_ACCOUNT_FILE[:50]}..."
        )
        return

    # Set environment variable for langchain-google-genai to use
    if credentials_file:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file
        logger.debug(f"Set GOOGLE_APPLICATION_CREDENTIALS to: {credentials_file}")
