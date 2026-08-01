"""Authentication and authorization middleware for aegra deployment (MR-22).

Provides configurable auth strategies for the LangGraph Platform API:
- ``noop``: No authentication (development)
- ``api_key``: Simple API key validation via X-API-Key header
- ``jwt``: JWT bearer token validation (production)

The active strategy is selected via the ``LANGGRAPH_AUTH_TYPE`` env var.
"""

import hashlib
import hmac
import os
import time
from typing import Any

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

AUTH_TYPE = os.environ.get("LANGGRAPH_AUTH_TYPE", "noop")
API_KEY = os.environ.get("LANGGRAPH_API_KEY", "")
JWT_SECRET = os.environ.get("LANGGRAPH_JWT_SECRET", "")
JWT_ALGORITHM = os.environ.get("LANGGRAPH_JWT_ALGORITHM", "HS256")


class AuthError(Exception):
    """Raised when authentication fails."""

    def __init__(self, message: str, status_code: int = 401):
        """Initialize with error message and HTTP status code."""
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def validate_api_key(provided_key: str) -> bool:
    """Constant-time comparison of API keys to prevent timing attacks."""
    if not API_KEY:
        logger.warning("LANGGRAPH_API_KEY not set — all keys accepted")
        return True
    return hmac.compare_digest(provided_key.encode(), API_KEY.encode())


def validate_jwt_token(token: str) -> dict[str, Any]:
    """Validate a JWT token and return its claims.

    Requires ``PyJWT`` to be installed. Falls back to a simple
    HMAC-based validation if PyJWT is unavailable.
    """
    try:
        import jwt

        claims: dict[str, Any] = jwt.decode(
            token, JWT_SECRET, algorithms=[JWT_ALGORITHM]
        )
        if claims.get("exp") and claims["exp"] < time.time():
            raise AuthError("Token expired")
        return claims
    except ImportError:
        logger.warning("PyJWT not installed — using HMAC fallback validation")
        return _hmac_validate(token)
    except Exception as exc:
        raise AuthError(f"JWT validation failed: {exc}") from exc


def _hmac_validate(token: str) -> dict[str, Any]:
    """Minimal HMAC-based token validation without PyJWT."""
    parts = token.split(".")
    if len(parts) != 3:
        raise AuthError("Malformed token")

    signature_input = f"{parts[0]}.{parts[1]}".encode()
    expected = hashlib.sha256(JWT_SECRET.encode() + signature_input).hexdigest()

    if not hmac.compare_digest(parts[2], expected):
        raise AuthError("Invalid token signature")

    return {"sub": "hmac-validated", "token_prefix": token[:20]}


def authenticate(headers: dict[str, str]) -> dict[str, Any]:
    """Authenticate a request based on the configured auth type.

    Args:
        headers: Request headers (case-insensitive keys).

    Returns:
        Auth context dict with user info (empty for noop).

    Raises:
        AuthError: If authentication fails.
    """
    if AUTH_TYPE == "noop":
        return {}

    if AUTH_TYPE == "api_key":
        key = headers.get("x-api-key") or headers.get("X-API-Key") or ""
        if not key:
            raise AuthError("Missing X-API-Key header")
        if not validate_api_key(key):
            raise AuthError("Invalid API key")
        return {"auth_type": "api_key"}

    if AUTH_TYPE == "jwt":
        auth_header = headers.get("authorization") or headers.get("Authorization") or ""
        if not auth_header.startswith("Bearer "):
            raise AuthError("Missing or malformed Authorization header")
        token = auth_header[7:]
        claims = validate_jwt_token(token)
        return {"auth_type": "jwt", "claims": claims}

    raise AuthError(f"Unknown auth type: {AUTH_TYPE}", status_code=500)
