"""Production security middleware for HTTP security headers and request validation.

Implements OWASP security recommendations for FastAPI applications.
"""

from typing import Any

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add OWASP-recommended security headers to all HTTP responses.

    Headers applied:
    - X-Content-Type-Options: nosniff
    - X-Frame-Options: DENY
    - X-XSS-Protection: 1; mode=block
    - Strict-Transport-Security: max-age=31536000; includeSubDomains (HTTPS only)
    - Content-Security-Policy: default-src 'self'
    - Referrer-Policy: strict-origin-when-cross-origin
    - Permissions-Policy: geolocation=(), microphone=(), camera=()
    """

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        """Add security headers to response."""
        response = await call_next(request)

        # Always set these headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )

        # CSP: allow self for API endpoints, adjust if serving web UI
        response.headers["Content-Security-Policy"] = "default-src 'self'"

        # HSTS: only set on HTTPS connections
        if request.url.scheme == "https" or settings.is_production:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )

        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Enforce request body size limits to prevent DoS attacks.

    Default limit: 10MB (configurable via REQUEST_BODY_MAX_SIZE).
    """

    def __init__(self, app: Any, max_size_bytes: int = 10 * 1024 * 1024):
        """Initialize with configurable max request body size."""
        super().__init__(app)
        self.max_size_bytes = max_size_bytes
        logger.info("Request body size limit: %d bytes", max_size_bytes)

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        """Check request body size before processing."""
        # Skip for GET/HEAD/OPTIONS (no body)
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return await call_next(request)

        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_size_bytes:
            logger.warning(
                "Request body too large: %s bytes (max %d)",
                content_length,
                self.max_size_bytes,
            )
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "detail": f"Request body exceeds maximum size of {self.max_size_bytes} bytes",
                    "error_type": "request_too_large",
                },
            )

        return await call_next(request)
