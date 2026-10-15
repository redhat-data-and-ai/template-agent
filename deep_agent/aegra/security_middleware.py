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

    Upload paths are exempt and governed by the attachment limits
    instead. A single global cap cannot serve both: 10MB is right for
    a JSON body and rejects every real video, while raising it
    globally would let any endpoint accept 50MB of JSON to parse.

    The exemption is a *larger* limit, not no limit — uploads are
    still bounded, by `attachments.max_file_bytes`, and the endpoint
    itself reads in chunks and stops at the cap.
    """

    # Prefix-matched. Kept narrow and explicit rather than a pattern,
    # so adding an exemption is a deliberate edit.
    _UPLOAD_PATHS = ("/knowledge/documents", "/knowledge/sources")
    # `/state` is how an attachment is actually written: the upload
    # endpoint base64-encodes the file and POSTs it back to the thread
    # over loopback. Without the exemption a 12 MB video uploads fine
    # and then fails on the internal write, surfacing as a 502 that
    # blames storage rather than size.
    _UPLOAD_SUFFIXES = ("/attachments", "/state")

    def __init__(
        self,
        app: Any,
        max_size_bytes: int = 10 * 1024 * 1024,
        upload_max_bytes: int | None = None,
    ):
        """Initialize with configurable max request body size."""
        super().__init__(app)
        self.max_size_bytes = max_size_bytes
        # Headroom over the per-file cap for multipart framing.
        self.upload_max_bytes = upload_max_bytes or (max_size_bytes * 6)
        logger.info(
            "Request body size limit: %d bytes (uploads: %d)",
            max_size_bytes,
            self.upload_max_bytes,
        )

    def _limit_for(self, path: str) -> int:
        if path.endswith(self._UPLOAD_SUFFIXES) or path.startswith(self._UPLOAD_PATHS):
            return self.upload_max_bytes
        return self.max_size_bytes

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        """Check request body size before processing."""
        # Skip for GET/HEAD/OPTIONS (no body)
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return await call_next(request)

        limit = self._limit_for(request.url.path)
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > limit:
            logger.warning(
                "Request body too large: %s bytes (max %d) on %s",
                content_length,
                limit,
                request.url.path,
            )
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "detail": f"Request body exceeds maximum size of {limit} bytes",
                    "error_type": "request_too_large",
                },
            )

        return await call_next(request)
