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

    def __init__(
        self,
        app: Any,
        max_size_bytes: int = 10 * 1024 * 1024,
        path_limits: dict[str, int] | None = None,
    ):
        """Initialize with configurable max request body size.

        Args:
            app: The ASGI app.
            max_size_bytes: Global body size limit.
            path_limits: Per-path overrides for routes that legitimately
                accept more (e.g. /upload). A path is never exempt: an
                unlimited route lets an unauthenticated caller stream
                arbitrary bytes, since the body is parsed before the route
                can check auth.
        """
        super().__init__(app)
        self.max_size_bytes = max_size_bytes
        self.path_limits = path_limits or {}
        logger.info(
            "Request body size limit: %d bytes (overrides: %s)",
            max_size_bytes,
            self.path_limits or "none",
        )

    def limit_for(self, path: str) -> int:
        """Return the body size limit that applies to a path."""
        return self.path_limits.get(path, self.max_size_bytes)

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        """Check declared request body size before processing."""
        # Skip for GET/HEAD/OPTIONS (no body)
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return await call_next(request)

        limit = self.limit_for(request.url.path)
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > limit:
            logger.warning(
                "Request body too large: %s bytes (max %d)", content_length, limit
            )
            return _too_large(limit)

        return await call_next(request)


def _too_large(limit: int) -> JSONResponse:
    """Build the 413 returned when a body exceeds its limit."""
    return JSONResponse(
        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        content={
            "detail": f"Request body exceeds maximum size of {limit} bytes",
            "error_type": "request_too_large",
        },
    )


class StreamingBodyLimitMiddleware:
    """Cap request body bytes as they arrive, before anything parses them.

    ``RequestSizeLimitMiddleware`` only inspects ``Content-Length``, which a
    caller controls and a chunked request omits entirely. That is not enough
    for a route that accepts uploads: FastAPI resolves ``UploadFile``
    parameters as *dependencies*, so python-multipart spools the whole body
    to disk **before** the handler runs — before the feature flag is read,
    before authorization, and before any in-handler size check. An
    unauthenticated caller could therefore fill the disk and only then be
    told 404 or 401.

    This is pure ASGI rather than ``BaseHTTPMiddleware`` because it has to
    wrap ``receive`` to count bytes as they stream in; ``BaseHTTPMiddleware``
    cannot see the body without consuming it.
    """

    def __init__(
        self,
        app: Any,
        max_size_bytes: int,
        path_limits: dict[str, int] | None = None,
    ):
        """Initialize with a default limit and optional per-path overrides."""
        self.app = app
        self.max_size_bytes = max_size_bytes
        self.path_limits = path_limits or {}

    def limit_for(self, path: str) -> int:
        """Return the body size limit that applies to a path."""
        return self.path_limits.get(path, self.max_size_bytes)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """Count streamed body bytes and abort the request past the limit."""
        if scope["type"] != "http" or scope.get("method") in (
            "GET",
            "HEAD",
            "OPTIONS",
        ):
            await self.app(scope, receive, send)
            return

        limit = self.limit_for(scope.get("path", ""))
        state = {"received": 0, "exceeded": False, "sent": False}

        async def limited_receive() -> Any:
            message = await receive()
            if message["type"] == "http.request":
                state["received"] += len(message.get("body", b""))
                if state["received"] > limit:
                    state["exceeded"] = True
                    logger.warning(
                        "Streamed body exceeded limit on %s: %d bytes (max %d)",
                        scope.get("path", ""),
                        state["received"],
                        limit,
                    )
                    # Unwind the app: it sees the client as gone, so the
                    # multipart parser stops reading instead of spooling on.
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message: Any) -> None:
            # Once over the limit we answer with our own 413; suppress
            # whatever the aborted app tries to emit.
            if state["exceeded"]:
                return
            state["sent"] = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            # A parser reacting to the synthetic disconnect is expected;
            # anything else is a genuine error and must surface.
            if not state["exceeded"]:
                raise

        if state["exceeded"] and not state["sent"]:
            await _too_large(limit)(scope, receive, send)
