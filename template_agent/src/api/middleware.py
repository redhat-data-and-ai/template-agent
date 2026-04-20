"""Middleware configuration for the FastAPI application.

This module provides custom middleware components including request/response logging
and CORS configuration. Middleware is executed for every request and can modify
requests/responses or perform cross-cutting concerns like logging and timing.

Why this exists:
    Middleware provides centralized request processing, logging, and CORS handling.
    Separating middleware from app setup keeps each concern focused and testable.

Classes:
    RequestLoggingMiddleware: Logs incoming requests and outgoing responses
"""

import time
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all incoming requests and outgoing responses."""

    async def dispatch(self, request: Request, call_next: Callable):
        """Process and log incoming requests and outgoing responses."""
        if not settings.REQUEST_LOGGING_ENABLED:
            return await call_next(request)

        start_time = time.time()

        # Capture request details
        request_data = {
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host if request.client else None,
            "query_params": dict(request.query_params)
            if request.query_params
            else None,
        }

        # Optionally log headers
        if settings.REQUEST_LOG_HEADERS:
            request_data["headers"] = dict(request.headers)

        # Optionally log request body
        if settings.REQUEST_LOG_BODY:
            try:
                body_bytes = await request.body()
                body_size = len(body_bytes)

                if body_size > 0:
                    request_data["body_size"] = body_size
                    if (
                        settings.REQUEST_LOG_BODY_MAX_SIZE == 0
                        or body_size <= settings.REQUEST_LOG_BODY_MAX_SIZE
                    ):
                        try:
                            body_str = body_bytes.decode("utf-8")
                            request_data["body"] = body_str
                        except UnicodeDecodeError:
                            request_data["body"] = "<binary data>"
                    else:
                        request_data["body"] = f"<truncated: {body_size} bytes>"

                # Rebuild request with body
                async def receive():
                    return {"type": "http.request", "body": body_bytes}

                request = Request(request.scope, receive)
            except Exception as e:
                logger.warning("failed_to_read_request_body", error=str(e))

        logger.info("incoming_request", **request_data)

        # Process request
        response = await call_next(request)

        # Capture response details
        duration_ms = (time.time() - start_time) * 1000
        response_data = {
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
        }

        # Optionally log response headers
        if settings.REQUEST_LOG_HEADERS:
            response_data["headers"] = dict(response.headers)

        logger.info("outgoing_response", **response_data)

        return response


def configure_middleware(app: FastAPI) -> None:
    """Configure all middleware for the FastAPI application.

    Args:
        app: The FastAPI application instance to configure.
    """
    # Register request logging middleware first to capture all requests
    app.add_middleware(RequestLoggingMiddleware)

    # Configure CORS middleware for cross-origin requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
