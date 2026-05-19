"""A2A Agent Identity middleware.

Validates inbound A2A requests on /a2a/ by requiring:
- X-Calling-Agent-ID in combined 'agent_name+uuid' format

The combined value must be present in the A2A_ALLOWED_INBOUND_AGENTS list
(when configured). Also captures X-Correlation-ID for end-to-end tracing.
"""

from __future__ import annotations

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)

_BASE_PROTECTED_PREFIXES = ("/a2a",)


class A2AIdentityMiddleware(BaseHTTPMiddleware):
    """Enforce agent identity verification on A2A endpoints."""

    def __init__(self, app, **kwargs):
        super().__init__(app, **kwargs)
        if settings.PROTECT_STREAM_ENDPOINT:
            self._protected_prefixes = _BASE_PROTECTED_PREFIXES + ("/v1/stream",)
        else:
            self._protected_prefixes = _BASE_PROTECTED_PREFIXES

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self._is_protected_path(request.url.path):
            return await call_next(request)

        # Capture correlation ID for tracing propagation
        correlation_id = (
            request.headers.get("X-Correlation-ID")
            or getattr(request.state, "request_id", None)
        )
        request.state.correlation_id = correlation_id

        # Validate X-Calling-Agent-ID presence and format
        identity = request.headers.get("X-Calling-Agent-ID")

        if not identity or "+" not in identity:
            logger.warning(
                "a2a_missing_identity",
                path=request.url.path,
                has_identity=bool(identity),
                correlation_id=correlation_id,
            )
            return self._forbidden(
                "Missing or malformed X-Calling-Agent-ID header "
                "(expected format: agent_name+uuid)"
            )

        agent_id, agent_uuid = identity.split("+", 1)
        request.state.calling_agent_id = agent_id
        request.state.calling_agent_uuid = agent_uuid

        # Allowlist check (skip if not configured)
        allowed = settings.A2A_ALLOWED_INBOUND_AGENTS
        if allowed is None:
            return await call_next(request)

        if identity not in allowed:
            logger.warning(
                "a2a_inbound_agent_denied",
                calling_agent_id=agent_id,
                reason="identity_not_in_allowlist",
                correlation_id=correlation_id,
            )
            return self._forbidden(
                f"Agent '{agent_id}' is not in the allowed inbound agents list."
            )

        return await call_next(request)

    def _is_protected_path(self, path: str) -> bool:
        return any(path.startswith(prefix) for prefix in self._protected_prefixes)

    @staticmethod
    def _forbidden(detail: str) -> JSONResponse:
        return JSONResponse(
            status_code=403,
            content={"detail": detail},
        )
