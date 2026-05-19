"""A2A Agent Identity and Authentication middleware.

Validates inbound A2A requests on /a2a/ (and optionally /v1/stream) by:
1. Requiring X-Calling-Agent-ID in combined 'agent_name+uuid' format
2. Checking the allowlist (A2A_ALLOWED_INBOUND_AGENTS, when configured)
3. Validating the bearer token (X-Token or Authorization: Bearer)

On success, populates request.state with access_token, token_format,
jwt_claims, and username so downstream code (the A2A context builder
and route handlers) can read them without re-validating.
"""

from __future__ import annotations

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

import jwt

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
        if allowed is not None and identity not in allowed:
            logger.warning(
                "a2a_inbound_agent_denied",
                calling_agent_id=agent_id,
                reason="identity_not_in_allowlist",
                correlation_id=correlation_id,
            )
            return self._forbidden(
                f"Agent '{agent_id}' is not in the allowed inbound agents list."
            )

        # Validate bearer token (X-Token or Authorization: Bearer)
        token = self._extract_token(request)
        if not token:
            logger.warning(
                "a2a_missing_token",
                path=request.url.path,
                calling_agent_id=agent_id,
                correlation_id=correlation_id,
            )
            return self._unauthorized(
                "Missing authentication token: send X-Token or Authorization: Bearer"
            )

        try:
            claims, token_format = self._validate_token(token)
        except Exception as e:
            logger.warning(
                "a2a_invalid_token",
                path=request.url.path,
                calling_agent_id=agent_id,
                correlation_id=correlation_id,
                error=str(e),
            )
            return self._unauthorized(f"Invalid bearer token: {e}")

        request.state.access_token = token
        request.state.token_format = token_format
        request.state.jwt_claims = claims
        if token_format == "jwt":
            request.state.username = (
                claims.get("preferred_username")
                or claims.get("sub")
                or claims.get("email")
                or "unknown"
            )
        else:
            request.state.username = "jwe-authenticated"

        return await call_next(request)

    def _is_protected_path(self, path: str) -> bool:
        return any(path.startswith(prefix) for prefix in self._protected_prefixes)

    @staticmethod
    def _forbidden(detail: str) -> JSONResponse:
        return JSONResponse(
            status_code=403,
            content={"detail": detail},
        )

    @staticmethod
    def _unauthorized(detail: str) -> JSONResponse:
        return JSONResponse(
            status_code=401,
            content={"detail": detail},
            headers={"WWW-Authenticate": "Bearer"},
        )

    @staticmethod
    def _extract_token(request: Request) -> str | None:
        """Extract token from X-Token header or Authorization: Bearer."""
        raw = request.headers.get("X-Token")
        if raw:
            return raw.strip()
        auth = request.headers.get("authorization")
        if not auth:
            return None
        parts = auth.strip().split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
        return None

    @staticmethod
    def _validate_token(token: str) -> tuple[dict, str]:
        """Detect JWT vs JWE and decode JWT claims without signature verification."""
        if token.count(".") == 4:
            logger.info("A2A auth: JWE token detected, storing for forwarding")
            return {"token_format": "jwe"}, "jwe"
        claims = jwt.decode(
            token,
            options={
                "verify_signature": False,
                "verify_exp": False,
                "verify_aud": False,
                "verify_iss": False,
            },
            algorithms=["RS256", "HS256", "ES256"],
        )
        return claims, "jwt"
