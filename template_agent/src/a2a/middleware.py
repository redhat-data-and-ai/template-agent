"""Starlette middleware for A2A HTTP-level auth and identity header extraction.

Handles three header categories:
1. Authentication  – ``Authorization: Bearer <token>`` → local JWT validation
2. Identity        – ``X-Calling-Agent-ID``
3. Correlation     – ``X-Correlation-ID``

All values are stashed in the ``a2a_request_ctx`` ContextVar so they are
visible to the executor and the delegation tool without threading arguments
through the A2A SDK's request handling.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from template_agent.src.a2a.context import A2ARequestContext, a2a_request_ctx

if TYPE_CHECKING:
    from template_agent.src.settings import Settings

logger = structlog.get_logger(__name__)

_WELL_KNOWN_SUFFIX = "/.well-known/agent-card.json"


def _extract_bearer(auth_header: str | None) -> str | None:
    if not auth_header:
        return None
    parts = auth_header.split()
    if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1].strip():
        return parts[1].strip()
    return None


def _validate_jwt(token: str, cfg: "Settings") -> bool:
    """Validate a JWT locally.

    Supports HS256 (shared secret) and RS256/ES256 (JWKS).
    Falls back to presence-only check when neither secret nor JWKS is configured.
    """
    import jwt

    if cfg.A2A_JWT_SECRET:
        try:
            jwt.decode(
                token,
                cfg.A2A_JWT_SECRET,
                algorithms=["HS256"],
                audience=cfg.A2A_JWT_AUDIENCE,
                issuer=cfg.A2A_JWT_ISSUER,
                options={
                    "verify_aud": cfg.A2A_JWT_AUDIENCE is not None,
                    "verify_iss": cfg.A2A_JWT_ISSUER is not None,
                },
            )
            return True
        except jwt.PyJWTError as exc:
            logger.warning("jwt_validation_failed", error=str(exc))
            return False

    if cfg.A2A_JWT_JWKS_URL:
        try:
            jwks_client = jwt.PyJWKClient(cfg.A2A_JWT_JWKS_URL)
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256", "ES256"],
                audience=cfg.A2A_JWT_AUDIENCE,
                issuer=cfg.A2A_JWT_ISSUER,
                options={
                    "verify_aud": cfg.A2A_JWT_AUDIENCE is not None,
                    "verify_iss": cfg.A2A_JWT_ISSUER is not None,
                },
            )
            return True
        except jwt.PyJWTError as exc:
            logger.warning("jwks_validation_failed", error=str(exc))
            return False

    # Neither secret nor JWKS configured → presence-only check (MCP does real validation)
    return True


_V1_METHODS = frozenset({
    "SendMessage", "SendStreamingMessage", "GetTask", "CancelTask",
    "ListTasks", "SubscribeToTask", "CreateTaskPushNotificationConfig",
    "GetTaskPushNotificationConfig", "ListTaskPushNotificationConfigs",
    "DeleteTaskPushNotificationConfig", "GetExtendedAgentCard",
})


class A2AVersionDefaultMiddleware(BaseHTTPMiddleware):
    """Inject ``A2A-Version: 1.0`` when the header is absent and the
    JSON-RPC method is a v1.0 method name.

    Clients like the A2A TCK send v1.0 method names (``SendMessage``) but
    omit the ``A2A-Version`` header. The SDK defaults missing headers to
    ``0.3`` and rejects the request. This middleware fills in the header
    only for v1.0 methods so the v0.3 compat path (``message/send`` etc.)
    still works for clients like the A2A Inspector.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method == "POST" and "a2a-version" not in request.headers:
            try:
                body = await request.body()
                import json as _json
                method = _json.loads(body).get("method", "")
                if method in _V1_METHODS:
                    request.scope["headers"] = list(request.scope["headers"]) + [
                        (b"a2a-version", b"1.0"),
                    ]
            except Exception:
                pass
        return await call_next(request)


class A2AAuthMiddleware(BaseHTTPMiddleware):
    """Extract auth + identity headers and populate ``a2a_request_ctx``."""

    def __init__(self, app, cfg: "Settings"):
        super().__init__(app)
        self.cfg = cfg

    async def dispatch(self, request: Request, call_next) -> Response:
        # Agent card endpoint is unauthenticated per A2A spec
        if request.url.path.endswith(_WELL_KNOWN_SUFFIX):
            return await call_next(request)

        token = _extract_bearer(request.headers.get("authorization"))

        if self.cfg.A2A_AUTH_REQUIRED and not token:
            return JSONResponse(
                status_code=401,
                content={"error": "Missing bearer token"},
                headers={"WWW-Authenticate": 'Bearer realm="a2a"'},
            )

        if token and self.cfg.A2A_AUTH_REQUIRED:
            if not _validate_jwt(token, self.cfg):
                return JSONResponse(
                    status_code=401,
                    content={"error": "Invalid or expired token"},
                    headers={"WWW-Authenticate": 'Bearer realm="a2a", error="invalid_token"'},
                )

        correlation_id = (
            request.headers.get("x-correlation-id") or str(uuid.uuid4())
        )

        ctx = A2ARequestContext(
            access_token=token,
            calling_agent_id=request.headers.get("x-calling-agent-id"),
            correlation_id=correlation_id,
        )
        reset_token = a2a_request_ctx.set(ctx)

        structlog.contextvars.bind_contextvars(
            correlation_id=ctx.correlation_id,
            calling_agent_id=ctx.calling_agent_id,
        )

        try:
            response = await call_next(request)
            return response
        finally:
            a2a_request_ctx.reset(reset_token)
            structlog.contextvars.unbind_contextvars(
                "correlation_id", "calling_agent_id"
            )
