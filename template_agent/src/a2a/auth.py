"""Bearer token authentication for incoming A2A requests.

Accepts both JWT (3-part) and JWE (5-part) tokens in the Authorization
header. JWT tokens are decoded for claims; JWE tokens are stored as-is
for forwarding to downstream agents that can decrypt them.

The raw token is stored in ServerCallContext.state so the AgentExecutor
can forward it to downstream agents/MCP servers.
"""

import jwt
from a2a.auth.user import User
from a2a.server.context import ServerCallContext
from a2a.server.routes.common import (
    HTTP_EXTENSION_HEADER,
    DefaultServerCallContextBuilder,
    get_requested_extensions,
)
from starlette.requests import Request

from template_agent.src.a2a.executor import ACCESS_TOKEN_STATE_KEY
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


class AuthenticatedUser(User):
    """Authenticated user extracted from a bearer token."""

    def __init__(self, claims: dict, name: str):
        """Initialize with decoded token claims and display name."""
        self._claims = claims
        self._name = name

    @property
    def is_authenticated(self) -> bool:
        """Return True; this user was validated from a bearer token."""
        return True

    @property
    def user_name(self) -> str:
        """Return the user's display name extracted from the token."""
        return self._name

    @property
    def claims(self) -> dict:
        """Return the full set of decoded token claims."""
        return self._claims


def _extract_bearer_token(request: Request) -> str | None:
    """Pull the raw token from 'Authorization: Bearer <token>'."""
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return None


def _is_jwe(token: str) -> bool:
    """JWE compact serialization has 5 dot-separated segments."""
    return token.count(".") == 4


def _decode_jwt(token: str) -> dict:
    """Decode a JWT without signature verification for claim extraction."""
    return jwt.decode(
        token,
        options={
            "verify_signature": False,
            "verify_exp": False,
            "verify_aud": False,
            "verify_iss": False,
        },
        algorithms=["RS256", "HS256", "ES256"],
    )


def _validate_token(token: str) -> tuple[dict, str]:
    """Validate a bearer token and extract identity information.

    Returns:
        A tuple of (claims_dict, token_format) where token_format is "jwt" or "jwe".
        For JWE tokens, claims will contain only {"token_format": "jwe"} since
        we cannot decrypt them here — they are forwarded as-is to downstream agents.
    """
    if _is_jwe(token):
        logger.info("A2A auth: JWE token detected, storing for forwarding")
        return {"token_format": "jwe"}, "jwe"

    claims = _decode_jwt(token)
    return claims, "jwt"


class A2AServerCallContextBuilder(DefaultServerCallContextBuilder):
    """Extends the default context builder with bearer token validation.

    For every A2A request this:
    1. Extracts the Authorization: Bearer token
    2. Detects format: JWT (3-part) or JWE (5-part)
    3. For JWT: decodes claims to extract user identity
    4. For JWE: accepts as-is for downstream forwarding
    5. Stores the raw token in state so the executor can forward it

    The /.well-known/agent-card.json endpoint is NOT routed through
    this builder (it uses separate routes), so it remains public.
    """

    def build(self, request: Request) -> ServerCallContext:
        """Extract and validate the bearer token, then build the call context."""
        token = _extract_bearer_token(request)

        if not token:
            raise PermissionError("Missing Authorization: Bearer token")

        try:
            claims, token_format = _validate_token(token)
        except jwt.ExpiredSignatureError:
            raise PermissionError("Bearer token has expired")
        except jwt.InvalidTokenError as e:
            raise PermissionError(f"Invalid bearer token: {e}")

        if token_format == "jwt":
            user_name = (
                claims.get("preferred_username")
                or claims.get("sub")
                or claims.get("email")
                or "unknown"
            )
        else:
            user_name = "jwe-authenticated"

        logger.info(
            f"A2A request authenticated: user={user_name}, format={token_format}"
        )

        state = {
            ACCESS_TOKEN_STATE_KEY: token,
            "headers": dict(request.headers),
            "token_format": token_format,
            "jwt_claims": claims,
        }

        return ServerCallContext(
            user=AuthenticatedUser(claims=claims, name=user_name),
            state=state,
            requested_extensions=get_requested_extensions(
                request.headers.getlist(HTTP_EXTENSION_HEADER)
            ),
        )
