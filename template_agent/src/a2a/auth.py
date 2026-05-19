"""Context bridge from A2AIdentityMiddleware into the A2A SDK.

A2AIdentityMiddleware validates agent identity and bearer tokens at the
HTTP layer, populating request.state with access_token, jwt_claims,
token_format, username, etc.  This builder reads that validated state
and carries it into ServerCallContext.state for the executor — it never
raises and never re-validates.

The /.well-known/agent-card.json endpoint is NOT routed through this
builder (it uses separate routes), so it remains public.
"""

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


class A2AServerCallContextBuilder(DefaultServerCallContextBuilder):
    """Thin bridge from request.state into ServerCallContext.

    A2AIdentityMiddleware has already validated the bearer token and
    populated request.state.  This builder reads that state and packages
    it into a ServerCallContext for the executor.  It never raises.
    """

    def build(self, request: Request) -> ServerCallContext:
        """Read validated auth state from request.state and build the call context."""
        token = getattr(request.state, "access_token", None)
        token_format = getattr(request.state, "token_format", "unknown")
        claims = getattr(request.state, "jwt_claims", {})
        user_name = getattr(request.state, "username", "unknown")
        calling_agent_id = getattr(request.state, "calling_agent_id", None)

        logger.info(
            f"A2A request authenticated: user={user_name}, format={token_format}"
            + (f", calling_agent={calling_agent_id}" if calling_agent_id else "")
        )

        state = {
            ACCESS_TOKEN_STATE_KEY: token,
            "headers": dict(request.headers),
            "token_format": token_format,
            "jwt_claims": claims,
            "correlation_id": getattr(request.state, "correlation_id", None),
            "calling_agent_id": calling_agent_id,
            "calling_agent_uuid": getattr(request.state, "calling_agent_uuid", None),
        }

        return ServerCallContext(
            user=AuthenticatedUser(claims=claims, name=user_name),
            state=state,
            requested_extensions=get_requested_extensions(
                request.headers.getlist(HTTP_EXTENSION_HEADER)
            ),
        )
