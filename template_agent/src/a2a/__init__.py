"""A2A protocol support: app, middleware, registry, delegation, and context."""

from template_agent.src.a2a.app import (
    build_a2a_starlette_app,
    build_agent_card,
    resolve_a2a_public_base_url,
)
from template_agent.src.a2a.context import A2ARequestContext, a2a_request_ctx
from template_agent.src.a2a.middleware import A2AAuthMiddleware
from template_agent.src.a2a.models import A2ATargetAgent
from template_agent.src.a2a.registry import (
    cleanup_registry,
    get_registry,
    initialize_registry,
)

__all__ = [
    "A2AAuthMiddleware",
    "A2ARequestContext",
    "A2ATargetAgent",
    "a2a_request_ctx",
    "build_a2a_starlette_app",
    "build_agent_card",
    "cleanup_registry",
    "get_registry",
    "initialize_registry",
    "resolve_a2a_public_base_url",
]
