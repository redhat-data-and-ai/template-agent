"""Build the A2A (Agent2Agent) Starlette sub-application for discovery and JSON-RPC."""

from __future__ import annotations

import os

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    HTTPAuthSecurityScheme,
    SecurityRequirement,
    SecurityScheme,
    StringList,
)
from starlette.applications import Starlette

from template_agent.src.a2a.middleware import A2AAuthMiddleware, A2AVersionDefaultMiddleware
from template_agent.src.a2a.registry import get_registry
from template_agent.src.core.a2a_executor import TemplateAgentA2AExecutor
from template_agent.src.settings import Settings, settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


def resolve_a2a_public_base_url(cfg: Settings) -> str:
    """Public base URL for the A2A JSON-RPC endpoint.

    Starlette ``app.mount()`` redirects requests without a trailing
    slash (307), so the URL **must** end with ``/`` to avoid redirect
    loops when A2A SDK clients POST to it.
    """
    prefix = (cfg.A2A_PATH_PREFIX or "/a2a").strip().rstrip("/") or "/a2a"
    if not prefix.startswith("/"):
        prefix = f"/{prefix}"

    explicit = (cfg.AGENT_PUBLIC_BASE_URL or "").strip().rstrip("/")
    if not explicit:
        explicit = os.getenv("AGENT_URL", "").strip().rstrip("/")
    if explicit:
        return f"{explicit}{prefix}/"

    host = cfg.AGENT_HOST
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    return f"http://{host}:{cfg.AGENT_PORT}{prefix}/"


def _get_downstream_skills() -> list[AgentSkill]:
    """Collect skills from discovered downstream agents in the registry."""
    try:
        registry = get_registry()
        skills: list[AgentSkill] = []
        for agent in registry.list_agents():
            for skill_name in agent.skills:
                skills.append(
                    AgentSkill(
                        id=f"downstream:{agent.agent_id}:{skill_name}",
                        name=f"{agent.agent_id}/{skill_name}",
                        description=agent.description or f"Skill from {agent.agent_id}",
                        tags=["downstream", agent.agent_id],
                        examples=[],
                    )
                )
        return skills
    except Exception:
        return []


def build_agent_card(cfg: Settings) -> AgentCard:
    """Agent card served at ``/.well-known/agent-card.json`` under the A2A mount."""
    base_url = resolve_a2a_public_base_url(cfg)

    primary_skill = AgentSkill(
        id="template-agent-mcp",
        name="Template agent",
        description=(
            "LangGraph agent with MCP tools (same stack as POST /v1/stream). "
            "Requires OAuth bearer via Authorization header."
        ),
        tags=["mcp", "template-agent", "langgraph"],
        examples=[
            "What is 2 multiplied by 3?",
            "List available tools.",
        ],
    )
    all_skills = [primary_skill] + _get_downstream_skills()

    security_schemes = None
    security_requirements = None
    if cfg.A2A_AUTH_REQUIRED:
        security_schemes = {
            "bearer": SecurityScheme(
                http_auth_security_scheme=HTTPAuthSecurityScheme(
                    scheme="bearer",
                    description="OAuth2 / SSO bearer in Authorization header",
                ),
            ),
        }
        security_requirements = [
            SecurityRequirement(schemes={"bearer": StringList()}),
        ]

    provider = None
    if cfg.A2A_PROVIDER_NAME:
        provider = AgentProvider(
            organization=cfg.A2A_PROVIDER_NAME,
            url=cfg.A2A_PROVIDER_URL or "",
        )

    return AgentCard(
        name="Template Agent",
        description=(
            "Template agent for MCP-backed tool use. "
            "Authenticate via Authorization: Bearer header. "
            "Application context (user_id, session_id, thread_id) goes in A2A message metadata."
        ),
        supported_interfaces=[
            AgentInterface(protocol_binding="JSONRPC", url=base_url),
        ],
        version=cfg.A2A_AGENT_VERSION,
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
        skills=all_skills,
        security_schemes=security_schemes,
        security_requirements=security_requirements,
        provider=provider,
    )


def build_a2a_starlette_app(cfg: Settings | None = None) -> Starlette:
    """Starlette app exposing A2A JSON-RPC and the agent card."""
    cfg = cfg or settings
    card = build_agent_card(cfg)
    handler = DefaultRequestHandler(
        agent_executor=TemplateAgentA2AExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )

    routes = []
    routes.extend(create_agent_card_routes(card))
    routes.extend(create_jsonrpc_routes(handler, rpc_url="/", enable_v0_3_compat=True))

    a2a_starlette = Starlette(routes=routes)
    a2a_starlette.add_middleware(A2AAuthMiddleware, cfg=cfg)
    a2a_starlette.add_middleware(A2AVersionDefaultMiddleware)

    base_url = card.supported_interfaces[0].url if card.supported_interfaces else "?"
    logger.info(
        "A2A enabled: jsonrpc url=%s, auth_required=%s (set AGENT_PUBLIC_BASE_URL or AGENT_URL if behind a proxy)",
        base_url,
        cfg.A2A_AUTH_REQUIRED,
    )
    return a2a_starlette
