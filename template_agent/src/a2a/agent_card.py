"""Agent Card builder for the template agent's A2A identity."""

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    APIKeySecurityScheme,
    HTTPAuthSecurityScheme,
    SecurityRequirement,
    SecurityScheme,
    StringList,
)

from template_agent.src.settings import settings


def build_agent_card() -> AgentCard:
    """Build the A2A Agent Card from current settings.

    The card advertises:
    - JSON-RPC v1.0 and v0.3 interfaces (preferred)
    - HTTP+JSON/REST v1.0 interface
    - Streaming capability
    - (Bearer OR X-Token) + X-Calling-Agent-ID auth requirements
    - Agent skills describing what this agent can do
    """
    base_url = settings.a2a_base_url

    provider = None
    if settings.A2A_PROVIDER_ORG and settings.A2A_PROVIDER_URL:
        provider = AgentProvider(
            organization=settings.A2A_PROVIDER_ORG,
            url=settings.A2A_PROVIDER_URL,
        )

    return AgentCard(
        name=settings.A2A_AGENT_NAME,
        description=settings.A2A_AGENT_DESCRIPTION,
        version=settings.A2A_AGENT_VERSION,
        provider=provider,
        supported_interfaces=[
            AgentInterface(
                url=f"{base_url}/a2a",
                protocol_binding="JSONRPC",
                protocol_version="1.0",
            ),
            AgentInterface(
                url=f"{base_url}/a2a",
                protocol_binding="JSONRPC",
                protocol_version="0.3",
            ),
            AgentInterface(
                url=f"{base_url}/a2a",
                protocol_binding="HTTP+JSON",
                protocol_version="1.0",
            ),
        ],
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain", "application/json"],
        skills=[
            AgentSkill(
                id="general-assistant",
                name="General Assistant",
                description=(
                    "General-purpose AI assistant with access to tools via MCP. "
                    "Can answer questions, perform calculations, generate code reviews, "
                    "and execute multi-step reasoning tasks."
                ),
                tags=["chat", "tools", "reasoning", "mcp"],
                examples=[
                    "What is 2 multiplied by 3?",
                    "Generate a code review for my changes",
                    "Help me analyze this data",
                ],
            ),
        ],
        security_schemes={
            "bearer": SecurityScheme(
                http_auth_security_scheme=HTTPAuthSecurityScheme(
                    scheme="Bearer",
                    bearer_format="JWT",
                    description="JWT bearer token via Authorization header",
                )
            ),
            "xToken": SecurityScheme(
                api_key_security_scheme=APIKeySecurityScheme(
                    description="Alternative bearer token passed via X-Token header",
                    location="header",
                    name="X-Token",
                )
            ),
            "agentIdentity": SecurityScheme(
                api_key_security_scheme=APIKeySecurityScheme(
                    description=(
                        "Calling agent identity in 'name+uuid' format "
                        "(e.g. 'my-agent+550e8400-e29b-41d4-a716-446655440000'). "
                        "Required for agent-to-agent identity verification "
                        "and allowlist enforcement."
                    ),
                    location="header",
                    name="X-Calling-Agent-ID",
                )
            ),
        },
        security_requirements=[
            # OR semantics: satisfy either option to authenticate
            SecurityRequirement(
                schemes={"bearer": StringList(), "agentIdentity": StringList()}
            ),
            SecurityRequirement(
                schemes={"xToken": StringList(), "agentIdentity": StringList()}
            ),
        ],
    )
