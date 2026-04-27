"""A2A Agent Registry – discovers and tracks downstream agents at startup."""

from __future__ import annotations

import httpx
import structlog

from template_agent.src.a2a.models import A2ATargetAgent
from template_agent.src.settings import settings

logger = structlog.get_logger(__name__)

_CARD_PATHS = [
    "/a2a/.well-known/agent-card.json",
    "/.well-known/agent-card.json",
    "/.well-known/agent.json",
]

_registry: A2AAgentRegistry | None = None


class A2AAgentRegistry:
    """In-memory registry of downstream A2A agents."""

    def __init__(self, timeout: float = 30.0):
        self._agents: dict[str, A2ATargetAgent] = {}
        self._client = httpx.AsyncClient(timeout=timeout)

    async def discover(self, targets: dict[str, dict[str, str]]) -> None:
        """Fetch agent cards for all configured targets.

        Each entry in *targets* maps ``agent_id`` to at least ``{"base_url": "..."}``
        with an optional ``"description"``.
        """
        for agent_id, info in targets.items():
            base_url = info["base_url"].rstrip("/")
            description = info.get("description")
            agent = A2ATargetAgent(
                agent_id=agent_id,
                base_url=base_url,
                description=description,
            )
            card = await self._fetch_card(base_url)
            if card:
                agent.card = card
                agent.skills = [s["id"] for s in card.get("skills", []) if "id" in s]
                agent.capabilities = card.get("capabilities")
                logger.info(
                    "a2a_agent_discovered",
                    agent_id=agent_id,
                    skills=agent.skills,
                )
            else:
                logger.warning(
                    "a2a_agent_card_fetch_failed",
                    agent_id=agent_id,
                    base_url=base_url,
                )
            self._agents[agent_id] = agent

    async def _fetch_card(self, base_url: str) -> dict | None:
        for path in _CARD_PATHS:
            url = f"{base_url}{path}"
            try:
                resp = await self._client.get(url)
                if resp.status_code == 200:
                    return resp.json()
            except Exception as exc:
                logger.debug("a2a_card_probe_failed", url=url, error=str(exc))
        return None

    def get(self, agent_id: str) -> A2ATargetAgent | None:
        return self._agents.get(agent_id)

    def list_agents(self) -> list[A2ATargetAgent]:
        return list(self._agents.values())

    def list_agent_ids(self) -> list[str]:
        return list(self._agents.keys())

    async def close(self) -> None:
        await self._client.aclose()


def get_registry() -> A2AAgentRegistry:
    """Return the module-level singleton (must be initialized first)."""
    global _registry
    if _registry is None:
        _registry = A2AAgentRegistry(timeout=settings.A2A_REQUEST_TIMEOUT)
    return _registry


async def initialize_registry() -> None:
    """Create the registry and discover downstream agents from settings."""
    global _registry
    _registry = A2AAgentRegistry(timeout=settings.A2A_REQUEST_TIMEOUT)
    targets = settings.A2A_TARGET_AGENTS
    if targets:
        await _registry.discover(targets)
        logger.info(
            "a2a_registry_initialized",
            agent_count=len(_registry.list_agents()),
        )
    else:
        logger.info("a2a_registry_initialized", agent_count=0, note="no targets configured")


async def cleanup_registry() -> None:
    """Close the registry's HTTP client."""
    global _registry
    if _registry:
        await _registry.close()
        _registry = None
