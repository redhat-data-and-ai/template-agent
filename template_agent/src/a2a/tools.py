"""Dynamic LangChain tools for calling downstream A2A agents.

Discovers downstream agents via the SDK's A2ACardResolver, then creates
one StructuredTool per agent (not per skill).  Skill descriptions are
aggregated into the tool description so the LLM can make informed routing
decisions, while the downstream agent retains ownership of internal skill
selection (A2A spec §4.4.5 — opaque execution).
"""

import httpx
from a2a.client import A2ACardResolver
from a2a.types import AgentCard
from langchain_core.tools import StructuredTool

from template_agent.src.a2a.client import send_to_downstream_agent
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


def _build_agent_description(card: AgentCard) -> str:
    """Build an LLM-facing description from the agent's card and skills."""
    skill_parts: list[str] = []
    for skill in card.skills:
        entry = skill.description
        if skill.examples:
            examples = ", ".join(f"'{e}'" for e in skill.examples[:3])
            entry += f" (e.g. {examples})"
        skill_parts.append(entry)

    skills_summary = "; ".join(skill_parts) if skill_parts else "general-purpose"
    return (
        f"Delegate to '{card.name}' agent: {card.description or 'downstream agent'}. "
        f"Skills: {skills_summary}"
    )


def _make_tool_for_agent(
    card: AgentCard,
    agent_url: str,
    access_token: str | None,
    context_id: str | None = None,
    correlation_id: str | None = None,
) -> StructuredTool:
    """Create one LangChain tool for a downstream A2A agent.

    The tool description aggregates the agent's skills so the LLM
    can make informed routing decisions, while the downstream agent
    retains ownership of internal skill selection (A2A opaque execution).
    """
    description = _build_agent_description(card)

    _url = agent_url
    _token = access_token
    _ctx = context_id
    _corr_id = correlation_id

    async def _invoke(
        query: str,
        url: str = _url,
        token: str | None = _token,
        ctx: str | None = _ctx,
        corr_id: str | None = _corr_id,
    ) -> str:
        return await send_to_downstream_agent(
            agent_url=url,
            message_text=query,
            access_token=token,
            context_id=ctx,
            correlation_id=corr_id,
        )

    tool_name = f"a2a_{card.name.lower().replace(' ', '_')}"
    tool_name = "".join(c if c.isalnum() or c == "_" else "_" for c in tool_name)[:64]

    return StructuredTool.from_function(
        coroutine=_invoke,
        name=tool_name,
        description=description,
    )


async def build_a2a_tools(
    access_token: str | None = None,
    context_id: str | None = None,
    correlation_id: str | None = None,
) -> list[StructuredTool]:
    """Discover downstream A2A agents and build LangChain tools.

    Fetches Agent Cards from configured URLs via the SDK's A2ACardResolver,
    then creates one tool per agent.  Auth credentials and correlation
    context are captured in each tool's closure.

    Returns an empty list if no downstream agents are configured or if
    discovery fails (non-fatal).
    """
    downstream_urls = settings.a2a_downstream_urls
    if not downstream_urls:
        return []

    all_tools: list[StructuredTool] = []

    async with httpx.AsyncClient(timeout=10.0, verify=False) as http_client:  # nosec B501
        for url in downstream_urls:
            try:
                resolver = A2ACardResolver(httpx_client=http_client, base_url=url)
                card = await resolver.get_agent_card()
                tool = _make_tool_for_agent(card, url, access_token, context_id, correlation_id)
                all_tools.append(tool)
                logger.info(
                    f"Discovered A2A agent '{card.name}' at {url} "
                    f"with {len(card.skills)} skill(s)"
                )
            except Exception as e:
                logger.warning(f"Failed to discover A2A agent at {url}: {e}. Skipping.")

    if all_tools:
        logger.info(f"Registered {len(all_tools)} downstream A2A tool(s) total")

    return all_tools
