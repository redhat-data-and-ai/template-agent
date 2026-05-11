"""Dynamic LangChain tools for calling downstream A2A agents."""

import httpx
from a2a.client import A2ACardResolver
from a2a.types import AgentCard
from langchain_core.tools import StructuredTool

from template_agent.src.a2a.client import send_to_downstream_agent
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


def _make_tool_for_agent(
    card: AgentCard,
    agent_url: str,
    access_token: str | None,
    context_id: str | None = None,
) -> list[StructuredTool]:
    """Create LangChain tools from an Agent Card's skills."""
    tools = []

    for skill in card.skills:
        examples_str = ""
        if skill.examples:
            examples_str = " Examples: " + ", ".join(
                f"'{e}'" for e in skill.examples[:3]
            )

        description = (
            f"Delegate to '{card.name}' agent — {skill.description}.{examples_str}"
        )

        # Capture in closure
        _url = agent_url
        _token = access_token
        _skill_id = skill.id
        _ctx = context_id

        async def _invoke(query: str, url=_url, token=_token, ctx=_ctx) -> str:
            return await send_to_downstream_agent(
                agent_url=url,
                message_text=query,
                access_token=token,
                context_id=ctx,
            )

        tool_name = f"a2a_{card.name.lower().replace(' ', '_')}_{_skill_id}"
        # LangChain tool names must be <= 64 chars and alphanumeric/underscore
        tool_name = "".join(c if c.isalnum() or c == "_" else "_" for c in tool_name)[
            :64
        ]

        tools.append(
            StructuredTool.from_function(
                coroutine=_invoke,
                name=tool_name,
                description=description,
            )
        )

    return tools


async def build_a2a_tools(
    access_token: str | None = None,
    context_id: str | None = None,
) -> list[StructuredTool]:
    """Discover downstream A2A agents and build LangChain tools from their skills.

    Fetches Agent Cards from configured URLs, then creates a tool per skill.
    The access token and context_id are captured in each tool's closure so
    downstream calls reuse the same conversation thread.

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
                agent_tools = _make_tool_for_agent(card, url, access_token, context_id)
                all_tools.extend(agent_tools)
                logger.info(
                    f"Discovered A2A agent '{card.name}' at {url} "
                    f"with {len(agent_tools)} tool(s)"
                )
            except Exception as e:
                logger.warning(f"Failed to discover A2A agent at {url}: {e}. Skipping.")

    if all_tools:
        logger.info(f"Registered {len(all_tools)} downstream A2A tool(s) total")

    return all_tools
