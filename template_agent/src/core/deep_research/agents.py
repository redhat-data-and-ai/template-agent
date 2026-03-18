"""Deep research agent factory and context management.

This module provides the ResearchContext factory and lightweight per-worker
agent execution for the deep research pipeline.  MCP tools are loaded
directly (not extracted from a compiled LangGraph agent) so that probe,
plan, and supervisor nodes have full tool awareness.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from template_agent.src.core.deep_research.mode_config import resolve_mode
from template_agent.src.core.deep_research.state import ResearchContext
from template_agent.src.core.deep_research.token_tracker import TokenUsageTracker
from template_agent.src.core.utils import truncate_text as _truncate_text
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


def _get_message_content(msg: Any) -> str:
    """Extract string content from a message object."""
    raw: Any
    if hasattr(msg, "content"):
        raw = msg.content
    elif isinstance(msg, dict):
        raw = msg.get("content", "")
    else:
        return str(msg)

    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(raw) if raw else ""


@asynccontextmanager
async def get_research_context(
    model_name: Optional[str] = None,
    checkpointer: Any = None,
    user_id: Optional[str] = None,
    event_queue: Optional[asyncio.Queue] = None,
    max_subqueries_override: Optional[int] = None,
    max_mode: bool = False,
    token_tracker: Optional[TokenUsageTracker] = None,
) -> AsyncGenerator[ResearchContext, None]:
    """Create a ResearchContext with MCP tools loaded directly.

    Tools are fetched from the MCP server via MultiServerMCPClient so that
    ctx.tools contains real LangChain tool objects.  Each worker creates its
    own lightweight create_react_agent using these tools.

    Args:
        model_name: Model name (e.g., 'gemini-2.5-flash', 'gemini-2.5-pro').
        checkpointer: Optional checkpointer for persistence.
        user_id: User ID for personalized prompts.
        event_queue: Optional queue for streaming events.
        max_subqueries_override: Optional max subqueries override (4-20).
        max_mode: Enable max context length mode for each model.
        token_tracker: Optional token usage tracker.

    Yields:
        ResearchContext with MCP tools and base model.
    """
    logger.info(
        "Initializing research context",
        user_id=user_id,
        model_name=model_name,
        max_subqueries_override=max_subqueries_override,
        max_mode=max_mode,
    )

    resolved_model_name = model_name or "gemini-2.5-flash"

    tools: list[Any] = []
    try:
        server_config: dict[str, Any] = {
            "url": settings.MCP_SERVER_URL,
            "transport": settings.MCP_TRANSPORT_PROTOCOL,
        }
        if not settings.MCP_SSL_VERIFY:
            server_config["verify"] = False
            logger.warning(
                "SSL certificate verification disabled for deep research MCP connection"
            )

        client = MultiServerMCPClient({settings.MCP_SERVER_NAME: server_config})
        tools = await asyncio.wait_for(
            client.get_tools(), timeout=settings.MCP_CONNECTION_TIMEOUT
        )
        logger.info("Deep research: loaded %d MCP tools", len(tools))
    except asyncio.TimeoutError:
        logger.warning(
            "Deep research: MCP connection timed out after %ds",
            settings.MCP_CONNECTION_TIMEOUT,
        )
    except Exception as exc:
        logger.warning("Deep research: MCP tools unavailable -- %s", exc)

    mode = resolve_mode(resolved_model_name, max_mode)

    base_model = ChatGoogleGenerativeAI(
        model=resolved_model_name,
        temperature=0.3,
    )

    llm_semaphore = asyncio.Semaphore(4)
    tracker = token_tracker or TokenUsageTracker(model_name=resolved_model_name)

    ctx = ResearchContext(
        tools=tools,
        base_model=base_model,
        checkpointer=checkpointer,
        user_id=user_id,
        event_queue=event_queue,
        token_tracker=tracker,
        llm_semaphore=llm_semaphore,
        max_subqueries_override=max_subqueries_override,
        _max_mode=max_mode,
        model_name=resolved_model_name,
        mode_config=mode,
    )

    logger.info(
        "Research context initialized (mode=%s)",
        mode.name,
        tool_count=len(tools),
        has_checkpointer=checkpointer is not None,
    )

    yield ctx


async def execute_with_research_agent(
    ctx: ResearchContext,
    query: str,
    thread_id: Optional[str] = None,
    timeout: int = 120,
) -> dict[str, Any]:
    """Execute a query using a lightweight per-worker ReAct agent.

    Creates a fresh create_react_agent for each invocation using the MCP
    tools already loaded in ctx.tools.  This avoids the overhead of
    routing through the full template agent and eliminates per-call MCP
    handshakes.

    Args:
        ctx: Research context with tools and base model.
        query: The query to execute.
        thread_id: Optional thread ID for context.
        timeout: Timeout in seconds for the invocation.

    Returns:
        Dict with 'answer' and 'tool_results' keys.
    """
    max_retries = 2
    input_payload = {"messages": [HumanMessage(content=query)]}
    last_exception: Optional[Exception] = None

    from template_agent.utils.tracing import langfuse_handler

    config: RunnableConfig | None = None
    if langfuse_handler is not None:
        config = RunnableConfig(callbacks=[langfuse_handler])

    for attempt in range(max_retries + 1):
        try:
            worker = create_react_agent(model=ctx.base_model, tools=ctx.tools)
            result = await asyncio.wait_for(
                worker.ainvoke(input_payload, config=config),
                timeout=timeout,
            )
            if isinstance(result, dict):
                answer = extract_answer_from_result(result)
                tool_results = extract_tool_results(result)
                return {"answer": answer, "tool_results": tool_results}
            return {"answer": "", "tool_results": []}
        except asyncio.TimeoutError as e:
            last_exception = e
            if attempt < max_retries:
                wait_time = 2**attempt
                logger.warning(
                    "Research worker timeout (attempt %d/%d), retrying in %ds",
                    attempt + 1,
                    max_retries + 1,
                    wait_time,
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error("Research worker timeout after %d retries", max_retries)
                return {
                    "answer": "",
                    "tool_results": [],
                    "error": f"Query timed out after {timeout}s",
                }
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                wait_time = 2**attempt
                logger.warning(
                    "Research worker error (attempt %d/%d), retrying in %ds: %s",
                    attempt + 1,
                    max_retries + 1,
                    wait_time,
                    str(e)[:100],
                )
                await asyncio.sleep(wait_time)
            else:
                raise

    if last_exception:
        raise last_exception
    return {"answer": "", "tool_results": []}


def extract_answer_from_result(result: dict[str, Any]) -> str:
    """Extract the final answer from a research agent result.

    Args:
        result: The result dict from agent.ainvoke().

    Returns:
        The extracted answer string.
    """
    messages = result.get("messages", [])
    if not messages:
        return ""

    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = _get_message_content(msg)
            if content:
                return content.strip()

    return ""


def extract_tool_results(result: dict[str, Any]) -> list[str]:
    """Extract tool results from a research agent result.

    Args:
        result: The result dict from agent.ainvoke().

    Returns:
        List of formatted tool result strings.
    """
    messages = result.get("messages", [])
    tool_results = []

    for msg in messages:
        if isinstance(msg, ToolMessage):
            name = getattr(msg, "name", "unknown_tool")
            content = _get_message_content(msg)
            if content:
                content = _truncate_text(content, 3000)
                tool_results.append(f"{name}: {content}")

    return tool_results
