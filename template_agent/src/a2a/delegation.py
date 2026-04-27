"""A2A delegation tool – LangChain StructuredTool for calling downstream agents."""

from __future__ import annotations

import uuid

import httpx
import structlog
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from template_agent.src.a2a.context import a2a_request_ctx
from template_agent.src.a2a.registry import get_registry
from template_agent.src.settings import settings

logger = structlog.get_logger(__name__)


class DelegateInput(BaseModel):
    """Input schema for the delegation tool."""

    agent_id: str = Field(description="ID of the downstream agent to delegate to")
    message: str = Field(
        description="The query or instruction to send to the downstream agent"
    )


async def delegate_to_a2a_agent(
    agent_id: str,
    message: str,
    *,
    access_token: str | None = None,
    user_id: str = "a2a",
    thread_id: str | None = None,
) -> str:
    """Send an A2A ``message/send`` JSON-RPC request to a downstream agent.

    Authentication and correlation are forwarded via HTTP headers (not metadata)
    per the A2A enterprise spec.
    """
    registry = get_registry()
    agent = registry.get(agent_id)
    if not agent:
        return f"Error: agent '{agent_id}' is not registered in the A2A registry."

    ctx = a2a_request_ctx.get()
    token = access_token or ctx.access_token

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    headers["X-Calling-Agent-ID"] = settings.A2A_AGENT_ID
    if ctx.correlation_id:
        headers["X-Correlation-ID"] = ctx.correlation_id

    task_id = str(uuid.uuid4())
    message_id = str(uuid.uuid4())
    payload = {
        "jsonrpc": "2.0",
        "id": task_id,
        "method": "message/send",
        "params": {
            "message": {
                "messageId": message_id,
                "role": "user",
                "parts": [{"kind": "text", "text": message}],
                "metadata": {
                    "user_id": user_id,
                    **({"thread_id": thread_id} if thread_id else {}),
                },
            },
        },
    }

    target_url = agent.base_url.rstrip("/") + "/a2a/"
    try:
        async with httpx.AsyncClient(timeout=settings.A2A_REQUEST_TIMEOUT) as client:
            resp = await client.post(target_url, json=payload, headers=headers)
            resp.raise_for_status()
            body = resp.json()
    except httpx.HTTPStatusError as exc:
        logger.error(
            "a2a_delegation_http_error",
            agent_id=agent_id,
            status=exc.response.status_code,
        )
        return f"Error: downstream agent '{agent_id}' returned HTTP {exc.response.status_code}."
    except Exception as exc:
        logger.error("a2a_delegation_failed", agent_id=agent_id, error=str(exc))
        return f"Error: failed to reach downstream agent '{agent_id}': {exc}"

    logger.debug("a2a_delegation_response", agent_id=agent_id, body=body)

    result = body.get("result", {})
    parts_text: list[str] = []

    result_kind = result.get("kind")
    if result_kind == "message":
        for part in result.get("parts", []):
            if part.get("kind") == "text" and part.get("text"):
                parts_text.append(part["text"])
    else:
        for artifact in result.get("artifacts", []):
            for part in artifact.get("parts", []):
                if part.get("kind") == "text" and part.get("text"):
                    parts_text.append(part["text"])

    if parts_text:
        return "\n".join(parts_text)

    status = result.get("status", {})
    if status.get("state") == "failed":
        err_msg = status.get("message", {})
        return f"Downstream agent '{agent_id}' failed: {err_msg}"

    return f"Downstream agent '{agent_id}' returned no text artifacts."


def build_a2a_delegation_tool(
    access_token: str | None = None,
    user_id: str = "a2a",
    thread_id: str | None = None,
) -> StructuredTool | None:
    """Build the delegation StructuredTool if the registry has agents.

    Returns ``None`` when no downstream agents are configured.
    """
    registry = get_registry()
    agents = registry.list_agents()
    if not agents:
        return None

    lines = ["Delegate a task to a downstream A2A agent. Available agents:"]
    for a in agents:
        skills_str = ", ".join(a.skills) if a.skills else "general"
        desc = a.description or "(no description)"
        lines.append(f"  - {a.agent_id}: {desc} [skills: {skills_str}]")
    tool_description = "\n".join(lines)

    async def _run(agent_id: str, message: str) -> str:
        return await delegate_to_a2a_agent(
            agent_id,
            message,
            access_token=access_token,
            user_id=user_id,
            thread_id=thread_id,
        )

    return StructuredTool.from_function(
        coroutine=_run,
        name="delegate_to_a2a_agent",
        description=tool_description,
        args_schema=DelegateInput,
    )
