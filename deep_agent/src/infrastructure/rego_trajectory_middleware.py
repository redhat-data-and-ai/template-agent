"""OPA-backed trajectory policy middleware for deep agents.

Evaluates the agent's message trajectory against Rego policies served by
Open Policy Agent before each LLM call and tool invocation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import hook_config
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

POLICY_DENIAL_MESSAGE = (
    "I'm unable to complete this request because it isn't allowed by our "
    "compliance policies. Please try a different approach or contact your "
    "administrator if you need access."
)


class RegoTrajectoryMiddleware(AgentMiddleware):
    """Enforce trajectory rules via OPA."""

    name = "rego_trajectory_policy"

    def __init__(
        self,
        opa_url: str = "http://localhost:8181/v1/data/agent/authz",
        *,
        timeout: float = 2.0,
    ) -> None:
        super().__init__()
        self.opa_url = opa_url
        self.timeout = timeout

    def _parse_trajectory(self, messages: list[Any]) -> list[dict[str, Any]]:
        """Convert message history into a structural array for Rego."""
        trajectory: list[dict[str, Any]] = []
        for msg in messages:
            msg_type = getattr(msg, "type", None)
            if msg_type == "ai" and getattr(msg, "tool_calls", None):
                trajectory.append(
                    {
                        "type": "agent_action",
                        "tools": [
                            {"name": tc["name"], "args": tc.get("args", {})}
                            for tc in msg.tool_calls
                        ],
                    }
                )
            elif msg_type == "tool":
                trajectory.append(
                    {
                        "type": "tool_response",
                        "name": getattr(msg, "name", None),
                        "status": "completed",
                    }
                )
        return trajectory

    def _evaluate_policy(
        self,
        trajectory: list[dict[str, Any]],
        current_intent: dict,
    ) -> bool:
        """Query OPA for policy evaluation.

        Args:
            trajectory: Agent's message history
            current_intent: Current action being evaluated

        Returns:
            True if allowed, False if denied
        """
        payload = {
            "input": {
                "trajectory": trajectory,
                "current_intent": current_intent,
            }
        }

        logger.debug(
            f"OPA eval: intent={current_intent.get('action')}, "
            f"trajectory_len={len(trajectory)}"
        )

        try:
            response = httpx.post(
                self.opa_url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json().get("result", {})
            allowed = result.get("allow", False)

            if not allowed:
                # Log denial reasons if available
                denial_reasons = result.get("denial_reasons", [])
                logger.warning(
                    f"OPA denied: intent={current_intent.get('action')}, "
                    f"trajectory_steps={len(trajectory)}, "
                    f"reasons={denial_reasons}"
                )

            return bool(allowed)

        except Exception as exc:
            logger.error(f"OPA policy check failed (fail-closed): {exc}")
            return False

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: dict, runtime: Any) -> dict[str, Any] | None:
        """Evaluate trajectory before the LLM is allowed to respond."""
        trajectory = self._parse_trajectory(state.get("messages", []))

        # Evaluate policy
        if not self._evaluate_policy(trajectory, {"action": "llm_request"}):
            logger.warning("Policy denied LLM request")
            return {
                "jump_to": "end",
                "messages": [AIMessage(content=POLICY_DENIAL_MESSAGE)],
            }

        logger.debug("Policy allowed LLM request")
        return None

    async def awrap_tool_call(self, request: Any, handler: Callable) -> Any:
        """Evaluate trajectory before executing a tool."""
        trajectory = self._parse_trajectory(request.state.get("messages", []))
        tool_call = request.tool_call
        tool_name = tool_call.get("name") if isinstance(tool_call, dict) else tool_call.name
        tool_args = tool_call.get("args", {}) if isinstance(tool_call, dict) else tool_call.args
        tool_call_id = tool_call.get("id") if isinstance(tool_call, dict) else tool_call.id

        current_intent = {
            "action": "tool_call",
            "name": tool_name,
            "args": tool_args,
        }

        # Evaluate policy
        if not self._evaluate_policy(trajectory, current_intent):
            logger.warning(f"Policy denied tool call {tool_name}")
            return Command[str](
                update={
                    "messages": [
                        ToolMessage(
                            content=POLICY_DENIAL_MESSAGE,
                            tool_call_id=tool_call_id,
                            name=tool_name,
                            status="error",
                        ),
                        AIMessage(content=POLICY_DENIAL_MESSAGE),
                    ],
                },
                goto="end",
            )

        logger.debug(f"Policy allowed tool call {tool_name}")
        return await handler(request)
