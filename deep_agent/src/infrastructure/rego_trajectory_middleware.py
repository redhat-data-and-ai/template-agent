"""OPA-backed trajectory policy middleware for deep agents.

Evaluates the agent's message trajectory against Rego policies served by
Open Policy Agent before each LLM call and tool invocation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
from langchain.agents.middleware import AgentMiddleware

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


class RegoTrajectoryMiddleware(AgentMiddleware):
    """Enforce trajectory rules via OPA before model and tool execution."""

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

    def _evaluate_policy(self, trajectory: list[dict[str, Any]], current_intent: dict) -> bool:
        """Query OPA; fail closed on errors or missing allow result."""
        payload = {
            "input": {
                "trajectory": trajectory,
                "current_intent": current_intent,
            }
        }
        try:
            response = httpx.post(
                self.opa_url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            allowed = response.json().get("result", {}).get("allow", False)
            if not allowed:
                logger.warning(
                    "OPA denied intent=%s trajectory_steps=%d",
                    current_intent.get("action"),
                    len(trajectory),
                )
            return bool(allowed)
        except Exception as exc:
            logger.error("OPA policy check failed (fail-closed): %s", exc)
            return False

    def before_model(self, state: dict, runtime: Any) -> dict[str, Any] | None:
        """Evaluate trajectory before the LLM is allowed to respond."""
        trajectory = self._parse_trajectory(state.get("messages", []))
        if not self._evaluate_policy(trajectory, {"action": "llm_request"}):
            raise PermissionError(
                "Policy Violation: Trajectory limits breached on LLM request."
            )
        return None

    def wrap_tool_call(self, request: Any, handler: Callable) -> Any:
        """Evaluate trajectory before executing a tool."""
        trajectory = self._parse_trajectory(request.state.get("messages", []))
        tool_call = request.tool_call
        tool_name = tool_call.get("name") if isinstance(tool_call, dict) else tool_call.name
        tool_args = tool_call.get("args", {}) if isinstance(tool_call, dict) else tool_call.args
        current_intent = {
            "action": "tool_call",
            "name": tool_name,
            "args": tool_args,
        }

        if not self._evaluate_policy(trajectory, current_intent):
            raise PermissionError(
                f"Policy Violation: Tool '{tool_name}' blocked by trajectory rules."
            )

        return handler(request)


def create_rego_trajectory_middleware() -> RegoTrajectoryMiddleware:
    """Factory for dynamic middleware import paths."""
    from deep_agent.src.settings import settings

    return RegoTrajectoryMiddleware(
        opa_url=settings.OPA_URL,
        timeout=settings.OPA_TIMEOUT,
    )
