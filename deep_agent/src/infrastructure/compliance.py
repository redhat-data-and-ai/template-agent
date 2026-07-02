"""OPA-backed compliance policy middleware."""

from __future__ import annotations

import json
from typing import Any

import httpx
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import hook_config
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.types import Command

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

POLICY_DENIAL_MESSAGE = (
    "This output is blocked by our compliance policies."
)


class RegoTrajectoryMiddleware(AgentMiddleware):
    """Enforce trajectory rules via OPA."""

    name = "rego_trajectory_policy"

    def __init__(
        self,
        opa_url: str = "http://localhost:8181/v1/data/agent/authz",
        *,
        timeout: float = 2.0,
        enable_retry: bool = True,
    ) -> None:
        super().__init__()
        self.opa_url = opa_url
        self.timeout = timeout
        self.enable_retry = enable_retry

    def _parse_trajectory(self, messages: list[Any]) -> list[dict[str, Any]]:
        trajectory: list[dict[str, Any]] = []
        for msg in messages:
            msg_type = getattr(msg, "type", None)
            content = getattr(msg, "content", "")

            if msg_type == "human" and content:
                trajectory.append({"type": "user_message", "content": str(content)})
            elif msg_type == "ai" and getattr(msg, "tool_calls", None):
                trajectory.append({
                    "type": "agent_action",
                    "tools": [
                        {"name": tc["name"], "args": tc.get("args", {})}
                        for tc in msg.tool_calls
                    ],
                })
            elif msg_type == "ai" and content:
                trajectory.append({"type": "agent_response", "content": str(content)})
            elif msg_type == "tool":
                trajectory.append({
                    "type": "tool_response",
                    "name": getattr(msg, "name", None),
                    "status": "completed",
                })

        return trajectory

    def _evaluate_policy(
        self, trajectory: list[dict[str, Any]], current_intent: dict
    ) -> tuple[bool, list[str]]:
        try:
            response = httpx.post(
                self.opa_url,
                json={"input": {"trajectory": trajectory, "current_intent": current_intent}},
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json().get("result", {})
            allowed = result.get("allow", False)
            denial_reasons = result.get("deny_reasons", [])

            if not allowed:
                logger.warning(
                    f"OPA denied: {current_intent.get('action')}, reasons={denial_reasons}"
                )

            return bool(allowed), denial_reasons
        except Exception as exc:
            logger.error(f"OPA check failed (fail-closed): {exc}")
            return False, ["Policy evaluation failed due to technical error"]

    def _format_retry_prompt(
        self,
        denial_reasons: list[str],
        *,
        call_input: str | dict[str, Any] | None = None,
    ) -> str:
        reasons_text = "\n".join(f"- {reason}" for reason in denial_reasons)
        if call_input is None:
            input_section = ""
        elif isinstance(call_input, dict):
            input_section = f"\n\Retry :\n{json.dumps(call_input, indent=2)}"
        else:
            input_section = f"\n\nRetry the prompt :\n{call_input}"
        return f"{POLICY_DENIAL_MESSAGE}\n\nReason(s):\n{reasons_text}\n{input_section}"

    def _format_denial(self, denial_reasons: list[str]) -> str:
        if not denial_reasons:
            return POLICY_DENIAL_MESSAGE
        reasons_text = "\n".join(f"- {reason}" for reason in denial_reasons)
        return f"{POLICY_DENIAL_MESSAGE}\n\nReason(s):\n{reasons_text}"

    @staticmethod
    def _extract_tool_content(result: Any) -> str:
        if isinstance(result, Command) and isinstance(getattr(result, "update", None), dict):
            messages = result.update.get("messages", [])
        elif isinstance(result, dict):
            messages = result.get("messages", [])
        elif hasattr(result, "content"):
            return str(result.content)
        else:
            return ""

        parts: list[str] = []
        for msg in messages:
            if getattr(msg, "type", None) in ("tool", "ai"):
                content = getattr(msg, "content", "")
                if content:
                    parts.append(str(content))
        return "\n".join(parts)

    @staticmethod
    def _latest_human_content(messages: list[Any]) -> str | None:
        for msg in reversed(messages):
            if getattr(msg, "type", None) == "human":
                content = str(getattr(msg, "content", "")).strip()
                return content or None
        return None

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: dict, runtime: Any) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        violation_context = state.get("policy_violation_context", {})

        if violation_context.get("retry_available") and self._latest_human_content(messages):
            reasons = violation_context.get("denial_reasons", [])
            reasons_text = "\n".join(f"- {reason}" for reason in reasons)
            return {
                "messages": [SystemMessage(content=f"""
IMPORTANT POLICY CONTEXT FOR THIS RESPONSE:
Your previous response violated compliance policies for the following reasons:
{reasons_text}

Please regenerate your response while strictly avoiding these policy violations.
Focus on providing helpful information within policy boundaries.""")],
                "policy_violation_context": {
                    **violation_context,
                    "retry_available": False,
                },
            }

        trajectory = self._parse_trajectory(messages)
        allowed, reasons = self._evaluate_policy(
            trajectory,
            {"action": "trajectory_validation", "trajectory_length": len(trajectory)},
        )
        if not allowed:
            return {
                "jump_to": "end",
                "messages": [AIMessage(content=self._format_denial(reasons))],
                "policy_violation_context": {
                    "checkpoint": "abefore_model",
                    "denial_reasons": reasons,
                    "retry_available": False,
                },
            }
        return None

    @hook_config(can_jump_to=["end"])
    async def aafter_model(self, state: dict, runtime: Any) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        last_ai_content = ""
        last_ai_index = -1
        for i in range(len(messages) - 1, -1, -1):
            if getattr(messages[i], "type", None) == "ai":
                last_ai_content = str(messages[i].content)
                last_ai_index = i
                break

        if not last_ai_content:
            return None

        allowed, reasons = self._evaluate_policy(
            self._parse_trajectory(messages),
            {"action": "llm_response", "agent_message": last_ai_content},
        )
        if allowed:
            return None

        violation_context = state.get("policy_violation_context", {})
        offer_retry = self.enable_retry and violation_context.get("retry_available", True)
        filtered_messages = messages[:last_ai_index] if last_ai_index > 0 else []

        return {
            "jump_to": "end",
            "messages": filtered_messages + [
                AIMessage(
                    content=self._format_retry_prompt(
                        reasons,
                        call_input=last_ai_content if offer_retry else None,
                    )
                )
            ],
            "policy_violation_context": {
                "checkpoint": "aafter_model",
                "denial_reasons": reasons,
                "retry_available": offer_retry,
                "violated_at": last_ai_index,
                **({"blocked_input": last_ai_content} if offer_retry else {}),
            },
        }

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        result = await handler(request)

        tool_call = request.tool_call
        is_dict = isinstance(tool_call, dict)
        tool_name = tool_call.get("name") if is_dict else tool_call.name
        tool_args = tool_call.get("args", {}) if is_dict else tool_call.args
        tool_id = tool_call.get("id") if is_dict else tool_call.id

        tool_content = self._extract_tool_content(result)
        if not tool_content:
            return result

        allowed, reasons = self._evaluate_policy(
            self._parse_trajectory(request.state.get("messages", [])),
            {"action": "tool_response", "name": tool_name, "result": tool_content},
        )
        if allowed:
            return result

        violation_context = request.state.get("policy_violation_context", {})
        offer_retry = self.enable_retry and violation_context.get("retry_available", True)
        content = self._format_retry_prompt(
            reasons,
            call_input=f"Call {tool_name} tool with arguments: {tool_args}" if offer_retry else None,
        )

        violation_update: dict[str, Any] = {
            "checkpoint": "awrap_tool_call",
            "denial_reasons": reasons,
            "retry_available": offer_retry,
            "tool_name": tool_name,
        }
        if offer_retry:
            violation_update["blocked_input"] = {"name": tool_name, "args": tool_args}

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=content,
                        tool_call_id=tool_id,
                        name=tool_name,
                        status="error",
                    ),
                    AIMessage(content=content),
                ],
                "policy_violation_context": violation_update,
            },
            goto="end",
        )
