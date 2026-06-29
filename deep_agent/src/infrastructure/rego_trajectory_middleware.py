"""OPA-backed trajectory policy middleware for deep agents.

This middleware enforces compliance policies at four checkpoints:
1. Before LLM (abefore_model): Validates user messages
2. Before Tool (awrap_tool_call): Validates tool calls
3. After LLM (aafter_model): Validates agent responses
4. After Tool (aafter_tool_call): Validates tool results

Policies are evaluated by an external OPA server using Rego rules.
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
        """Convert message history into structural array for Rego."""
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
        """Query OPA for policy evaluation."""
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

    def _format_denial(self, denial_reasons: list[str]) -> str:
        """Format denial message with reasons."""
        if denial_reasons:
            reasons_text = "\n".join(f"- {reason}" for reason in denial_reasons)
            return f"{POLICY_DENIAL_MESSAGE}\n\nReason(s):\n{reasons_text}"
        return POLICY_DENIAL_MESSAGE

    def _get_last_message_content(self, messages: list[Any], msg_type: str) -> str:
        """Extract content from last message of given type."""
        for msg in reversed(messages):
            if getattr(msg, "type", None) == msg_type:
                return str(getattr(msg, "content", ""))
        return ""

    def _extract_tool_attrs(self, tool_call: Any) -> tuple[str, dict, str]:
        """Extract name, args, and id from tool call."""
        is_dict = isinstance(tool_call, dict)
        return (
            tool_call.get("name") if is_dict else tool_call.name,
            tool_call.get("args", {}) if is_dict else tool_call.args,
            tool_call.get("id") if is_dict else tool_call.id,
        )

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: dict, runtime: Any) -> dict[str, Any] | None:
        """Evaluate trajectory before LLM responds."""
        messages = state.get("messages", [])
        trajectory = self._parse_trajectory(messages)
        intent = {
            "action": "llm_request",
            "user_message": self._get_last_message_content(messages, "human"),
        }

        allowed, reasons = self._evaluate_policy(trajectory, intent)
        if not allowed:
            return {
                "jump_to": "end",
                "messages": [AIMessage(content=self._format_denial(reasons))],
            }
        return None

    async def awrap_tool_call(self, request: Any, handler: Callable) -> Any:
        """Evaluate trajectory before executing tool."""
        trajectory = self._parse_trajectory(request.state.get("messages", []))
        tool_name, tool_args, tool_id = self._extract_tool_attrs(request.tool_call)
        intent = {"action": "tool_call", "name": tool_name, "args": tool_args}

        allowed, reasons = self._evaluate_policy(trajectory, intent)
        if not allowed:
            denial_msg = self._format_denial(reasons)
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=denial_msg,
                            tool_call_id=tool_id,
                            name=tool_name,
                            status="error",
                        ),
                        AIMessage(content=denial_msg),
                    ],
                },
                goto="end",
            )
        return await handler(request)

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        """Intercept model execution to validate response before streaming.

        This runs DURING model execution and buffers the complete response
        before it can be streamed to the UI, allowing us to validate and
        replace denied content.
        """
        from langchain_core.messages import AIMessage as LCCoreAIMessage
        from langchain.agents.middleware.types import ModelResponse

        logger.warning("awrap_model_call invoked - intercepting model execution")

        # Execute the model and get the complete response
        response = await handler(request)

        # Extract the AI message content from the response
        ai_content = ""
        if response.result:
            for msg in response.result:
                if getattr(msg, "type", None) == "ai":
                    ai_content = str(getattr(msg, "content", ""))
                    break

        # Build trajectory from current state
        messages = request.state.get("messages", [])
        trajectory = self._parse_trajectory(messages)

        intent = {
            "action": "llm_response",
            "agent_message": ai_content,
        }

        allowed, reasons = self._evaluate_policy(trajectory, intent)
        if not allowed:
            logger.warning(f"awrap_model_call BLOCKING response, reasons={reasons}")
            # Replace the response with a denial message
            denial_content = self._format_denial(reasons)
            return ModelResponse(
                result=[LCCoreAIMessage(content=denial_content)],
                structured_response=response.structured_response,
            )

        logger.warning("awrap_model_call ALLOWING response")
        return response

    @hook_config(can_jump_to=["end"])
    async def aafter_model(self, state: dict, runtime: Any) -> dict[str, Any] | None:
        """Validate LLM response after generation.

        NOTE: In streaming mode, the response has already been streamed to the client
        by the time this hook runs. This hook ensures the denied message is NOT persisted
        in the conversation state, but cannot prevent the brief flash of banned content
        in real-time streaming UIs.

        For non-streaming scenarios, this effectively blocks the content.
        """
        messages = state.get("messages", [])
        trajectory = self._parse_trajectory(messages)

        # Get the last AI message that was just generated
        last_ai_content = ""
        last_ai_index = -1
        for i in range(len(messages) - 1, -1, -1):
            if getattr(messages[i], "type", None) == "ai":
                last_ai_content = str(getattr(messages[i], "content", ""))
                last_ai_index = i
                break

        if not last_ai_content:
            logger.debug("aafter_model: No AI message found in state")
            return None

        intent = {
            "action": "llm_response",
            "agent_message": last_ai_content,
        }

        allowed, reasons = self._evaluate_policy(trajectory, intent)
        if not allowed:
            logger.warning(f"aafter_model BLOCKING response, reasons={reasons}")
            # Replace the agent's response with denial message
            # Remove all messages after and including the denied AI message
            filtered_messages = messages[:last_ai_index] if last_ai_index > 0 else []
            return {
                "jump_to": "end",
                "messages": filtered_messages + [AIMessage(content=self._format_denial(reasons))],
            }

        logger.debug("aafter_model: Response allowed by policy")
        return None

    async def aafter_tool_call(self, result: Any, request: Any) -> Any:
        """Evaluate trajectory after tool executes.

        This hook runs after subagent/tool execution completes. Like aafter_model,
        it cannot prevent streaming of tool output, but ensures denied content
        is not persisted in the conversation state.
        """
        trajectory = self._parse_trajectory(request.state.get("messages", []))
        tool_name, _, tool_id = self._extract_tool_attrs(request.tool_call)

        tool_content = ""
        if isinstance(result, dict) and "messages" in result:
            for msg in result["messages"]:
                if getattr(msg, "type", None) == "tool":
                    tool_content = str(getattr(msg, "content", ""))
                    break
        elif hasattr(result, "content"):
            tool_content = str(result.content)

        logger.warning(f"aafter_tool_call: tool={tool_name}, content_length={len(tool_content)}, result_type={type(result)}")

        if not tool_content:
            logger.debug("aafter_tool_call: No tool content found")
            return result

        intent = {"action": "tool_response", "name": tool_name, "result": tool_content}

        allowed, reasons = self._evaluate_policy(trajectory, intent)
        if not allowed:
            logger.warning(f"aafter_tool_call BLOCKING tool result from {tool_name}, reasons={reasons}")
            denial_msg = self._format_denial(reasons)
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=denial_msg,
                            tool_call_id=tool_id,
                            name=tool_name,
                            status="error",
                        ),
                        AIMessage(content=denial_msg),
                    ],
                },
                goto="end",
            )

        logger.debug(f"aafter_tool_call: Tool result from {tool_name} allowed by policy")
        return result
