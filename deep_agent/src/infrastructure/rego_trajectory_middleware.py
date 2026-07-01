"""OPA-backed trajectory policy middleware for deep agents.

This middleware enforces compliance policies at three checkpoints:
1. Before LLM (abefore_model): Validates entire conversation trajectory
2. After LLM (aafter_model): Validates agent responses
3. After Tool (aafter_tool_call): Validates tool results

Policies are evaluated by an external OPA server using Rego rules.
"""

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
        retry_keywords: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.opa_url = opa_url
        self.timeout = timeout
        self.enable_retry = enable_retry
        self.retry_keywords = retry_keywords or ["yes", "retry", "/retry"]

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

    def _format_retry_prompt(
        self,
        denial_reasons: list[str],
        *,
        call_input: str | dict[str, Any] | None = None,
    ) -> str:
        """Format interrupt message with the blocked call input and denial reasons."""
        reasons_text = "\n".join(f"- {reason}" for reason in denial_reasons)
        if call_input is None:
            input_section = ""
        elif isinstance(call_input, dict):
            input_section = f"\n\Retry :\n{json.dumps(call_input, indent=2)}"
        else:
            input_section = f"\n\nRetry the prompt :\n{call_input}"
        return f"{POLICY_DENIAL_MESSAGE}\n\nReason(s):\n{reasons_text}\n{input_section}"

    def _build_policy_context_prompt(self, violation_context: dict) -> str:
        """Build system prompt addition with policy violation context."""
        reasons = violation_context.get("denial_reasons", [])
        reasons_text = "\n".join(f"- {reason}" for reason in reasons)
        return f"""
IMPORTANT POLICY CONTEXT FOR THIS RESPONSE:
Your previous response violated compliance policies for the following reasons:
{reasons_text}

Please regenerate your response while strictly avoiding these policy violations.
Focus on providing helpful information within policy boundaries."""

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
        """Evaluate entire trajectory before LLM responds.

        This hook validates the complete conversation history to ensure
        the trajectory as a whole complies with policy rules before
        allowing the LLM to generate a response.

        Additionally, this hook detects retry confirmations and injects
        policy violation context into the system prompt for retry attempts.
        """
        messages = state.get("messages", [])
        trajectory = self._parse_trajectory(messages)

        logger.warning(f"abefore_model: evaluating trajectory with {len(trajectory)} steps")

        # Check if this is a retry attempt
        violation_context = state.get("policy_violation_context", {})
        if violation_context.get("retry_available") and messages:
            # Check if the last user message is a retry confirmation
            last_user_message = None
            for msg in reversed(messages):
                if getattr(msg, "type", None) == "human":
                    last_user_message = str(getattr(msg, "content", "")).strip().lower()
                    break

            # Any user message after violation triggers retry (no keywords needed)
            if last_user_message:
                logger.info(f"Detected user message after policy violation - triggering retry")
                # Inject policy context as a system message and mark retry as used
                policy_prompt = self._build_policy_context_prompt(violation_context)

                # Insert policy context as a SystemMessage before LLM processes
                # This provides the context without modifying base system prompt
                policy_message = SystemMessage(content=policy_prompt)

                return {
                    "messages": [policy_message],
                    "policy_violation_context": {
                        **violation_context,
                        "retry_available": False,
                    },
                }

        # Evaluate the entire trajectory
        intent = {
            "action": "trajectory_validation",
            "trajectory_length": len(trajectory),
        }

        allowed, reasons = self._evaluate_policy(trajectory, intent)
        if not allowed:
            logger.warning(f"abefore_model BLOCKING trajectory, reasons={reasons}")
            return {
                "jump_to": "end",
                "messages": [AIMessage(content=self._format_denial(reasons))],
            }

        logger.warning("abefore_model ALLOWING trajectory to proceed")
        return None

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
            # Remove all messages after and including the denied AI message
            filtered_messages = messages[:last_ai_index] if last_ai_index > 0 else []

            # Check if retry is enabled and this is the first violation (not a retry)
            violation_context = state.get("policy_violation_context", {})
            logger.warning(f"aafter_model violation_context: {violation_context}")
            retry_available = violation_context.get("retry_available", True)
            logger.warning(f"aafter_model retry_available: {retry_available}")

            if self.enable_retry and retry_available:
                # Offer retry with HITL confirmation
                logger.info("Offering HITL retry for policy violation")
                return {
                    "jump_to": "end",
                    "messages": filtered_messages + [
                        AIMessage(
                            content=self._format_retry_prompt(
                                reasons, call_input=last_ai_content
                            )
                        )
                    ],
                    "policy_violation_context": {
                        "checkpoint": "aafter_model",
                        "denial_reasons": reasons,
                        "retry_available": True,
                        "violated_at": last_ai_index,
                        "blocked_input": last_ai_content,
                    },
                }
            else:
                # No retry available - final denial
                logger.warning("No retry available - returning final denial")
                return {
                    "jump_to": "end",
                    "messages": filtered_messages + [AIMessage(content=self._format_denial(reasons))],
                    "policy_violation_context": {
                        "checkpoint": "aafter_model",
                        "denial_reasons": reasons,
                        "retry_available": False,
                        "violated_at": last_ai_index,
                    },
                }

        logger.debug("aafter_model: Response allowed by policy")
        return None

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        """Wrap tool execution to evaluate results against policy.

        This wraps the tool call handler to intercept and validate tool results.
        Like aafter_model, it cannot prevent streaming of tool output, but ensures
        denied content is not persisted in the conversation state.

        Args:
            request: The tool call request containing state and tool_call info.
            handler: The actual tool execution handler to call.

        Returns:
            Tool result or Command with denial/retry prompt.
        """
        # Call the actual tool handler first
        result = await handler(request)

        # Now validate the result
        trajectory = self._parse_trajectory(request.state.get("messages", []))
        tool_name, tool_args, tool_id = self._extract_tool_attrs(request.tool_call)

        # Extract tool content from various result types
        tool_content = ""

        # Handle Command objects (what subagents return)
        if isinstance(result, Command):
            if hasattr(result, "update") and isinstance(result.update, dict):
                messages = result.update.get("messages", [])
                for msg in messages:
                    # Get all AI message content from subagent results
                    if getattr(msg, "type", None) == "ai":
                        content = getattr(msg, "content", "")
                        if content:
                            tool_content += str(content) + "\n"
                    elif getattr(msg, "type", None) == "tool":
                        content = getattr(msg, "content", "")
                        if content:
                            tool_content += str(content) + "\n"
                tool_content = tool_content.strip()
        # Handle dict results
        elif isinstance(result, dict) and "messages" in result:
            for msg in result["messages"]:
                if getattr(msg, "type", None) in ("tool", "ai"):
                    content = getattr(msg, "content", "")
                    if content:
                        tool_content += str(content) + "\n"
            tool_content = tool_content.strip()
        # Handle ToolMessage results
        elif hasattr(result, "content"):
            tool_content = str(result.content)

        logger.warning(f"awrap_tool_call: tool={tool_name}, content_length={len(tool_content)}, result_type={type(result)}")

        if not tool_content:
            logger.debug("awrap_tool_call: No tool content found")
            return result

        intent = {"action": "tool_response", "name": tool_name, "result": tool_content}

        allowed, reasons = self._evaluate_policy(trajectory, intent)
        if not allowed:
            logger.warning(f"awrap_tool_call BLOCKING tool result from {tool_name}, reasons={reasons}")

            # Check if retry is enabled and this is the first violation (not a retry)
            violation_context = request.state.get("policy_violation_context", {})
            logger.warning(f"awrap_tool_call violation_context: {violation_context}")
            retry_available = violation_context.get("retry_available", True)
            logger.warning(f"awrap_tool_call retry_available: {retry_available}")
            logger.warning(f"awrap_tool_call enable_retry: {self.enable_retry}")

            if self.enable_retry and retry_available:
                # Offer retry with HITL confirmation
                logger.warning(f"Offering HITL retry for tool policy violation: {tool_name}")
                retry_prompt = self._format_retry_prompt(
                    reasons,
                    call_input=f"Call {tool_name} tool with arguments: {tool_args}",
                )
                logger.warning(f"awrap_tool_call retry_prompt: {retry_prompt}")
                return Command(
                    update={
                        "messages": [
                            ToolMessage(
                                content=retry_prompt,
                                tool_call_id=tool_id,
                                name=tool_name,
                                status="error",
                            ),
                            AIMessage(content=retry_prompt),
                        ],
                        "policy_violation_context": {
                            "checkpoint": "awrap_tool_call",
                            "denial_reasons": reasons,
                            "retry_available": True,
                            "tool_name": tool_name,
                            "blocked_input": {"name": tool_name, "args": tool_args},
                        },
                    },
                    goto="end",
                )
            else:
                # No retry available - final denial
                logger.warning(f"No retry available for tool {tool_name} - returning final denial")
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
                        "policy_violation_context": {
                            "checkpoint": "awrap_tool_call",
                            "denial_reasons": reasons,
                            "retry_available": False,
                            "tool_name": tool_name,
                        },
                    },
                    goto="end",
                )

        logger.debug(f"awrap_tool_call: Tool result from {tool_name} allowed by policy")
        return result
