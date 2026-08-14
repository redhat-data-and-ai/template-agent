"""OPA policy evaluation service.

Sends policy inputs to the OPA REST API and returns structured results.

Two entry points:
  evaluate_message()    — evaluates a single agent/tool message
                          (actions: llm_response, tool_response)
  evaluate_trajectory() — evaluates a full message trajectory
                          (action: trajectory_validation)

OPA input shapes (defined by agent.authz policies):

  llm_response:
    {"current_intent": {"action": "llm_response", "agent_message": "<text>"}}

  tool_response:
    {"current_intent": {"action": "tool_response", "result": "<text>"}}

  trajectory_validation:
    {"current_intent": {"action": "trajectory_validation"},
     "trajectory": [{"type": "...", "content": "..."}, ...]}

OPA response shape:
  {"result": {"deny_reasons": ["..."]}}

Allowed is derived from deny_reasons being empty — the allow flag is not read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from langchain_core.messages import BaseMessage

from deep_agent.src.opa.config import get_opa_fail_open, get_opa_timeout, get_opa_url
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)

MessageAction = Literal["llm_response", "tool_response"]


@dataclass
class OpaResult:
    """Result of an OPA policy evaluation."""

    allowed: bool
    denial_reasons: list[str] = field(default_factory=list)


def _parse_result(data: dict[str, Any]) -> OpaResult:
    result = data.get("result")
    if not isinstance(result, dict) or "deny_reasons" not in result:
        logger.warning(
            "OPA response missing expected decision keys (result=%s) -- denying by default",
            type(result).__name__
            if result and not isinstance(result, dict)
            else ("empty" if not result else list(result.keys())),
        )
        return OpaResult(
            allowed=False,
            denial_reasons=["OPA response missing expected decision keys"],
        )
    raw_reasons = result.get("deny_reasons")
    if raw_reasons is None or not isinstance(raw_reasons, list):
        logger.warning(
            "OPA deny_reasons is %s, expected list -- denying by default",
            type(raw_reasons).__name__,
        )
        return OpaResult(
            allowed=False,
            denial_reasons=["OPA deny_reasons missing or invalid type"],
        )
    return OpaResult(allowed=len(raw_reasons) == 0, denial_reasons=raw_reasons)


def _serialize_message(msg: BaseMessage) -> dict[str, str]:
    return {"type": msg.type, "content": str(msg.content)}


async def evaluate_message(
    action: MessageAction,
    *,
    agent_message: str | None = None,
    result: str | None = None,
) -> OpaResult:
    """Evaluate a single agent or tool message against OPA policies.

    Args:
        action: ``"llm_response"`` for LLM output, ``"tool_response"`` for tool output.
        agent_message: The agent's text output. Required when action is ``llm_response``.
        result: The tool result text. Required when action is ``tool_response``.

    Returns:
        OpaResult with allowed flag and any denial reasons.
    """
    current_intent: dict[str, str] = {"action": action}
    if action == "llm_response" and agent_message is not None:
        current_intent["agent_message"] = agent_message
    elif action == "tool_response" and result is not None:
        current_intent["result"] = result

    logger.debug("OPA evaluate_message: action=%s", action)
    opa_result = await _query({"current_intent": current_intent})
    _log_result(action, opa_result)
    return opa_result


async def evaluate_trajectory(trajectory: list[BaseMessage]) -> OpaResult:
    """Evaluate a full message trajectory against OPA policies.

    Args:
        trajectory: Ordered list of LangChain messages representing the conversation.

    Returns:
        OpaResult with allowed flag and any denial reasons.
    """
    logger.debug("OPA evaluate_trajectory: %d message(s)", len(trajectory))
    payload = {
        "current_intent": {"action": "trajectory_validation"},
        "trajectory": [_serialize_message(m) for m in trajectory],
    }
    opa_result = await _query(payload)
    _log_result("trajectory_validation", opa_result)
    return opa_result


def _log_result(action: str, opa_result: OpaResult) -> None:
    if opa_result.allowed:
        logger.debug("OPA allowed: action=%s", action)
    else:
        logger.info(
            "OPA denied: action=%s reasons=%s",
            action,
            opa_result.denial_reasons,
        )


def _error_result(reason: str, fail_open: bool) -> OpaResult:
    """Return an OpaResult for an OPA communication error.

    When *fail_open* is True the request is allowed (backwards-compatible).
    When False the request is denied and an error-level log is emitted.
    """
    disposition = "allowed" if fail_open else "denied"
    message = f"{reason} -- {disposition} by default"
    if fail_open:
        logger.warning("%s", message)
    else:
        logger.error("%s", message)
    return OpaResult(allowed=fail_open, denial_reasons=[message])


async def _query(opa_input: dict[str, Any]) -> OpaResult:
    url = get_opa_url()
    timeout = get_opa_timeout()
    fail_open = get_opa_fail_open()

    logger.debug(
        "OPA request: POST %s action=%s payload_keys=%s",
        url,
        opa_input.get("current_intent", {}).get("action"),
        list(opa_input.keys()),
    )
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json={"input": opa_input})
            response.raise_for_status()
            data = response.json()
            logger.debug("OPA response: status=%d", response.status_code)
            return _parse_result(data)
    except httpx.TimeoutException:
        return _error_result(f"OPA request timed out after {timeout:.1f}s", fail_open)
    except httpx.HTTPStatusError as exc:
        return _error_result(f"OPA returned HTTP {exc.response.status_code}", fail_open)
    except Exception as exc:
        return _error_result(f"OPA unreachable ({exc})", fail_open)


def _query_sync(opa_input: dict[str, Any]) -> OpaResult:
    url = get_opa_url()
    timeout = get_opa_timeout()
    fail_open = get_opa_fail_open()

    logger.debug(
        "OPA request (sync): POST %s action=%s payload_keys=%s",
        url,
        opa_input.get("current_intent", {}).get("action"),
        list(opa_input.keys()),
    )
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json={"input": opa_input})
            response.raise_for_status()
            data = response.json()
            logger.debug("OPA response (sync): status=%d", response.status_code)
            return _parse_result(data)
    except httpx.TimeoutException:
        return _error_result(f"OPA request timed out after {timeout:.1f}s", fail_open)
    except httpx.HTTPStatusError as exc:
        return _error_result(f"OPA returned HTTP {exc.response.status_code}", fail_open)
    except Exception as exc:
        return _error_result(f"OPA unreachable ({exc})", fail_open)


def evaluate_message_sync(
    action: MessageAction,
    *,
    agent_message: str | None = None,
    result: str | None = None,
) -> OpaResult:
    """Synchronous variant of evaluate_message for use in non-async contexts."""
    current_intent: dict[str, str] = {"action": action}
    if action == "llm_response" and agent_message is not None:
        current_intent["agent_message"] = agent_message
    elif action == "tool_response" and result is not None:
        current_intent["result"] = result

    logger.debug("OPA evaluate_message_sync: action=%s", action)
    opa_result = _query_sync({"current_intent": current_intent})
    _log_result(action, opa_result)
    return opa_result
