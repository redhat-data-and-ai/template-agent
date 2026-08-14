"""State serialization and deserialization for aegra deployment (MR-16).

Converts LangGraph agent state to/from JSON-safe representations for
persistence, API responses, and cross-service communication. Handles
LangChain message objects, tool calls, and nested state structures.
"""

import json
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


def serialize_message(msg: BaseMessage) -> dict[str, Any]:
    """Serialize a single LangChain message to a JSON-safe dict."""
    data: dict[str, Any] = {
        "type": msg.type,
        "content": msg.content,
        "id": getattr(msg, "id", None),
    }

    if isinstance(msg, AIMessage) and msg.tool_calls:
        data["tool_calls"] = [
            {"id": tc.get("id"), "name": tc["name"], "args": tc["args"]}
            for tc in msg.tool_calls
        ]

    if isinstance(msg, ToolMessage):
        data["tool_call_id"] = msg.tool_call_id
        data["name"] = getattr(msg, "name", None)

    if msg.response_metadata:
        data["response_metadata"] = _safe_serialize(msg.response_metadata)

    return data


def deserialize_message(data: dict[str, Any]) -> BaseMessage:
    """Reconstruct a LangChain message from a serialized dict."""
    msg_type = data.get("type", "human")
    content = data.get("content", "")
    msg_id = data.get("id")

    if msg_type == "human":
        return HumanMessage(content=content, id=msg_id)
    elif msg_type == "ai":
        kwargs: dict[str, Any] = {"content": content, "id": msg_id}
        if "tool_calls" in data:
            kwargs["tool_calls"] = data["tool_calls"]
        return AIMessage(**kwargs)
    elif msg_type == "system":
        return SystemMessage(content=content, id=msg_id)
    elif msg_type == "tool":
        return ToolMessage(
            content=content,
            tool_call_id=data.get("tool_call_id", ""),
            name=data.get("name"),
            id=msg_id,
        )
    else:
        return HumanMessage(content=content, id=msg_id)


def serialize_state(state: dict[str, Any]) -> dict[str, Any]:
    """Serialize full LangGraph state to a JSON-safe dict.

    Walks the state dict, converting LangChain messages and any other
    non-serializable objects into JSON-compatible representations.
    """
    result: dict[str, Any] = {}

    for key, value in state.items():
        if key == "messages" and isinstance(value, list):
            result[key] = [
                serialize_message(m)
                if isinstance(m, BaseMessage)
                else _safe_serialize(m)
                for m in value
            ]
        else:
            result[key] = _safe_serialize(value)

    result["_serialized_at"] = datetime.now(UTC).isoformat()
    return result


def deserialize_state(data: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct LangGraph state from a serialized dict."""
    result: dict[str, Any] = {}

    for key, value in data.items():
        if key == "_serialized_at":
            continue
        elif key == "messages" and isinstance(value, list):
            result[key] = [
                deserialize_message(m) if isinstance(m, dict) and "type" in m else m
                for m in value
            ]
        else:
            result[key] = value

    return result


def state_to_json(state: dict[str, Any], indent: int | None = None) -> str:
    """Serialize state to a JSON string."""
    return json.dumps(serialize_state(state), indent=indent, default=str)


def state_from_json(json_str: str) -> dict[str, Any]:
    """Deserialize state from a JSON string."""
    return deserialize_state(json.loads(json_str))


def _safe_serialize(obj: Any) -> Any:
    """Recursively convert an object to a JSON-safe representation."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(item) for item in obj]
    if isinstance(obj, BaseMessage):
        return serialize_message(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return str(obj)
