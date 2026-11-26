"""State conversion utilities for Aegra integration.

Bridges the template-agent's StreamRequest/response format with
Aegra's standard state format. When the agent runs via
`aegra dev` / `aegra serve`, messages arrive in LangGraph's native
format. These helpers convert between the two worlds.
"""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def stream_request_to_langgraph_input(message: str) -> dict[str, Any]:
    """Convert a raw user message into LangGraph-compatible input.

    Args:
        message: The user's text message.

    Returns:
        Dict with ``messages`` key containing a HumanMessage list.
    """
    return {"messages": [HumanMessage(content=message)]}


def langgraph_messages_to_dicts(messages: list) -> list[dict[str, Any]]:
    """Convert LangChain message objects to plain dicts for serialization.

    Useful for logging, testing, and API responses that need JSON-safe
    representations of conversation history.
    """
    result = []
    for msg in messages:
        entry: dict[str, Any] = {"content": msg.content}
        if isinstance(msg, HumanMessage):
            entry["role"] = "human"
        elif isinstance(msg, AIMessage):
            entry["role"] = "ai"
        elif isinstance(msg, SystemMessage):
            entry["role"] = "system"
        else:
            entry["role"] = "unknown"

        if hasattr(msg, "tool_calls") and msg.tool_calls:
            entry["tool_calls"] = [
                {"name": tc["name"], "args": tc["args"]} for tc in msg.tool_calls
            ]
        result.append(entry)
    return result


def _answer_text(content: Any) -> str:
    """Return the visible answer, skipping thought blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        if block.get("type") in {"thinking", "reasoning", "redacted_thinking"}:
            continue
        if block.get("thought") is True:
            continue
        text = block.get("text")
        if block.get("type") == "text" and isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def extract_final_response(state: dict[str, Any]) -> str | None:
    """Extract the final AI response text from a LangGraph state snapshot.

    Returns the content of the last AIMessage, or None if the conversation
    has no AI responses yet. Thought blocks are skipped when content is a list.
    """
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        text = _answer_text(msg.content)
        if text:
            return text
    return None
