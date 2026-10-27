"""Parse SSE lines from the local agent and from AI Factory."""

from __future__ import annotations

import json
from typing import Any


def parse_sse_lines(lines: list[str]) -> list[tuple[str, Any]]:
    """Parse ``event:`` / ``data:`` lines into ``(event_type, payload)`` pairs."""
    events: list[tuple[str, Any]] = []
    current_event = "message"
    for line in lines:
        text = line.rstrip("\n")
        if text.startswith("event:"):
            current_event = text[len("event:") :].strip() or "message"
        elif text.startswith("data:"):
            raw = text[len("data:") :].strip()
            try:
                events.append((current_event, json.loads(raw)))
            except json.JSONDecodeError:
                pass
            current_event = "message"
    return events


def _unwrap_update(data: Any) -> dict[str, Any]:
    """Return the update dict from a top-level or subgraph stream event."""
    if isinstance(data, list) and len(data) == 2 and isinstance(data[1], dict):
        return data[1]
    if isinstance(data, dict):
        return data
    return {}


def thread_id_from_events(
    events: list[tuple[str, Any]],
    header_thread_id: str | None = None,
) -> str | None:
    """Return the thread id from the response header or a metadata event."""
    if header_thread_id:
        return header_thread_id
    for _event_type, data in events:
        if isinstance(data, dict) and data.get("thread_id"):
            return str(data["thread_id"])
    return None


def end_status(events: list[tuple[str, Any]]) -> str | None:
    """Return the last SSE end status, when the stream sent one."""
    status: str | None = None
    for event_type, data in events:
        if (
            event_type == "end"
            and isinstance(data, dict)
            and isinstance(data.get("status"), str)
        ):
            status = data["status"]
    return status


def interrupts_from_events(events: list[tuple[str, Any]]) -> list[Any]:
    """Collect ``__interrupt__`` payloads from a stream."""
    found: list[Any] = []
    for _event_type, data in events:
        update = _unwrap_update(data)
        raw = update.get("__interrupt__")
        if raw is None:
            continue
        found.extend(raw if isinstance(raw, list) else [raw])
    return found


def _text_from_content(content: Any) -> str:
    """Flatten a model message content value to text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for chunk in content:
            if isinstance(chunk, dict) and chunk.get("type") == "text":
                text = str(chunk.get("text", "")).strip()
                if text:
                    parts.append(text)
            elif isinstance(chunk, str) and chunk.strip():
                parts.append(chunk.strip())
        return "\n".join(parts)
    return ""


def message_texts(events: list[tuple[str, Any]]) -> list[str]:
    """Return assistant text blocks from local updates and AI Factory messages."""
    texts: list[str] = []
    for _event_type, data in events:
        update = _unwrap_update(data)
        blobs: list[Any] = []
        for value in update.values():
            if isinstance(value, dict):
                messages = value.get("messages")
                if isinstance(messages, list):
                    blobs.extend(messages)
        for message in blobs:
            if not isinstance(message, dict):
                continue
            role = message.get("type") or message.get("role") or ""
            if role not in ("ai", "assistant"):
                continue
            text = _text_from_content(message.get("content"))
            if text:
                texts.append(text)
    return texts
