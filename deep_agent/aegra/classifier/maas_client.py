"""Streaming chat completion against MaaS.

The classifier needs the full JSON document before it can route, so this
client streams the tokens and joins them. The agent answer is a separate
stream and is not buffered here.
"""

from __future__ import annotations

import json
import logging

import httpx

from deep_agent.aegra.classifier.config import MaasSettings

logger = logging.getLogger(__name__)


def _delta_text(choice: dict) -> tuple[str, str]:
    """Return (content, reasoning) text from one streamed choice."""
    delta = choice.get("delta") or choice.get("message") or {}
    if not isinstance(delta, dict):
        return "", ""
    content = delta.get("content") or ""
    reasoning = delta.get("reasoning") or ""
    return str(content), str(reasoning)


def consume_sse_line(line: str, content: list[str], reasoning: list[str]) -> bool:
    """Append one SSE line. Return True when the stream is finished."""
    if not line.startswith("data:"):
        return False
    data = line[5:].strip()
    if data == "[DONE]":
        return True
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return False
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return False
    piece, thought = _delta_text(choices[0])
    if piece:
        content.append(piece)
    if thought:
        reasoning.append(thought)
    return False


def _join_stream(content: list[str], reasoning: list[str]) -> str:
    """Prefer streamed content, and use reasoning when content is empty."""
    text = "".join(content).strip()
    if text:
        return text
    return "".join(reasoning).strip()


def _from_json_body(raw: str) -> str:
    """Read a non-streaming chat completion body."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip()
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    piece, thought = _delta_text(choices[0])
    return piece.strip() or thought.strip()


async def stream_completion(
    *,
    query: str,
    system_prompt: str,
    settings: MaasSettings,
) -> str | None:
    """Stream one MaaS completion and return the joined assistant text."""
    if not settings.api_key or not system_prompt.strip():
        return None
    body = {
        "model": settings.model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        "max_tokens": settings.max_tokens,
        "stream": True,
    }
    timeout = httpx.Timeout(settings.timeout)
    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json",
    }
    content: list[str] = []
    reasoning: list[str] = []
    async with httpx.AsyncClient(verify=settings.verify_ssl, timeout=timeout) as client:
        async with client.stream(
            "POST",
            f"{settings.base_url}/chat/completions",
            json=body,
            headers=headers,
        ) as response:
            if response.status_code not in (200, 202):
                detail = (await response.aread()).decode("utf-8", errors="replace")
                logger.warning("MaaS HTTP %s: %s", response.status_code, detail[:300])
                return None
            media = response.headers.get("content-type", "")
            if "text/event-stream" not in media:
                raw = (await response.aread()).decode("utf-8", errors="replace")
                text = _from_json_body(raw)
                return text or None
            async for line in response.aiter_lines():
                if consume_sse_line(line, content, reasoning):
                    break
    text = _join_stream(content, reasoning)
    return text or None
