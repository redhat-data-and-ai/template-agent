"""Run the agent on behalf of a channel, then deliver the answer.

Calls the agent's own HTTP API over loopback rather than reaching into
aegra's service layer. That keeps this decoupled from a dependency whose
internals move between versions — the Agent Protocol is the stable
interface, and it is the same path the UI takes, so a channel run and a
UI run go through identical auth, ownership and checkpointing.

Deliberately ack-then-work: Slack retries anything it cannot deliver in
three seconds, and a retried event is a second run of the same prompt.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

import httpx

from deep_agent.src.channels.adapters import InboundMessage
from deep_agent.src.channels.config import ChannelConfig
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

# Remembered so a retry does not run twice. Bounded because this is a
# process-local cache, not a durability mechanism — Slack retries within
# about a minute, well inside any sane bound.
_SEEN_EVENTS: dict[str, float] = {}
_SEEN_LIMIT = 2048


def _base_url() -> str:
    return f"http://127.0.0.1:{settings.AGENT_PORT}"


def already_handled(event_key: str) -> bool:
    """True if this exact delivery has been seen before.

    Slack resends an event when the ack is slow, and an agent that acts
    twice on one message is worse than one that acts late.
    """
    if not event_key:
        return False
    if event_key in _SEEN_EVENTS:
        return True

    loop_time = asyncio.get_event_loop().time()
    if len(_SEEN_EVENTS) >= _SEEN_LIMIT:
        oldest = sorted(_SEEN_EVENTS, key=_SEEN_EVENTS.__getitem__)[: _SEEN_LIMIT // 2]
        for key in oldest:
            _SEEN_EVENTS.pop(key, None)
    _SEEN_EVENTS[event_key] = loop_time
    return False


def thread_key(conversation: str) -> str:
    """Deterministic agent thread id for a conversation.

    A Slack thread should keep its context across messages, and the
    agent's thread ids are UUID-shaped, so hash the conversation key
    into one rather than storing a mapping table.
    """
    digest = hashlib.sha256(conversation.encode()).hexdigest()
    return (
        f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"
    )


async def _ensure_thread(client: httpx.AsyncClient, thread_id: str) -> None:
    """Create the thread if this is the first message in the conversation."""
    existing = await client.get(f"/threads/{thread_id}")
    if existing.status_code == 200:
        return
    created = await client.post("/threads", json={"thread_id": thread_id})
    # 409 means another delivery of the same conversation won the race,
    # which is success for our purposes.
    if created.status_code not in (200, 201, 409):
        created.raise_for_status()


async def run_for_channel(
    message: InboundMessage,
    channel: ChannelConfig,
    user_id: str,
    *,
    service_token: str | None = None,
) -> str:
    """Run the agent and return its reply text."""
    headers: dict[str, str] = {}
    if service_token:
        headers["Authorization"] = f"Bearer {service_token}"
    # The run is attributed to the mapped platform user, not to the
    # channel, so ownership and authz behave as they do for the UI.
    headers["X-Channel-User-Id"] = user_id

    thread_id = thread_key(message.conversation)

    async with httpx.AsyncClient(
        base_url=_base_url(), headers=headers, timeout=channel.timeout_seconds
    ) as client:
        await _ensure_thread(client, thread_id)

        response = await client.post(
            f"/threads/{thread_id}/runs/wait",
            json={
                "assistant_id": channel.assistant_id,
                "input": {"messages": [{"type": "human", "content": message.text}]},
            },
        )
        response.raise_for_status()
        return _reply_text(response.json())


def _content_text(content: Any) -> str:
    """Flatten a message's content to plain text.

    Gemini 3 returns content as a list of typed blocks rather than a
    string — ``[{"type": "text", "text": "..."}]`` — with non-text
    blocks like thought signatures mixed in. ``str()`` on that posts a
    Python repr to Slack, which is how a model upgrade turns into a
    user-visible mess.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        return "\n".join(p for p in parts if p.strip())
    return ""


def _reply_text(payload: Any) -> str:
    """Pull the assistant's last message out of a run result."""
    messages = (payload or {}).get("messages") or []
    for message in reversed(messages):
        if message.get("type") == "ai":
            text = _content_text(message.get("content"))
            if text:
                return text
    error = (payload or {}).get("__error__")
    if error:
        return f"The agent could not complete that: {error.get('message', 'unknown error')}"
    return "The agent produced no reply."


async def deliver(reply: str, message: InboundMessage, channel: ChannelConfig) -> None:
    """Post the answer back to wherever it came from."""
    if channel.type == "slack":
        await _deliver_slack(reply, message)
    elif message.reply_to:
        await _deliver_webhook(reply, message)


async def _deliver_slack(reply: str, message: InboundMessage) -> None:
    import os

    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        logger.warning("SLACK_BOT_TOKEN unset — cannot post reply")
        return

    async with httpx.AsyncClient(timeout=30) as client:
        result = await client.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "channel": message.metadata.get("channel"),
                # Reply in-thread, so the agent does not shout into the
                # channel for every mention.
                "thread_ts": message.metadata.get("thread_ts"),
                "text": reply,
            },
        )
        # Slack signals failure in the body with HTTP 200.
        body = result.json()
        if not body.get("ok"):
            logger.warning("Slack rejected the reply: %s", body.get("error"))


async def _deliver_webhook(reply: str, message: InboundMessage) -> None:
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            await client.post(
                str(message.reply_to),
                json={"conversation": message.conversation, "text": reply},
            )
        except httpx.HTTPError as exc:
            logger.warning("Callback to %s failed: %s", message.reply_to, exc)
