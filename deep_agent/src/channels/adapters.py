"""Turn a channel's payload into something the agent can run.

Each channel speaks its own shape. The rest of the system should not
know that, so everything is normalised to :class:`InboundMessage` here
and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class InboundMessage:
    """A normalised request to run the agent."""

    # Channel-native sender id, before identity mapping.
    sender: str
    text: str
    # Stable per-conversation key, so a Slack thread maps to one agent
    # thread rather than starting fresh on every message.
    conversation: str
    # Where to deliver the answer, if the channel gave us somewhere.
    reply_to: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AdapterError(ValueError):
    """The payload was authentic but not something we can act on."""


def _require(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise AdapterError(f"payload missing required field {key!r}")
    return payload[key]


def parse_slack(payload: dict[str, Any]) -> InboundMessage | None:
    """Parse a Slack Events API callback.

    Returns None for events we deliberately ignore. Bot messages are the
    important case: answering one is how an agent ends up in an infinite
    conversation with itself.
    """
    event = payload.get("event") or {}

    if payload.get("type") != "event_callback":
        raise AdapterError(f"unsupported Slack payload type {payload.get('type')!r}")

    if event.get("type") not in ("app_mention", "message"):
        return None

    # `bot_id` covers other apps; any `subtype` at all covers the
    # rest. Matching only `bot_message` let `message_changed`,
    # `message_deleted`, `channel_join` and file shares through, and
    # each one started a run: editing a message re-ran it, and
    # deleting one ran the tombstone.
    if event.get("bot_id") or event.get("subtype"):
        return None

    text = (event.get("text") or "").strip()
    if not text:
        return None

    channel = event.get("channel") or ""
    # Reply in the thread that mentioned us, starting one if needed, so
    # a busy channel does not interleave several conversations.
    thread_ts = event.get("thread_ts") or event.get("ts") or ""

    return InboundMessage(
        sender=str(_require(event, "user")),
        text=text,
        conversation=f"slack:{channel}:{thread_ts}",
        reply_to=channel or None,
        metadata={
            "channel": channel,
            "thread_ts": thread_ts,
            "team": payload.get("team_id", ""),
            "event_id": payload.get("event_id", ""),
        },
    )


def parse_webhook(payload: dict[str, Any]) -> InboundMessage:
    """Parse the generic webhook shape.

    ``{"sender": "...", "text": "...", "conversation": "...", ...}`` —
    deliberately small, because every field is one more thing a caller
    can get wrong.
    """
    sender = str(_require(payload, "sender"))
    text = str(_require(payload, "text")).strip()
    if not text:
        raise AdapterError("text is empty")

    # Without a conversation key every call is a new thread. That is a
    # reasonable default for CI-style senders, which are stateless.
    conversation = str(payload.get("conversation") or f"webhook:{sender}")

    return InboundMessage(
        sender=sender,
        text=text,
        conversation=conversation,
        reply_to=payload.get("callback_url"),
        metadata={k: v for k, v in payload.items() if k not in ("sender", "text")},
    )


PARSERS = {"slack": parse_slack, "webhook": parse_webhook}


def parse(channel_type: str, payload: dict[str, Any]) -> InboundMessage | None:
    """Normalise a channel payload. Raises AdapterError for unknown types."""
    parser = PARSERS.get(channel_type)
    if parser is None:
        raise AdapterError(f"no parser for channel type {channel_type!r}")
    return parser(payload)
