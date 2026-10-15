"""Inbound channel endpoints.

Everything under ``/channels`` is reachable from outside the cluster —
it is the only part of the agent API the external Route exposes. Treat
each handler as hostile input until the signature check passes.

The ordering in :func:`receive` is deliberate and worth preserving:

1. Is the channel configured and on? An unknown path must look the same
   as a disabled one, or the 404/403 split enumerates our channels.
2. Does the signature verify? Before parsing, before logging the body,
   before anything that could be influenced by the payload.
3. Only then parse, map identity, and run.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response, status

from deep_agent.src.channels.adapters import AdapterError, InboundMessage, parse
from deep_agent.src.channels.config import ChannelConfig
from deep_agent.src.channels.dispatch import (
    already_handled,
    deliver,
    run_for_channel,
)
from deep_agent.src.channels.signatures import verify_slack, verify_webhook
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

router = APIRouter(prefix="/channels", tags=["channels"])


def _load_channels() -> dict[str, ChannelConfig]:
    """Active channels, re-read per request so config can hot-reload."""
    from deep_agent.src.channels.loader import load_channels

    return load_channels().active()


def _verify(channel: ChannelConfig, request: Request, body: bytes) -> None:
    """Raise 401 unless the request is authentically from this channel."""
    secret = channel.signing_secret
    if not secret:
        # Misconfiguration, but the caller learns nothing either way.
        logger.error(
            "Channel %s has no secret in %s", channel.type, channel.signing_secret_env
        )
        raise HTTPException(status_code=401, detail="Signature verification failed")

    if channel.type == "slack":
        result = verify_slack(
            body=body,
            timestamp=request.headers.get("X-Slack-Request-Timestamp", ""),
            signature=request.headers.get("X-Slack-Signature", ""),
            secret=secret,
            max_age_seconds=channel.max_age_seconds,
        )
    else:
        result = verify_webhook(
            body=body,
            timestamp=request.headers.get("X-Timestamp", ""),
            signature=request.headers.get("X-Signature", ""),
            secret=secret,
            max_age_seconds=channel.max_age_seconds,
        )

    if not result:
        # Logged, never returned: the reason tells an attacker which of
        # freshness or digest failed, which is a tuning signal.
        logger.warning("Channel signature rejected: %s", result.reason)
        raise HTTPException(status_code=401, detail="Signature verification failed")


async def _run_and_deliver(
    message: InboundMessage, channel: ChannelConfig, user_id: str
) -> None:
    """Background half of ack-then-work."""
    try:
        reply = await run_for_channel(
            message,
            channel,
            user_id,
            service_token=os.environ.get("CHANNEL_SERVICE_TOKEN"),
        )
        await deliver(reply, message, channel)
    except Exception:
        # A background task that raises disappears silently, and a
        # channel that goes quiet is indistinguishable from one nobody
        # used.
        logger.exception("Channel run failed for conversation %s", message.conversation)


@router.post("/{name}")
async def receive(name: str, request: Request) -> Response:
    """Accept an event from a configured channel."""
    channels = _load_channels()
    channel = channels.get(name)
    if channel is None:
        raise HTTPException(status_code=404, detail="Unknown channel")

    body = await request.body()
    _verify(channel, request, body)

    try:
        payload: dict[str, Any] = json.loads(body or b"{}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Body is not JSON") from None

    # Slack proves ownership of a new endpoint with this, before any
    # event is ever sent. It is signed, so it is handled after _verify.
    if payload.get("type") == "url_verification":
        return Response(
            content=json.dumps({"challenge": payload.get("challenge", "")}),
            media_type="application/json",
        )

    try:
        message = parse(channel.type, payload)
    except AdapterError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    if message is None:
        # Something we deliberately ignore, like a bot's own post.
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    event_key = message.metadata.get("event_id") or ""
    if already_handled(event_key):
        logger.info("Ignoring duplicate delivery of %s", event_key)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    user_id = channel.identity.resolve(message.sender)
    if user_id is None:
        logger.warning("Channel %s rejected unmapped sender %s", name, message.sender)
        raise HTTPException(status_code=403, detail="Sender is not authorised")

    # Ack now, work after: Slack retries anything slower than three
    # seconds, and a retry is a second run of the same prompt.
    asyncio.create_task(_run_and_deliver(message, channel, user_id))
    return Response(status_code=status.HTTP_202_ACCEPTED)


@router.get("/")
async def list_channels() -> dict[str, Any]:
    """Which channels are live. No secrets, no identity maps."""
    return {
        "channels": [
            {"name": name, "type": c.type, "assistant_id": c.assistant_id}
            for name, c in _load_channels().items()
        ]
    }
