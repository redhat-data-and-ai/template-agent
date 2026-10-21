"""Put the agent's own description in front of the model.

Replaces the MCP status middleware rather than sitting beside it.
Two middlewares each appending their own block left the model with
two descriptions of itself to reconcile, and no way to tell that
"datadog is not connected" and "you may delegate to investigator,
which reaches datadog" were about the same thing.

Added to the *request*, never to state. Returning it from
`before_model` would append a copy per turn through the
`add_messages` reducer, and the thread would carry one for every
exchange, forever.
"""

from __future__ import annotations

import time
from typing import Any

from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

# Connection state changes only when somebody completes an OAuth
# flow. Asking per model call would put a round trip in front of
# every turn.
_CACHE_SECONDS = settings.SELF_KNOWLEDGE_CACHE_SECONDS


def build_self_knowledge_middleware(
    user_id: str | None = None, model_name: str = ""
) -> Any:
    """Middleware describing the agent to itself, or None if there is nothing to say."""
    from deep_agent.src.self_knowledge.describe import build_self_description

    try:
        from langchain.agents.middleware.types import AgentMiddleware
        from langchain_core.messages import SystemMessage
    except ImportError:
        logger.warning("AgentMiddleware unavailable — self-knowledge disabled")
        return None

    # The config-derived half never changes within a process, so it
    # is built once; only the connection state is per-user and
    # refreshed.
    cache: dict[str, tuple[float, str, tuple[str, ...]]] = {}

    def _notice(connected: list[str], unconnected: list[str]) -> str:
        return build_self_description(model_name, connected, unconnected)

    static_notice = _notice([], [])

    class SelfKnowledgeMiddleware(AgentMiddleware):
        """Tell the model what it is, what it can reach, and what is refused."""

        async def awrap_model_call(self, request: Any, handler: Any) -> Any:
            notice = await self._notice()
            response = await handler(_apply(request, notice, SystemMessage))
            return _offer_connect(response, self._unconnected(), _asked_for(request))

        def wrap_model_call(self, request: Any, handler: Any) -> Any:
            # The sync path cannot await the connection lookup, and
            # blocking the event loop to find out would cost more
            # than the integration list is worth. It uses whatever
            # the async path cached, falling back to the half that
            # needs no lookup.
            hit = cache.get(user_id or "")
            response = handler(
                _apply(request, hit[1] if hit else static_notice, SystemMessage)
            )
            return _offer_connect(response, self._unconnected(), _asked_for(request))

        @staticmethod
        def _unconnected() -> tuple[str, ...]:
            hit = cache.get(user_id or "")
            return hit[2] if hit and len(hit) > 2 else ()

        async def _notice(self) -> str:
            key = user_id or ""
            hit = cache.get(key)
            now = time.monotonic()
            if hit and now - hit[0] < _CACHE_SECONDS:
                return hit[1]

            connected: list[str] = []
            unconnected: list[str] = []
            if user_id:
                # Without an identity the connection state is not
                # knowable, and guessing produces the exact wrong
                # answer: "nothing is connected".
                try:
                    from deep_agent.src.mcp_status.middleware import connection_state

                    connected, unconnected = await connection_state(user_id)
                except Exception:
                    logger.debug("Could not read MCP connection state", exc_info=True)

            notice = _notice(connected, unconnected)
            cache[key] = (now, notice, tuple(unconnected))
            return notice

    if not static_notice:
        logger.info("Self-knowledge: nothing to announce")
        return None

    logger.info("Self-knowledge active for user=%s model=%s", user_id, model_name)
    return SelfKnowledgeMiddleware()


def _asked_for(request: Any) -> str:
    """The user's own words for this turn, lowercased.

    A Connect button answers "I could not do what you asked". Whether
    the *answer* happens to name an integration does not say that —
    "which integrations can you use right now?" names four of them
    and needs none. What the user asked for does.
    """
    try:
        messages = list(getattr(request, "messages", None) or [])
    except Exception:
        return ""
    for message in reversed(messages):
        if getattr(message, "type", "") != "human":
            continue
        text = getattr(message, "text", "") or ""
        return text.lower() if isinstance(text, str) else ""
    return ""


def _offer_connect(
    response: Any, unconnected: tuple[str, ...], asked_for: str = ""
) -> Any:
    """Put a Connect button on a reply that declined for authorization.

    The button was only ever built from a tool *failure*: something
    had to call an unreachable server and raise. But this middleware
    tells the model up front which servers are unconnected, so it
    declines without calling anything — nothing raises, nothing is
    recorded, and the reply sends the reader to a settings page
    instead of offering the one click that fixes it. The better the
    model behaves, the worse the answer gets.

    Derived here rather than asked of the model: a prompt rule to
    emit the marker was tried, and it paraphrased the sentence and
    dropped the marker.
    """
    if not unconnected:
        return response
    try:
        from deep_agent.aegra.mcp_tool_auth import narrow_auth_requests

        message: Any = getattr(response, "result", None)
        if isinstance(message, list):
            message = message[0] if message else None
        if message is None:
            return response

        text = getattr(message, "text", "") or ""
        if not isinstance(text, str) or not text:
            return response
        haystack = text.lower()

        # Named in the answer *and* wanted by the question.
        #
        # Matching the answer alone turned "which integrations can
        # you use right now?" into four Connect buttons: the reply
        # listed them, so every one looked like a blocked request.
        # A button is an offer to unblock something the user asked
        # for, so the question has to have asked for it.
        named = [
            n for n in unconnected if n.lower() in haystack and n.lower() in asked_for
        ]
        if not named:
            return response

        extra = getattr(message, "additional_kwargs", None)
        if not isinstance(extra, dict):
            extra = {}
            message.additional_kwargs = extra

        # A tool failure is the better signal when both fire: it
        # knows whether the grant expired or was never made. Only
        # fill the gaps it left.
        existing = extra.get("auth_required")
        requests = list(existing) if isinstance(existing, list) else []
        already = {r.get("mcp_name") for r in requests if isinstance(r, dict)}
        for name in named:
            if name in already:
                continue
            requests.append({"mcp_name": name, "reason": "not_connected"})

        # Assign unconditionally. Guarding on a non-empty result left
        # the wider discovery list in place exactly when narrowing had
        # decided none of it applied.
        requests = narrow_auth_requests(requests, text)
        extra["auth_required"] = requests
        if requests:
            logger.info(
                "Offering connect action for %s",
                ", ".join(str(r.get("mcp_name")) for r in requests),
            )
    except Exception:
        # A missing button is a worse answer; a failed turn is none.
        # Warning, not debug: a button that silently fails to appear
        # is indistinguishable from one that was never wanted, and
        # that cost several rounds of looking in the wrong place.
        logger.warning("Could not offer connect action", exc_info=True)
    return response


def _apply(request: Any, notice: str, system_message_cls: Any) -> Any:
    """Append the notice to the request's system message."""
    if not notice:
        return request
    try:
        existing = getattr(request, "system_message", None)
        if existing is None:
            updated = system_message_cls(content=notice)
        elif isinstance(getattr(existing, "content", None), str):
            updated = existing.model_copy(
                update={"content": f"{existing.content}\n\n{notice}".strip()}
            )
        else:
            blocks = list(existing.content or [])
            blocks.append({"type": "text", "text": notice})
            updated = existing.model_copy(update={"content": blocks})
        return request.override(system_message=updated)
    except Exception:
        # A missing notice costs the model some context; failing
        # here would cost the answer.
        logger.debug("Could not attach the self-knowledge notice", exc_info=True)
        return request
