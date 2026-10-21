"""Tell the model which MCP servers it can actually reach.

Every enabled server contributes a tool to the model's tool list
whether or not anyone has connected to it. So the model sees thirteen
integrations and has no way to tell the two it can use from the eleven
that will answer "authorization required" — it picks one, fails, and
has nothing useful to say to the user about why.

The fix is to say so. A short note on each model call lists what is
connected and what is not, so an unusable tool is visibly unusable
*before* it is called, and the model can tell the user to go and
connect it rather than reporting a mysterious error.

Built the same way as the attachment notice, and for the same reason:
into the request's system message via `wrap_model_call`, never into
the transcript. The `messages` channel reduces with `add_messages`,
which appends — a notice returned from `before_model` accumulates one
copy per turn and is stored in the thread forever.
"""

from __future__ import annotations

from typing import Any

from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

# Connection state is per user and lives in Redis. Re-reading it on
# every model call would put a round trip in front of each turn, and
# it changes only when somebody completes an OAuth flow.
_CACHE_SECONDS = settings.MCP_STATUS_CACHE_SECONDS


def build_notice(connected: list[str], unconnected: list[str]) -> str:
    """The note describing which integrations are usable."""
    lines: list[str] = []
    if connected:
        lines.append("Connected and usable: " + ", ".join(sorted(connected)) + ".")
    if unconnected:
        lines.append(
            "Not connected, so their tools will fail: "
            + ", ".join(sorted(unconnected))
            + ". If one of these is needed, tell the user to connect it in "
            f"{settings.MCP_SETTINGS_PATH} rather than calling it and "
            "reporting the error."
        )
    return "\n".join(lines)


async def connection_state(user_id: str) -> tuple[list[str], list[str]]:
    """`(connected, unconnected)` server names for *user_id*.

    Any failure returns two empty lists, which suppresses the notice
    entirely. A wrong list is worse than none: telling the model an
    integration is unavailable when it works would stop it using a
    tool the user is waiting on.
    """
    try:
        from deep_agent.aegra.mcp_oauth_handlers import handle_mcp_connections

        result = await handle_mcp_connections(user_id)
        entries = (result or {}).get("connections") or []
    except Exception:
        logger.debug("Could not read MCP connection state", exc_info=True)
        return [], []

    # The connected list carries the account each integration is
    # acting as. The same person is one identity in the agent and
    # often another in the integration, and "who is my manager"
    # answered from the agent's login quietly answers for the wrong
    # person whenever those differ.
    connected = [
        f"{e['mcp_name']} (as {e['account']})" if e.get("account") else e["mcp_name"]
        for e in entries
        if e.get("connected")
    ]
    unconnected = [e["mcp_name"] for e in entries if not e.get("connected")]
    return connected, unconnected


# Strong references to the cache-priming tasks.
_PRIME_TASKS: set[Any] = set()


def build_mcp_status_middleware(user_id: str | None = None) -> Any:
    """Middleware announcing which MCP servers are connected."""
    import asyncio
    import time

    from langchain.agents.middleware.types import AgentMiddleware
    from langchain_core.messages import SystemMessage

    # (fetched_at, notice, unconnected server names)
    cache: dict[str, tuple[float, str, tuple[str, ...]]] = {}

    async def _prime() -> None:
        """Fill the cache before the first model call needs it.

        `wrap_model_call` is synchronous and cannot await a Redis
        lookup, so it can only serve what is already cached. Without
        priming, a run that takes the sync path emits no notice at
        all and the model says it has no integrations — which is the
        bug this middleware exists to fix, reappearing through a
        different door.
        """
        if not user_id:
            return
        try:
            connected, unconnected = await connection_state(user_id)
            if connected or unconnected:
                cache[user_id] = (
                    time.monotonic(),
                    build_notice(connected, unconnected),
                    tuple(unconnected),
                )
        except Exception:
            logger.debug("Could not prime MCP status", exc_info=True)

    try:
        # Held: asyncio keeps only a weak reference, so a task
        # nobody holds can be collected before it fills the cache.
        _task = asyncio.get_running_loop().create_task(_prime())
        _PRIME_TASKS.add(_task)
        _task.add_done_callback(_PRIME_TASKS.discard)
    except RuntimeError:
        # No loop yet: the async path will fill the cache instead.
        pass

    class McpStatusMiddleware(AgentMiddleware):
        """Name the integrations the model can and cannot use."""

        async def awrap_model_call(self, request: Any, handler: Any) -> Any:
            notice = await self._notice(request)
            response = await handler(self._apply(request, notice))
            return self._offer_connect(response, self._unconnected(request))

        def wrap_model_call(self, request: Any, handler: Any) -> Any:
            # The sync path cannot await the connection lookup, and
            # blocking the event loop to find out would cost more than
            # the notice is worth. It falls back to whatever the async
            # path cached.
            user_id = self._user_id(request)
            cached = cache.get(user_id)
            notice = cached[1] if cached else ""
            response = handler(self._apply(request, notice))
            return self._offer_connect(response, self._unconnected(request))

        def _unconnected(self, request: Any) -> tuple[str, ...]:
            hit = cache.get(self._user_id(request))
            return hit[2] if hit and len(hit) > 2 else ()

        @staticmethod
        def _offer_connect(response: Any, unconnected: tuple[str, ...]) -> Any:
            """Put a Connect button on a reply that declined for auth.

            The button already existed, but it was only ever built
            from a tool *failure*: something had to call an
            unreachable server and raise. Told up front that
            a server was not connected, the model correctly declined
            without calling anything — so no failure was recorded, no
            marker was emitted, and the reply sent the reader off to
            a settings page instead of offering the one click that
            fixes it.

            Derived here instead of asked of the model. Telling it to
            emit a marker was tried and it paraphrased the sentence
            and dropped the marker, which is why provenance exists.
            """
            if not unconnected:
                return response
            try:
                message: Any = getattr(response, "result", None)
                if isinstance(message, list):
                    message = message[0] if message else None
                if message is None:
                    return response

                text = getattr(message, "text", "") or ""
                if not isinstance(text, str) or not text:
                    return response
                haystack = text.lower()

                named = [n for n in unconnected if n.lower() in haystack]
                if not named:
                    return response

                extra = getattr(message, "additional_kwargs", None)
                if not isinstance(extra, dict):
                    extra = {}
                    message.additional_kwargs = extra

                # A tool failure is the better signal when both fire:
                # it knows whether the grant expired or was never
                # made. Only fill gaps it left.
                existing = extra.get("auth_required")
                requests = list(existing) if isinstance(existing, list) else []
                already = {r.get("mcp_name") for r in requests if isinstance(r, dict)}
                for name in named:
                    if name in already:
                        continue
                    requests.append({"mcp_name": name, "reason": "not_connected"})

                # Then keep only what this answer is about. Shared
                # with the tool-failure path so that whichever
                # middleware writes last reaches the same answer.
                from deep_agent.aegra.mcp_tool_auth import narrow_auth_requests

                requests = narrow_auth_requests(requests, text)
                if requests:
                    extra["auth_required"] = requests
            except Exception:
                # A missing button is a worse answer; a failed turn is
                # no answer.
                logger.debug("Could not offer connect action", exc_info=True)
            return response

        @staticmethod
        def _user_id(request: Any) -> str:
            # Supplied by the graph, which resolved it for the tool
            # list. Anything else is a fallback for callers that did
            # not pass one.
            if user_id:
                return str(user_id)

            # The per-request contextvar is where the rest of the
            # graph gets this. Reading `state["user_id"]` was wrong:
            # the key is not in graph state, so every lookup ran as
            # "anonymous", found no tokens, and the model was told
            # confidently that nothing was connected while eight
            # servers were. A notice that is wrong is worse than no
            # notice, which is the whole reason this falls silent on
            # failure — so it must not quietly use the wrong identity.
            from deep_agent.utils.pylogger import _user_id_var

            # A distinct name: assigning to `user_id` here made it a
            # local of this function, shadowing the closure argument
            # above and raising UnboundLocalError on the very first
            # line that read it.
            from_context = _user_id_var.get()
            if from_context:
                return str(from_context)

            state = getattr(request, "state", None)
            if isinstance(state, dict) and state.get("user_id"):
                return str(state["user_id"])
            return ""

        async def _notice(self, request: Any) -> str:
            user_id = self._user_id(request)
            if not user_id:
                # Without an identity the connection state is not
                # knowable, and guessing produces the exact wrong
                # answer: "nothing is connected".
                return ""
            hit = cache.get(user_id)
            now = time.monotonic()
            if hit and now - hit[0] < _CACHE_SECONDS:
                return hit[1]

            connected, unconnected = await connection_state(user_id)
            notice = (
                build_notice(connected, unconnected)
                if (connected or unconnected)
                else ""
            )
            cache[user_id] = (now, notice, tuple(unconnected))
            return notice

        @staticmethod
        def _apply(request: Any, notice: str) -> Any:
            if not notice:
                return request
            existing = getattr(request, "system_message", None)
            try:
                # `system_message` is a SystemMessage, not a string.
                if existing is None:
                    updated = SystemMessage(content=notice)
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
                logger.debug("Could not attach MCP status notice", exc_info=True)
                return request

    return McpStatusMiddleware()
