"""Model routing middleware.

Routing lives in the harness rather than in a gateway, because choosing a
model needs the agent's own context — its prompt, tools and task mix —
which a generic proxy does not have.

Implemented over ``wrap_model_call``, so the agent graph, tools and
prompts are untouched; only the model backing a call changes.
"""

from __future__ import annotations

from typing import Any

from deep_agent.src.routing.classifier import choose_tier
from deep_agent.src.routing.config import RouterConfig, load_router_config
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

# Key under which the chosen tier is remembered on thread state, so
# "first_message" routing survives subsequent turns.
STATE_KEY = settings.ROUTING_STATE_KEY

# Thread id -> tier, for `strategy: first_message`.
#
# In-process and bounded: losing it on restart costs one extra
# classification for a live thread, which is the behaviour this
# replaces rather than a regression.
_ROUTED_TIERS: dict[str, str] = {}
_ROUTED_TIERS_LIMIT = 2048


def _thread_id() -> str:
    """The current thread, or "" when there is no graph context."""
    try:
        from langgraph.config import get_config

        config = get_config() or {}
        return str((config.get("configurable") or {}).get("thread_id") or "")
    except Exception:
        return ""


def _first_human_text(messages: list[Any]) -> str:
    """The text that routing is decided on."""
    for message in messages:
        mtype = getattr(message, "type", None) or (
            message.get("type") if isinstance(message, dict) else None
        )
        role = getattr(message, "role", None) or (
            message.get("role") if isinstance(message, dict) else None
        )
        if mtype in ("human", "user") or role in ("human", "user"):
            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict)
                )
    return ""


def _latest_human_text(messages: list[Any]) -> str:
    """The most recent user turn, for per-turn routing."""
    return _first_human_text(list(reversed(messages)))


def build_model_router_middleware(
    config: RouterConfig | None = None,
) -> Any | None:
    """Build the routing middleware, or ``None`` when routing is off."""
    cfg = config or load_router_config()

    if not cfg.enabled or len(cfg.tiers) < 2:
        return None

    try:
        from langchain.agents.middleware import wrap_model_call
    except ImportError:
        logger.warning("wrap_model_call unavailable — model routing disabled")
        return None

    from deep_agent.src.agent.llm import create_model

    # Built once per tier rather than per request: constructing a client
    # per call adds latency to every turn for no benefit.
    clients: dict[str, Any] = {}

    def _client_for(tier_name: str) -> Any | None:
        if tier_name in clients:
            return clients[tier_name]
        tier = cfg.tier_by_name(tier_name)
        if tier is None:
            return None
        try:
            kwargs: dict[str, Any] = {"model_name": tier.model}
            if tier.max_output_tokens:
                kwargs["max_output_tokens"] = tier.max_output_tokens
            clients[tier_name] = create_model(**kwargs)
        except Exception as exc:
            logger.error("Could not build model for tier '%s': %s", tier_name, exc)
            clients[tier_name] = None
        return clients[tier_name]

    def _annotate(response: Any, fields: dict[str, Any]) -> None:
        """Attach fields to the returned message, best effort.

        Never raises: a missing annotation costs a line of provenance,
        and failing the turn over it would trade the answer for the
        footnote.
        """
        message = getattr(response, "result", None) or response
        if isinstance(message, list) and message:
            message = message[0]
        existing = getattr(message, "additional_kwargs", None)
        if isinstance(existing, dict):
            existing.update(fields)

    async def route(request: Any, handler: Any) -> Any:
        state = getattr(request, "state", None) or {}
        messages = list(getattr(request, "messages", []) or [])

        thread = _thread_id()
        remembered = state.get(STATE_KEY) if isinstance(state, dict) else None
        if not remembered and thread:
            remembered = _ROUTED_TIERS.get(thread)

        if cfg.strategy == "first_message" and remembered:
            # Keep the thread on one model: switching mid-thread discards
            # the provider's prompt cache and re-reads the conversation at
            # full price.
            chosen_name, reason, method = remembered, "already routed", "sticky"
        else:
            text = (
                _latest_human_text(messages)
                if cfg.strategy == "every_turn"
                else _first_human_text(messages)
            )
            decision = await choose_tier(text, cfg)
            chosen_name = decision.tier.name
            reason, method = decision.reason, decision.method

            logger.info("Routed to tier '%s' (%s: %s)", chosen_name, method, reason)

            # Remember it, so "first_message" means the first message.
            #
            # Nothing wrote `STATE_KEY`, so `remembered` was always
            # empty and the sticky branch never ran: every model call
            # in the thread re-applied the rules and, where they did
            # not match, paid for another classifier call. This
            # middleware wraps the model call and cannot write graph
            # state, so the thread is the key here instead.
            if thread and cfg.strategy == "first_message":
                if len(_ROUTED_TIERS) >= _ROUTED_TIERS_LIMIT:
                    for stale in list(_ROUTED_TIERS)[: _ROUTED_TIERS_LIMIT // 2]:
                        _ROUTED_TIERS.pop(stale, None)
                _ROUTED_TIERS[thread] = chosen_name

        client = _client_for(chosen_name)
        if client is None:
            # Routing must never be able to stop the agent answering.
            logger.warning(
                "Tier '%s' has no usable model — using the agent's default",
                chosen_name,
            )
            return await handler(request)

        routed = request.override(model=client)
        response = await handler(routed)

        # Record the choice on the message that came back, so the
        # reader can see which tier answered. Without it routing is
        # invisible: two turns in one conversation can come from
        # different models and the transcript looks identical.
        _annotate(response, {"routed_tier": chosen_name, "routed_by": method})
        return response

    middleware = wrap_model_call(route, name="ModelRouterMiddleware")
    logger.info(
        "Model router active: %s",
        ", ".join(f"{t.name}={t.model}" for t in cfg.tiers),
    )
    return middleware
