"""Pick a model tier for a request.

A rule pass runs before the LLM classifier, because a large share of
traffic is decidable without a model call and the classifier's own cost
is charged against the savings it produces.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from deep_agent.src.routing.config import ModelTier, RouterConfig
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_CLASSIFIER_PROMPT = """You route a request to the cheapest model tier that \
can plausibly complete it.

Tiers, cheapest first:
{tiers}

Reply with the tier name alone. No explanation.

Request:
{request}"""

# Short, closed-form asks that no large model improves on.
_TRIVIAL = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|ok|okay|yes|no|ping)\b[\s!.?]*$",
    re.IGNORECASE,
)

# Words that reliably mark multi-step work. Deliberately a small list:
# every entry here is a bet that the expensive tier pays for itself.
_COMPLEX_MARKERS = (
    "refactor",
    "migrate",
    "architect",
    "design a",
    "debug",
    "root cause",
    "trade-off",
    "tradeoff",
    "step by step",
    "end to end",
)


@dataclass
class RoutingDecision:
    """Which tier was chosen, and why — the 'why' is for the audit trail."""

    tier: ModelTier
    reason: str
    method: str  # "rule" | "classifier" | "fallback"


def classify_by_rules(text: str, config: RouterConfig) -> RoutingDecision | None:
    """Decide without a model call where the answer is obvious."""
    if not config.tiers:
        return None

    cheapest = config.tiers[0]
    dearest = config.tiers[-1]
    stripped = text.strip()

    if not stripped:
        return RoutingDecision(cheapest, "empty request", "rule")

    if _TRIVIAL.match(stripped):
        return RoutingDecision(cheapest, "greeting or acknowledgement", "rule")

    lowered = stripped.lower()
    for marker in _COMPLEX_MARKERS:
        if marker in lowered:
            return RoutingDecision(
                dearest, f"contains complexity marker {marker!r}", "rule"
            )

    # Very long requests tend to carry multi-part requirements.
    if len(stripped) > 2000:
        return RoutingDecision(dearest, "long multi-part request", "rule")

    return None


async def classify_with_model(
    text: str, config: RouterConfig
) -> RoutingDecision | None:
    """Ask a small model which tier fits.

    Returns ``None`` on any failure so the caller falls back rather than
    failing the user's request — a router that can break the agent is
    worse than no router.
    """
    if not config.classifier_model:
        return None

    tier_lines = "\n".join(f"- {tier.name}: {tier.criteria}" for tier in config.tiers)
    prompt = _CLASSIFIER_PROMPT.format(tiers=tier_lines, request=text[:4000])

    try:
        from deep_agent.src.agent.llm import create_model

        model = create_model(model_name=config.classifier_model)
        response = await asyncio.wait_for(
            model.ainvoke(prompt), timeout=config.classifier_timeout
        )
    except TimeoutError:
        logger.warning(
            "Tier classifier timed out after %.1fs", config.classifier_timeout
        )
        return None
    except Exception as exc:
        logger.warning("Tier classifier failed: %s", exc)
        return None

    raw = getattr(response, "content", response)
    answer = (raw if isinstance(raw, str) else str(raw)).strip().lower()

    # Exact answer first, then whole words, and only then
    # containment. Containment alone took the first tier whose name
    # appeared anywhere, so "not fast, use deep" routed to `fast` and
    # a tier called `pro` matched the reply "pro-max". Small models
    # do add punctuation and a leading "Tier:", which is what the
    # last two passes are for.
    import re as _re

    by_name = {tier.name.lower(): tier for tier in config.tiers}
    exact = by_name.get(answer)
    if exact is not None:
        return RoutingDecision(exact, f"classifier chose {exact.name}", "classifier")

    tokens = set(_re.findall(r"[a-z0-9_.-]+", answer))
    for tier in config.tiers:
        if tier.name.lower() in tokens:
            return RoutingDecision(tier, f"classifier chose {tier.name}", "classifier")

    for tier in config.tiers:
        if tier.name.lower() in answer:
            return RoutingDecision(tier, f"classifier chose {tier.name}", "classifier")

    logger.warning("Classifier returned an unrecognised tier: %r", answer[:80])
    return None


async def choose_tier(text: str, config: RouterConfig) -> RoutingDecision:
    """Resolve a request to a tier: rules, then classifier, then fallback."""
    decided = classify_by_rules(text, config)
    if decided is not None:
        return decided

    decided = await classify_with_model(text, config)
    if decided is not None:
        return decided

    default = config.default_tier
    if default is None:
        raise ValueError("router has no tiers configured")

    # Falls back to the most capable tier: under-serving a hard request
    # costs more to discover than over-serving an easy one.
    return RoutingDecision(default, "classification unavailable", "fallback")
