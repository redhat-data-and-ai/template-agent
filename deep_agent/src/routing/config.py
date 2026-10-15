"""Model router configuration.

Tiers and their selection criteria are declarative, because the criteria
are the part that needs tuning against a real task mix and the part a
non-Python owner should be able to change.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


class ModelTier(BaseModel):
    """One routable model and the work it should absorb."""

    name: str
    model: str
    # Plain language, handed to the classifier verbatim. This is the knob
    # that actually moves routing quality.
    criteria: str
    max_output_tokens: int | None = None


class RouterConfig(BaseModel):
    """Parsed ``router.yaml``."""

    enabled: bool = False

    # "first_message" routes once and keeps that model for the thread —
    # switching mid-thread invalidates the provider's prompt cache, so the
    # next turn re-reads the whole conversation at full price.
    # "every_turn" re-evaluates each turn; better for threads that change
    # shape, worse for cost on long ones.
    strategy: Literal["first_message", "every_turn"] = "first_message"

    # Ordered cheapest-first. Order is load-bearing: the classifier is
    # told to pick the first tier that can plausibly finish the job.
    tiers: list[ModelTier] = Field(default_factory=list)

    # Used when classification fails or returns something unrecognised.
    # Defaults to the *most* capable tier: a wrong cheap answer costs more
    # to discover than a right expensive one.
    fallback_tier: str | None = None

    # Model that performs classification. Must be cheap and fast; its cost
    # is paid on every routed request and eats the savings otherwise.
    classifier_model: str = ""
    classifier_timeout: float = 10.0

    def tier_by_name(self, name: str) -> ModelTier | None:
        """The tier with this name, or None when nothing matches."""
        for tier in self.tiers:
            if tier.name == name:
                return tier
        return None

    @property
    def default_tier(self) -> ModelTier | None:
        """Where to land when classification cannot be trusted."""
        if self.fallback_tier:
            explicit = self.tier_by_name(self.fallback_tier)
            if explicit:
                return explicit
        return self.tiers[-1] if self.tiers else None

    def model_post_init(self, _context: object) -> None:
        """Reject a routing config that cannot route: enabled needs two tiers."""
        if not self.enabled:
            return
        if len(self.tiers) < 2:
            raise ValueError("router requires at least two tiers to route between")
        if self.fallback_tier and not self.tier_by_name(self.fallback_tier):
            raise ValueError(
                f"fallback_tier '{self.fallback_tier}' is not a defined tier"
            )
        names = [t.name for t in self.tiers]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate tier names: {names}")


def _candidate_paths() -> list[Path]:
    explicit = os.getenv("ROUTER_CONFIG_PATH")
    agent_dir = Path(
        os.getenv(
            "CONFIG_PATH",
            str(Path(__file__).parent.parent.parent.parent / "config" / "agent"),
        )
    )
    candidates = [
        agent_dir / "runtime" / "router.yaml",
        agent_dir.parent / "routing" / "router.yaml",
    ]
    if explicit:
        candidates.insert(0, Path(explicit))
    return candidates


def load_router_config(path: Path | None = None) -> RouterConfig:
    """Load ``router.yaml``; absent means routing is off."""
    if path is not None:
        found: Path | None = path if path.is_file() else None
    else:
        found = next((p for p in _candidate_paths() if p.is_file()), None)

    if found is None:
        return RouterConfig(enabled=False)

    with found.open() as handle:
        raw = yaml.safe_load(handle) or {}

    config = RouterConfig(**raw)
    if config.enabled:
        logger.info(
            "Model router enabled: %d tier(s), strategy=%s",
            len(config.tiers),
            config.strategy,
        )
    return config
