"""Per-request model tier selection."""

from deep_agent.src.routing.classifier import (
    RoutingDecision,
    choose_tier,
    classify_by_rules,
)
from deep_agent.src.routing.config import ModelTier, RouterConfig, load_router_config
from deep_agent.src.routing.middleware import build_model_router_middleware

__all__ = [
    "ModelTier",
    "RouterConfig",
    "RoutingDecision",
    "build_model_router_middleware",
    "choose_tier",
    "classify_by_rules",
    "load_router_config",
]
