"""Tests for model tier routing."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from deep_agent.src.routing.classifier import (
    choose_tier,
    classify_by_rules,
    classify_with_model,
)
from deep_agent.src.routing.config import RouterConfig, load_router_config
from deep_agent.src.routing.middleware import build_model_router_middleware

SHIPPED = Path(__file__).parents[3] / "config" / "agent" / "runtime" / "router.yaml"


def cfg(**overrides) -> RouterConfig:
    base = {
        "enabled": True,
        "tiers": [
            {"name": "fast", "model": "m-fast", "criteria": "simple"},
            {"name": "balanced", "model": "m-bal", "criteria": "medium"},
            {"name": "deep", "model": "m-deep", "criteria": "hard"},
        ],
        "fallback_tier": "deep",
        "classifier_model": "m-fast",
    }
    base.update(overrides)
    return RouterConfig(**base)


class TestShippedConfig:
    def test_is_off_by_default(self):
        """Routing to models a deployment cannot reach is worse than none."""
        assert load_router_config(SHIPPED).enabled is False

    def test_tiers_are_ordered_cheapest_first(self):
        config = load_router_config(SHIPPED)
        assert [t.name for t in config.tiers] == ["fast", "balanced", "deep"]

    def test_fallback_is_the_most_capable_tier(self):
        config = load_router_config(SHIPPED)
        assert config.default_tier.name == "deep"

    def test_every_tier_has_criteria(self):
        """Criteria are what the classifier actually reads."""
        for tier in load_router_config(SHIPPED).tiers:
            assert tier.criteria.strip()


class TestConfigValidation:
    def test_rejects_a_single_tier(self):
        with pytest.raises(ValueError, match="at least two tiers"):
            RouterConfig(
                enabled=True,
                tiers=[{"name": "only", "model": "m", "criteria": "c"}],
            )

    def test_rejects_unknown_fallback(self):
        with pytest.raises(ValueError, match="fallback_tier"):
            cfg(fallback_tier="nonexistent")

    def test_rejects_duplicate_tier_names(self):
        with pytest.raises(ValueError, match="duplicate"):
            RouterConfig(
                enabled=True,
                tiers=[
                    {"name": "a", "model": "m1", "criteria": "c"},
                    {"name": "a", "model": "m2", "criteria": "c"},
                ],
            )

    def test_disabled_config_skips_validation(self):
        """A half-written config must not stop the agent booting."""
        assert RouterConfig(enabled=False, tiers=[]).enabled is False

    def test_missing_file_disables_routing(self, tmp_path: Path):
        assert load_router_config(tmp_path / "absent.yaml").enabled is False

    def test_reads_a_real_file(self, tmp_path: Path):
        path = tmp_path / "router.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "enabled": True,
                    "strategy": "every_turn",
                    "tiers": [
                        {"name": "a", "model": "m1", "criteria": "x"},
                        {"name": "b", "model": "m2", "criteria": "y"},
                    ],
                }
            )
        )
        assert load_router_config(path).strategy == "every_turn"


class TestRuleClassification:
    """The rule pass exists so the classifier's own cost is not charged
    against requests whose tier is obvious."""

    @pytest.mark.parametrize("text", ["hi", "hello", "thanks", "ok", "  yes  "])
    def test_greetings_go_cheap(self, text: str):
        decision = classify_by_rules(text, cfg())
        assert decision.tier.name == "fast"
        assert decision.method == "rule"

    @pytest.mark.parametrize(
        "text",
        [
            "refactor the auth module",
            "debug why the stream hangs",
            "what are the tradeoffs here",
            "migrate us off this library",
        ],
    )
    def test_complexity_markers_go_deep(self, text: str):
        assert classify_by_rules(text, cfg()).tier.name == "deep"

    def test_long_requests_go_deep(self):
        assert classify_by_rules("x" * 2500, cfg()).tier.name == "deep"

    def test_empty_request_goes_cheap(self):
        assert classify_by_rules("   ", cfg()).tier.name == "fast"

    def test_ordinary_request_defers_to_the_classifier(self):
        """Rules must not guess at the ambiguous middle."""
        assert classify_by_rules("summarise this ticket", cfg()) is None


class TestModelClassification:
    @pytest.mark.asyncio
    async def test_uses_the_tier_the_model_names(self):
        response = type("R", (), {"content": "balanced"})()
        model = AsyncMock()
        model.ainvoke = AsyncMock(return_value=response)

        with patch("deep_agent.src.agent.llm.create_model", return_value=model):
            decision = await classify_with_model("do a thing", cfg())

        assert decision.tier.name == "balanced"
        assert decision.method == "classifier"

    @pytest.mark.asyncio
    async def test_tolerates_chatty_output(self):
        """Small models add punctuation and preamble despite instruction."""
        response = type("R", (), {"content": "Tier: FAST."})()
        model = AsyncMock()
        model.ainvoke = AsyncMock(return_value=response)

        with patch("deep_agent.src.agent.llm.create_model", return_value=model):
            decision = await classify_with_model("do a thing", cfg())

        assert decision.tier.name == "fast"

    @pytest.mark.asyncio
    async def test_unrecognised_answer_yields_none(self):
        response = type("R", (), {"content": "I think maybe medium-ish?"})()
        model = AsyncMock()
        model.ainvoke = AsyncMock(return_value=response)

        with patch("deep_agent.src.agent.llm.create_model", return_value=model):
            assert await classify_with_model("x", cfg()) is None

    @pytest.mark.asyncio
    async def test_classifier_failure_yields_none(self):
        model = AsyncMock()
        model.ainvoke = AsyncMock(side_effect=RuntimeError("provider down"))

        with patch("deep_agent.src.agent.llm.create_model", return_value=model):
            assert await classify_with_model("x", cfg()) is None

    @pytest.mark.asyncio
    async def test_no_classifier_configured_yields_none(self):
        assert await classify_with_model("x", cfg(classifier_model="")) is None


class TestChooseTier:
    @pytest.mark.asyncio
    async def test_rules_win_before_any_model_call(self):
        model = AsyncMock()
        with patch("deep_agent.src.agent.llm.create_model", return_value=model):
            decision = await choose_tier("hello", cfg())

        assert decision.method == "rule"
        model.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_back_to_the_capable_tier_when_classification_fails(self):
        """Under-serving a hard request costs more than over-serving an easy one."""
        model = AsyncMock()
        model.ainvoke = AsyncMock(side_effect=RuntimeError("down"))

        with patch("deep_agent.src.agent.llm.create_model", return_value=model):
            decision = await choose_tier("summarise this ticket", cfg())

        assert decision.tier.name == "deep"
        assert decision.method == "fallback"


class TestMiddlewareConstruction:
    def test_disabled_config_builds_nothing(self):
        assert build_model_router_middleware(cfg(enabled=False)) is None

    def test_single_tier_builds_nothing(self):
        """Enabled, with one tier. There is nothing to route between.

        This was written with `enabled=False`, which made it a second
        copy of the test above: it passed with the `len(tiers) < 2`
        guard deleted, which is the only thing it is here to check.
        """
        config = RouterConfig(
            enabled=False, tiers=[{"name": "a", "model": "m", "criteria": "c"}]
        )
        # Past the model validator, which refuses this combination
        # outright. The builder's own guard is still the thing under
        # test: config can also arrive from a cached or hand-built
        # object that never went through validation.
        object.__setattr__(config, "enabled", True)
        assert build_model_router_middleware(config) is None

    def test_enabled_config_builds_middleware(self):
        with patch("deep_agent.src.agent.llm.create_model"):
            assert build_model_router_middleware(cfg()) is not None
