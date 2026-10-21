"""The /safeguards route.

PII scrubbing, guardrails and the token budget all change what the
model sees or what comes back, and all of them are silent in the
transcript. This route is the only place a reader can find out.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import deep_agent.aegra.skill_routes as sr


def _config(pii=None, guardrail=None, budget=None):
    cfg = MagicMock()
    cfg.get_custom_pii_config.return_value = pii
    cfg.get_guardrails_config.return_value = guardrail
    cfg.get_token_budget_config.return_value = budget
    return cfg


def _pii(enabled=True, names=("email", "credit_card"), strategy="hash"):
    return SimpleNamespace(
        enabled=enabled,
        trace_strategy=strategy,
        rules=[SimpleNamespace(name=n) for n in names],
    )


class TestSafeguards:
    @pytest.mark.asyncio
    async def test_reports_pii_state_and_what_it_matches(self):
        with patch("deep_agent.src.agent.config.agent_config", _config(pii=_pii())):
            body = await sr.list_safeguards()

        assert body["pii"]["enabled"] is True
        assert body["pii"]["entities"] == ["credit_card", "email"]
        assert body["pii"]["trace_strategy"] == "hash"

    @pytest.mark.asyncio
    async def test_reports_what_is_matched_never_how(self):
        # A redaction pattern is a map of what evades it.
        rules = [SimpleNamespace(name="email", regex=r"\\S+@\\S+", label="EMAIL")]
        pii = SimpleNamespace(enabled=True, trace_strategy="hash", rules=rules)
        with patch("deep_agent.src.agent.config.agent_config", _config(pii=pii)):
            body = await sr.list_safeguards()

        assert body["pii"]["entities"] == ["email"]
        assert "regex" not in str(body)
        assert "@" not in str(body)

    @pytest.mark.asyncio
    async def test_guardrail_reports_only_the_switch(self):
        # Not whether a model is named: the loader drops `model` when
        # guardrails are off, so any such flag reports "none" for a
        # config that has one.
        guardrail = SimpleNamespace(enabled=False, model=None)
        with patch(
            "deep_agent.src.agent.config.agent_config", _config(guardrail=guardrail)
        ):
            body = await sr.list_safeguards()

        assert body["guardrail"] == {"enabled": False}

    @pytest.mark.asyncio
    async def test_reports_the_token_budget_switch(self):
        budget = SimpleNamespace(enabled=True)
        with patch("deep_agent.src.agent.config.agent_config", _config(budget=budget)):
            body = await sr.list_safeguards()
        assert body["token_budget"]["enabled"] is True

    @pytest.mark.asyncio
    async def test_a_broken_section_reports_off_rather_than_failing(self):
        cfg = MagicMock()
        cfg.get_custom_pii_config.side_effect = RuntimeError("bad yaml")
        cfg.get_guardrails_config.side_effect = RuntimeError("bad yaml")
        cfg.get_token_budget_config.side_effect = RuntimeError("bad yaml")
        with patch("deep_agent.src.agent.config.agent_config", cfg):
            body = await sr.list_safeguards()

        assert body["pii"]["enabled"] is False
        assert body["pii"]["entities"] == []
        assert body["guardrail"]["enabled"] is False
