"""Reasoning models must leave room to actually answer.

Found against real Vertex AI: `gemini-2.5-pro` with the shipped
`max_output_tokens: 8192` and no thinking budget returns zero content
chunks and no error — the run succeeds with an empty message. Since
2.5-pro is the model the shipped PROMPT.md names, that was the
out-of-the-box experience with a real provider.
"""

from __future__ import annotations

import pytest

from deep_agent.src.settings import settings
from deep_agent.src.agent.llm import (
    _thinking_budget_for,
)


class TestWhoGetsABudget:
    @pytest.mark.parametrize("model", ["gemini-2.5-pro", "gemini-3.1-pro-preview"])
    def test_reasoning_models_get_one(self, model):
        assert _thinking_budget_for(model, 8192) is not None

    @pytest.mark.parametrize(
        "model", ["gemini-2.5-flash", "claude-sonnet-4", "mock-model"]
    )
    def test_other_models_are_left_alone(self, model):
        """Passing a budget to a model that does not think is noise."""
        assert _thinking_budget_for(model, 8192) is None


class TestTheBudgetLeavesRoom:
    def test_reasoning_cannot_consume_the_whole_allowance(self):
        allowance = 8192
        budget = _thinking_budget_for("gemini-2.5-pro", allowance)
        assert budget < allowance, "no tokens left to answer with"

    def test_it_is_a_fraction_of_the_allowance(self):
        assert _thinking_budget_for("gemini-2.5-pro", 8192) == int(
            8192 * settings.THINKING_BUDGET_SHARE
        )

    def test_a_small_allowance_is_raised_toward_a_usable_floor(self):
        """A 25% share of a small number is not enough to think with.

        Capped at half the allowance, so the floor can never starve the
        answer — see TestTheFloorCannotStarveTheAnswer.
        """
        # 25% of 1000 is 250, below the floor, so it is raised — but
        # only as far as half the allowance allows.
        assert _thinking_budget_for("gemini-2.5-pro", 1000) == 500
        # 25% of 3000 is 750, above the floor, so the share wins.
        assert _thinking_budget_for("gemini-2.5-pro", 3000) == 750

    def test_no_allowance_means_no_budget_rather_than_a_guess(self):
        assert _thinking_budget_for("gemini-2.5-pro", None) is None
        assert _thinking_budget_for("gemini-2.5-pro", 0) is None


class TestTheFloorCannotStarveTheAnswer:
    """The minimum budget must not exceed the allowance it comes out of.

    Caught against a real `gemini-3-flash-preview` call with
    `max_output_tokens=128`: the 512-token floor left nothing to answer
    with and the model returned two chunks and zero characters — the
    same empty-reply failure this function exists to prevent, reached
    from the other side.
    """

    @pytest.mark.parametrize("allowance", [64, 128, 256, 512, 1024, 8192])
    def test_at_least_half_the_allowance_is_left_for_output(self, allowance):
        budget = _thinking_budget_for("gemini-2.5-pro", allowance)
        assert budget is not None
        assert budget <= allowance // 2, (
            f"budget {budget} of {allowance} leaves too little to answer with"
        )

    def test_a_tiny_allowance_scales_down_rather_than_using_the_floor(self):
        assert _thinking_budget_for("gemini-2.5-pro", 128) == 64
