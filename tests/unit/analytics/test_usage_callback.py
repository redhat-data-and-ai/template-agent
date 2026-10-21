"""Token accounting for a completed LLM call.

The subtle part is *which* run id the tokens belong to: LangChain's
`run_id` keyword is this single call, while the row in `runs` is the
Aegra run pinned into `configurable.run_id` and copied into metadata.
Attributing to the wrong one silently loses every token.
"""

from __future__ import annotations

from types import SimpleNamespace

from deep_agent.src.token_budget.service import extract_cached_tokens
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, Generation, LLMResult

from deep_agent.src.analytics.usage_callback import (
    UsageRecorderCallbackHandler,
    _model_from_response,
    _run_id_from_metadata,
    _tokens_from_generations,
)


def _result(message: AIMessage | None = None, llm_output: dict | None = None):
    generations = [[ChatGeneration(message=message)]] if message else [[]]
    return LLMResult(generations=generations, llm_output=llm_output)


class TestRunIdFromMetadata:
    def test_reads_the_aegra_run_id(self):
        assert _run_id_from_metadata({"run_id": "abc"}) == "abc"

    def test_stringifies_a_non_string_id(self):
        assert _run_id_from_metadata({"run_id": 7}) == "7"

    @pytest.mark.parametrize("metadata", [None, {}, {"run_id": ""}, {"run_id": None}])
    def test_absent_means_nothing_to_attribute_to(self, metadata):
        assert _run_id_from_metadata(metadata) is None


class TestModelFromResponse:
    def test_prefers_the_generation_response_metadata(self):
        message = AIMessage(content="hi", response_metadata={"model_name": "gemini-x"})
        assert _model_from_response(_result(message)) == "gemini-x"

    def test_falls_back_to_llm_output_model_name(self):
        assert _model_from_response(_result(llm_output={"model_name": "claude-y"})) == (
            "claude-y"
        )

    def test_falls_back_to_llm_output_model(self):
        assert _model_from_response(_result(llm_output={"model": "gpt-z"})) == "gpt-z"

    def test_none_when_nothing_names_a_model(self):
        assert _model_from_response(_result()) is None

    def test_ignores_a_non_chat_generation(self):
        result = LLMResult(generations=[[Generation(text="plain")]], llm_output=None)
        assert _model_from_response(result) is None


class TestTokensFromGenerations:
    def test_sums_usage_across_generations(self):
        message = AIMessage(
            content="hi",
            usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
        )
        assert _tokens_from_generations(_result(message)) == (10, 4)

    def test_zero_when_the_model_reported_nothing(self):
        assert _tokens_from_generations(_result(AIMessage(content="hi"))) == (0, 0)


class TestOnLlmEnd:
    @pytest.mark.asyncio
    async def test_records_against_the_aegra_run(self):
        message = AIMessage(
            content="hi",
            response_metadata={"model_name": "gemini-x"},
            usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        )
        record = AsyncMock()
        with patch(
            "deep_agent.src.analytics.usage_callback.record_llm_usage", new=record
        ):
            await UsageRecorderCallbackHandler().on_llm_end(
                _result(message), run_id="langchain-call-id", metadata={"run_id": "r1"}
            )

        # "r1", not "langchain-call-id": the latter matches no row.
        record.assert_awaited_once_with("r1", "gemini-x", 3, 2, 0)

    @pytest.mark.asyncio
    async def test_skips_a_call_with_no_run_to_attribute_to(self):
        # Evals and warmups run outside a request. Not an error.
        record = AsyncMock()
        with patch(
            "deep_agent.src.analytics.usage_callback.record_llm_usage", new=record
        ):
            await UsageRecorderCallbackHandler().on_llm_end(_result(), metadata={})
        record.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_records_zeroes_rather_than_dropping_the_call(self):
        # A row with no tokens still says the call happened.
        record = AsyncMock()
        with patch(
            "deep_agent.src.analytics.usage_callback.record_llm_usage", new=record
        ):
            await UsageRecorderCallbackHandler().on_llm_end(
                _result(AIMessage(content="hi")), metadata={"run_id": "r2"}
            )
        record.assert_awaited_once_with("r2", None, 0, 0, 0)


class TestRunIdFromConfig:
    """Where the Aegra run id actually comes from.

    The previous version read `metadata["run_id"]`, on the strength
    of LangChain copying scalar configurable values into callback
    metadata. It does not — that copy feeds the LangSmith tracer —
    and `on_llm_end` is passed no metadata at all, so the lookup
    returned None for every call and every token was dropped.
    """

    def test_reads_the_run_id_from_the_active_config(self):
        from deep_agent.src.analytics.usage_callback import _run_id_from_config

        with patch(
            "langgraph.config.get_config",
            return_value={"configurable": {"run_id": "r1", "thread_id": "t1"}},
        ):
            assert _run_id_from_config() == "r1"

    def test_none_outside_a_run(self):
        from deep_agent.src.analytics.usage_callback import _run_id_from_config

        with patch("langgraph.config.get_config", return_value={}):
            assert _run_id_from_config() is None

    def test_none_when_there_is_no_config_at_all(self):
        from deep_agent.src.analytics.usage_callback import _run_id_from_config

        with patch("langgraph.config.get_config", side_effect=RuntimeError("no run")):
            assert _run_id_from_config() is None

    @pytest.mark.asyncio
    async def test_records_with_no_metadata_at_all(self):
        # The shape every real call has: on_llm_end receives none.
        message = AIMessage(
            content="hi",
            response_metadata={"model_name": "gemini-3.8-flash"},
            usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        )
        record = AsyncMock()
        with (
            patch(
                "langgraph.config.get_config",
                return_value={"configurable": {"run_id": "r9"}},
            ),
            patch(
                "deep_agent.src.analytics.usage_callback.record_llm_usage", new=record
            ),
        ):
            await UsageRecorderCallbackHandler().on_llm_end(_result(message))

        record.assert_awaited_once_with("r9", "gemini-3.8-flash", 7, 3, 0)

    @pytest.mark.asyncio
    async def test_metadata_still_works_as_a_fallback(self):
        record = AsyncMock()
        with (
            patch("langgraph.config.get_config", return_value={}),
            patch(
                "deep_agent.src.analytics.usage_callback.record_llm_usage", new=record
            ),
        ):
            await UsageRecorderCallbackHandler().on_llm_end(
                _result(), metadata={"run_id": "r2"}
            )
        record.assert_awaited_once()


# ── cached prompt tokens ──────────────────────────────────────────────


class TestExtractCachedTokens:
    """Cached input tokens are the only read on whether caching works.

    They are a *subset* of input_tokens — billed at a discount, not
    free — so a hit and a miss record the same input count. Without
    this figure the prompt-prefix fix is unmeasurable, which is how
    a per-second timestamp sat in the cacheable prefix unnoticed.
    """

    @staticmethod
    def _result(usage):
        message = SimpleNamespace(usage_metadata=usage, response_metadata={})
        generation = SimpleNamespace(message=message, generation_info=None)
        return SimpleNamespace(generations=[[generation]], llm_output=None)

    def test_reads_langchain_normalised_details(self):
        result = self._result(
            {"input_tokens": 1000, "input_token_details": {"cache_read": 900}}
        )
        assert extract_cached_tokens(result) == 900

    def test_reads_gemini_field_name(self):
        result = self._result({"input_tokens": 1000, "cached_content_token_count": 850})
        assert extract_cached_tokens(result) == 850

    def test_reads_anthropic_field_name(self):
        result = self._result({"input_tokens": 1000, "cache_read_input_tokens": 700})
        assert extract_cached_tokens(result) == 700

    def test_zero_when_the_provider_says_nothing(self):
        """Indistinguishable from a real miss, and reported as such."""
        assert extract_cached_tokens(self._result({"input_tokens": 1000})) == 0

    def test_zero_on_a_result_with_no_usage_at_all(self):
        empty = SimpleNamespace(generations=[], llm_output=None)
        assert extract_cached_tokens(empty) == 0

    def test_sums_across_generations(self):
        def gen(cached):
            msg = SimpleNamespace(
                usage_metadata={"input_tokens": 10, "cache_read_input_tokens": cached},
                response_metadata={},
            )
            return SimpleNamespace(message=msg, generation_info=None)

        result = SimpleNamespace(generations=[[gen(5), gen(7)]], llm_output=None)
        assert extract_cached_tokens(result) == 12

    def test_falls_back_to_llm_output(self):
        result = SimpleNamespace(
            generations=[],
            llm_output={"token_usage": {"cached_tokens": 42}},
        )
        assert extract_cached_tokens(result) == 42
