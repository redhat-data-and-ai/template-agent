"""Per-message provenance.

The question it answers is "why does this answer look like this" —
which model, cut short or not, routed or fallen back. All of it was
invisible in the transcript before.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from deep_agent.aegra.provenance import describe_message


def _msg(meta=None, extra=None, content="hello", tool_calls=None):
    return SimpleNamespace(
        response_metadata=meta or {},
        additional_kwargs=extra or {},
        content=content,
        tool_calls=tool_calls,
        type="ai",
    )


class TestModel:
    def test_reports_the_model_that_answered(self):
        assert describe_message(_msg({"model_name": "gemini-3.8-flash"}))["model"] == (
            "gemini-3.8-flash"
        )

    @pytest.mark.parametrize("key", ["model_name", "model", "model_version"])
    def test_accepts_the_spellings_providers_use(self, key):
        assert describe_message(_msg({key: "m"}))["model"] == "m"

    def test_absent_rather_than_unknown(self):
        # The UI renders the line only when it says something; a
        # block of nulls is noise on every turn.
        assert "model" not in describe_message(_msg({}))


class TestFinishReason:
    def test_an_ordinary_stop_is_still_reported(self):
        # "Stopped normally" is the answer to "was this cut off".
        out = describe_message(_msg({"finish_reason": "STOP"}))
        assert out["finish_reason"] == "STOP"
        assert "truncated" not in out

    @pytest.mark.parametrize(
        "reason", ["MAX_TOKENS", "max_tokens", "length", "max-output-tokens"]
    )
    def test_flags_the_provider_spellings_of_running_out_of_room(self, reason):
        assert describe_message(_msg({"finish_reason": reason}))["truncated"] is True

    def test_reads_camel_case_and_stop_reason(self):
        assert (
            describe_message(_msg({"finishReason": "STOP"}))["finish_reason"] == "STOP"
        )
        assert describe_message(_msg({"stop_reason": "end_turn"}))["finish_reason"] == (
            "end_turn"
        )


class TestMiddlewareAnnotations:
    def test_reports_the_tier_routing_chose(self):
        out = describe_message(_msg(extra={"routed_tier": "fast", "routed_by": "rule"}))
        assert out["routed_tier"] == "fast"
        assert out["routed_by"] == "rule"

    def test_reports_a_fallback(self):
        out = describe_message(_msg(extra={"fallback_from": "gemini-3.8-flash"}))
        assert out["fallback_from"] == "gemini-3.8-flash"

    def test_absent_means_the_feature_did_not_fire(self):
        out = describe_message(_msg(extra={"routed_tier": None, "cached": False}))
        assert "routed_tier" not in out
        assert "cached" not in out


class TestEmptyReply:
    def test_names_the_silent_failure(self):
        # A model that returns nothing and no error reads as having
        # had nothing to say. One boolean says otherwise.
        assert describe_message(_msg(content=""))["empty"] is True

    def test_a_tool_call_with_no_text_is_not_empty(self):
        out = describe_message(_msg(content="", tool_calls=[{"name": "search"}]))
        assert "empty" not in out

    def test_nothing_to_say_is_an_empty_dict(self):
        assert describe_message(_msg(content="hi")) == {}


class TestShape:
    def test_carries_no_endpoints_or_credentials(self):
        out = describe_message(
            _msg({"model_name": "m", "base_url": "https://internal.example/v1"})
        )
        assert "base_url" not in out
        assert "https" not in str(out)

    def test_tolerates_junk_metadata(self):
        assert (
            describe_message(
                SimpleNamespace(
                    response_metadata="not a dict",
                    additional_kwargs=None,
                    content="hi",
                    tool_calls=None,
                )
            )
            == {}
        )


class TestPiiAnnotation:
    def test_reports_that_a_substitution_happened(self):
        # The restored text reads exactly like the original, so
        # without this the fact that the model saw something else is
        # unrecoverable.
        out = describe_message(_msg(extra={"pii_scrubbed": 3}))
        assert out["pii_scrubbed"] == 3

    def test_absent_when_nothing_was_scrubbed(self):
        assert "pii_scrubbed" not in describe_message(_msg(extra={}))
