"""Payload normalisation, and the loops it has to avoid."""

from __future__ import annotations

import pytest

from deep_agent.src.channels.adapters import (
    AdapterError,
    parse,
    parse_slack,
    parse_webhook,
)


def slack_event(**event) -> dict:
    return {
        "type": "event_callback",
        "team_id": "T1",
        "event_id": "Ev1",
        "event": {"type": "app_mention", "user": "U1", "text": "hi", **event},
    }


class TestSlack:
    def test_an_app_mention_becomes_a_message(self):
        message = parse_slack(slack_event(channel="C1", ts="1.0"))
        assert message is not None
        assert message.sender == "U1"
        assert message.text == "hi"

    def test_a_bot_post_is_ignored(self):
        """Answering one is how an agent talks to itself forever."""
        assert parse_slack(slack_event(bot_id="B1")) is None

    def test_a_legacy_bot_message_is_ignored(self):
        assert parse_slack(slack_event(subtype="bot_message")) is None

    def test_an_empty_message_is_ignored(self):
        assert parse_slack(slack_event(text="   ")) is None

    def test_an_unrelated_event_type_is_ignored(self):
        assert parse_slack(slack_event(type="reaction_added")) is None

    def test_a_non_event_payload_is_an_error(self):
        with pytest.raises(AdapterError):
            parse_slack({"type": "something_else"})

    def test_replies_land_in_the_thread_that_mentioned_us(self):
        message = parse_slack(slack_event(channel="C1", ts="2.0", thread_ts="1.0"))
        assert message.metadata["thread_ts"] == "1.0"

    def test_a_top_level_mention_starts_a_thread_at_its_own_ts(self):
        message = parse_slack(slack_event(channel="C1", ts="2.0"))
        assert message.metadata["thread_ts"] == "2.0"

    def test_one_slack_thread_is_one_conversation(self):
        first = parse_slack(slack_event(channel="C1", ts="1.0"))
        second = parse_slack(slack_event(channel="C1", ts="9.9", thread_ts="1.0"))
        assert first.conversation == second.conversation

    def test_different_threads_are_different_conversations(self):
        first = parse_slack(slack_event(channel="C1", ts="1.0"))
        other = parse_slack(slack_event(channel="C1", ts="2.0"))
        assert first.conversation != other.conversation

    def test_a_missing_user_is_an_error(self):
        payload = slack_event(channel="C1", ts="1.0")
        del payload["event"]["user"]
        with pytest.raises(AdapterError):
            parse_slack(payload)


class TestWebhook:
    def test_a_minimal_payload_parses(self):
        message = parse_webhook({"sender": "ci", "text": "run it"})
        assert message.sender == "ci"
        assert message.text == "run it"

    def test_a_missing_field_is_an_error(self):
        with pytest.raises(AdapterError):
            parse_webhook({"text": "no sender"})

    def test_empty_text_is_an_error(self):
        with pytest.raises(AdapterError):
            parse_webhook({"sender": "ci", "text": "  "})

    def test_a_stable_conversation_key_is_kept(self):
        message = parse_webhook(
            {"sender": "ci", "text": "x", "conversation": "build-1"}
        )
        assert message.conversation == "build-1"

    def test_without_one_the_sender_scopes_the_conversation(self):
        message = parse_webhook({"sender": "ci", "text": "x"})
        assert message.conversation == "webhook:ci"


class TestDispatchByType:
    def test_an_unknown_type_is_an_error(self):
        with pytest.raises(AdapterError):
            parse("carrier-pigeon", {})


class TestContentFlattening:
    """Gemini 3 returns content blocks, not a string.

    `str()` on a block list posts a Python repr to Slack — which is how
    a model upgrade becomes a user-visible mess rather than an error.
    """

    def test_a_plain_string_passes_through(self):
        from deep_agent.src.channels.dispatch import _content_text

        assert _content_text("hello") == "hello"

    def test_text_blocks_are_joined(self):
        from deep_agent.src.channels.dispatch import _content_text

        content = [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
        assert _content_text(content) == "first\nsecond"

    def test_non_text_blocks_are_dropped(self):
        """Thought signatures are binary and must never reach a channel."""
        from deep_agent.src.channels.dispatch import _content_text

        content = [
            {"type": "text", "text": "visible"},
            {"extras": {"signature": "AY89a1/Uwk..."}},
        ]
        assert _content_text(content) == "visible"

    def test_an_unexpected_shape_yields_nothing_rather_than_a_repr(self):
        from deep_agent.src.channels.dispatch import _content_text

        assert _content_text({"unexpected": "dict"}) == ""
