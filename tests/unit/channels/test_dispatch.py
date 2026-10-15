"""Channel dispatch: deduplication, thread mapping, reply extraction.

The recurring hazard here is a model or platform changing shape
underneath us — Slack resending an event, or Gemini returning typed
content blocks where a string used to be.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from deep_agent.src.channels import dispatch
from deep_agent.src.channels.adapters import InboundMessage
from deep_agent.src.channels.config import ChannelConfig


@pytest.fixture(autouse=True)
def _clear_seen():
    dispatch._SEEN_EVENTS.clear()
    yield
    dispatch._SEEN_EVENTS.clear()


def _message(**kwargs) -> InboundMessage:
    return InboundMessage(
        **{
            "sender": "U1",
            "text": "hello",
            "conversation": "C1:1700000000.1",
            **kwargs,
        }
    )


class TestAlreadyHandled:
    """Slack resends when the ack is slow, and acting twice on one
    message is worse than acting late."""

    @pytest.mark.asyncio
    async def test_first_delivery_is_not_a_duplicate(self):
        assert dispatch.already_handled("evt-1") is False

    @pytest.mark.asyncio
    async def test_second_delivery_of_the_same_event_is(self):
        dispatch.already_handled("evt-1")
        assert dispatch.already_handled("evt-1") is True

    @pytest.mark.asyncio
    async def test_an_empty_key_is_never_treated_as_seen(self):
        # No key means no way to tell, and refusing to act would drop
        # real messages.
        assert dispatch.already_handled("") is False
        assert dispatch.already_handled("") is False

    @pytest.mark.asyncio
    async def test_evicts_the_oldest_half_rather_than_growing_forever(self):
        for i in range(dispatch._SEEN_LIMIT):
            dispatch.already_handled(f"evt-{i}")
        dispatch.already_handled("one-more")
        assert len(dispatch._SEEN_EVENTS) <= dispatch._SEEN_LIMIT
        # The newest survives; that is the one a retry would repeat.
        assert dispatch.already_handled("one-more") is True


class TestThreadKey:
    def test_is_deterministic(self):
        assert dispatch.thread_key("C1:123") == dispatch.thread_key("C1:123")

    def test_differs_per_conversation(self):
        assert dispatch.thread_key("C1:123") != dispatch.thread_key("C1:124")

    def test_is_uuid_shaped(self):
        # The agent's thread ids are UUIDs, so the hash has to look
        # like one rather than needing a mapping table.
        key = dispatch.thread_key("C1:123")
        assert [len(part) for part in key.split("-")] == [8, 4, 4, 4, 12]


class TestContentText:
    def test_passes_a_plain_string_through(self):
        assert dispatch._content_text("hello") == "hello"

    def test_flattens_typed_blocks(self):
        # Gemini 3 returns a list of blocks; str() on that posts a
        # Python repr to Slack.
        content = [
            {"type": "text", "text": "first"},
            {"type": "thought_signature", "signature": "..."},
            {"type": "text", "text": "second"},
        ]
        assert dispatch._content_text(content) == "first\nsecond"

    def test_drops_blank_blocks(self):
        assert dispatch._content_text([{"text": "  "}, {"text": "kept"}]) == "kept"

    def test_empty_for_anything_else(self):
        assert dispatch._content_text(None) == ""
        assert dispatch._content_text(42) == ""


class TestReplyText:
    def test_takes_the_last_ai_message(self):
        payload = {
            "messages": [
                {"type": "ai", "content": "older"},
                {"type": "human", "content": "question"},
                {"type": "ai", "content": "newest"},
            ]
        }
        assert dispatch._reply_text(payload) == "newest"

    def test_skips_an_empty_ai_message(self):
        payload = {
            "messages": [
                {"type": "ai", "content": "real answer"},
                {"type": "ai", "content": ""},
            ]
        }
        assert dispatch._reply_text(payload) == "real answer"

    def test_surfaces_a_run_error_rather_than_going_silent(self):
        payload = {"messages": [], "__error__": {"message": "model unavailable"}}
        assert "model unavailable" in dispatch._reply_text(payload)

    def test_says_so_when_there_is_nothing_at_all(self):
        assert dispatch._reply_text({}) == "The agent produced no reply."
        assert dispatch._reply_text(None) == "The agent produced no reply."


class _Client:
    def __init__(self, get=None, post=None):
        self.get = get or AsyncMock()
        self.post = post or AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _response(status_code: int, body=None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = body or {}
    response.raise_for_status = MagicMock()
    return response


class TestEnsureThread:
    @pytest.mark.asyncio
    async def test_does_nothing_when_the_thread_exists(self):
        client = _Client(get=AsyncMock(return_value=_response(200)))
        await dispatch._ensure_thread(client, "t1")
        client.post.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_creates_the_thread_on_the_first_message(self):
        client = _Client(
            get=AsyncMock(return_value=_response(404)),
            post=AsyncMock(return_value=_response(201)),
        )
        await dispatch._ensure_thread(client, "t1")
        client.post.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_409_counts_as_success(self):
        # Another delivery of the same conversation won the race.
        created = _response(409)
        client = _Client(
            get=AsyncMock(return_value=_response(404)),
            post=AsyncMock(return_value=created),
        )
        await dispatch._ensure_thread(client, "t1")
        created.raise_for_status.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_on_a_real_failure(self):
        created = _response(500)
        created.raise_for_status.side_effect = httpx.HTTPError("boom")
        client = _Client(
            get=AsyncMock(return_value=_response(404)),
            post=AsyncMock(return_value=created),
        )
        with pytest.raises(httpx.HTTPError):
            await dispatch._ensure_thread(client, "t1")


class TestRunForChannel:
    @pytest.mark.asyncio
    async def test_attributes_the_run_to_the_mapped_user(self):
        client = _Client(
            get=AsyncMock(return_value=_response(200)),
            post=AsyncMock(
                return_value=_response(
                    200, {"messages": [{"type": "ai", "content": "hi"}]}
                )
            ),
        )
        channel = ChannelConfig(
            type="webhook", signing_secret_env="SECRET", assistant_id="agent"
        )
        with patch("httpx.AsyncClient", return_value=client) as factory:
            reply = await dispatch.run_for_channel(
                _message(), channel, "alice", service_token="tok"
            )

        assert reply == "hi"
        headers = factory.call_args.kwargs["headers"]
        assert headers["X-Channel-User-Id"] == "alice"
        assert headers["Authorization"] == "Bearer tok"


class TestDeliver:
    @pytest.mark.asyncio
    async def test_slack_goes_to_slack(self):
        channel = ChannelConfig(type="slack", signing_secret_env="S")
        with patch.object(dispatch, "_deliver_slack", new=AsyncMock()) as slack:
            await dispatch.deliver("answer", _message(), channel)
        slack.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_webhook_only_when_there_is_somewhere_to_reply(self):
        channel = ChannelConfig(type="webhook", signing_secret_env="S")
        with patch.object(dispatch, "_deliver_webhook", new=AsyncMock()) as hook:
            await dispatch.deliver("answer", _message(reply_to=None), channel)
        hook.assert_not_awaited()

        with patch.object(dispatch, "_deliver_webhook", new=AsyncMock()) as hook:
            await dispatch.deliver(
                "answer", _message(reply_to="https://example.test/cb"), channel
            )
        hook.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_slack_without_a_token_warns_rather_than_raising(self):
        with patch.dict("os.environ", {}, clear=True):
            await dispatch._deliver_slack("answer", _message())

    @pytest.mark.asyncio
    async def test_slack_failure_is_in_the_body_not_the_status(self):
        # Slack signals failure with HTTP 200 and ok: false.
        post = AsyncMock(return_value=_response(200, {"ok": False, "error": "no_perm"}))
        with (
            patch.dict("os.environ", {"SLACK_BOT_TOKEN": "xoxb"}),
            patch("httpx.AsyncClient", return_value=_Client(post=post)),
        ):
            await dispatch._deliver_slack("answer", _message())
        post.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_failing_callback_does_not_propagate(self):
        post = AsyncMock(side_effect=httpx.HTTPError("unreachable"))
        with patch("httpx.AsyncClient", return_value=_Client(post=post)):
            await dispatch._deliver_webhook(
                "answer", _message(reply_to="https://example.test/cb")
            )
