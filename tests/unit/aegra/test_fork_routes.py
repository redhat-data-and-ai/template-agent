"""Fork helpers: which messages get copied, and what is stripped."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from deep_agent.aegra.fork_routes import (
    _forked_title,
    _latest_messages,
    _messages_at,
    _strip_for_replay,
)


def _entry(checkpoint_id: str, messages: list[dict]) -> dict:
    return {"checkpoint_id": checkpoint_id, "values": {"messages": messages}}


class TestMessagesAt:
    def test_returns_the_messages_at_that_checkpoint(self):
        history = [_entry("c2", [{"type": "human"}]), _entry("c1", [])]
        assert _messages_at(history, "c2") == [{"type": "human"}]

    def test_404_for_a_checkpoint_in_another_thread(self):
        with pytest.raises(HTTPException) as exc:
            _messages_at([_entry("c1", [])], "nope")
        assert exc.value.status_code == 404

    def test_tolerates_an_entry_without_values(self):
        assert _messages_at([{"checkpoint_id": "c1"}], "c1") == []


class TestLatestMessages:
    def test_picks_the_longest_list(self):
        # History is newest-first, but "newest" and "longest" can come
        # apart; the full conversation is the longest one.
        history = [_entry("c1", [{"a": 1}]), _entry("c2", [{"a": 1}, {"b": 2}])]
        assert len(_latest_messages(history)) == 2

    def test_empty_history_gives_nothing(self):
        assert _latest_messages([]) == []


class TestStripForReplay:
    def test_drops_ids_that_would_collide_in_a_new_thread(self):
        # Re-using ids makes LangGraph treat seeded messages as updates
        # to messages that do not exist there, and they vanish.
        out = _strip_for_replay([{"type": "human", "id": "m1", "run_id": "r1"}])
        assert out == [{"type": "human"}]

    def test_keeps_only_replayable_message_types(self):
        messages = [
            {"type": "human"},
            {"type": "ai"},
            {"type": "system"},
            {"type": "tool"},
            {"type": "something-else"},
        ]
        assert [m["type"] for m in _strip_for_replay(messages)] == [
            "human",
            "ai",
            "system",
            "tool",
        ]

    def test_skips_entries_that_are_not_mappings(self):
        assert _strip_for_replay(["not a message", None, {"type": "ai"}]) == [
            {"type": "ai"}
        ]


class TestForkedTitle:
    def test_marks_the_copy(self):
        assert _forked_title({"thread_name": "Incident 42"}) == "Incident 42 (fork)"

    def test_names_an_untitled_thread(self):
        assert _forked_title({}) == "Forked conversation"
        assert _forked_title({"thread_name": "   "}) == "Forked conversation"

    def test_truncates_rather_than_overflowing_the_column(self):
        assert len(_forked_title({"thread_name": "x" * 500})) == 200


from unittest.mock import AsyncMock, MagicMock, patch  # noqa: E402

import deep_agent.aegra.fork_routes as fr  # noqa: E402
from deep_agent.aegra.fork_routes import ForkRequest  # noqa: E402


def _request(auth: str | None = None) -> MagicMock:
    request = MagicMock()
    request.headers = {"Authorization": auth} if auth else {}
    return request


def _response(status_code: int, body=None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = body if body is not None else {}
    return response


class _Client:
    def __init__(self, responses: dict):
        self._responses = responses

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, path):
        return self._responses[path]

    async def post(self, path, **kwargs):
        return self._responses.get(("POST", path), _response(200, {}))


class TestClientCredentials:
    def test_forwards_the_callers_authorization(self):
        # Borrowing a service identity here would let anyone fork any
        # thread.
        client = fr._client(_request("Bearer tok"))
        assert client.headers["authorization"] == "Bearer tok"

    def test_omits_the_header_when_there_is_none(self):
        client = fr._client(_request())
        assert "authorization" not in client.headers


class TestForkThread:
    @pytest.mark.asyncio
    async def test_400_without_a_fork_point(self):
        with (
            patch.object(fr, "authenticated_user_id", new=AsyncMock(return_value="u1")),
            pytest.raises(HTTPException) as exc,
        ):
            await fr.fork_thread("t1", ForkRequest(), _request())
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_404_for_a_thread_the_caller_cannot_see(self):
        # Reading history through the API is also the ownership check.
        client = _Client({"/threads/t1/history": _response(404)})
        with (
            patch.object(fr, "authenticated_user_id", new=AsyncMock(return_value="u1")),
            patch.object(fr, "_client", return_value=client),
            pytest.raises(HTTPException) as exc,
        ):
            await fr.fork_thread("t1", ForkRequest(message_index=1), _request())
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_502_when_history_cannot_be_read(self):
        client = _Client({"/threads/t1/history": _response(500)})
        with (
            patch.object(fr, "authenticated_user_id", new=AsyncMock(return_value="u1")),
            patch.object(fr, "_client", return_value=client),
            pytest.raises(HTTPException) as exc,
        ):
            await fr.fork_thread("t1", ForkRequest(message_index=1), _request())
        assert exc.value.status_code == 502

    @pytest.mark.asyncio
    async def test_400_for_an_empty_history(self):
        client = _Client({"/threads/t1/history": _response(200, [])})
        with (
            patch.object(fr, "authenticated_user_id", new=AsyncMock(return_value="u1")),
            patch.object(fr, "_client", return_value=client),
            pytest.raises(HTTPException) as exc,
        ):
            await fr.fork_thread("t1", ForkRequest(message_index=1), _request())
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_400_when_the_fork_point_leaves_nothing(self):
        history = [_entry("c1", [{"type": "human", "content": "hi"}])]
        client = _Client({"/threads/t1/history": _response(200, history)})
        with (
            patch.object(fr, "authenticated_user_id", new=AsyncMock(return_value="u1")),
            patch.object(fr, "_client", return_value=client),
            pytest.raises(HTTPException) as exc,
        ):
            await fr.fork_thread("t1", ForkRequest(message_index=0), _request())
        assert exc.value.status_code == 400
