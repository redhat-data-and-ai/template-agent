"""Forking a conversation at an earlier turn."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from deep_agent.aegra.fork_routes import (
    _forked_title,
    _latest_messages,
    _messages_at,
    _strip_for_replay,
)


def entry(checkpoint_id: str, *contents: str) -> dict:
    return {
        "checkpoint_id": checkpoint_id,
        "values": {
            "messages": [
                {"type": "human", "content": c, "id": f"m{i}"}
                for i, c in enumerate(contents)
            ]
        },
    }


class TestStripForReplay:
    def test_message_ids_are_dropped(self):
        """Re-used ids make LangGraph treat seeded messages as updates
        to messages that do not exist, and they silently vanish."""
        out = _strip_for_replay([{"type": "human", "content": "x", "id": "m1"}])
        assert "id" not in out[0]

    def test_run_ids_are_dropped(self):
        out = _strip_for_replay([{"type": "ai", "content": "x", "run_id": "r1"}])
        assert "run_id" not in out[0]

    def test_content_survives(self):
        out = _strip_for_replay([{"type": "human", "content": "keep me"}])
        assert out[0]["content"] == "keep me"

    def test_unknown_message_types_are_dropped(self):
        assert _strip_for_replay([{"type": "mystery", "content": "x"}]) == []

    def test_non_dict_entries_are_skipped(self):
        assert _strip_for_replay(["not a message", None]) == []

    def test_the_four_real_types_are_kept(self):
        messages = [
            {"type": t, "content": "x"} for t in ("human", "ai", "system", "tool")
        ]
        assert len(_strip_for_replay(messages)) == 4


class TestCutPoint:
    def test_a_checkpoint_selects_its_own_message_list(self):
        history = [entry("c2", "a", "b"), entry("c1", "a")]
        assert len(_messages_at(history, "c1")) == 1

    def test_an_unknown_checkpoint_is_404(self):
        with pytest.raises(HTTPException) as exc:
            _messages_at([entry("c1", "a")], "nope")
        assert exc.value.status_code == 404

    def test_the_latest_state_is_the_longest_message_list(self):
        """History is newest-first, but entries are not all full states."""
        history = [entry("c3", "a"), entry("c2", "a", "b", "c"), entry("c1", "a", "b")]
        assert len(_latest_messages(history)) == 3

    def test_empty_history_yields_nothing(self):
        assert _latest_messages([]) == []


class TestTitle:
    def test_a_fork_is_named_after_its_source(self):
        assert _forked_title({"thread_name": "Deploy plan"}) == "Deploy plan (fork)"

    def test_an_unnamed_source_gets_a_generic_title(self):
        assert _forked_title({}) == "Forked conversation"

    def test_a_blank_name_counts_as_unnamed(self):
        assert _forked_title({"thread_name": "   "}) == "Forked conversation"

    def test_titles_are_bounded(self):
        assert len(_forked_title({"thread_name": "z" * 400})) == 200
