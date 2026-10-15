"""The share snapshot — what leaves the building.

`build_snapshot` is an allowlist on purpose: thread state grows new
keys as middleware is added, and a denylist would publish each new
one by default. These tests are that allowlist, written down.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from deep_agent.src.sharing.store import (
    MAX_MESSAGE_CHARS,
    MAX_SNAPSHOT_MESSAGES,
    Share,
    build_snapshot,
)


def _share(**kwargs) -> Share:
    base = {
        "share_id": "s1",
        "thread_id": "t1",
        "owner_id": "alice",
        "title": "A chat",
        "scope": "authenticated",
        "created_at": datetime.now(UTC),
        "expires_at": None,
        "revoked_at": None,
        "views": 0,
        "last_viewed_at": None,
    }
    return Share(**{**base, **kwargs})


class TestShareActive:
    def test_a_fresh_share_is_active(self):
        assert _share().active is True

    def test_a_revoked_share_is_not(self):
        assert _share(revoked_at=datetime.now(UTC)).active is False

    def test_an_expired_share_is_not(self):
        past = datetime.now(UTC) - timedelta(days=1)
        assert _share(expires_at=past).active is False

    def test_a_future_expiry_is_still_active(self):
        future = datetime.now(UTC) + timedelta(days=1)
        assert _share(expires_at=future).active is True

    def test_revocation_beats_a_future_expiry(self):
        future = datetime.now(UTC) + timedelta(days=1)
        assert _share(expires_at=future, revoked_at=datetime.now(UTC)).active is False

    def test_as_dict_is_json_serialisable(self):
        body = _share().as_dict()
        assert body["share_id"] == "s1"
        assert body["title"] == "A chat"


class TestBuildSnapshot:
    def test_keeps_the_conversation(self):
        values = {
            "messages": [
                {"type": "human", "content": "question"},
                {"type": "ai", "content": "answer"},
            ]
        }
        snapshot = build_snapshot(values, "Title")
        assert [m["role"] for m in snapshot["messages"]] == ["user", "assistant"]
        assert snapshot["title"] == "Title"

    def test_normalises_role_names(self):
        values = {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ]
        }
        assert [m["role"] for m in build_snapshot(values, "T")["messages"]] == [
            "user",
            "assistant",
        ]

    def test_drops_tool_and_system_messages(self):
        # These carry retrieved documents, file contents and prompt
        # internals. None of that is what somebody meant to share.
        values = {
            "messages": [
                {"type": "system", "content": "secret prompt"},
                {"type": "tool", "content": "contents of /documents/salaries.csv"},
                {"type": "human", "content": "safe"},
            ]
        }
        snapshot = build_snapshot(values, "T")
        assert len(snapshot["messages"]) == 1
        assert snapshot["messages"][0]["content"] == "safe"

    def test_keeps_text_blocks_and_drops_binary_ones(self):
        # Image and audio blocks are the attachment bytes by another
        # name.
        values = {
            "messages": [
                {
                    "type": "human",
                    "content": [
                        {"type": "text", "text": "look at this"},
                        {"type": "image_url", "image_url": {"url": "data:..."}},
                    ],
                }
            ]
        }
        snapshot = build_snapshot(values, "T")
        assert snapshot["messages"][0]["content"] == "look at this"

    def test_skips_empty_messages(self):
        values = {"messages": [{"type": "ai", "content": "   "}]}
        assert build_snapshot(values, "T")["messages"] == []

    def test_skips_entries_that_are_not_mappings(self):
        values = {"messages": ["not a message", {"type": "ai", "content": "kept"}]}
        assert len(build_snapshot(values, "T")["messages"]) == 1

    def test_truncates_a_very_long_message(self):
        values = {
            "messages": [{"type": "ai", "content": "x" * (MAX_MESSAGE_CHARS + 50)}]
        }
        assert len(build_snapshot(values, "T")["messages"][0]["content"]) == (
            MAX_MESSAGE_CHARS
        )

    def test_caps_the_message_count(self):
        values = {
            "messages": [
                {"type": "ai", "content": f"m{i}"}
                for i in range(MAX_SNAPSHOT_MESSAGES + 20)
            ]
        }
        assert len(build_snapshot(values, "T")["messages"]) == MAX_SNAPSHOT_MESSAGES

    def test_names_attachments_without_including_them(self):
        # A reader can see the conversation was grounded in a document
        # without the document travelling with it.
        values = {
            "messages": [],
            "files": {"/uploads/report.pdf": "…bytes…", "/uploads/a.csv": "…"},
        }
        snapshot = build_snapshot(values, "T")
        assert snapshot["attachment_names"] == ["a.csv", "report.pdf"]
        assert "bytes" not in str(snapshot)

    def test_tolerates_empty_state(self):
        snapshot = build_snapshot({}, "T")
        assert snapshot["messages"] == []
        assert snapshot["attachment_names"] == []
        assert snapshot["captured_at"]

    def test_does_not_carry_unknown_state_keys(self):
        # The allowlist: a new middleware key must not publish itself.
        values = {"messages": [], "internal_scratchpad": "secret"}
        assert "internal_scratchpad" not in build_snapshot(values, "T")
