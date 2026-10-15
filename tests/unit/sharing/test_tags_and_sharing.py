"""Tags and shared conversations.

The SQL runs against a live database in the cluster checks; what is
pinned here is the normalisation and, more importantly, what a
snapshot refuses to carry.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from deep_agent.aegra.tag_routes import (
    MAX_TAG_LENGTH,
    MAX_TAGS_PER_THREAD,
    normalise_tag,
    normalise_tags,
)
from deep_agent.src.sharing.store import (
    MAX_MESSAGE_CHARS,
    Share,
    build_snapshot,
)


class TestTagNormalisation:
    def test_case_is_folded(self):
        # `Billing` and `billing` are the same tag; a UI showing both
        # is a UI that lost half the conversations.
        assert normalise_tag("Billing") == "billing"

    def test_surrounding_and_inner_whitespace_collapses(self):
        assert normalise_tag("  q3   planning ") == "q3 planning"

    def test_duplicates_collapse_after_folding(self):
        assert normalise_tags(["Billing", "billing", " BILLING "]) == ["billing"]

    def test_tags_come_back_sorted(self):
        # Stable order, so a tag list does not reshuffle between saves.
        assert normalise_tags(["zeta", "alpha"]) == ["alpha", "zeta"]

    def test_an_empty_tag_is_rejected(self):
        with pytest.raises(ValueError):
            normalise_tag("   ")

    def test_an_overlong_tag_is_rejected(self):
        with pytest.raises(ValueError, match="longer than"):
            normalise_tag("x" * (MAX_TAG_LENGTH + 1))

    @pytest.mark.parametrize("bad", ["drop/table", "tag;delete", "a\x00b", "<b>x</b>"])
    def test_punctuation_is_rejected(self, bad):
        with pytest.raises(ValueError):
            normalise_tag(bad)

    def test_too_many_tags_is_rejected(self):
        with pytest.raises(ValueError, match="at most"):
            normalise_tags([f"tag{i}" for i in range(MAX_TAGS_PER_THREAD + 1)])


class TestSnapshotLeavesThingsOut:
    """Each test is a thing a naive snapshot would have published."""

    def test_tool_messages_are_dropped(self):
        # Tool results carry retrieved documents and file contents.
        values = {
            "messages": [
                {"type": "human", "content": "what does the contract say"},
                {"type": "tool", "content": "SECRET CLAUSE 7: ..."},
                {"type": "ai", "content": "It says the term is 3 years."},
            ]
        }
        snapshot = build_snapshot(values, "t")
        joined = " ".join(m["content"] for m in snapshot["messages"])
        assert "SECRET CLAUSE" not in joined
        assert len(snapshot["messages"]) == 2

    def test_system_messages_are_dropped(self):
        values = {
            "messages": [
                {
                    "type": "system",
                    "content": "You are an internal assistant. API key=x",
                },
                {"type": "human", "content": "hello"},
            ]
        }
        snapshot = build_snapshot(values, "t")
        assert all("API key" not in m["content"] for m in snapshot["messages"])

    def test_attachment_bytes_are_never_included(self):
        values = {
            "messages": [{"type": "human", "content": "summarise this"}],
            "files": {
                "/uploads/contract.pdf": {
                    "content": "BASE64PAYLOAD",
                    "encoding": "base64",
                }
            },
        }
        snapshot = build_snapshot(values, "t")
        assert snapshot["attachment_names"] == ["contract.pdf"]
        assert "BASE64PAYLOAD" not in str(snapshot)

    def test_multimodal_image_blocks_are_dropped_but_text_kept(self):
        # An image block is the attachment bytes under another name.
        values = {
            "messages": [
                {
                    "type": "human",
                    "content": [
                        {"type": "text", "text": "what is in this picture"},
                        {"type": "image", "source": {"data": "IMAGEBYTES"}},
                    ],
                }
            ]
        }
        snapshot = build_snapshot(values, "t")
        assert snapshot["messages"][0]["content"] == "what is in this picture"
        assert "IMAGEBYTES" not in str(snapshot)

    def test_unknown_state_keys_are_not_carried(self):
        # An allowlist, because thread state grows new keys as
        # middleware is added and a denylist publishes each new one.
        values = {
            "messages": [{"type": "human", "content": "hi"}],
            "memories": ["user's home address"],
            "todos": ["call the bank"],
        }
        snapshot = build_snapshot(values, "t")
        assert "home address" not in str(snapshot)
        assert "call the bank" not in str(snapshot)

    def test_a_very_long_message_is_truncated(self):
        values = {
            "messages": [{"type": "ai", "content": "x" * (MAX_MESSAGE_CHARS + 50)}]
        }
        snapshot = build_snapshot(values, "t")
        assert len(snapshot["messages"][0]["content"]) == MAX_MESSAGE_CHARS

    def test_roles_are_normalised_for_the_reader(self):
        values = {
            "messages": [
                {"type": "human", "content": "a"},
                {"role": "assistant", "content": "b"},
            ]
        }
        roles = [m["role"] for m in build_snapshot(values, "t")["messages"]]
        assert roles == ["user", "assistant"]


def _share(**overrides) -> Share:
    base = dict(
        share_id="abc",
        thread_id="t1",
        owner_id="alice",
        title="x",
        scope="authenticated",
        created_at=datetime.now(UTC),
        expires_at=None,
        revoked_at=None,
        views=0,
        last_viewed_at=None,
    )
    base.update(overrides)
    return Share(**base)  # type: ignore[arg-type]


class TestShareLifetime:
    def test_a_fresh_share_is_active(self):
        assert _share(expires_at=datetime.now(UTC) + timedelta(days=1)).active

    def test_an_expired_share_is_not_active(self):
        # The risk is not the moment of sharing, it is the year after.
        assert not _share(expires_at=datetime.now(UTC) - timedelta(seconds=1)).active

    def test_a_revoked_share_is_not_active_even_before_expiry(self):
        assert not _share(
            expires_at=datetime.now(UTC) + timedelta(days=30),
            revoked_at=datetime.now(UTC),
        ).active

    def test_view_counts_are_reported_to_the_owner(self):
        # A share nobody opened is a share worth revoking.
        payload = _share(views=4).as_dict()
        assert payload["views"] == 4
        assert "last_viewed_at" in payload
