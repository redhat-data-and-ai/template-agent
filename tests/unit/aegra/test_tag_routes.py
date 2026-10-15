"""Tag normalisation and the endpoints over it.

Normalisation happens on write, so these tests are the contract for
what ends up in the database — not just what a caller typed.
"""

from __future__ import annotations

import pytest

from deep_agent.aegra.tag_routes import (
    MAX_TAG_LENGTH,
    MAX_TAGS_PER_THREAD,
    normalise_tag,
    normalise_tags,
)


class TestNormaliseTag:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Urgent", "urgent"),
            ("  spaced  ", "spaced"),
            ("Two   Words", "two words"),
            ("with-hyphen", "with-hyphen"),
            ("with_underscore", "with_underscore"),
            ("digits123", "digits123"),
        ],
    )
    def test_folds_to_a_canonical_form(self, raw, expected):
        assert normalise_tag(raw) == expected

    @pytest.mark.parametrize("raw", ["", "   ", None])
    def test_rejects_an_empty_tag(self, raw):
        with pytest.raises(ValueError, match="cannot be empty"):
            normalise_tag(raw)

    def test_rejects_an_over_long_tag(self):
        with pytest.raises(ValueError, match="longer than"):
            normalise_tag("x" * (MAX_TAG_LENGTH + 1))

    @pytest.mark.parametrize("raw", ["semi;colon", "slash/es", "star*", "quote'"])
    def test_rejects_punctuation(self, raw):
        # Tags reach a LIKE query and a URL path segment; keeping the
        # character set small is cheaper than escaping everywhere.
        with pytest.raises(ValueError, match="may only contain"):
            normalise_tag(raw)


class TestNormaliseTags:
    def test_deduplicates_after_folding(self):
        assert normalise_tags(["Urgent", "urgent", "  URGENT "]) == ["urgent"]

    def test_sorts_the_result(self):
        assert normalise_tags(["beta", "alpha"]) == ["alpha", "beta"]

    def test_empty_list_is_allowed(self):
        # A PUT is a replace, so clearing every tag is a real request.
        assert normalise_tags([]) == []

    def test_rejects_too_many(self):
        with pytest.raises(ValueError, match="at most"):
            normalise_tags([f"tag{i}" for i in range(MAX_TAGS_PER_THREAD + 1)])

    def test_counts_duplicates_once_against_the_limit(self):
        raw = [f"tag{i}" for i in range(MAX_TAGS_PER_THREAD)] + ["tag0"]
        assert len(normalise_tags(raw)) == MAX_TAGS_PER_THREAD


from unittest.mock import AsyncMock, MagicMock, patch  # noqa: E402

from fastapi import HTTPException  # noqa: E402

import deep_agent.aegra.tag_routes as tr  # noqa: E402
from deep_agent.aegra.tag_routes import TagsRequest  # noqa: E402


class _Conn:
    """Async-context connection whose execute returns a fixed cursor."""

    def __init__(self, row=None, rows=None):
        self._row = row
        self._rows = rows or []
        self.commit = AsyncMock()
        self.executed: list[tuple] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, sql, params=None):
        self.executed.append((sql, params))
        cursor = MagicMock()
        cursor.fetchone = AsyncMock(return_value=self._row)
        cursor.fetchall = AsyncMock(return_value=self._rows)
        return cursor


def _connect(conn: _Conn):
    return patch.object(tr, "_connect", new=AsyncMock(return_value=conn))


def _auth(user_id: str = "alice"):
    return patch.object(
        tr, "authenticated_user_id", new=AsyncMock(return_value=user_id)
    )


class TestSetTags:
    @pytest.mark.asyncio
    async def test_replaces_the_set(self):
        conn = _Conn(row={"thread_id": "t1"})
        with _auth(), _connect(conn):
            body = await tr.set_tags(
                "t1", TagsRequest(tags=["Beta", "alpha"]), MagicMock()
            )

        assert body == {"thread_id": "t1", "tags": ["alpha", "beta"]}
        conn.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_422_for_an_invalid_tag(self):
        with (
            _auth(),
            pytest.raises(HTTPException) as exc,
        ):
            await tr.set_tags("t1", TagsRequest(tags=["bad;tag"]), MagicMock())
        assert exc.value.status_code == 422

    @pytest.mark.asyncio
    async def test_404_for_someone_elses_thread(self):
        # Not 403: that would confirm the thread exists, which is more
        # than a stranger should learn from a URL.
        conn = _Conn(row=None)
        with _auth(), _connect(conn), pytest.raises(HTTPException) as exc:
            await tr.set_tags("t1", TagsRequest(tags=["ok"]), MagicMock())
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_ownership_is_part_of_the_update(self):
        # One statement, so the thread cannot change hands between a
        # check and the write.
        conn = _Conn(row={"thread_id": "t1"})
        with _auth("alice"), _connect(conn):
            await tr.set_tags("t1", TagsRequest(tags=["ok"]), MagicMock())

        sql, params = conn.executed[0]
        assert "user_id = %(user_id)s" in sql
        assert params["user_id"] == "alice"


class TestGetTags:
    @pytest.mark.asyncio
    async def test_returns_the_tags(self):
        conn = _Conn(row={"tags": ["alpha"]})
        with _auth(), _connect(conn):
            assert await tr.get_tags("t1", MagicMock()) == {
                "thread_id": "t1",
                "tags": ["alpha"],
            }

    @pytest.mark.asyncio
    async def test_an_untagged_thread_has_an_empty_list(self):
        conn = _Conn(row={"tags": None})
        with _auth(), _connect(conn):
            assert (await tr.get_tags("t1", MagicMock()))["tags"] == []

    @pytest.mark.asyncio
    async def test_404_when_the_thread_is_not_the_callers(self):
        conn = _Conn(row=None)
        with _auth(), _connect(conn), pytest.raises(HTTPException) as exc:
            await tr.get_tags("t1", MagicMock())
        assert exc.value.status_code == 404
