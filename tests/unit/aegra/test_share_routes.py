"""Share endpoints.

The reader-facing decisions are the ones worth pinning: a public
share needs no sign-in, an `authenticated` one does, and expired is
reported apart from revoked because "ask them to share it again" and
"they took it down" are different instructions.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import deep_agent.aegra.share_routes as sr
from deep_agent.src.sharing.store import Share


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


def _request(auth: str | None = None) -> MagicMock:
    request = MagicMock()
    request.headers = {"Authorization": auth} if auth else {}
    return request


class _Client:
    def __init__(self, responses):
        self._responses = responses

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, path):
        return self._responses[path]


def _response(status_code, body=None):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = body or {}
    return response


class TestShareThread:
    @pytest.mark.asyncio
    async def test_publishes_a_snapshot(self):
        client = _Client(
            {
                "/threads/t1/state": _response(
                    200, {"values": {"messages": [{"type": "human", "content": "hi"}]}}
                ),
                "/threads/t1": _response(200, {"metadata": {"title": "My chat"}}),
            }
        )
        with (
            patch.object(
                sr, "authenticated_user_id", new=AsyncMock(return_value="alice")
            ),
            patch("httpx.AsyncClient", return_value=client),
            patch.object(
                sr.store,
                "create_share",
                new=AsyncMock(return_value=_share(title="My chat")),
            ),
        ):
            body = await sr.share_thread("t1", sr.ShareRequest(), _request("Bearer t"))

        assert body["path"] == "/shared/s1"
        assert body["messages"] == 1

    @pytest.mark.asyncio
    async def test_404_when_the_thread_is_not_there(self):
        client = _Client({"/threads/t1/state": _response(404)})
        with (
            patch.object(
                sr, "authenticated_user_id", new=AsyncMock(return_value="alice")
            ),
            patch("httpx.AsyncClient", return_value=client),
            pytest.raises(HTTPException) as exc,
        ):
            await sr.share_thread("t1", sr.ShareRequest(), _request())
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_422_rather_than_a_link_to_nothing(self):
        # An empty share reads as a broken link to the recipient.
        client = _Client(
            {
                "/threads/t1/state": _response(200, {"values": {"messages": []}}),
                "/threads/t1": _response(200, {}),
            }
        )
        with (
            patch.object(
                sr, "authenticated_user_id", new=AsyncMock(return_value="alice")
            ),
            patch("httpx.AsyncClient", return_value=client),
            pytest.raises(HTTPException) as exc,
        ):
            await sr.share_thread("t1", sr.ShareRequest(), _request())
        assert exc.value.status_code == 422

    @pytest.mark.asyncio
    async def test_422_when_the_store_rejects_the_request(self):
        client = _Client(
            {
                "/threads/t1/state": _response(
                    200, {"values": {"messages": [{"type": "ai", "content": "hi"}]}}
                ),
                "/threads/t1": _response(200, {}),
            }
        )
        with (
            patch.object(
                sr, "authenticated_user_id", new=AsyncMock(return_value="alice")
            ),
            patch("httpx.AsyncClient", return_value=client),
            patch.object(
                sr.store,
                "create_share",
                new=AsyncMock(side_effect=ValueError("scope not allowed")),
            ),
            pytest.raises(HTTPException) as exc,
        ):
            await sr.share_thread("t1", sr.ShareRequest(), _request())
        assert exc.value.status_code == 422


class TestGetAndRevoke:
    @pytest.mark.asyncio
    async def test_reports_an_unshared_thread(self):
        with (
            patch.object(
                sr, "authenticated_user_id", new=AsyncMock(return_value="alice")
            ),
            patch.object(
                sr.store, "get_share_for_thread", new=AsyncMock(return_value=None)
            ),
        ):
            assert await sr.get_thread_share("t1", _request()) == {"shared": False}

    @pytest.mark.asyncio
    async def test_reports_a_shared_thread_with_its_path(self):
        with (
            patch.object(
                sr, "authenticated_user_id", new=AsyncMock(return_value="alice")
            ),
            patch.object(
                sr.store, "get_share_for_thread", new=AsyncMock(return_value=_share())
            ),
        ):
            body = await sr.get_thread_share("t1", _request())
        assert body["shared"] is True
        assert body["path"] == "/shared/s1"

    @pytest.mark.asyncio
    async def test_revoke_reports_whether_anything_was_revoked(self):
        with (
            patch.object(
                sr, "authenticated_user_id", new=AsyncMock(return_value="alice")
            ),
            patch.object(sr.store, "revoke_share", new=AsyncMock(return_value=True)),
        ):
            assert await sr.unshare_thread("t1", _request()) == {"revoked": True}

    @pytest.mark.asyncio
    async def test_my_shares_lists_everything_published(self):
        # Without this the only way to find out what you have
        # published is to remember.
        with (
            patch.object(
                sr, "authenticated_user_id", new=AsyncMock(return_value="alice")
            ),
            patch.object(
                sr.store, "list_shares", new=AsyncMock(return_value=[_share()])
            ),
        ):
            body = await sr.my_shares(_request())
        assert body["shares"][0]["path"] == "/shared/s1"


class TestReadShared:
    @pytest.mark.asyncio
    async def test_404_for_an_unknown_link(self):
        with (
            patch.object(sr.store, "read_share", new=AsyncMock(return_value=None)),
            pytest.raises(HTTPException) as exc,
        ):
            await sr.read_shared("nope", _request())
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_public_scope_needs_no_sign_in(self):
        snapshot = {
            "messages": [{"role": "user", "content": "hi"}],
            "captured_at": "now",
        }
        auth = AsyncMock()
        with (
            patch.object(
                sr.store,
                "read_share",
                new=AsyncMock(return_value=(_share(scope="public"), snapshot)),
            ),
            patch.object(sr, "authenticated_user_id", new=auth),
        ):
            body = await sr.read_shared("s1", _request())
        auth.assert_not_awaited()
        assert body["messages"] == snapshot["messages"]

    @pytest.mark.asyncio
    async def test_authenticated_scope_requires_a_signed_in_reader(self):
        # Still a link rather than a grant — but not one that works
        # for the whole internet.
        snapshot = {"messages": [{"role": "user", "content": "hi"}]}
        auth = AsyncMock(return_value="bob")
        with (
            patch.object(
                sr.store, "read_share", new=AsyncMock(return_value=(_share(), snapshot))
            ),
            patch.object(sr, "authenticated_user_id", new=auth),
        ):
            await sr.read_shared("s1", _request())
        auth.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_revoked_and_expired_are_reported_apart(self):
        snapshot = {"messages": []}
        revoked = _share(scope="public", revoked_at=datetime.now(UTC))
        with (
            patch.object(
                sr.store, "read_share", new=AsyncMock(return_value=(revoked, snapshot))
            ),
            pytest.raises(HTTPException) as exc,
        ):
            await sr.read_shared("s1", _request())
        assert exc.value.status_code == 410
        assert "revoked" in exc.value.detail

        expired = _share(
            scope="public", expires_at=datetime.now(UTC) - timedelta(days=1)
        )
        with (
            patch.object(
                sr.store, "read_share", new=AsyncMock(return_value=(expired, snapshot))
            ),
            pytest.raises(HTTPException) as exc,
        ):
            await sr.read_shared("s1", _request())
        assert exc.value.status_code == 410
        assert "expired" in exc.value.detail

    @pytest.mark.asyncio
    async def test_attachments_are_named_not_included(self):
        snapshot = {
            "messages": [],
            "attachment_names": ["report.pdf"],
            "captured_at": "now",
        }
        with patch.object(
            sr.store,
            "read_share",
            new=AsyncMock(return_value=(_share(scope="public"), snapshot)),
        ):
            body = await sr.read_shared("s1", _request())
        assert body["attachment_names"] == ["report.pdf"]
