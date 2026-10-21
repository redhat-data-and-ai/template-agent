"""One MCP session per server per turn, not one per call."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_agent.aegra import mcp_host


@pytest.fixture(autouse=True)
def _clean_scope():
    mcp_host._open_sessions.set(None)
    mcp_host._session_stacks.clear()
    yield
    mcp_host._open_sessions.set(None)
    mcp_host._session_stacks.clear()


def _patched(session_obj, opened):
    """Patch everything mcp_session touches, counting handshakes."""

    @asynccontextmanager
    async def fake_session(_name):
        opened.append(1)
        yield session_obj

    client = MagicMock()
    client.session = fake_session
    return (
        patch.object(mcp_host, "ensure_mcp_apps_capability_advertised", MagicMock()),
        patch.object(mcp_host, "_get_enabled_server", MagicMock(return_value={})),
        patch.object(mcp_host, "_resolve_bearer", AsyncMock(return_value="tok")),
        patch.object(mcp_host, "_build_server_config", MagicMock(return_value={})),
        patch.object(mcp_host, "MultiServerMCPClient", MagicMock(return_value=client)),
    )


class TestSessionReuse:
    async def test_without_a_scope_every_call_opens_its_own(self):
        """The safe default: a session must not outlive its request."""
        opened: list[int] = []
        a, b, c, d, e = _patched(MagicMock(), opened)
        with a, b, c, d, e:
            async with mcp_host.mcp_session("dv", user_id="u", sso_token=None):
                pass
            async with mcp_host.mcp_session("dv", user_id="u", sso_token=None):
                pass
        assert len(opened) == 2

    async def test_inside_a_scope_the_session_is_reused(self):
        """Two listings used to mean two `initialize` handshakes.

        Measured at roughly twelve seconds against a server that
        answers each call in well under one, run back to back —
        which is what "things happen sequentially" looked like.
        """
        opened: list[int] = []
        a, b, c, d, e = _patched(MagicMock(), opened)
        with a, b, c, d, e:
            async with mcp_host.mcp_session_scope():
                async with mcp_host.mcp_session(
                    "dv", user_id="u", sso_token=None
                ) as s1:
                    pass
                async with mcp_host.mcp_session(
                    "dv", user_id="u", sso_token=None
                ) as s2:
                    pass
        assert len(opened) == 1
        assert s1 is s2

    async def test_a_different_server_gets_its_own_session(self):
        opened: list[int] = []
        a, b, c, d, e = _patched(MagicMock(), opened)
        with a, b, c, d, e:
            async with mcp_host.mcp_session_scope():
                async with mcp_host.mcp_session("dv", user_id="u", sso_token=None):
                    pass
                async with mcp_host.mcp_session("atlan", user_id="u", sso_token=None):
                    pass
        assert len(opened) == 2

    async def test_a_different_user_never_shares_a_session(self):
        """A session is established with one user's bearer.

        Handing it to another user would send their request under
        someone else's authorization.
        """
        opened: list[int] = []
        a, b, c, d, e = _patched(MagicMock(), opened)
        with a, b, c, d, e:
            async with mcp_host.mcp_session_scope():
                async with mcp_host.mcp_session("dv", user_id="alice", sso_token=None):
                    pass
                async with mcp_host.mcp_session("dv", user_id="bob", sso_token=None):
                    pass
        assert len(opened) == 2

    async def test_the_scope_closes_what_it_opened(self):
        opened: list[int] = []
        a, b, c, d, e = _patched(MagicMock(), opened)
        with a, b, c, d, e:
            async with mcp_host.mcp_session_scope():
                async with mcp_host.mcp_session("dv", user_id="u", sso_token=None):
                    pass
        assert mcp_host._open_sessions.get() is None
        assert mcp_host._session_stacks == {}


class TestSessionStacksAreBounded:
    """`enter_mcp_session_scope` has no exit, so it must not grow for ever.

    The graph is built in the request context and runs in a worker
    task, so there is nowhere to close a scope that spans the turn.
    That trade is deliberate. Nothing removing the entry afterwards
    was not: the dict, and the MCP sessions each stack holds open,
    accumulated for the life of the process.
    """

    async def test_the_stack_table_stops_growing(self):
        from contextlib import AsyncExitStack

        from deep_agent.aegra import mcp_host as mh

        original = dict(mh._session_stacks)
        mh._session_stacks.clear()
        try:
            for i in range(mh._MAX_SESSION_STACKS * 2):
                mh._session_stacks[i] = ({}, AsyncExitStack())
                await mh._evict_old_session_stacks()
            assert len(mh._session_stacks) < mh._MAX_SESSION_STACKS
        finally:
            mh._session_stacks.clear()
            mh._session_stacks.update(original)

    async def test_eviction_closes_what_it_drops(self):
        from contextlib import AsyncExitStack

        from deep_agent.aegra import mcp_host as mh

        closed: list[str] = []

        class _Tracked(AsyncExitStack):
            async def aclose(self) -> None:  # type: ignore[override]
                closed.append("yes")

        original = dict(mh._session_stacks)
        mh._session_stacks.clear()
        try:
            for i in range(mh._MAX_SESSION_STACKS + 2):
                mh._session_stacks[i] = ({}, _Tracked())
                await mh._evict_old_session_stacks()
            assert closed, "an evicted stack must be closed, not just dropped"
        finally:
            mh._session_stacks.clear()
            mh._session_stacks.update(original)

    async def test_eviction_empties_the_cache_it_drops(self):
        """A dropped stack must not leave its sessions reachable.

        `aclose()` here runs in whichever task tripped the cap, not
        the one that entered the stack, so anyio raises and the close
        is best-effort. A cache still holding those sessions would
        hand a dead one to the next call in that turn.
        """
        from contextlib import AsyncExitStack

        from deep_agent.aegra import mcp_host as mh

        original = dict(mh._session_stacks)
        mh._session_stacks.clear()
        caches = []
        try:
            for i in range(mh._MAX_SESSION_STACKS + 2):
                cache = {("server", "user"): object()}
                caches.append(cache)
                mh._session_stacks[i] = (cache, AsyncExitStack())
                await mh._evict_old_session_stacks()
            assert caches[0] == {}
        finally:
            mh._session_stacks.clear()
            mh._session_stacks.update(original)

    async def test_the_cap_clears_concurrent_runs(self):
        """At 32 against 40 concurrent runs, eviction hit live turns."""
        from deep_agent.aegra import mcp_host as mh
        from deep_agent.src.settings import settings

        assert mh._MAX_SESSION_STACKS > settings.MAX_CONCURRENT_RUNS

    async def test_a_failing_close_does_not_stop_eviction(self):
        """A stack whose task is gone raises; the entry must still go."""
        from contextlib import AsyncExitStack

        from deep_agent.aegra import mcp_host as mh

        class _Angry(AsyncExitStack):
            async def aclose(self) -> None:  # type: ignore[override]
                raise RuntimeError("task is gone")

        original = dict(mh._session_stacks)
        mh._session_stacks.clear()
        try:
            for i in range(mh._MAX_SESSION_STACKS + 3):
                mh._session_stacks[i] = ({}, _Angry())
                await mh._evict_old_session_stacks()
            assert len(mh._session_stacks) < mh._MAX_SESSION_STACKS
        finally:
            mh._session_stacks.clear()
            mh._session_stacks.update(original)
