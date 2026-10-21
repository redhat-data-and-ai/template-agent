"""Tests for run admission control."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from deep_agent.aegra.admission import AdmissionControlMiddleware


def _app(max_concurrent: int, hold: asyncio.Event | None = None) -> FastAPI:
    app = FastAPI()

    @app.post("/threads/{thread_id}/runs")
    async def start_run(thread_id: str):
        if hold is not None:
            await hold.wait()
        return {"ok": thread_id}

    @app.post("/runs")
    async def start_bare_run():
        if hold is not None:
            await hold.wait()
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/threads/{thread_id}/runs")
    async def list_runs(thread_id: str):
        return {"runs": []}

    app.add_middleware(AdmissionControlMiddleware, max_concurrent=max_concurrent)
    return app


async def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


class TestAdmissionControl:
    async def test_admits_below_the_limit(self):
        async with await _client(_app(2)) as c:
            r = await c.post("/threads/t1/runs")
        assert r.status_code == 200

    async def test_sheds_at_the_limit(self):
        """A full pod says so immediately rather than queueing.

        Queueing would hold the caller — and its connection — behind
        a turn that may run for minutes, and the failure would land
        on requests already in flight instead of the one that should
        never have been admitted.
        """
        hold = asyncio.Event()
        app = _app(1, hold)
        async with await _client(app) as c:
            first = asyncio.create_task(c.post("/runs"))
            await asyncio.sleep(0)  # let it reach the handler
            second = await c.post("/runs")

            assert second.status_code == 503
            assert second.json()["error_type"] == "at_capacity"
            assert second.headers["Retry-After"] == "5"

            hold.set()
            assert (await first).status_code == 200

    async def test_capacity_is_returned_after_a_run_finishes(self):
        hold = asyncio.Event()
        app = _app(1, hold)
        async with await _client(app) as c:
            first = asyncio.create_task(c.post("/runs"))
            await asyncio.sleep(0)
            hold.set()
            await first

            second = await c.post("/runs")
        assert second.status_code == 200

    @pytest.mark.parametrize("path", ["/health", "/threads/t1/runs"])
    async def test_reads_are_never_gated(self, path):
        """A pod at capacity still answers probes and polling.

        Shedding a readiness check would take the pod out of
        rotation for being busy, and shedding a GET on a run would
        break the UI of someone whose turn is already running.
        """
        hold = asyncio.Event()
        app = _app(1, hold)
        async with await _client(app) as c:
            running = asyncio.create_task(c.post("/runs"))
            await asyncio.sleep(0)

            assert (await c.get(path)).status_code == 200

            hold.set()
            await running

    async def test_zero_disables_the_gate(self):
        hold = asyncio.Event()
        app = _app(0, hold)
        async with await _client(app) as c:
            tasks = [asyncio.create_task(c.post("/runs")) for _ in range(5)]
            await asyncio.sleep(0)
            hold.set()
            results = await asyncio.gather(*tasks)
        assert [r.status_code for r in results] == [200] * 5


class TestThePermitSpansTheWholeResponse:
    """`call_next` returns when the response starts, not when it ends.

    So `async with self._sem` released the permit before a
    `/runs/stream` had sent a single token, and the only route this
    middleware really limited was `/runs/wait`, which returns at the
    end.
    """

    @staticmethod
    def _middleware(max_concurrent: int = 1):
        from deep_agent.aegra.admission import AdmissionControlMiddleware

        return AdmissionControlMiddleware(app=None, max_concurrent=max_concurrent)

    @staticmethod
    def _streaming_request():
        from unittest.mock import MagicMock

        request = MagicMock()
        request.method = "POST"
        request.url.path = "/threads/abc/runs/stream"
        return request

    async def test_the_permit_is_still_held_while_the_body_streams(self):
        mw = self._middleware()

        async def body():
            yield b"chunk-1"
            yield b"chunk-2"

        from unittest.mock import MagicMock

        response = MagicMock()
        response.body_iterator = body()

        async def call_next(_request):
            return response

        out = await mw.dispatch(self._streaming_request(), call_next)
        # Started, not finished: nobody else may start a run yet.
        assert mw._sem.locked()

        chunks = [c async for c in out.body_iterator]
        assert chunks == [b"chunk-1", b"chunk-2"]
        # Finished, so the permit is back.
        assert not mw._sem.locked()

    async def test_a_plain_response_releases_immediately(self):
        from unittest.mock import MagicMock

        mw = self._middleware()
        response = MagicMock()
        del response.body_iterator

        async def call_next(_request):
            return response

        await mw.dispatch(self._streaming_request(), call_next)
        assert not mw._sem.locked()

    async def test_a_failure_does_not_strand_the_permit(self):
        mw = self._middleware()

        async def call_next(_request):
            raise RuntimeError("upstream died")

        import pytest

        with pytest.raises(RuntimeError):
            await mw.dispatch(self._streaming_request(), call_next)
        assert not mw._sem.locked()

    async def test_a_client_that_disconnects_mid_stream_releases(self):
        mw = self._middleware()

        async def body():
            yield b"chunk-1"
            raise ConnectionResetError("client went away")

        from unittest.mock import MagicMock

        response = MagicMock()
        response.body_iterator = body()

        async def call_next(_request):
            return response

        out = await mw.dispatch(self._streaming_request(), call_next)
        import pytest

        with pytest.raises(ConnectionResetError):
            async for _ in out.body_iterator:
                pass
        assert not mw._sem.locked()
