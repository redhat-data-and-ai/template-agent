"""Tests for the controllable clock, and a real use of it.

Backoff, expiry and TTL paths are the ones where correctness matters and
wall-clock tests are slowest and flakiest. The last class proves the
fixture works against real agent code, not just against itself.
"""

import asyncio
import time

import pytest

from tests.support.clock import FakeClock


class TestFakeClock:
    def test_time_does_not_move_on_its_own(self):
        clock = FakeClock()
        assert clock.time() == clock.time()

    def test_advance_moves_both_sources_together(self):
        """Code mixing time.time and monotonic must see consistent order."""
        clock = FakeClock()
        before_wall, before_mono = clock.time(), clock.monotonic()

        clock.advance(30)

        assert clock.time() - before_wall == 30
        assert clock.monotonic() - before_mono == 30

    def test_cannot_go_backwards(self):
        with pytest.raises(ValueError):
            FakeClock().advance(-1)

    def test_now_tracks_advance(self):
        clock = FakeClock()
        before = clock.now()
        clock.advance(3600)
        assert (clock.now() - before).total_seconds() == 3600

    @pytest.mark.asyncio
    async def test_async_sleep_records_and_advances(self):
        """Recording durations is what lets a test assert backoff shape."""
        clock = FakeClock()
        start = clock.time()

        await clock.asleep(5)
        await clock.asleep(10)

        assert clock.sleeps == [5, 10]
        assert clock.time() - start == 15


class TestInstall:
    def test_patches_the_global_time_module(self, fake_clock):
        before = time.time()
        fake_clock.advance(100)
        assert time.time() - before == 100

    @pytest.mark.asyncio
    async def test_asyncio_sleep_does_not_actually_wait(self, fake_clock):
        real_start = time.monotonic  # patched; use wall-clock via perf_counter
        import time as _t

        wall = _t.perf_counter()
        await asyncio.sleep(3600)

        assert _t.perf_counter() - wall < 1.0, "should not really sleep an hour"
        assert fake_clock.sleeps == [3600]
        assert real_start is not None


class TestAgainstRealCode:
    """The fixture is only useful if it reaches real time-dependent paths."""

    def test_token_expiry_is_controllable(self, fake_clock):
        import base64
        import json

        from deep_agent.aegra import mcp as mcp_mod

        header = base64.urlsafe_b64encode(b'{"alg":"RS256"}').rstrip(b"=").decode()
        exp = fake_clock.time() + 600
        payload = (
            base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode())
            .rstrip(b"=")
            .decode()
        )
        token = f"{header}.{payload}.sig"

        assert mcp_mod._token_remaining(token) == pytest.approx(600, abs=1)

        fake_clock.advance(599)
        assert mcp_mod._token_remaining(token) == pytest.approx(1, abs=1)

        fake_clock.advance(10)
        assert mcp_mod._token_remaining(token) < 0

    def test_opaque_token_stays_immeasurable_as_time_passes(self, fake_clock):
        import math

        from deep_agent.aegra import mcp as mcp_mod

        assert mcp_mod._token_remaining("opaque") == math.inf
        fake_clock.advance(86_400)
        assert mcp_mod._token_remaining("opaque") == math.inf
