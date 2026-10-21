"""Tests for the `/runs/wait` fast-poll patch."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from deep_agent.aegra import run_wait
from deep_agent.src.settings import settings


class TestBackoffSchedule:
    """Poll fast, then back off.

    Upstream sleeps a flat 2s between checks, so a run finishing in
    300ms is reported at 2.0s. Turn latencies clustered at 2.09s,
    4.10s and 6.11s — spaced 2.01s apart — and cutting the prompt
    fifteenfold did not move them.
    """

    def test_starts_at_the_floor_and_ends_at_the_ceiling(self):
        with (
            patch.object(settings, "RUN_WAIT_MIN_POLL_SECONDS", 0.05),
            patch.object(settings, "RUN_WAIT_MAX_POLL_SECONDS", 2.0),
        ):
            delays = run_wait._backoff_schedule()

        assert delays[0] == 0.05
        assert delays[-1] == 2.0

    def test_doubles(self):
        with (
            patch.object(settings, "RUN_WAIT_MIN_POLL_SECONDS", 0.1),
            patch.object(settings, "RUN_WAIT_MAX_POLL_SECONDS", 1.6),
        ):
            assert run_wait._backoff_schedule() == [0.1, 0.2, 0.4, 0.8, 1.6]

    def test_a_floor_above_the_ceiling_still_yields_one_delay(self):
        """Misconfiguration must not produce an empty schedule.

        An empty list would index-error on the first sleep, turning a
        latency optimisation into a crash on every run.
        """
        with (
            patch.object(settings, "RUN_WAIT_MIN_POLL_SECONDS", 5.0),
            patch.object(settings, "RUN_WAIT_MAX_POLL_SECONDS", 1.0),
        ):
            assert run_wait._backoff_schedule() == [1.0]

    def test_first_delay_is_far_below_the_upstream_flat_interval(self):
        delays = run_wait._backoff_schedule()
        assert delays[0] < 0.5


class TestInstall:
    @pytest.fixture(autouse=True)
    def _unpatched(self):
        run_wait._patched = False
        yield
        run_wait._patched = False

    def test_respects_the_off_switch(self):
        with patch.object(settings, "RUN_WAIT_FAST_POLL", False):
            assert run_wait.install_fast_run_wait() == "skipped: disabled"

    def test_is_idempotent(self):
        run_wait._patched = True
        assert run_wait.install_fast_run_wait() == "ok: already patched"

    def test_an_upstream_rename_does_not_stop_startup(self):
        """Losing the optimisation is acceptable; failing to boot is not.

        This patches a vendored package by name, so a rename upstream
        has to surface as a warning at startup rather than as an
        exception on the first request.
        """
        with (
            patch.object(settings, "RUN_WAIT_FAST_POLL", True),
            patch.dict("sys.modules", {"aegra_api.services.worker_executor": None}),
        ):
            result = run_wait.install_fast_run_wait()

        assert result.startswith("skipped:")
        assert run_wait._patched is False
