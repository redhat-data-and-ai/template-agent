"""Interrupted runs are re-checked after boot, not only at boot.

A pod shutting down marks its in-flight runs `interrupted`. When that
write lands after the replacement pod has already scanned, nothing
looks again until the next restart. Seen on a rolling update: new pods
scanned at 09:53:55 and 09:54:15, the old ones marked five runs
interrupted at 09:54:31, and two threads sat "busy" with nobody coming.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from deep_agent.aegra import startup as st
from deep_agent.src.settings import settings


@pytest.fixture(autouse=True)
def _no_leftover_tasks():
    yield
    for task in list(st._BACKGROUND_TASKS):
        task.cancel()
    st._BACKGROUND_TASKS.clear()


class TestSweeperRegistration:
    def test_disabled_by_a_zero_interval(self):
        with patch.object(settings, "LIFECYCLE_SWEEP_INTERVAL_SECONDS", 0):
            assert st._start_interrupted_run_sweeper() == "skipped: sweep disabled"

    def test_disabled_with_lifecycle_persistence_off(self):
        with patch.object(settings, "LIFECYCLE_PERSISTENCE_ENABLED", False):
            result = st._start_interrupted_run_sweeper()
        assert "lifecycle persistence disabled" in result

    async def test_enabled_reports_its_cadence_and_holds_the_task(self):
        with (
            patch.object(settings, "LIFECYCLE_PERSISTENCE_ENABLED", True),
            patch.object(settings, "LIFECYCLE_SWEEP_INTERVAL_SECONDS", 30),
        ):
            result = st._start_interrupted_run_sweeper()
        assert result == "ok: every 30s"
        # Held deliberately: asyncio keeps only a weak reference, so a
        # task nobody holds can be collected mid-flight.
        assert len(st._BACKGROUND_TASKS) == 1

    async def test_it_actually_rechecks(self):
        resume = AsyncMock(return_value="ok: 2 reset, 0 stale")
        with (
            patch.object(settings, "LIFECYCLE_PERSISTENCE_ENABLED", True),
            patch.object(settings, "LIFECYCLE_SWEEP_INTERVAL_SECONDS", 1),
            patch.object(settings, "LIFECYCLE_SWEEP_MIN_AGE_SECONDS", 45),
            patch.object(st, "_resume_interrupted_runs", resume),
        ):
            st._start_interrupted_run_sweeper()
            await asyncio.sleep(1.2)

        assert resume.await_count >= 1
        # The age floor keeps it off runs a pod is still shutting
        # down, and `startup=False` means an operator who turned
        # off resume-at-boot still gets the periodic recovery.
        assert resume.await_args.kwargs == {
            "min_age_seconds": 45,
            "startup": False,
        }

    async def test_a_failing_sweep_does_not_kill_the_loop(self):
        calls = {"n": 0}

        async def flaky(**kwargs):
            calls["n"] += 1
            raise RuntimeError("database down")

        with (
            patch.object(settings, "LIFECYCLE_PERSISTENCE_ENABLED", True),
            patch.object(settings, "LIFECYCLE_SWEEP_INTERVAL_SECONDS", 1),
            patch.object(st, "_resume_interrupted_runs", flaky),
        ):
            st._start_interrupted_run_sweeper()
            await asyncio.sleep(2.2)

        assert calls["n"] >= 2
