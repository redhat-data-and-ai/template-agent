"""Tests for the in-flight run gauge.

The gauge decides whether a scale-to-zero autoscaler may remove the
last pod. A wrong value is not a cosmetic bug: a false zero lets KEDA
kill a running agent mid-answer.
"""

from unittest.mock import AsyncMock, patch

import pytest

from deep_agent.aegra import inflight_metrics as metrics


@pytest.fixture(autouse=True)
def _reset():
    metrics._task = None
    yield
    metrics._task = None


class TestActiveStatuses:
    def test_pending_counts_as_in_flight(self):
        """A queued run with no pod to claim it is exactly the case where
        scaling to zero strands work."""
        assert "pending" in metrics._ACTIVE_STATUSES

    def test_running_counts_as_in_flight(self):
        assert "running" in metrics._ACTIVE_STATUSES

    def test_finished_runs_do_not_count(self):
        for status in ("success", "error", "interrupted"):
            assert status not in metrics._ACTIVE_STATUSES


class TestGauge:
    def test_gauge_is_created_lazily(self):
        """Importing the module must not register a metric as a side effect."""
        gauge = metrics._get_gauge()
        assert gauge is metrics._get_gauge()

    def test_gauge_is_exported_under_the_queried_name(self):
        import prometheus_client

        metrics._get_gauge().set(7)
        exported = prometheus_client.generate_latest().decode()

        assert "aegra_runs_in_flight 7.0" in exported


class TestRefreshLoop:
    @pytest.mark.asyncio
    async def test_a_failed_query_does_not_report_zero(self):
        """A false zero is the dangerous failure: it tells the scaler
        there is no work and the last pod can go."""
        gauge = metrics._get_gauge()
        gauge.set(4)

        with (
            patch.object(
                metrics,
                "count_active_runs",
                AsyncMock(side_effect=RuntimeError("db down")),
            ),
            patch.object(metrics, "REFRESH_INTERVAL_SECONDS", 0),
        ):
            import asyncio

            task = asyncio.create_task(metrics._refresh_loop())
            await asyncio.sleep(0.01)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        sample = next(
            s
            for metric in gauge.collect()
            for s in metric.samples
            if s.name == "aegra_runs_in_flight"
        )
        assert sample.value == 4, "stale value is safer than a false zero"

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self):
        with patch.object(metrics, "count_active_runs", AsyncMock(return_value=0)):
            metrics.start_inflight_metrics()
            first = metrics._task
            metrics.start_inflight_metrics()

            assert metrics._task is first

            await metrics.stop_inflight_metrics()

    @pytest.mark.asyncio
    async def test_stop_is_safe_when_never_started(self):
        await metrics.stop_inflight_metrics()


class TestStartupWiring:
    def test_enabled_by_default(self, monkeypatch):
        """An operator who forgets the flag should still get working
        scale-down, not a silent fallback to CPU-only scaling."""
        from unittest.mock import patch as _patch

        from deep_agent.aegra import startup

        monkeypatch.delenv("ENABLE_PROMETHEUS_METRICS", raising=False)
        with _patch(
            "deep_agent.aegra.inflight_metrics.start_inflight_metrics"
        ) as started:
            assert startup._start_inflight_metrics() == "ok"
            started.assert_called_once()

    def test_can_be_turned_off_explicitly(self, monkeypatch):
        from deep_agent.aegra.startup import _start_inflight_metrics

        monkeypatch.setenv("ENABLE_PROMETHEUS_METRICS", "false")
        assert "skipped" in _start_inflight_metrics()

    def test_failure_does_not_block_startup(self, monkeypatch):
        """Autoscaling telemetry must never stop the agent booting."""
        from deep_agent.aegra import startup

        monkeypatch.setenv("ENABLE_PROMETHEUS_METRICS", "true")
        with patch(
            "deep_agent.aegra.inflight_metrics.start_inflight_metrics",
            side_effect=RuntimeError("no event loop"),
        ):
            result = startup._start_inflight_metrics()

        assert result.startswith("error")
