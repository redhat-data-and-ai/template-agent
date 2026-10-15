"""The analytics endpoints.

Two decisions are load-bearing and worth pinning: a database that
cannot be reached returns 503 rather than zeroes, and per-user data
returns 404 rather than an empty list when the deployment has not
enabled it. Both have the same failure mode — a chart that reads as
"nobody used it" when the truth is "we did not ask".
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

import deep_agent.aegra.analytics_routes as ar
from deep_agent.src.analytics import queries


class TestWindow:
    def test_clamps_to_the_maximum(self):
        assert ar._window(queries.MAX_WINDOW_DAYS + 50).days == queries.MAX_WINDOW_DAYS

    def test_clamps_to_at_least_one_day(self):
        assert ar._window(0).days == 1


class TestUnavailable:
    def test_is_a_503_rather_than_zeroes(self):
        # A chart of flat zeroes is indistinguishable from "nobody
        # used it", which is a conclusion somebody will act on.
        exc = ar._unavailable(RuntimeError("connection refused"))
        assert isinstance(exc, HTTPException)
        assert exc.status_code == 503


class TestSummary:
    @pytest.mark.asyncio
    async def test_reports_the_counters(self):
        result = queries.Summary(
            window_days=7, runs=10, threads=4, users=2, successes=9, errors=1
        )
        with patch.object(queries, "summary", new=AsyncMock(return_value=result)):
            body = await ar.analytics_summary(days=7, _token="t")

        assert body["runs"] == 10
        assert body["errors"] == 1
        assert body["window_days"] == 7

    @pytest.mark.asyncio
    async def test_503_when_the_database_is_unreachable(self):
        with (
            patch.object(
                queries, "summary", new=AsyncMock(side_effect=RuntimeError("down"))
            ),
            pytest.raises(HTTPException) as exc,
        ):
            await ar.analytics_summary(days=7, _token="t")
        assert exc.value.status_code == 503


class TestTimeseries:
    @pytest.mark.asyncio
    async def test_returns_the_points_and_the_bucket(self):
        with patch.object(
            queries, "timeseries", new=AsyncMock(return_value=[{"t": 1}])
        ):
            body = await ar.analytics_timeseries(days=7, bucket="day", _token="t")
        assert body == {"bucket": "day", "points": [{"t": 1}]}

    @pytest.mark.asyncio
    async def test_503_on_failure(self):
        with (
            patch.object(
                queries, "timeseries", new=AsyncMock(side_effect=RuntimeError("down"))
            ),
            pytest.raises(HTTPException) as exc,
        ):
            await ar.analytics_timeseries(days=7, bucket="day", _token="t")
        assert exc.value.status_code == 503


class TestModelsAndErrors:
    @pytest.mark.asyncio
    async def test_models_returns_rows(self):
        with patch.object(queries, "by_model", new=AsyncMock(return_value=[{"m": 1}])):
            body = await ar.analytics_models(days=7, _token="t")
        assert body["models"] == [{"m": 1}]

    @pytest.mark.asyncio
    async def test_models_503_on_failure(self):
        with (
            patch.object(
                queries, "by_model", new=AsyncMock(side_effect=RuntimeError("down"))
            ),
            pytest.raises(HTTPException) as exc,
        ):
            await ar.analytics_models(days=7, _token="t")
        assert exc.value.status_code == 503

    @pytest.mark.asyncio
    async def test_errors_returns_classes_not_messages(self):
        # An exception string routinely contains the input that
        # produced it; a dashboard is the wrong place for someone's
        # document path.
        with patch.object(
            queries, "errors", new=AsyncMock(return_value=[{"class": "TimeoutError"}])
        ):
            body = await ar.analytics_errors(days=7, _token="t")
        assert body["classes"] == [{"class": "TimeoutError"}]

    @pytest.mark.asyncio
    async def test_errors_503_on_failure(self):
        with (
            patch.object(
                queries, "errors", new=AsyncMock(side_effect=RuntimeError("down"))
            ),
            pytest.raises(HTTPException) as exc,
        ):
            await ar.analytics_errors(days=7, _token="t")
        assert exc.value.status_code == 503


class TestUsers:
    @pytest.mark.asyncio
    async def test_404_when_the_deployment_has_not_enabled_it(self):
        breakdown = queries.UserBreakdown(enabled=False, reason="disabled by config")
        with (
            patch.object(ar, "_expose_users", return_value=False),
            patch.object(queries, "by_user", new=AsyncMock(return_value=breakdown)),
            pytest.raises(HTTPException) as exc,
        ):
            await ar.analytics_users(days=7, limit=50, _token="t")

        # Not an empty list: that reads as "nobody used it", which is
        # the wrong answer to a question nobody asked.
        assert exc.value.status_code == 404
        assert exc.value.detail == "disabled by config"

    @pytest.mark.asyncio
    async def test_returns_rows_when_enabled(self):
        breakdown = queries.UserBreakdown(
            enabled=True, rows=[queries.UserRow(user_id="alice", runs=3)]
        )
        with (
            patch.object(ar, "_expose_users", return_value=True),
            patch.object(queries, "by_user", new=AsyncMock(return_value=breakdown)),
        ):
            body = await ar.analytics_users(days=7, limit=50, _token="t")

        assert body["users"][0]["user_id"] == "alice"

    @pytest.mark.asyncio
    async def test_503_on_failure(self):
        with (
            patch.object(ar, "_expose_users", return_value=True),
            patch.object(
                queries, "by_user", new=AsyncMock(side_effect=RuntimeError("down"))
            ),
            pytest.raises(HTTPException) as exc,
        ):
            await ar.analytics_users(days=7, limit=50, _token="t")
        assert exc.value.status_code == 503
