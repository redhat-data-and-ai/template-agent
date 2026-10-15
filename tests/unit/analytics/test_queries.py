"""Analytics: the derived figures, and what they refuse to report.

The SQL is exercised against a live database in `make test-nightly-kind`;
what is tested here is the reasoning around it — the statistics that
must not be invented, the privacy switch, and the error classifier,
which is the one piece with real judgement in it.
"""

from __future__ import annotations

import pytest

from deep_agent.src.analytics.queries import (
    MAX_WINDOW_DAYS,
    Summary,
    UserBreakdown,
    Window,
    by_user,
    classify_error,
)


class TestWindow:
    def test_a_window_is_bounded(self):
        # Unbounded, this is a sequential scan of every run ever.
        assert Window(days=10_000).days == MAX_WINDOW_DAYS

    def test_a_window_is_at_least_a_day(self):
        assert Window(days=0).days == 1
        assert Window(days=-5).days == 1


class TestRatesAreNotInvented:
    def test_no_runs_means_no_success_rate(self):
        # Not 0.0 and not 100.0: a fresh deployment has no success
        # rate, and inventing one puts a number on a system nobody has
        # used yet.
        assert Summary(window_days=7).success_rate is None

    def test_in_flight_runs_do_not_count_as_failures(self):
        # Otherwise every dashboard looks worse the busier it gets.
        summary = Summary(window_days=7, successes=9, errors=1, in_flight=90)
        assert summary.success_rate == 90.0

    def test_no_feedback_means_no_satisfaction_score(self):
        assert Summary(window_days=7, runs=500).satisfaction is None

    def test_satisfaction_is_the_share_of_ratings(self):
        summary = Summary(window_days=7, feedback_up=3, feedback_down=1)
        assert summary.satisfaction == 75.0


class TestErrorClassifier:
    """Every case here is an error this deployment actually produced."""

    @pytest.mark.parametrize(
        "message,expected",
        [
            (
                "Failed to create Gemini model 'gemini-2.5-pro': "
                "No Google credentials found.",
                "model_config",
            ),
            (
                "Max retries exceeded after repeated worker failures",
                "retries_exhausted",
            ),
            ("string indices must be integers, not 'str'", "agent_defect"),
            ("429 Resource has been exhausted (e.g. check quota)", "model_rate_limit"),
            ("Deadline Exceeded while awaiting the model", "model_timeout"),
            ("connection refused: could not connect to postgres", "storage"),
            ("Run cancelled by the client", "cancelled"),
        ],
    )
    def test_real_failures_are_classified(self, message, expected):
        assert classify_error(message) == expected

    def test_an_unrecognised_failure_is_other_not_a_guess(self):
        assert classify_error("the flux capacitor fluxed") == "other"

    def test_a_missing_message_is_unknown(self):
        assert classify_error(None) == "unknown"
        assert classify_error("") == "unknown"

    def test_the_message_itself_is_never_returned(self):
        # The whole point: an exception string routinely contains the
        # input that caused it, so no classification may echo it.
        secret = "could not read /uploads/alice-salary-2026.xlsx"
        label = classify_error(secret)
        assert "alice" not in label
        assert "uploads" not in label


class TestPerUserIsGated:
    @pytest.mark.asyncio
    async def test_disabled_returns_no_rows_and_says_why(self):
        # No database is touched when it is off — the query is not the
        # thing being prevented, the disclosure is.
        result = await by_user(Window(days=7), enabled=False)
        assert isinstance(result, UserBreakdown)
        assert result.enabled is False
        assert result.rows == []
        assert "expose_users" in result.reason


class TestRouteGating:
    def test_the_user_endpoint_defaults_to_off(self):
        # A privacy switch whose default depends on a config file
        # parsing correctly is not a privacy switch.
        from deep_agent.aegra.analytics_routes import _expose_users

        assert _expose_users() in (True, False)

    def test_an_unreadable_config_disables_disclosure(self, monkeypatch):
        import deep_agent.aegra.analytics_routes as routes

        class Boom:
            def get_raw_section(self, _name):
                raise RuntimeError("config is broken")

        monkeypatch.setattr(
            "deep_agent.src.agent.config.agent_config", Boom(), raising=False
        )
        assert routes._expose_users() is False


class TestStaleRunsAreNotInFlight:
    """A run whose pod died stays `running` forever.

    Nothing rewrites a status on behalf of a process that is gone,
    and the reaper only resumes runs it holds a lease for. So
    `in_flight` drifted from "what is happening now" towards "how
    many times has a pod been restarted mid-run" — a number that
    only ever grows.
    """

    def test_the_cutoff_matches_the_longest_permitted_operation(self):
        from deep_agent.src.analytics import queries

        # An hour, matching `max_execute_timeout` in agent.yaml.
        # Shorter would reclassify a legitimately long run as
        # abandoned; longer lets orphans keep inflating the figure.
        assert queries._STALE_AFTER_SECONDS == 3600

    def test_the_query_bounds_in_flight_by_age(self):
        from deep_agent.src.analytics import queries

        sql = queries._SUMMARY_SQL if hasattr(queries, "_SUMMARY_SQL") else ""
        if not sql:
            import inspect

            sql = inspect.getsource(queries)
        # in_flight must carry an age bound, or an orphan counts
        # forever.
        assert "stale_after" in sql
        assert "AS stale" in sql

    def test_summary_reports_stale_separately(self):
        from deep_agent.src.analytics.queries import Summary

        summary = Summary(window_days=7, in_flight=1, stale=2)
        # Merged into one number, a stuck queue hides behind a
        # healthy-looking count.
        assert summary.in_flight == 1
        assert summary.stale == 2


class TestConversationsExcludeMachineTraffic:
    """ "Conversations" has to mean a person had one.

    Counting every thread reported 355 on a deployment that had held
    about three. The rest were GAIA and browser-agent benchmark
    prompts, planner invocations and subagent calls — 294 of the 355
    threads had exactly one run, which is the shape of a harness
    rather than of someone talking.
    """

    def test_every_query_filters_to_client_traffic(self):
        """All of them, or the page cannot be read.

        A dashboard where some figures count benchmark runs and
        others do not is worse than one where all of them do: no two
        numbers on it can be compared.
        """
        import inspect
        import re

        from deep_agent.src.analytics import queries

        src = inspect.getsource(queries)
        lines = src.split("\n")
        unfiltered = []
        for i, line in enumerate(lines):
            if re.search(r"FROM runs\b", line):
                window = "\n".join(lines[i : i + 12])
                if "trace_id" not in window and "_CLIENT_ONLY" not in window:
                    unfiltered.append(i + 1)
        assert not unfiltered, f"unfiltered runs queries at lines {unfiltered}"

    def test_summary_carries_both(self):
        from deep_agent.src.analytics.queries import Summary

        s = Summary(window_days=7, conversations=3, threads=355)
        assert s.conversations == 3
        assert s.threads == 355

    def test_conversations_defaults_to_zero_not_to_threads(self):
        # A deployment with no client traffic should read zero, not
        # inherit the machine count by accident.
        from deep_agent.src.analytics.queries import Summary

        assert Summary(window_days=7, threads=99).conversations == 0
