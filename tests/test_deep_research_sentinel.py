"""Tests for deep research sentinel (loop circuit breaker) module."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from template_agent.src.core.deep_research.sentinel import (
    PHASE_COMPLETE,
    PHASE_SYNTHESIZE,
    check_cancellation,
    check_loop_sentinel,
    trim_follow_ups,
)
from template_agent.src.core.deep_research.state import DeepResearchState


def _make_state(**kwargs: object) -> DeepResearchState:
    """Build a minimal DeepResearchState-like dict for tests."""
    defaults: dict[str, object] = {
        "query": "test query",
        "thread_id": "test-thread",
        "current_phase": "planning",
        "findings_board": {},
        "findings": {},
        "execution_start_time": time.time(),
        "max_session_seconds": 600.0,
        "total_subqueries_executed": 0,
        "max_total_subqueries": 50,
        "total_node_transitions": 0,
        "max_node_transitions": 100,
        "findings_count_history": [],
    }
    defaults.update(kwargs)
    return defaults  # type: ignore[return-value]


class TestCheckCancellation:
    """Test cases for check_cancellation async function."""

    @pytest.mark.asyncio
    async def test_check_cancellation_no_thread_id_returns_false(self) -> None:
        """When thread_id is empty, cancellation is not checked."""
        state = _make_state(thread_id="")
        cancelled, reason, phase = await check_cancellation(state)
        assert cancelled is False
        assert reason is None
        assert phase is None

    @pytest.mark.asyncio
    async def test_check_cancellation_not_cancelled_returns_false(self) -> None:
        """When store says not cancelled, returns (False, None, None)."""
        state = _make_state()
        mock_store = AsyncMock()
        mock_store.is_cancelled.return_value = False

        with patch(
            "template_agent.src.core.deep_research.sentinel.get_cancel_store",
            return_value=mock_store,
        ):
            cancelled, reason, phase = await check_cancellation(state)

        assert cancelled is False
        assert reason is None
        assert phase is None
        mock_store.is_cancelled.assert_called_once_with("test-thread")

    @pytest.mark.asyncio
    async def test_check_cancellation_cancelled_with_findings_returns_synthesize(
        self,
    ) -> None:
        """When cancelled and findings exist, forced_phase is PHASE_SYNTHESIZE."""
        state = _make_state(
            findings_board={
                "sq1": {"finding": {"subquery": "q", "answer": "a", "error": None}},
            }
        )
        mock_store = AsyncMock()
        mock_store.is_cancelled.return_value = True

        with patch(
            "template_agent.src.core.deep_research.sentinel.get_cancel_store",
            return_value=mock_store,
        ):
            cancelled, reason, phase = await check_cancellation(state)

        assert cancelled is True
        assert reason == "Cancelled by user"
        assert phase == PHASE_SYNTHESIZE

    @pytest.mark.asyncio
    async def test_check_cancellation_cancelled_no_findings_returns_complete(
        self,
    ) -> None:
        """When cancelled and no valid findings, forced_phase is PHASE_COMPLETE."""
        state = _make_state(findings_board={}, findings={})
        mock_store = AsyncMock()
        mock_store.is_cancelled.return_value = True

        with patch(
            "template_agent.src.core.deep_research.sentinel.get_cancel_store",
            return_value=mock_store,
        ):
            cancelled, reason, phase = await check_cancellation(state)

        assert cancelled is True
        assert reason == "Cancelled by user"
        assert phase == PHASE_COMPLETE


class TestCheckLoopSentinel:
    """Test cases for check_loop_sentinel function."""

    @patch("template_agent.src.core.deep_research.sentinel._get_setting")
    def test_check_loop_sentinel_no_findings_skips_soft_budgets(
        self, mock_get_setting: MagicMock
    ) -> None:
        """When no findings, supervisor/completeness nodes skip soft budget checks."""
        mock_get_setting.side_effect = lambda name, default: default
        state = _make_state(
            findings_board={},
            execution_start_time=time.time() - 10,
            max_session_seconds=600,
        )
        ctx = None

        should_stop, reason, phase = check_loop_sentinel(
            state, ctx, node_name="supervisor"
        )

        assert should_stop is False
        assert reason is None
        assert phase is None

    @patch("template_agent.src.core.deep_research.sentinel._get_setting")
    def test_check_loop_sentinel_subquery_budget_exhausted(
        self, mock_get_setting: MagicMock
    ) -> None:
        """Subquery budget exhausted triggers sentinel."""
        mock_get_setting.side_effect = lambda name, default: default
        state = _make_state(
            findings_board={"sq1": {"finding": {"subquery": "q", "answer": "a"}}},
            total_subqueries_executed=50,
            max_total_subqueries=50,
            execution_start_time=time.time() - 5,
        )

        should_stop, reason, phase = check_loop_sentinel(
            state, None, node_name="supervisor"
        )

        assert should_stop is True
        assert "Subquery budget exhausted" in (reason or "")
        assert phase == PHASE_SYNTHESIZE

    @patch("template_agent.src.core.deep_research.sentinel._get_setting")
    def test_check_loop_sentinel_node_transition_budget_exhausted(
        self, mock_get_setting: MagicMock
    ) -> None:
        """Node transition budget exhausted triggers sentinel."""
        mock_get_setting.side_effect = lambda name, default: default
        state = _make_state(
            findings_board={"sq1": {"finding": {"subquery": "q", "answer": "a"}}},
            total_node_transitions=100,
            max_node_transitions=100,
            total_subqueries_executed=5,
            max_total_subqueries=50,
        )

        should_stop, reason, phase = check_loop_sentinel(
            state, None, node_name="supervisor"
        )

        assert should_stop is True
        assert "Node transition budget exhausted" in (reason or "")
        assert phase == PHASE_SYNTHESIZE

    @patch("template_agent.src.core.deep_research.sentinel._get_setting")
    @patch("template_agent.src.core.deep_research.sentinel.time")
    def test_check_loop_sentinel_wall_clock_timeout(
        self, mock_time: MagicMock, mock_get_setting: MagicMock
    ) -> None:
        """Wall clock timeout triggers sentinel."""
        mock_get_setting.side_effect = lambda name, default: default
        start = 1000.0
        mock_time.time.return_value = start + 700
        state = _make_state(
            findings_board={"sq1": {"finding": {"subquery": "q", "answer": "a"}}},
            execution_start_time=start,
            max_session_seconds=600,
            total_subqueries_executed=5,
            total_node_transitions=5,
        )

        should_stop, reason, phase = check_loop_sentinel(
            state, None, node_name="supervisor"
        )

        assert should_stop is True
        assert "Session timeout" in (reason or "")
        assert phase == PHASE_SYNTHESIZE

    @patch("template_agent.src.core.deep_research.sentinel._get_setting")
    @patch("template_agent.src.core.deep_research.sentinel.time")
    def test_check_loop_sentinel_hard_ceiling_breached(
        self, mock_time: MagicMock, mock_get_setting: MagicMock
    ) -> None:
        """Hard ceiling (2x budget) always fires regardless of findings."""
        mock_get_setting.side_effect = lambda name, default: default
        start = 1000.0
        mock_time.time.return_value = start + 1300
        state = _make_state(
            findings_board={},
            execution_start_time=start,
            max_session_seconds=600,
        )

        should_stop, reason, phase = check_loop_sentinel(
            state, None, node_name="supervisor"
        )

        assert should_stop is True
        assert "Hard ceiling breached" in (reason or "")
        assert phase == PHASE_COMPLETE

    @patch("template_agent.src.core.deep_research.sentinel._get_setting")
    def test_check_loop_sentinel_stagnation_detected(
        self, mock_get_setting: MagicMock
    ) -> None:
        """Stagnation (unchanged findings count for N rounds) triggers sentinel."""
        mock_get_setting.side_effect = lambda name, default: (
            3 if "STAGNATION" in str(name) else default
        )
        state = _make_state(
            findings_board={"sq1": {"finding": {"subquery": "q", "answer": "a"}}},
            execution_start_time=time.time() - 5,
            total_subqueries_executed=5,
            total_node_transitions=5,
            findings_count_history=[2, 2, 2],
        )

        should_stop, reason, phase = check_loop_sentinel(
            state, None, node_name="supervisor"
        )

        assert should_stop is True
        assert "stagnated" in (reason or "").lower()
        assert phase == PHASE_SYNTHESIZE

    @patch("template_agent.src.core.deep_research.sentinel._get_setting")
    def test_check_loop_sentinel_no_stagnation_when_count_varies(
        self, mock_get_setting: MagicMock
    ) -> None:
        """Stagnation not triggered when findings count varies."""
        mock_get_setting.side_effect = lambda name, default: (
            3 if "STAGNATION" in str(name) else default
        )
        state = _make_state(
            findings_board={"sq1": {"finding": {"subquery": "q", "answer": "a"}}},
            execution_start_time=time.time() - 5,
            total_subqueries_executed=5,
            total_node_transitions=5,
            findings_count_history=[1, 2, 3],
        )

        should_stop, reason, _ = check_loop_sentinel(
            state, None, node_name="supervisor"
        )

        assert should_stop is False
        assert reason is None

    @patch("template_agent.src.core.deep_research.sentinel._get_setting")
    def test_check_loop_sentinel_zero_start_time_skips_wall_clock(
        self, mock_get_setting: MagicMock
    ) -> None:
        """Zero execution_start_time skips wall clock check."""
        mock_get_setting.side_effect = lambda name, default: default
        state = _make_state(
            findings_board={"sq1": {"finding": {"subquery": "q", "answer": "a"}}},
            execution_start_time=0,
            max_session_seconds=600,
            total_subqueries_executed=5,
            total_node_transitions=5,
            findings_count_history=[],
        )

        should_stop, reason, _ = check_loop_sentinel(
            state, None, node_name="synthesize"
        )

        assert should_stop is False
        assert reason is None


class TestTrimFollowUps:
    """Test cases for trim_follow_ups function."""

    def test_trim_follow_ups_within_budget_returns_all(self) -> None:
        """When within budget, all follow-ups are returned."""
        follow_ups = ["q1", "q2", "q3"]
        result = trim_follow_ups(follow_ups, total_executed=5, max_total=20)
        assert result == follow_ups

    def test_trim_follow_ups_exceeds_budget_trims(self) -> None:
        """When exceeding budget, follow-ups are trimmed to remaining."""
        follow_ups = ["q1", "q2", "q3", "q4", "q5"]
        result = trim_follow_ups(follow_ups, total_executed=18, max_total=20)
        assert result == ["q1", "q2"]

    def test_trim_follow_ups_zero_max_returns_all(self) -> None:
        """When max_total is 0, budget check is skipped (returns all)."""
        follow_ups = ["q1", "q2"]
        result = trim_follow_ups(follow_ups, total_executed=10, max_total=0)
        assert result == follow_ups

    def test_trim_follow_ups_already_at_budget_returns_empty(self) -> None:
        """When already at budget, no follow-ups remain."""
        follow_ups = ["q1", "q2"]
        result = trim_follow_ups(follow_ups, total_executed=20, max_total=20)
        assert result == []

    def test_trim_follow_ups_over_budget_returns_empty(self) -> None:
        """When over budget, remaining is 0."""
        follow_ups = ["q1", "q2"]
        result = trim_follow_ups(follow_ups, total_executed=25, max_total=20)
        assert result == []
