"""Unit tests for exponential decay scoring."""

from datetime import datetime, timedelta, timezone

from deep_agent.src.memory.scoring import (
    MIN_SCORE,
    apply_access_boost,
    compute_decay_score,
)


class TestComputeDecayScore:
    def test_fresh_memory_keeps_score(self):
        now = datetime.now(timezone.utc)
        score = compute_decay_score(1.0, now, now)
        assert score == 1.0

    def test_old_memory_decays(self):
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=30)
        score = compute_decay_score(1.0, old, now)
        assert score < 1.0
        assert score > MIN_SCORE

    def test_very_old_memory_near_min(self):
        now = datetime.now(timezone.utc)
        ancient = now - timedelta(days=365)
        score = compute_decay_score(1.0, ancient, now)
        assert score <= 0.05

    def test_never_below_min(self):
        now = datetime.now(timezone.utc)
        ancient = now - timedelta(days=10000)
        score = compute_decay_score(1.0, ancient, now)
        assert score >= MIN_SCORE

    def test_naive_datetime_handled(self):
        now = datetime.now(timezone.utc)
        naive = datetime.utcnow()
        score = compute_decay_score(1.0, naive, now)
        assert 0.99 < score <= 1.0

    def test_zero_age(self):
        now = datetime.now(timezone.utc)
        assert compute_decay_score(0.5, now, now) == 0.5


class TestAccessBoost:
    def test_boost_increases_score(self):
        assert apply_access_boost(0.5) == 0.6

    def test_boost_capped_at_one(self):
        assert apply_access_boost(0.95) == 1.0

    def test_boost_from_zero(self):
        assert apply_access_boost(0.0) == 0.1
