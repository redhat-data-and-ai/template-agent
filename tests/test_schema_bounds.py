"""Tests for input validation bounds on deep research fields."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from template_agent.src.schema import StreamRequest


class TestDeepResearchFieldBounds:
    """Verify ge/le constraints on deep research override fields."""

    def test_max_subqueries_valid(self):
        """Should accept values within bounds."""
        req = StreamRequest(message="test", deep_research_max_subqueries=15)
        assert req.deep_research_max_subqueries == 15

    def test_max_subqueries_lower_bound(self):
        """Should reject values below 1."""
        with pytest.raises(ValidationError):
            StreamRequest(message="test", deep_research_max_subqueries=0)

    def test_max_subqueries_upper_bound(self):
        """Should reject values above 30."""
        with pytest.raises(ValidationError):
            StreamRequest(message="test", deep_research_max_subqueries=31)

    def test_max_subqueries_extreme(self):
        """Should reject extremely large values."""
        with pytest.raises(ValidationError):
            StreamRequest(message="test", deep_research_max_subqueries=10000)

    def test_max_supervisor_rounds_valid(self):
        """Should accept values within bounds."""
        req = StreamRequest(message="test", deep_research_max_supervisor_rounds=5)
        assert req.deep_research_max_supervisor_rounds == 5

    def test_max_supervisor_rounds_lower_bound(self):
        """Should reject values below 1."""
        with pytest.raises(ValidationError):
            StreamRequest(message="test", deep_research_max_supervisor_rounds=0)

    def test_max_supervisor_rounds_upper_bound(self):
        """Should reject values above 10."""
        with pytest.raises(ValidationError):
            StreamRequest(message="test", deep_research_max_supervisor_rounds=11)

    def test_max_review_iterations_valid_zero(self):
        """Should accept 0 (disable review)."""
        req = StreamRequest(message="test", deep_research_max_review_iterations=0)
        assert req.deep_research_max_review_iterations == 0

    def test_max_review_iterations_upper_bound(self):
        """Should reject values above 5."""
        with pytest.raises(ValidationError):
            StreamRequest(message="test", deep_research_max_review_iterations=6)

    def test_max_review_iterations_negative(self):
        """Should reject negative values."""
        with pytest.raises(ValidationError):
            StreamRequest(message="test", deep_research_max_review_iterations=-1)

    def test_none_values_still_allowed(self):
        """None should still be accepted (no override)."""
        req = StreamRequest(
            message="test",
            deep_research_max_subqueries=None,
            deep_research_max_supervisor_rounds=None,
            deep_research_max_review_iterations=None,
        )
        assert req.deep_research_max_subqueries is None
        assert req.deep_research_max_supervisor_rounds is None
        assert req.deep_research_max_review_iterations is None
