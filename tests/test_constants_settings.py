"""Tests for constants.py referencing settings.py values."""

from __future__ import annotations

from template_agent.src.core.deep_research.constants import (
    DEEP_RESEARCH_COMPLETENESS_THRESHOLD,
    DEEP_RESEARCH_MAX_SUPERVISOR_ROUNDS,
    DEEP_RESEARCH_MAX_TOTAL_SUBQUERIES,
)
from template_agent.src.settings import settings


class TestConstantsMatchSettings:
    """Verify constants are driven from settings."""

    def test_max_supervisor_rounds(self):
        """Should equal the settings value."""
        assert (
            DEEP_RESEARCH_MAX_SUPERVISOR_ROUNDS
            == settings.DEEP_RESEARCH_MAX_SUPERVISOR_ROUNDS
        )

    def test_max_total_subqueries(self):
        """Should equal the settings value."""
        assert (
            DEEP_RESEARCH_MAX_TOTAL_SUBQUERIES
            == settings.DEEP_RESEARCH_MAX_TOTAL_SUBQUERIES
        )

    def test_completeness_threshold(self):
        """Should equal the settings value."""
        assert (
            DEEP_RESEARCH_COMPLETENESS_THRESHOLD
            == settings.DEEP_RESEARCH_COMPLETENESS_THRESHOLD
        )

    def test_default_values(self):
        """Settings defaults should match historical values."""
        assert settings.DEEP_RESEARCH_MAX_SUPERVISOR_ROUNDS == 3
        assert settings.DEEP_RESEARCH_MAX_TOTAL_SUBQUERIES == 20
        assert settings.DEEP_RESEARCH_COMPLETENESS_THRESHOLD == 70
