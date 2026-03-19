"""Shared constants for the deep research pipeline.

Extracted from supervisor, streaming, and completeness modules to avoid
duplication and ensure consistency across the pipeline.
Values are driven by settings.py so they can be overridden via environment
variables without code changes.
"""

from template_agent.src.settings import settings

DEEP_RESEARCH_MAX_SUPERVISOR_ROUNDS: int = settings.DEEP_RESEARCH_MAX_SUPERVISOR_ROUNDS
DEEP_RESEARCH_MAX_TOTAL_SUBQUERIES: int = settings.DEEP_RESEARCH_MAX_TOTAL_SUBQUERIES
DEEP_RESEARCH_COMPLETENESS_THRESHOLD: int = (
    settings.DEEP_RESEARCH_COMPLETENESS_THRESHOLD
)
