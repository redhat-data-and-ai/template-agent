"""Comprehensive pytest tests for the deep research nodes __init__ module."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from template_agent.src.core.deep_research.nodes import (
    PHASE_COMPLETE,
    PHASE_COMPLETENESS,
    PHASE_PLAN,
    PHASE_PROBE,
    PHASE_SUPERVISOR,
    PHASE_SYNTHESIZE,
    PHASE_TRIAGE,
    _aput_checkpoint,
    _hash_subquery,
    assess_complexity_node,
    complete_node,
    completeness_evaluator_node,
    context_answer_node,
    findings_from_board,
    format_full_cached_findings_for_triage,
    load_cached_findings,
    plan_node,
    probe_node,
    research_supervisor_node,
    review_node,
    synthesize_node,
    triage_node,
    visualize_node,
)
from template_agent.src.core.deep_research.state import Finding


class TestFindingsFromBoard:
    """Test findings_from_board utility."""

    def test_findings_from_board_valid_dict(self):
        """Valid board with entries returns legacy findings dict."""
        board = {
            "sq1": {"finding": {"subquery": "q1", "answer": "a1"}},
            "sq2": {"finding": {"subquery": "q2", "answer": "a2"}},
        }
        result = findings_from_board(board)
        assert result == {
            "sq1": {"subquery": "q1", "answer": "a1"},
            "sq2": {"subquery": "q2", "answer": "a2"},
        }

    def test_findings_from_board_non_dict_returns_empty(self):
        """Non-dict input returns empty dict."""
        assert findings_from_board(None) == {}
        assert findings_from_board([]) == {}
        assert findings_from_board("not a dict") == {}

    def test_findings_from_board_empty_dict(self):
        """Empty board returns empty dict."""
        result = findings_from_board({})
        assert result == {}

    def test_findings_from_board_entry_without_finding_uses_empty_dict(self):
        """Entry without 'finding' key uses empty dict."""
        board = {"sq1": {"other": "data"}}
        result = findings_from_board(board)
        assert result == {"sq1": {}}

    def test_findings_from_board_none_finding_uses_empty_dict(self):
        """Entry with finding=None uses empty dict."""
        board = {"sq1": {"finding": None}}
        result = findings_from_board(board)
        assert result == {"sq1": {}}


class TestFormatFullCachedFindingsForTriage:
    """Test format_full_cached_findings_for_triage utility."""

    def test_format_full_cached_findings_empty(self):
        """Empty findings returns empty string."""
        result = format_full_cached_findings_for_triage({})
        assert result == ""

    def test_format_full_cached_findings_with_entries(self):
        """Findings with subquery and answer are formatted."""
        findings = {
            "h1": Finding(subquery="What is X?", answer="X is 42", cached=False),
            "h2": Finding(subquery="What is Y?", answer="Y is 99", cached=False),
        }
        result = format_full_cached_findings_for_triage(findings)
        assert "### Subquery: What is X?" in result
        assert "X is 42" in result
        assert "### Subquery: What is Y?" in result
        assert "Y is 99" in result

    def test_format_full_cached_findings_respects_max_chars(self):
        """Output is truncated when exceeding max_chars."""
        findings = {
            "h1": Finding(
                subquery="Q1",
                answer="A" * 5000,
                cached=False,
            ),
            "h2": Finding(
                subquery="Q2",
                answer="B" * 5000,
                cached=False,
            ),
        }
        result = format_full_cached_findings_for_triage(findings, max_chars=200)
        assert "truncated for length" in result or len(result) <= 250

    def test_format_full_cached_findings_skips_empty_subquery(self):
        """Findings without subquery are skipped."""
        findings = {
            "h1": Finding(subquery="", answer="orphan answer", cached=False),
        }
        result = format_full_cached_findings_for_triage(findings)
        assert result == ""


class TestHashSubquery:
    """Test _hash_subquery helper."""

    def test_hash_subquery_deterministic(self):
        """Same subquery produces same hash."""
        h1 = _hash_subquery("What is the revenue?")
        h2 = _hash_subquery("What is the revenue?")
        assert h1 == h2
        assert len(h1) == 16

    def test_hash_subquery_case_insensitive(self):
        """Hash is case-insensitive (lowercased before hashing)."""
        h1 = _hash_subquery("REVENUE")
        h2 = _hash_subquery("revenue")
        assert h1 == h2

    def test_hash_subquery_strips_whitespace(self):
        """Leading/trailing whitespace is stripped."""
        h1 = _hash_subquery("  revenue  ")
        h2 = _hash_subquery("revenue")
        assert h1 == h2

    def test_hash_subquery_empty_returns_hash(self):
        """Empty string still returns 16-char hash."""
        h = _hash_subquery("")
        assert len(h) == 16
        assert isinstance(h, str)


class TestLoadCachedFindings:
    """Test load_cached_findings async utility."""

    @pytest.mark.asyncio
    async def test_load_cached_findings_no_checkpointer(self):
        """None checkpointer returns empty dict."""
        result = await load_cached_findings(None, "thread-1")
        assert result == {}

    @pytest.mark.asyncio
    async def test_load_cached_findings_no_thread_id(self):
        """None thread_id returns empty dict."""
        result = await load_cached_findings(MagicMock(), None)
        assert result == {}

    @pytest.mark.asyncio
    async def test_load_cached_findings_success(self):
        """Valid checkpointer and thread_id loads findings."""
        checkpointer = AsyncMock()
        mock_result = MagicMock()
        mock_result.metadata = {
            "deep_research_findings": [
                {"subquery": "Q1", "answer": "A1", "tool_results": []},
                {"subquery": "Q2", "answer": "A2"},
            ],
        }
        checkpointer.aget_tuple = AsyncMock(return_value=mock_result)

        result = await load_cached_findings(checkpointer, "thread-123")

        assert len(result) == 2
        for finding in result.values():
            assert "subquery" in finding
            assert "answer" in finding

    @pytest.mark.asyncio
    async def test_load_cached_findings_empty_metadata_returns_empty(self):
        """Checkpointer returns result without metadata."""
        checkpointer = AsyncMock()
        checkpointer.aget_tuple = AsyncMock(return_value=MagicMock(metadata=None))

        result = await load_cached_findings(checkpointer, "thread-1")

        assert result == {}

    @pytest.mark.asyncio
    async def test_load_cached_findings_exception_returns_empty(self):
        """Exception during load returns empty dict."""
        checkpointer = AsyncMock()
        checkpointer.aget_tuple = AsyncMock(side_effect=RuntimeError("DB error"))

        result = await load_cached_findings(checkpointer, "thread-1")

        assert result == {}


class TestAputCheckpoint:
    """Test _aput_checkpoint async utility."""

    @pytest.mark.asyncio
    async def test_aput_checkpoint_calls_aput_tuple(self):
        """Checkpointer with aput_tuple has it called when aput is absent."""
        checkpointer = MagicMock(spec=["aput_tuple"])
        checkpointer.aput_tuple = AsyncMock()

        config = MagicMock()
        checkpoint = {}
        metadata = {}
        channel_versions = {}

        await _aput_checkpoint(
            checkpointer, config, checkpoint, metadata, channel_versions
        )

        checkpointer.aput_tuple.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_aput_checkpoint_prefers_aput_when_present(self):
        """Checkpointer with aput uses it instead of aput_tuple."""
        checkpointer = MagicMock()
        checkpointer.aput = AsyncMock()
        checkpointer.aput_tuple = AsyncMock()

        config = MagicMock()
        checkpoint = {}
        metadata = {}

        await _aput_checkpoint(checkpointer, config, checkpoint, metadata, {})

        checkpointer.aput.assert_awaited_once()
        checkpointer.aput_tuple.assert_not_awaited()


class TestPhaseConstants:
    """Test phase constants are exported."""

    def test_phase_constants_exist(self):
        """All phase constants are defined and non-empty."""
        constants = [
            PHASE_TRIAGE,
            PHASE_PROBE,
            PHASE_PLAN,
            PHASE_SUPERVISOR,
            PHASE_COMPLETENESS,
            PHASE_SYNTHESIZE,
            PHASE_COMPLETE,
        ]
        for c in constants:
            assert c is not None
            assert isinstance(c, str)
            assert len(c) > 0

    def test_phase_constants_have_expected_values(self):
        """Phase constants have expected string values."""
        assert PHASE_TRIAGE == "triage"
        assert PHASE_PROBE == "probe"
        assert PHASE_PLAN == "plan"
        assert PHASE_SYNTHESIZE == "synthesize"
        assert PHASE_COMPLETE == "complete"


class TestNodeExports:
    """Test node functions are exported from __init__."""

    def test_node_functions_are_callable(self):
        """All node functions are exported and callable."""
        nodes = [
            assess_complexity_node,
            context_answer_node,
            triage_node,
            probe_node,
            plan_node,
            research_supervisor_node,
            completeness_evaluator_node,
            synthesize_node,
            visualize_node,
            review_node,
            complete_node,
        ]
        for node_fn in nodes:
            assert callable(node_fn)
