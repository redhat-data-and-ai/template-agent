"""Comprehensive pytest tests for deep research node cache utilities."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from template_agent.src.core.deep_research.nodes._cache import (
    find_matching_cached_finding,
    format_cached_findings_for_prompt,
    format_cached_findings_for_triage,
    format_conversation_for_prompt,
    hash_subquery,
    load_cached_findings,
    load_conversation_history,
    load_findings_in_memory,
    normalize_subquery,
    save_cached_findings,
    save_conversation_turn,
    save_findings_in_memory,
)

# ---------------------------------------------------------------------------
# Normalization and hashing
# ---------------------------------------------------------------------------


class TestNormalizeSubquery:
    """Tests for normalize_subquery."""

    def test_lowercases_and_strips(self):
        """Lowercase and strip whitespace."""
        result = normalize_subquery("  What Is Revenue?  ")
        assert result == "what is revenue"

    def test_removes_punctuation(self):
        """Remove common punctuation."""
        result = normalize_subquery("What is revenue? Yes!")
        assert "?" not in result
        assert "!" not in result

    def test_collapses_whitespace(self):
        """Collapse multiple spaces to single space."""
        result = normalize_subquery("What   is   revenue")
        assert "  " not in result

    def test_handles_none_and_empty(self):
        """Handle None and empty string."""
        assert normalize_subquery(None) == ""
        assert normalize_subquery("") == ""


class TestHashSubquery:
    """Tests for hash_subquery."""

    def test_returns_consistent_hash_for_same_input(self):
        """Return same hash for same normalized input."""
        h1 = hash_subquery("What is revenue?")
        h2 = hash_subquery("What is revenue?")
        assert h1 == h2

    def test_returns_16_char_hex_string(self):
        """Return 16-character hex string."""
        result = hash_subquery("test query")
        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)

    def test_different_queries_produce_different_hashes(self):
        """Different queries produce different hashes."""
        h1 = hash_subquery("Query one")
        h2 = hash_subquery("Query two")
        assert h1 != h2


# ---------------------------------------------------------------------------
# Find matching cached finding
# ---------------------------------------------------------------------------


class TestFindMatchingCachedFinding:
    """Tests for find_matching_cached_finding."""

    def test_returns_none_when_no_cached_findings(self):
        """Return None when cached_findings is empty."""
        result = find_matching_cached_finding("What is X?", {})
        assert result is None

    def test_finds_by_hash(self):
        """Find by query hash when subquery matches."""
        finding = {"subquery": "What is revenue?", "answer": "100"}
        cached = {hash_subquery("What is revenue?"): finding}
        result = find_matching_cached_finding("What is revenue?", cached)
        assert result == finding

    def test_finds_by_normalized_match(self):
        """Find by normalized string match when hash differs."""
        finding = {"subquery": "What is revenue?", "answer": "100"}
        cached = {"some_hash": finding}
        result = find_matching_cached_finding("What is revenue?", cached)
        assert result == finding

    def test_returns_none_when_no_match(self):
        """Return None when no matching finding."""
        finding = {"subquery": "Different query", "answer": "x"}
        cached = {"h1": finding}
        result = find_matching_cached_finding("Completely unrelated query", cached)
        assert result is None


# ---------------------------------------------------------------------------
# Load / save cached findings (async, with checkpointer)
# ---------------------------------------------------------------------------


class TestLoadCachedFindings:
    """Tests for load_cached_findings."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_checkpointer(self):
        """Return empty dict when checkpointer is None."""
        result = await load_cached_findings(None, "t1")
        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_thread_id(self):
        """Return empty dict when thread_id is None."""
        result = await load_cached_findings(MagicMock(), None)
        assert result == {}

    @pytest.mark.asyncio
    async def test_loads_findings_from_metadata(self):
        """Load findings from checkpointer metadata."""
        checkpointer = AsyncMock()
        metadata = {
            "deep_research_findings": [
                {
                    "subquery": "Q1",
                    "answer": "A1",
                    "tool_results": [],
                    "error": None,
                    "cached": True,
                }
            ]
        }
        result_tuple = MagicMock()
        result_tuple.metadata = metadata
        result_tuple.checkpoint = {}
        checkpointer.aget_tuple = AsyncMock(return_value=result_tuple)

        result = await load_cached_findings(checkpointer, "t1")
        assert len(result) == 1
        finding = list(result.values())[0]
        assert finding["subquery"] == "Q1"
        assert finding["answer"] == "A1"

    @pytest.mark.asyncio
    async def test_returns_empty_on_exception(self):
        """Return empty dict when checkpointer raises."""
        checkpointer = AsyncMock()
        checkpointer.aget_tuple = AsyncMock(side_effect=Exception("DB error"))
        result = await load_cached_findings(checkpointer, "t1")
        assert result == {}


class TestSaveCachedFindings:
    """Tests for save_cached_findings."""

    @pytest.mark.asyncio
    async def test_no_op_when_no_checkpointer(self):
        """No-op when checkpointer is None."""
        await save_cached_findings(None, "t1", [{"subquery": "Q1", "answer": "A1"}])
        # Should not raise

    @pytest.mark.asyncio
    async def test_no_op_when_no_thread_id(self):
        """No-op when thread_id is None."""
        await save_cached_findings(MagicMock(), None, [{"subquery": "Q1"}])
        # Should not raise

    @pytest.mark.asyncio
    async def test_no_op_when_findings_empty(self):
        """No-op when findings_list is empty."""
        checkpointer = AsyncMock()
        await save_cached_findings(checkpointer, "t1", [])
        checkpointer.aget_tuple.assert_not_called()

    @pytest.mark.asyncio
    async def test_saves_findings_via_aput(self):
        """Save findings via checkpointer aput."""
        checkpointer = AsyncMock()
        result_tuple = MagicMock()
        result_tuple.metadata = {}
        result_tuple.checkpoint = {"channel_values": {}}  # truthy for if check
        checkpointer.aget_tuple = AsyncMock(return_value=result_tuple)
        checkpointer.aput = AsyncMock()

        await save_cached_findings(
            checkpointer,
            "t1",
            [{"subquery": "Q1", "answer": "A1", "tool_results": [], "error": None}],
        )
        checkpointer.aput.assert_called_once()
        call_args = checkpointer.aput.call_args
        assert "deep_research_findings" in call_args[0][2]

    @pytest.mark.asyncio
    async def test_saves_via_aput_tuple_when_no_aput(self):
        """Save via aput_tuple when aput not available."""
        checkpointer = MagicMock()
        del checkpointer.aput  # ensure no aput - hasattr will be False
        result_tuple = MagicMock()
        result_tuple.metadata = {}
        result_tuple.checkpoint = {"channel_values": {}}
        checkpointer.aget_tuple = AsyncMock(return_value=result_tuple)
        checkpointer.aput_tuple = AsyncMock()

        await save_cached_findings(
            checkpointer,
            "t1",
            [{"subquery": "Q1", "answer": "A1"}],
        )
        checkpointer.aput_tuple.assert_called_once()


# ---------------------------------------------------------------------------
# Format cached findings
# ---------------------------------------------------------------------------


class TestFormatCachedFindingsForPrompt:
    """Tests for format_cached_findings_for_prompt."""

    def test_returns_empty_when_no_findings(self):
        """Return empty string when no findings."""
        result = format_cached_findings_for_prompt({})
        assert result == ""

    def test_formats_q_a_pairs(self):
        """Format as Q: / A: pairs."""
        findings = {
            "h1": {"subquery": "What is X?", "answer": "X is 42"},
        }
        result = format_cached_findings_for_prompt(findings)
        assert "Q: What is X?" in result
        assert "A:" in result
        assert "42" in result

    def test_respects_max_chars(self):
        """Stop adding entries when max_chars exceeded."""
        findings = {
            f"h{i}": {"subquery": f"Q{i}?", "answer": "A" * 500} for i in range(10)
        }
        result = format_cached_findings_for_prompt(findings, max_chars=500)
        assert len(result) <= 550  # Some buffer


class TestFormatCachedFindingsForTriage:
    """Tests for format_cached_findings_for_triage."""

    def test_returns_empty_when_no_findings(self):
        """Return empty string when no findings."""
        result = format_cached_findings_for_triage({})
        assert result == ""

    def test_formats_with_full_answers(self):
        """Format with full answers (not truncated like prompt version)."""
        long_answer = "This is a long answer. " * 50
        findings = {
            "h1": {"subquery": "What is revenue?", "answer": long_answer},
        }
        result = format_cached_findings_for_triage(findings)
        assert "### Subquery:" in result
        assert "What is revenue?" in result
        assert long_answer[:100] in result

    def test_truncates_when_over_max_chars(self):
        """Add truncation note when over max_chars."""
        findings = {
            f"h{i}": {"subquery": f"Q{i}?", "answer": "A" * 2000} for i in range(20)
        }
        result = format_cached_findings_for_triage(findings, max_chars=5000)
        assert "truncated" in result


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------


class TestFormatConversationForPrompt:
    """Tests for format_conversation_for_prompt."""

    def test_returns_empty_when_no_history(self):
        """Return empty string when history empty."""
        result = format_conversation_for_prompt([])
        assert result == ""

    def test_formats_user_assistant_turns(self):
        """Format as User: / Assistant: turns."""
        history = [
            {"query": "Q1", "answer": "A1"},
            {"query": "Q2", "answer": "A2"},
        ]
        result = format_conversation_for_prompt(history)
        assert "User: Q1" in result
        assert "Assistant: A1" in result
        assert "User: Q2" in result
        assert "Assistant: A2" in result

    def test_truncates_when_over_max_chars(self):
        """Truncate with note when over max_chars."""
        history = [{"query": "Q" * 500, "answer": "A" * 500}] * 10
        result = format_conversation_for_prompt(history, max_chars=500)
        assert "truncated" in result


class TestSaveAndLoadConversationTurn:
    """Tests for save_conversation_turn and load_conversation_history."""

    def test_save_and_load_conversation_turn(self):
        """Save turn and load history."""
        thread_id = "test_thread_conv_1"
        save_conversation_turn(thread_id, "User question", "Assistant answer")
        history = load_conversation_history(thread_id)
        assert len(history) >= 1
        assert history[-1]["query"] == "User question"
        assert history[-1]["answer"] == "Assistant answer"

    def test_save_no_op_when_no_thread_id(self):
        """Save no-op when thread_id empty."""
        save_conversation_turn("", "Q", "A")
        save_conversation_turn(None, "Q", "A")
        # Should not raise

    def test_save_no_op_when_no_query(self):
        """Save no-op when query empty."""
        save_conversation_turn("t_no_query", "", "A")
        # Should not raise

    def test_load_returns_empty_for_unknown_thread(self):
        """Load returns empty list for unknown thread."""
        result = load_conversation_history("nonexistent_thread_xyz_123")
        assert result == []


# ---------------------------------------------------------------------------
# In-memory findings
# ---------------------------------------------------------------------------


class TestSaveAndLoadFindingsInMemory:
    """Tests for save_findings_in_memory and load_findings_in_memory."""

    def test_save_and_load_findings(self):
        """Save findings and load them back."""
        thread_id = "test_thread_mem_1"
        findings_board = {
            "q1": {
                "finding": {
                    "subquery": "What is X?",
                    "answer": "X is 42",
                    "tool_results": [],
                    "error": None,
                }
            },
        }
        save_findings_in_memory(thread_id, findings_board)
        loaded = load_findings_in_memory(thread_id)
        assert len(loaded) >= 1
        finding = list(loaded.values())[0]
        assert finding["subquery"] == "What is X?"
        assert finding["answer"] == "X is 42"

    def test_save_no_op_when_no_thread_id(self):
        """Save no-op when thread_id empty."""
        save_findings_in_memory(
            "", {"q1": {"finding": {"subquery": "Q", "answer": "A"}}}
        )
        # Should not raise

    def test_save_no_op_when_empty_board(self):
        """Save no-op when findings_board empty."""
        save_findings_in_memory("t1", {})

    def test_load_returns_empty_for_unknown_thread(self):
        """Load returns empty dict for unknown thread."""
        result = load_findings_in_memory("nonexistent_mem_xyz")
        assert result == {}
