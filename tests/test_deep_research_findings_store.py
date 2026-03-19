"""Tests for deep research findings store module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from template_agent.src.core.deep_research.findings_store import (
    FINDINGS_NAMESPACE_SUFFIX,
    cross_chat_to_finding_dict,
    delete_findings_for_thread,
    format_cross_chat_findings,
    hash_subquery,
    save_findings_to_store,
    search_cross_chat_findings,
)
from template_agent.src.core.deep_research.state import Finding


def _make_finding(subquery: str, answer: str, **kwargs: object) -> Finding:
    """Build a minimal Finding for tests."""
    base: dict[str, object] = {"subquery": subquery, "answer": answer, **kwargs}
    return base  # type: ignore[return-value]


class TestHashSubquery:
    """Test cases for hash_subquery function."""

    def test_hash_subquery_deterministic(self) -> None:
        """Same subquery produces same hash."""
        h1 = hash_subquery("What is the revenue?")
        h2 = hash_subquery("What is the revenue?")
        assert h1 == h2

    def test_hash_subquery_normalizes_whitespace(self) -> None:
        """Extra whitespace is normalized before hashing."""
        h1 = hash_subquery("What is the revenue?")
        h2 = hash_subquery("  What   is   the   revenue?  ")
        assert h1 == h2

    def test_hash_subquery_normalizes_punctuation(self) -> None:
        """Punctuation is stripped before hashing."""
        h1 = hash_subquery("What is the revenue")
        h2 = hash_subquery("What is the revenue?")
        assert h1 == h2

    def test_hash_subquery_different_queries_different_hashes(self) -> None:
        """Different subqueries produce different hashes."""
        h1 = hash_subquery("What is revenue?")
        h2 = hash_subquery("What is profit?")
        assert h1 != h2

    def test_hash_subquery_empty_returns_hash(self) -> None:
        """Empty string returns a valid hash (16 chars)."""
        h = hash_subquery("")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_subquery_case_insensitive(self) -> None:
        """Hash is case insensitive (normalized to lower)."""
        h1 = hash_subquery("What is the revenue?")
        h2 = hash_subquery("WHAT IS THE REVENUE?")
        assert h1 == h2


class TestSaveFindingsToStore:
    """Test cases for save_findings_to_store async function."""

    @pytest.mark.asyncio
    async def test_save_findings_no_user_id_returns_zero(self) -> None:
        """Missing user_id returns 0."""
        store = MagicMock()
        count = await save_findings_to_store(
            store,
            user_id=None,
            thread_id="t1",
            query="q",
            findings=[_make_finding("sq", "ans")],
        )
        assert count == 0

    @pytest.mark.asyncio
    async def test_save_findings_no_thread_id_returns_zero(self) -> None:
        """Missing thread_id returns 0."""
        store = MagicMock()
        count = await save_findings_to_store(
            store,
            user_id="u1",
            thread_id=None,
            query="q",
            findings=[_make_finding("sq", "ans")],
        )
        assert count == 0

    @pytest.mark.asyncio
    async def test_save_findings_empty_list_returns_zero(self) -> None:
        """Empty findings list returns 0."""
        store = MagicMock()
        count = await save_findings_to_store(
            store, user_id="u1", thread_id="t1", query="q", findings=[]
        )
        assert count == 0

    @pytest.mark.asyncio
    async def test_save_findings_no_aput_returns_zero(self) -> None:
        """Store without aput returns 0."""
        store = MagicMock(spec=[])
        count = await save_findings_to_store(
            store,
            user_id="u1",
            thread_id="t1",
            query="q",
            findings=[_make_finding("sq", "ans")],
        )
        assert count == 0

    @pytest.mark.asyncio
    async def test_save_findings_persists_valid_findings(self) -> None:
        """Valid findings are persisted via aput."""
        store = MagicMock()
        store.aput = AsyncMock()
        findings = [
            _make_finding("What is revenue?", "Revenue is $1M"),
            _make_finding("What is profit?", "Profit is $500K"),
        ]

        count = await save_findings_to_store(
            store, user_id="u1", thread_id="t1", query="main query", findings=findings
        )

        assert count == 2
        assert store.aput.await_count == 2
        calls = store.aput.await_args_list
        ns = ("u1", *FINDINGS_NAMESPACE_SUFFIX)
        assert all(c[1]["namespace"] == ns for c in calls)

    @pytest.mark.asyncio
    async def test_save_findings_skips_empty_subquery(self) -> None:
        """Findings with empty subquery are skipped."""
        store = MagicMock()
        store.aput = AsyncMock()
        findings = [_make_finding("", "answer")]

        count = await save_findings_to_store(
            store, user_id="u1", thread_id="t1", query="q", findings=findings
        )

        assert count == 0
        store.aput.assert_not_called()

    @pytest.mark.asyncio
    async def test_save_findings_skips_empty_answer(self) -> None:
        """Findings with empty answer are skipped."""
        store = MagicMock()
        store.aput = AsyncMock()
        findings = [_make_finding("subquery", "")]

        count = await save_findings_to_store(
            store, user_id="u1", thread_id="t1", query="q", findings=findings
        )

        assert count == 0
        store.aput.assert_not_called()

    @pytest.mark.asyncio
    async def test_save_findings_includes_resources_used(self) -> None:
        """resources_used or data_products_used are stored."""
        store = MagicMock()
        store.aput = AsyncMock()
        findings = [
            _make_finding("q", "a", resources_used=["tool1"]),
        ]

        await save_findings_to_store(
            store, user_id="u1", thread_id="t1", query="q", findings=findings
        )

        value = store.aput.await_args[1]["value"]
        assert "resources_used" in value
        assert value["resources_used"] == ["tool1"]


class TestSearchCrossChatFindings:
    """Test cases for search_cross_chat_findings async function."""

    @pytest.mark.asyncio
    async def test_search_cross_chat_no_user_id_returns_empty(self) -> None:
        """Missing user_id returns empty list."""
        store = MagicMock()
        results = await search_cross_chat_findings(store, user_id=None, query="q")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_cross_chat_no_asearch_returns_empty(self) -> None:
        """Store without asearch returns empty list."""
        store = MagicMock(spec=[])
        results = await search_cross_chat_findings(store, user_id="u1", query="q")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_cross_chat_returns_matched_findings(self) -> None:
        """asearch results are filtered and returned."""
        store = MagicMock()
        item1 = MagicMock()
        item1.value = {"subquery": "q1", "answer": "a1", "thread_id": "t1"}
        item2 = MagicMock()
        item2.value = {"subquery": "q2", "answer": "a2", "thread_id": "t2"}
        store.asearch = AsyncMock(return_value=[item1, item2])

        with patch(
            "template_agent.src.core.deep_research.findings_store._get_setting",
            return_value=10,
        ):
            results = await search_cross_chat_findings(
                store, user_id="u1", query="revenue"
            )

        assert len(results) == 2
        assert results[0]["subquery"] == "q1"
        assert results[1]["subquery"] == "q2"

    @pytest.mark.asyncio
    async def test_search_cross_chat_excludes_thread_id(self) -> None:
        """exclude_thread_id filters out matching thread."""
        store = MagicMock()
        item1 = MagicMock()
        item1.value = {"subquery": "q1", "answer": "a1", "thread_id": "t1"}
        item2 = MagicMock()
        item2.value = {"subquery": "q2", "answer": "a2", "thread_id": "t2"}
        store.asearch = AsyncMock(return_value=[item1, item2])

        with patch(
            "template_agent.src.core.deep_research.findings_store._get_setting",
            return_value=10,
        ):
            results = await search_cross_chat_findings(
                store, user_id="u1", query="q", exclude_thread_id="t1"
            )

        assert len(results) == 1
        assert results[0]["thread_id"] == "t2"


class TestCrossChatToFindingDict:
    """Test cases for cross_chat_to_finding_dict function."""

    def test_cross_chat_to_finding_dict_empty_returns_empty(self) -> None:
        """Empty input returns empty dict."""
        result = cross_chat_to_finding_dict([])
        assert result == {}

    def test_cross_chat_to_finding_dict_converts_valid_items(self) -> None:
        """Valid items are converted to Finding dict keyed by subquery hash."""
        items = [
            {"subquery": "What is revenue?", "answer": "Revenue is $1M"},
            {"subquery": "What is profit?", "answer": "Profit is $500K"},
        ]
        result = cross_chat_to_finding_dict(items)
        assert len(result) == 2
        for key, finding in result.items():
            assert len(key) == 16
            assert finding["subquery"] in ["What is revenue?", "What is profit?"]
            assert finding["cached"] is True
            assert finding["error"] is None

    def test_cross_chat_to_finding_dict_skips_empty_subquery(self) -> None:
        """Items with empty subquery are skipped."""
        items = [
            {"subquery": "", "answer": "a1"},
            {"subquery": "q2", "answer": "a2"},
        ]
        result = cross_chat_to_finding_dict(items)
        assert len(result) == 1
        assert list(result.values())[0]["subquery"] == "q2"

    def test_cross_chat_to_finding_dict_uses_hash_as_key(self) -> None:
        """Keys are subquery hashes."""
        items = [{"subquery": "What is revenue?", "answer": "Revenue is $1M"}]
        result = cross_chat_to_finding_dict(items)
        key = list(result.keys())[0]
        assert key == hash_subquery("What is revenue?")


class TestDeleteFindingsForThread:
    """Test cases for delete_findings_for_thread async function."""

    @pytest.mark.asyncio
    async def test_delete_findings_no_user_id_returns_zero(self) -> None:
        """Missing user_id returns 0."""
        store = MagicMock()
        count = await delete_findings_for_thread(store, user_id=None, thread_id="t1")
        assert count == 0

    @pytest.mark.asyncio
    async def test_delete_findings_no_thread_id_returns_zero(self) -> None:
        """Missing thread_id returns 0."""
        store = MagicMock()
        count = await delete_findings_for_thread(store, user_id="u1", thread_id="")
        assert count == 0

    @pytest.mark.asyncio
    async def test_delete_findings_no_asearch_adelete_returns_zero(self) -> None:
        """Store without asearch/adelete returns 0."""
        store = MagicMock(spec=[])
        count = await delete_findings_for_thread(store, user_id="u1", thread_id="t1")
        assert count == 0

    @pytest.mark.asyncio
    async def test_delete_findings_deletes_matching_entries(self) -> None:
        """Matching entries are deleted via adelete."""
        store = MagicMock()
        item = MagicMock()
        item.value = {"subquery": "q", "thread_id": "t1"}
        item.key = "k1"
        store.asearch = AsyncMock(return_value=[item])
        store.adelete = AsyncMock()

        count = await delete_findings_for_thread(store, user_id="u1", thread_id="t1")

        assert count == 1
        store.adelete.assert_awaited_once()


class TestFormatCrossChatFindings:
    """Test cases for format_cross_chat_findings function."""

    def test_format_cross_chat_empty_returns_empty_string(self) -> None:
        """Empty list returns empty string."""
        result = format_cross_chat_findings([])
        assert result == ""

    def test_format_cross_chat_formats_entries(self) -> None:
        """Entries are formatted with subquery and answer."""
        items = [
            {
                "subquery": "What is revenue?",
                "answer": "Revenue is $1M",
                "thread_id": "t1",
            },
        ]
        result = format_cross_chat_findings(items)
        assert "What is revenue?" in result
        assert "Revenue is $1M" in result
        assert "Cross-chat findings" in result
        assert "1 results" in result

    def test_format_cross_chat_respects_max_chars(self) -> None:
        """Output is truncated when exceeding max_chars."""
        items = [
            {"subquery": "q1", "answer": "a" * 5000, "thread_id": "t1"},
        ]
        result = format_cross_chat_findings(items, max_chars=100)
        assert len(result) <= 150
