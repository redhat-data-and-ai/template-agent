"""Unit tests for memory clustering module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_agent.src.memory.clustering import (
    _build_tfidf,
    _cosine_sim,
    _normalize_text,
    _tokenize,
    cluster_memories,
    cluster_store_memories,
)


class TestNormalizeText:
    def test_lowercase(self):
        assert _normalize_text("Hello World") == "hello world"

    def test_separates_units(self):
        assert "70 kg" in _normalize_text("70kg")
        assert "180 cm" in _normalize_text("180cm")

    def test_strips_punctuation(self):
        result = _normalize_text("hello, world!")
        assert "," not in result
        assert "!" not in result


class TestTokenize:
    def test_stems_known_words(self):
        tokens = _tokenize("user prefers dark mode")
        assert "prefer" in tokens

    def test_preserves_unknown_words(self):
        tokens = _tokenize("python developer")
        assert "python" in tokens
        assert "developer" in tokens

    def test_handles_empty_string(self):
        assert _tokenize("") == []


class TestBuildTfidf:
    def test_empty_list(self):
        assert _build_tfidf([]) == []

    def test_single_document(self):
        vectors = _build_tfidf(["hello world"])
        assert len(vectors) == 1
        assert "hello" in vectors[0]
        assert "world" in vectors[0]

    def test_multiple_documents(self):
        vectors = _build_tfidf(["hello world", "goodbye world"])
        assert len(vectors) == 2


class TestCosineSim:
    def test_identical_vectors(self):
        v = {"a": 1.0, "b": 2.0}
        assert _cosine_sim(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = {"x": 1.0}
        b = {"y": 1.0}
        assert _cosine_sim(a, b) == 0.0

    def test_empty_vectors(self):
        assert _cosine_sim({}, {}) == 0.0

    def test_one_empty(self):
        assert _cosine_sim({"a": 1.0}, {}) == 0.0


class TestClusterMemories:
    def test_empty_list(self):
        assert cluster_memories([]) == []

    def test_single_item(self):
        assert cluster_memories(["hello"]) == []

    def test_identical_strings_cluster(self):
        clusters = cluster_memories(["user weighs 70kg", "user weighs 70kg"])
        assert len(clusters) == 1
        assert sorted(clusters[0]) == [0, 1]

    def test_similar_strings_cluster(self):
        clusters = cluster_memories(
            [
                "user weighs 70kg",
                "user weight is 70 kg",
                "user likes python programming",
            ]
        )
        assert len(clusters) >= 1
        weight_cluster = next((c for c in clusters if 0 in c or 1 in c), None)
        assert weight_cluster is not None
        assert 0 in weight_cluster and 1 in weight_cluster

    def test_dissimilar_strings_no_cluster(self):
        clusters = cluster_memories(
            [
                "user likes dark mode",
                "the weather is sunny today",
            ]
        )
        assert clusters == []

    def test_high_threshold_clusters_nothing(self):
        clusters = cluster_memories(
            ["user weighs 70kg", "user weight is 70 kg"],
            threshold=1.0,
        )
        assert clusters == []

    def test_low_threshold_clusters_everything(self):
        clusters = cluster_memories(
            ["hello world", "goodbye world", "world peace"],
            threshold=0.01,
        )
        assert len(clusters) >= 1

    def test_singletons_excluded(self):
        clusters = cluster_memories(
            [
                "identical fact",
                "identical fact",
                "completely unique unrelated text about quantum physics",
            ]
        )
        for group in clusters:
            assert len(group) >= 2
        assert len(clusters) >= 1


class TestNearDuplicateFacts:
    def test_weight_restatements_are_duplicates(self):
        from deep_agent.src.memory.clustering import are_near_duplicate_facts

        assert are_near_duplicate_facts("user weighs 70kg", "user weight is 70 kg")

    def test_birth_and_joining_dates_are_not_duplicates(self):
        from deep_agent.src.memory.clustering import (
            are_near_duplicate_facts,
            near_duplicate_groups,
        )

        birth = "The user's date of birth is June 11, 2003."
        joining = "The user's joining date is 2 June 2020."
        truncated = "The user's joining date is"
        assert not are_near_duplicate_facts(birth, joining)
        assert not are_near_duplicate_facts(birth, truncated)
        assert near_duplicate_groups([birth, joining, truncated]) == []

    def test_same_birth_dates_are_duplicates(self):
        from deep_agent.src.memory.clustering import near_duplicate_groups

        groups = near_duplicate_groups(
            [
                "The user's date of birth is June 11, 2003.",
                "The user's date of birth is June 12, 2003.",
            ]
        )
        assert len(groups) == 1
        assert sorted(groups[0]) == [0, 1]


def _store_item(key: str, content: str | list[str]) -> MagicMock:
    item = MagicMock()
    item.key = key
    item.value = {"content": content}
    return item


class TestClusterStoreMemories:
    @pytest.mark.asyncio
    async def test_returns_empty_when_fewer_than_two_facts(self):
        item = _store_item("user_profile.md", "only one fact")
        store = AsyncMock()
        store.setup = AsyncMock()
        store.asearch = AsyncMock(return_value=[item])
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=store)
        cm.__aexit__ = AsyncMock(return_value=False)
        with patch("langgraph.store.postgres.aio.AsyncPostgresStore") as store_cls:
            store_cls.from_conn_string.return_value = cm
            assert await cluster_store_memories("postgresql://x", ("user", "u1")) == []

    @pytest.mark.asyncio
    async def test_skips_non_memory_keys_and_clusters_facts(self):
        profile = _store_item(
            "user_profile.md",
            "user weighs 70kg\nuser weight is 70 kg\nlikes python programming",
        )
        other = _store_item("quarterly.md", "ignore this file")
        store = AsyncMock()
        store.setup = AsyncMock()
        store.asearch = AsyncMock(return_value=[profile, other])
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=store)
        cm.__aexit__ = AsyncMock(return_value=False)
        with patch("langgraph.store.postgres.aio.AsyncPostgresStore") as store_cls:
            store_cls.from_conn_string.return_value = cm
            clusters = await cluster_store_memories("postgresql://x", ("user", "u1"))
        assert clusters
        assert any(len(group) >= 2 for group in clusters)
