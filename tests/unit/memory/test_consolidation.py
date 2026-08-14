"""Unit tests for memory consolidation."""

from deep_agent.src.memory.consolidation import (
    find_duplicates,
    pick_representative,
    token_similarity,
)


class TestTokenSimilarity:
    def test_identical(self):
        assert token_similarity("hello world", "hello world") == 1.0

    def test_disjoint(self):
        assert token_similarity("hello world", "foo bar") == 0.0

    def test_partial_overlap(self):
        sim = token_similarity("I like Python", "I love Python")
        assert 0.3 < sim < 0.9

    def test_empty_string(self):
        assert token_similarity("", "hello") == 0.0

    def test_case_insensitive(self):
        assert token_similarity("Python", "python") == 1.0


class TestFindDuplicates:
    def test_no_duplicates(self):
        memories = [
            {"content": "I like cats"},
            {"content": "The weather is sunny"},
            {"content": "Python is great for data science"},
        ]
        groups = find_duplicates(memories, threshold=0.5)
        assert groups == []

    def test_finds_duplicates(self):
        memories = [
            {"content": "I prefer Python programming"},
            {"content": "I prefer Python for programming"},
            {"content": "The weather is nice today"},
        ]
        groups = find_duplicates(memories, threshold=0.5)
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1}

    def test_single_memory(self):
        memories = [{"content": "just one"}]
        assert find_duplicates(memories) == []

    def test_empty_list(self):
        assert find_duplicates([]) == []


class TestPickRepresentative:
    def test_picks_longest(self):
        memories = [
            {"content": "short", "score": "0.5"},
            {"content": "this is much longer content", "score": "0.5"},
        ]
        assert pick_representative(memories, [0, 1]) == 1

    def test_breaks_tie_by_score(self):
        memories = [
            {"content": "same length!", "score": "0.9"},
            {"content": "same length!", "score": "0.3"},
        ]
        assert pick_representative(memories, [0, 1]) == 0
