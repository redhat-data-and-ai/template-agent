"""Unit tests for relationship inference."""

from deep_agent.src.memory.relationships import (
    extract_keywords,
    find_related_pairs,
)


class TestExtractKeywords:
    def test_basic(self):
        keywords = extract_keywords("Python is a great programming language")
        assert "python" in keywords
        assert "programming" in keywords
        assert "language" in keywords

    def test_filters_stopwords(self):
        keywords = extract_keywords("I am a very good person")
        assert "good" in keywords
        assert "person" in keywords
        assert "very" not in keywords

    def test_filters_short_tokens(self):
        keywords = extract_keywords("Go is ok")
        assert "go" not in keywords
        assert "ok" not in keywords

    def test_empty(self):
        assert extract_keywords("") == []


class TestFindRelatedPairs:
    def test_finds_related(self):
        memories = [
            {"content": "I work at Red Hat on OpenShift platform engineering"},
            {"content": "Red Hat OpenShift is my primary deployment target"},
            {"content": "I like pizza and pasta for dinner"},
        ]
        pairs = find_related_pairs(memories, min_shared=2)
        assert len(pairs) == 1
        assert pairs[0][0] == 0
        assert pairs[0][1] == 1
        assert "openshift" in pairs[0][2]

    def test_no_related(self):
        memories = [
            {"content": "I like cats and dogs"},
            {"content": "The weather is sunny today"},
        ]
        assert find_related_pairs(memories, min_shared=2) == []

    def test_empty(self):
        assert find_related_pairs([], min_shared=2) == []
