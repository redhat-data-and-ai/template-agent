"""Unit tests for semantic clustering."""

from deep_agent.src.memory.clustering import cluster_memories


class TestClusterMemories:
    def test_clusters_similar(self):
        contents = [
            "I like Python programming language",
            "Python is my favorite programming language",
            "The weather today is very sunny",
        ]
        clusters = cluster_memories(contents, threshold=0.3)
        assert len(clusters) == 1
        assert set(clusters[0]) == {0, 1}

    def test_no_clusters_when_disjoint(self):
        contents = [
            "I like cats",
            "The sky is blue",
            "Databases are useful",
        ]
        clusters = cluster_memories(contents, threshold=0.5)
        assert clusters == []

    def test_empty_list(self):
        assert cluster_memories([], threshold=0.5) == []

    def test_single_item(self):
        assert cluster_memories(["hello world"], threshold=0.5) == []

    def test_all_similar(self):
        contents = [
            "Python is great for data science",
            "Python data science is great",
            "Data science with Python is great",
        ]
        clusters = cluster_memories(contents, threshold=0.3)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_high_threshold_no_match(self):
        contents = [
            "I like Python",
            "Python is good",
        ]
        clusters = cluster_memories(contents, threshold=0.99)
        assert clusters == []
