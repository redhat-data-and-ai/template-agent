"""Chunking quietly decides retrieval quality."""

from __future__ import annotations

import pytest

from deep_agent.src.knowledge.chunking import chunk_markdown, chunk_text


class TestSizing:
    def test_short_text_is_one_chunk(self):
        assert chunk_text("just a sentence", size=200) == ["just a sentence"]

    def test_text_that_fits_does_not_care_about_overlap(self):
        """Validating overlap on a single-chunk call fails for no reason."""
        assert chunk_text("short", size=50, overlap=50) == ["short"]

    def test_empty_text_yields_nothing(self):
        assert chunk_text("   \n  ") == []

    def test_long_text_is_split(self):
        chunks = chunk_text("word " * 500, size=200, overlap=50)
        assert len(chunks) > 1

    def test_no_chunk_greatly_exceeds_the_target(self):
        chunks = chunk_text("word " * 500, size=200, overlap=50)
        assert all(len(c) <= 220 for c in chunks), [len(c) for c in chunks]

    def test_overlap_must_be_smaller_than_size(self):
        """Otherwise the window never advances."""
        with pytest.raises(ValueError):
            chunk_text("x" * 100, size=50, overlap=50)


class TestTermination:
    def test_pathological_input_terminates(self):
        """A tiny chunk must not push the cursor backwards forever."""
        chunks = chunk_text("a" * 1000, size=10, overlap=9)
        assert len(chunks) < 1000

    def test_text_with_no_boundaries_still_splits(self):
        assert len(chunk_text("x" * 1000, size=100, overlap=10)) > 1


class TestBoundaries:
    def test_paragraphs_are_preferred_break_points(self):
        text = "First para here.\n\nSecond para here."
        chunks = chunk_text(text, size=20, overlap=5)
        # The first chunk should not run into the second paragraph.
        assert not chunks[0].startswith("Second")

    def test_overlap_keeps_straddling_facts_whole_somewhere(self):
        text = "alpha beta gamma delta epsilon zeta eta theta"
        chunks = chunk_text(text, size=20, overlap=10)
        joined = " ".join(chunks)
        for word in text.split():
            assert word in joined


class TestMarkdown:
    def test_headings_split_sections(self):
        md = "# One\ncontent one\n\n## Two\ncontent two"
        chunks = chunk_markdown(md, size=1000)
        assert len(chunks) == 2

    def test_a_heading_stays_with_its_section(self):
        md = "# Alpha\nbody alpha\n\n# Beta\nbody beta"
        chunks = chunk_markdown(md, size=1000)
        assert chunks[0].startswith("# Alpha")
        assert "body beta" not in chunks[0]

    def test_markdown_without_headings_falls_back_to_size(self):
        text = "plain " * 500
        assert chunk_markdown(text, size=200, overlap=50) == chunk_text(
            text, size=200, overlap=50
        )

    def test_preamble_before_the_first_heading_is_kept(self):
        md = "intro text\n\n# One\nbody"
        assert any("intro text" in c for c in chunk_markdown(md, size=1000))
