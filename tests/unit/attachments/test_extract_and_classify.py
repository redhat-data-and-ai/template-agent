"""Extraction dispatch and provenance lines, plus error bucketing.

`describe` is the line the model reads above any extracted text, so
it has to be honest about truncation and about which engine produced
the content. `classify_error` must never return the exception string:
those routinely contain the input that produced them.
"""

from __future__ import annotations

import pytest

from deep_agent.src.analytics.queries import classify_error
from deep_agent.src.attachments.extract import (
    describe,
    extract_text,
    is_image,
    needs_base64,
    suffix_of,
)


class TestSuffixHelpers:
    def test_suffix_is_lower_cased(self):
        assert suffix_of("REPORT.PDF") == "pdf"

    def test_no_suffix_is_an_empty_string(self):
        assert suffix_of("Makefile") == ""

    def test_images_are_read_directly_by_the_model(self):
        assert is_image("photo.png") is True
        assert is_image("notes.txt") is False

    def test_multimodal_files_travel_as_binary(self):
        assert needs_base64("photo.png") is True
        assert needs_base64("notes.txt") is False


class TestExtractText:
    def test_reads_a_text_file(self):
        text, complete = extract_text("notes.txt", b"hello", 100)
        assert (text, complete) == ("hello", True)

    def test_one_bad_byte_does_not_cost_the_upload(self):
        text, complete = extract_text("app.log", b"ok \xff here", 100)
        assert "ok" in text and complete is True

    def test_reports_truncation_so_the_model_does_not_assume_otherwise(self):
        text, complete = extract_text("notes.txt", b"abcdefghij", 4)
        assert text == "abcd"
        assert complete is False

    def test_an_unknown_binary_format_yields_nothing_rather_than_noise(self):
        assert extract_text("archive.zip", b"PK\x03\x04", 100) == ("", True)


class TestDescribe:
    def test_names_the_kind_and_size(self):
        line = describe("report.pdf", 2048, "body", complete=True)
        assert "report.pdf" in line
        assert "2 KB" in line
        assert "pdf" in line

    def test_rounds_a_tiny_file_up_to_one_kb(self):
        assert "1 KB" in describe("a.txt", 10, "body", complete=True)

    def test_calls_an_image_an_image(self):
        assert "image" in describe("photo.png", 1024, "", complete=True)

    def test_says_when_content_was_truncated(self):
        assert "truncated" in describe("a.txt", 1024, "body", complete=False)

    def test_says_when_nothing_could_be_extracted(self):
        line = describe("scan.pdf", 1024, "", complete=True)
        assert "not extractable" in line

    def test_names_a_remote_engine(self):
        line = describe("a.pdf", 1024, "body", complete=True, engine="docling")
        assert "read by docling" in line

    def test_reports_a_degraded_read(self):
        line = describe(
            "a.pdf", 1024, "body", complete=True, degraded="fell back to built-in"
        )
        assert "fell back to built-in" in line

    def test_degraded_is_reported_even_with_no_text(self):
        line = describe("a.pdf", 1024, "", complete=True, degraded="engine timed out")
        assert "engine timed out" in line


class TestClassifyError:
    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ("Deadline Exceeded after 60s", "model_timeout"),
            ("429 Too Many Requests", "model_rate_limit"),
            ("Resource exhausted: quota", "model_rate_limit"),
            ("blocked for safety", "model_refused"),
            ("No Google credentials found", "model_config"),
        ],
    )
    def test_buckets_known_failures(self, message, expected):
        assert classify_error(message) == expected

    def test_unknown_for_no_message(self):
        assert classify_error(None) == "unknown"
        assert classify_error("") == "unknown"

    def test_other_for_an_unrecognised_message(self):
        assert classify_error("something nobody has seen before") == "other"

    def test_never_returns_the_message_itself(self):
        # Exception strings routinely contain the input that produced
        # them; a dashboard is the wrong place for someone's file path.
        secret = "failed reading /documents/salaries-2026.csv"
        assert secret not in classify_error(secret)
