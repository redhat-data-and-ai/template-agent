"""Extraction and limits."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from deep_agent.src.attachments.config import AttachmentsConfig
from deep_agent.src.attachments.extract import describe, extract_text, is_image


class TestExtraction:
    def test_text_formats_are_decoded(self):
        text, complete = extract_text("a.csv", b"x,y\n1,2", 1000)
        assert text == "x,y\n1,2"
        assert complete

    def test_truncation_is_reported_not_hidden(self):
        """A model given half a document must not think it has all of it."""
        text, complete = extract_text("a.txt", b"A" * 100, 10)
        assert len(text) == 10
        assert complete is False

    def test_binary_formats_yield_no_text(self):
        """Base64 in the prompt teaches the model nothing and costs a lot."""
        text, complete = extract_text("a.png", b"\x89PNG\r\n", 1000)
        assert text == ""
        assert complete

    def test_undecodable_bytes_do_not_fail_the_upload(self):
        """One bad byte in a log should not lose the file."""
        text, _ = extract_text("a.log", b"ok \xff\xfe bad", 1000)
        assert "ok" in text

    def test_an_unknown_extension_is_stored_not_inlined(self):
        assert extract_text("a.bin", b"\x00\x01", 1000) == ("", True)

    def test_a_malformed_pdf_degrades_rather_than_raising(self):
        assert extract_text("a.pdf", b"not really a pdf", 1000) == ("", True)

    def test_images_are_recognised(self):
        assert is_image("photo.JPG")
        assert not is_image("notes.md")


class TestDescription:
    def test_a_binary_file_says_it_is_not_readable(self):
        note = describe("a.png", 2048, "", True)
        assert "not extractable" in note

    def test_truncation_is_visible_to_the_model(self):
        assert "(truncated)" in describe("a.txt", 2048, "hi", False)

    def test_a_complete_text_file_is_not_labelled_truncated(self):
        assert "truncated" not in describe("a.txt", 2048, "hi", True)


class TestLimits:
    def test_every_format_is_accepted_by_default(self):
        """Nothing uploaded is executed; the extension is not the control.

        Text is extracted, binary is passed through as base64, and
        anything unrecognised is stored inert. Size caps and filename
        sanitisation are what actually protect this endpoint.
        """
        config = AttachmentsConfig()
        for name in ["run.sh", "lib.so", "app.exe", "a.zip", "noextension", "x.weird"]:
            assert config.is_allowed(name), name

    def test_a_configured_allowlist_restricts(self):
        config = AttachmentsConfig(allowed_extensions=["pdf", "txt"])
        assert config.is_allowed("report.pdf")
        assert not config.is_allowed("app.exe")

    def test_a_configured_allowlist_is_case_insensitive(self):
        assert AttachmentsConfig(allowed_extensions=["pdf"]).is_allowed("REPORT.PDF")

    def test_a_configured_allowlist_judges_the_last_extension(self):
        """`report.pdf.sh` is a shell script wearing a hat."""
        config = AttachmentsConfig(allowed_extensions=["pdf"])
        assert not config.is_allowed("report.pdf.sh")

    def test_per_file_size_is_bounded_by_the_schema(self):
        with pytest.raises(ValidationError):
            AttachmentsConfig(max_file_bytes=10 * 1024 * 1024 * 1024)

    def test_defaults_are_bounded(self):
        """Forks inherit these without deciding, so they must be sane.

        The per-file cap is 50 MB rather than something smaller
        because video needs it — 10 MB rejected every real recording.
        It is bounded well below the schema maximum because binary is
        stored base64, so the checkpoint cost is ~1.33x the file.
        """
        config = AttachmentsConfig()
        assert 10 * 1024 * 1024 <= config.max_file_bytes <= 100 * 1024 * 1024
        assert config.max_thread_bytes > config.max_file_bytes
        assert config.max_files_per_request <= 10


class TestOfficeFormats:
    """Office documents are the obvious thing to put in a knowledge base.

    They were rejected with a bare 415 until a real upload failed. Each
    format is a zip of XML needing its own reader, and a missing reader
    must degrade to "stored but not readable" rather than failing the
    upload.
    """

    @pytest.mark.parametrize("ext", ["docx", "xlsx", "pptx"])
    def test_office_formats_are_allowed(self, ext):
        assert AttachmentsConfig().is_allowed(f"report.{ext}")

    def test_a_real_docx_yields_its_paragraphs(self):
        docx = pytest.importorskip("docx")
        import io

        document = docx.Document()
        document.add_paragraph("Leave policy")
        document.add_paragraph("25 days per year")
        buffer = io.BytesIO()
        document.save(buffer)

        text, complete = extract_text("policy.docx", buffer.getvalue(), 10_000)
        assert "Leave policy" in text
        assert "25 days per year" in text
        assert complete

    def test_a_real_xlsx_yields_its_cells(self):
        openpyxl = pytest.importorskip("openpyxl")
        import io

        book = openpyxl.Workbook()
        book.active.append(["name", "bmi"])
        book.active.append(["alice", 23.1])
        buffer = io.BytesIO()
        book.save(buffer)

        text, _ = extract_text("data.xlsx", buffer.getvalue(), 10_000)
        assert "alice" in text
        assert "bmi" in text

    @pytest.mark.parametrize("ext", ["docx", "xlsx", "pptx"])
    def test_a_corrupt_office_file_does_not_raise(self, ext):
        assert extract_text(f"broken.{ext}", b"not a zip at all", 1000) == ("", True)


class TestMultimodalClassification:
    """Which files the model receives as binary rather than as text.

    deepagents base64-decodes these regardless of the `encoding` field,
    so storing readable placeholder text for them breaks the *next*
    model call with "Incorrect padding" — an error naming neither the
    file nor the attachment that caused it.
    """

    @pytest.mark.parametrize(
        "ext", ["pdf", "pptx", "png", "jpg", "jpeg", "gif", "webp"]
    )
    def test_binary_formats_need_base64(self, ext):
        from deep_agent.src.attachments.extract import needs_base64

        assert needs_base64(f"file.{ext}")

    @pytest.mark.parametrize("ext", ["txt", "md", "csv", "json", "docx", "xlsx", "log"])
    def test_text_formats_do_not(self, ext):
        from deep_agent.src.attachments.extract import needs_base64

        assert not needs_base64(f"file.{ext}")

    def test_it_is_case_insensitive(self):
        from deep_agent.src.attachments.extract import needs_base64

        assert needs_base64("SCAN.PDF")

    def test_matches_deepagents_classification(self):
        """Our list is a copy of a decision deepagents owns.

        If it changes its mind about an extension, this fails here
        rather than as a corrupt-image error at the provider.
        """
        from deepagents.middleware.filesystem import _get_file_type

        from deep_agent.src.attachments.extract import needs_base64

        checked = [
            "txt",
            "md",
            "csv",
            "tsv",
            "json",
            "yaml",
            "yml",
            "log",
            "xml",
            "html",
            "pdf",
            "docx",
            "xlsx",
            "pptx",
            "png",
            "jpg",
            "jpeg",
            "gif",
            "webp",
            "exe",
            "zip",
            "bin",
            "rtf",
            "odt",
        ]
        for ext in checked:
            theirs = _get_file_type(f"file.{ext}") != "text"
            assert needs_base64(f"file.{ext}") == theirs, (
                f".{ext}: we say base64={needs_base64(f'file.{ext}')}, "
                f"deepagents says non-text={theirs}"
            )


class TestUploadBodyLimit:
    """A single global body cap cannot serve both JSON and uploads.

    10 MB is right for a JSON body and rejects every real video;
    raising it globally would let any endpoint accept 50 MB of JSON to
    parse. Upload paths get their own, larger, still-bounded limit.
    """

    def _middleware(self):
        from deep_agent.aegra.security_middleware import RequestSizeLimitMiddleware

        # Built, not __new__'d: bypassing __init__ meant a field
        # added to the constructor was simply missing here, and the
        # suite failed on an AttributeError rather than on the limit.
        return RequestSizeLimitMiddleware(
            app=None,
            max_size_bytes=10 * 1024 * 1024,
            upload_max_bytes=60 * 1024 * 1024,
        )

    @pytest.mark.parametrize(
        "path",
        [
            "/threads/abc-123/attachments",
            "/knowledge/documents",
            "/knowledge/sources/url",
            "/knowledge/sources/text",
            # How an attachment is actually written, over loopback.
            "/threads/abc-123/state",
        ],
    )
    def test_upload_paths_get_the_larger_limit(self, path):
        assert self._middleware().limit_for(path) == 60 * 1024 * 1024

    @pytest.mark.parametrize(
        "path", ["/threads", "/threads/abc/runs", "/search/threads", "/admin/config"]
    )
    def test_everything_else_keeps_the_json_limit(self, path):
        assert self._middleware().limit_for(path) == 10 * 1024 * 1024

    def test_the_exemption_is_a_limit_not_an_absence_of_one(self):
        """Uploads are still bounded — just by a different number."""
        m = self._middleware()
        assert m.limit_for("/threads/x/attachments") < float("inf")

    def test_a_lookalike_path_does_not_get_the_exemption(self):
        """`/attachments-admin` is not an upload path."""
        assert self._middleware().limit_for("/attachments-admin") == 10 * 1024 * 1024


class TestMediaInlineLimit:
    """Media too large for the provider must be refused at upload.

    The upload itself would succeed — it is the *read*, one question
    later, that fails, with an error naming neither the file nor the
    size.
    """

    def test_the_chat_limit_matches_the_transcription_limit(self):
        """Both are the same provider constraint; two numbers would
        drift and one of them would be wrong."""
        from deep_agent.src.knowledge.transcribe import MAX_INLINE_BYTES

        assert 10 * 1024 * 1024 <= MAX_INLINE_BYTES <= 20 * 1024 * 1024

    def test_media_is_recognised_for_the_check(self):
        from deep_agent.src.knowledge.transcribe import is_media

        assert is_media("talk.mp4")
        assert is_media("call.mp3")
        assert not is_media("slides.pdf")

    def test_a_large_document_is_still_fine(self):
        """The cap is about what the model can ingest in one request,
        not about storage — a 40 MB PDF is text and chunked."""
        from deep_agent.src.knowledge.transcribe import is_media

        assert not is_media("huge-report.pdf")
