"""The external extraction engines, and the rules that make them safe.

Each test here corresponds to a way this goes wrong in production, not
to a line of code: an extractor that fails the upload, one that fails
invisibly, and one that returns worse text than the parser it replaced.
"""

from __future__ import annotations

import httpx
import pytest

from deep_agent.src.attachments.config import ExtractionConfig
from deep_agent.src.attachments.engines import (
    ExtractionUnavailable,
    extract_remote,
    probe,
)
from deep_agent.src.attachments.extract import extract_best

DOCLING = ExtractionConfig(engine="docling", url="http://docling:5001")


def _transport(handler) -> None:
    """Patch httpx.AsyncClient so no test touches the network."""
    import deep_agent.src.attachments.engines as engines

    original = httpx.AsyncClient

    class Client(original):  # type: ignore[misc]
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    engines.httpx = type("mod", (), {"AsyncClient": Client, "Response": httpx.Response})
    return original


@pytest.fixture(autouse=True)
def restore_httpx():
    import deep_agent.src.attachments.engines as engines

    saved = engines.httpx
    yield
    engines.httpx = saved


def _ok(body: dict) -> callable:
    return lambda request: httpx.Response(200, json=body)


class TestRouting:
    def test_builtin_never_calls_out(self):
        assert ExtractionConfig().handles("a.pdf", 10) is False

    def test_an_engine_without_a_url_is_not_used(self):
        assert ExtractionConfig(engine="docling").handles("a.pdf", 10) is False

    def test_formats_parsed_perfectly_well_locally_stay_local(self):
        for name in ("a.csv", "a.json", "a.md", "a.txt", "a.xlsx"):
            assert DOCLING.handles(name, 10) is False, name

    def test_layout_heavy_formats_go_out(self):
        for name in ("a.pdf", "a.docx", "scan.tiff", "deck.pptx"):
            assert DOCLING.handles(name, 10) is True, name

    def test_large_files_are_parsed_locally(self):
        assert DOCLING.handles("a.pdf", 50 * 1024 * 1024) is False

    def test_a_url_must_be_http(self):
        with pytest.raises(ValueError):
            ExtractionConfig(engine="docling", url="file:///etc/passwd")


class TestDocling:
    @pytest.mark.asyncio
    async def test_markdown_is_returned(self):
        _transport(_ok({"status": "success", "document": {"md_content": "# Title"}}))
        result = await extract_remote("a.pdf", b"x", DOCLING)
        assert result.text == "# Title"
        assert result.engine == "docling"

    @pytest.mark.asyncio
    async def test_a_failed_conversion_inside_a_200_is_not_treated_as_text(self):
        # Docling reports conversion failure in the body with HTTP 200.
        # Trusting the status code stores the error as the document.
        _transport(
            _ok(
                {
                    "status": "failure",
                    "errors": [{"error_message": "encrypted pdf"}],
                    "document": {"md_content": ""},
                }
            )
        )
        with pytest.raises(ExtractionUnavailable, match="encrypted pdf"):
            await extract_remote("a.pdf", b"x", DOCLING)

    @pytest.mark.asyncio
    async def test_skipped_is_also_a_failure(self):
        _transport(_ok({"status": "skipped", "document": {}}))
        with pytest.raises(ExtractionUnavailable, match="skipped"):
            await extract_remote("a.pdf", b"x", DOCLING)

    @pytest.mark.asyncio
    async def test_an_http_error_carries_the_detail(self):
        _transport(lambda r: httpx.Response(422, json={"detail": "bad mime"}))
        with pytest.raises(ExtractionUnavailable, match="bad mime"):
            await extract_remote("a.pdf", b"x", DOCLING)

    @pytest.mark.asyncio
    async def test_an_unreachable_service_raises_our_error_not_httpx(self):
        def boom(request):
            raise httpx.ConnectError("no route")

        _transport(boom)
        with pytest.raises(ExtractionUnavailable, match="unreachable"):
            await extract_remote("a.pdf", b"x", DOCLING)

    @pytest.mark.asyncio
    async def test_empty_output_is_a_failure_not_an_empty_document(self):
        _transport(_ok({"status": "success", "document": {"md_content": "   "}}))
        with pytest.raises(ExtractionUnavailable):
            await extract_remote("a.pdf", b"x", DOCLING)


class TestTika:
    @pytest.mark.asyncio
    async def test_plain_text_is_returned(self):
        _transport(lambda r: httpx.Response(200, text="hello from tika"))
        config = ExtractionConfig(engine="tika", url="http://tika:9998")
        result = await extract_remote("a.docx", b"x", config)
        assert result.text == "hello from tika"
        assert result.engine == "tika"


class TestFallback:
    @pytest.mark.asyncio
    async def test_a_failing_engine_never_fails_the_upload(self):
        def boom(request):
            raise httpx.ConnectError("down")

        _transport(boom)
        result = await extract_best("a.txt.pdf", b"", 1000, DOCLING)
        assert result.engine == "builtin"

    @pytest.mark.asyncio
    async def test_the_fallback_is_visible(self):
        # The one real bug from hosting Docling was a silent fallback.
        def boom(request):
            raise httpx.ConnectError("down")

        _transport(boom)
        result = await extract_best("a.pdf", b"", 1000, DOCLING)
        assert "docling unavailable" in result.degraded

    @pytest.mark.asyncio
    async def test_fallback_can_be_turned_off(self):
        def boom(request):
            raise httpx.ConnectError("down")

        _transport(boom)
        strict = DOCLING.model_copy(update={"fallback": False})
        with pytest.raises(ExtractionUnavailable):
            await extract_best("a.pdf", b"", 1000, strict)

    @pytest.mark.asyncio
    async def test_a_thin_remote_result_loses_to_a_better_local_one(self):
        # A layout model returning a placeholder for a document the
        # built-in parser reads fine is a regression, and only a
        # comparison catches it.
        _transport(_ok({"status": "success", "document": {"md_content": "<no text>"}}))
        rich = "word " * 200
        config = DOCLING.model_copy(update={"suffixes": ["txt"]})
        result = await extract_best("a.txt", rich.encode(), 10_000, config)
        assert result.engine == "builtin"
        assert result.text.startswith("word")

    @pytest.mark.asyncio
    async def test_a_long_remote_result_is_kept_without_a_local_parse(self):
        _transport(_ok({"status": "success", "document": {"md_content": "x" * 500}}))
        config = DOCLING.model_copy(update={"suffixes": ["txt"]})
        result = await extract_best("a.txt", b"different content", 10_000, config)
        assert result.engine == "docling"

    @pytest.mark.asyncio
    async def test_remote_output_respects_the_inline_cap(self):
        _transport(_ok({"status": "success", "document": {"md_content": "y" * 900}}))
        config = DOCLING.model_copy(update={"suffixes": ["txt"]})
        result = await extract_best("a.txt", b"x", 100, config)
        assert len(result.text) == 100
        assert result.complete is False


class TestProbe:
    @pytest.mark.asyncio
    async def test_builtin_is_always_ready(self):
        ok, _ = await probe(ExtractionConfig())
        assert ok

    @pytest.mark.asyncio
    async def test_an_engine_without_a_url_is_reported_as_misconfigured(self):
        ok, detail = await probe(ExtractionConfig(engine="docling"))
        assert not ok
        assert "url is unset" in detail

    @pytest.mark.asyncio
    async def test_an_unreachable_engine_is_reported_before_an_upload_finds_it(self):
        def boom(request):
            raise httpx.ConnectError("refused")

        _transport(boom)
        ok, detail = await probe(DOCLING)
        assert not ok
        assert "unreachable" in detail
