"""Transcription: how media becomes an indexable document."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from deep_agent.src.knowledge.transcribe import (
    MAX_INLINE_BYTES,
    TranscriptionUnavailable,
    is_media,
    transcribe,
)


class TestMediaDetection:
    @pytest.mark.parametrize("ext", ["mp4", "mov", "webm", "mp3", "wav", "m4a"])
    def test_audio_and_video_are_media(self, ext):
        assert is_media(f"recording.{ext}")

    @pytest.mark.parametrize("ext", ["pdf", "txt", "png", "docx"])
    def test_everything_else_is_not(self, ext):
        assert not is_media(f"file.{ext}")

    def test_it_is_case_insensitive(self):
        assert is_media("CLIP.MP4")


class TestRouting:
    """Inline below the payload cap, Files API above it.

    A 17 MB screen recording is ordinary, so the cap is a routing
    decision rather than a limit on what is supported.
    """

    def test_small_media_is_sent_inline(self):
        model = MagicMock()
        model.invoke.return_value = MagicMock(content="transcript")
        with patch("deep_agent.src.agent.llm.create_model", return_value=model):
            transcribe("a.mp3", b"small", "m")
        block = next(
            b
            for b in model.invoke.call_args[0][0][0].content
            if b.get("type") == "media"
        )
        assert "data" in block and "file_uri" not in block

    def test_large_media_is_uploaded_and_referenced(self):
        model = MagicMock()
        model.invoke.return_value = MagicMock(content="transcript")
        with (
            patch(
                "deep_agent.src.knowledge.media_upload.upload_media",
                return_value="gs://bucket/clip",
            ),
            patch("deep_agent.src.agent.llm.create_model", return_value=model),
        ):
            transcribe("big.mp4", b"x" * (MAX_INLINE_BYTES + 1), "m")
        block = next(
            b
            for b in model.invoke.call_args[0][0][0].content
            if b.get("type") == "media"
        )
        assert block["file_uri"] == "gs://bucket/clip"
        # The bytes must not also be inlined — that is the whole point.
        assert "data" not in block

    def test_an_upload_failure_surfaces_as_a_readable_message(self):
        from deep_agent.src.knowledge.media_upload import MediaUploadError

        with patch(
            "deep_agent.src.knowledge.media_upload.upload_media",
            side_effect=MediaUploadError("clip.mp4: upload failed (TimeoutError)"),
        ):
            with pytest.raises(TranscriptionUnavailable, match="upload failed"):
                transcribe("clip.mp4", b"x" * (MAX_INLINE_BYTES + 1), "m")


class TestTranscription:
    def _model(self, content):
        model = MagicMock()
        model.invoke.return_value = MagicMock(content=content)
        return model

    def test_a_transcript_is_returned(self):
        with patch(
            "deep_agent.src.agent.llm.create_model",
            return_value=self._model("Speaker 1: hello"),
        ):
            assert transcribe("a.mp3", b"audio", "m") == "Speaker 1: hello"

    def test_block_content_is_flattened(self):
        """Gemini 3 returns content blocks rather than a string."""
        blocks = [
            {"type": "text", "text": "line one"},
            {"type": "text", "text": "line two"},
        ]
        with patch(
            "deep_agent.src.agent.llm.create_model", return_value=self._model(blocks)
        ):
            out = transcribe("a.mp4", b"video", "m")
        assert "line one" in out and "line two" in out

    def test_silence_is_reported_rather_than_indexed_as_empty(self):
        """An empty document in the corpus matches nothing and looks
        like a successful upload."""
        with patch(
            "deep_agent.src.agent.llm.create_model", return_value=self._model("  ")
        ):
            with pytest.raises(TranscriptionUnavailable, match="no transcript"):
                transcribe("a.mp4", b"video", "m")

    def test_a_provider_failure_becomes_a_readable_message(self):
        model = MagicMock()
        model.invoke.side_effect = RuntimeError("InvalidArgument: unsupported")
        with patch("deep_agent.src.agent.llm.create_model", return_value=model):
            with pytest.raises(
                TranscriptionUnavailable, match="could not be transcribed"
            ):
                transcribe("a.mp4", b"video", "m")

    def test_the_prompt_asks_for_verbatim_text(self):
        """A summary would make the corpus answer questions the source
        never did."""
        model = self._model("ok")
        with patch("deep_agent.src.agent.llm.create_model", return_value=model):
            transcribe("a.mp4", b"v", "m")
        sent = model.invoke.call_args[0][0][0].content
        instruction = next(b["text"] for b in sent if b.get("type") == "text")
        assert "verbatim" in instruction.lower()
