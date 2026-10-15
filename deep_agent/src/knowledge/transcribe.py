"""Turn audio and video into text so it can be indexed.

A knowledge corpus stores embeddings of text. Media has none — which
is why adding an MP4 used to fail with "no extractable text", a true
statement that reads as a refusal.

Gemini accepts audio and video inline and will transcribe them, so the
extraction step for media is a model call rather than a parser.
"""

from __future__ import annotations

import base64
import mimetypes

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_AUDIO = {"mp3", "wav", "ogg", "m4a", "aac", "flac"}
_VIDEO = {"mp4", "mov", "webm", "avi", "mkv"}

# Gemini accepts inline media up to roughly 20 MB of request payload.
# Base64 inflates by ~4/3, so the file itself must stay well under
# that. Larger files go through the Files API instead — uploaded once
# and referenced by URI. A 17 MB screen recording is ordinary, so this
# cap is a routing decision, not a limit on what is supported.
MAX_INLINE_BYTES = 14 * 1024 * 1024

# Transcription is slower than any other extraction path; a few
# minutes of audio takes tens of seconds.
_TIMEOUT_SECONDS = 300


class TranscriptionUnavailable(RuntimeError):
    """The file cannot be transcribed, with a reason worth showing."""


def is_media(filename: str) -> bool:
    """Does this filename look like audio or video needing transcription?"""
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return suffix in _AUDIO or suffix in _VIDEO


def transcribe(filename: str, data: bytes, model_name: str) -> str:
    """Transcribe audio or video to text.

    Raises :class:`TranscriptionUnavailable` with a message meant for
    the user — "too large", not "InvalidArgument".
    """
    mime = mimetypes.guess_type(filename)[0]
    if not mime:
        suffix = filename.rsplit(".", 1)[-1].lower()
        mime = f"{'audio' if suffix in _AUDIO else 'video'}/{suffix}"

    from langchain_core.messages import HumanMessage

    from deep_agent.src.agent.llm import create_model

    # Prepare the media before building the model: an upload that fails
    # means there is no request to make, and constructing a client for
    # it first only moves the error somewhere less relevant.
    #
    # Small enough to inline, or large enough to need an upload first.
    if len(data) > MAX_INLINE_BYTES:
        from deep_agent.src.knowledge.media_upload import (
            MediaUploadError,
            upload_media,
        )

        try:
            uri = upload_media(filename, data)
        except MediaUploadError as exc:
            raise TranscriptionUnavailable(str(exc)) from exc
        media_block: dict = {"type": "media", "file_uri": uri, "mime_type": mime}
        logger.info("Transcribing %s via uploaded file %s", filename, uri)
    else:
        media_block = {
            "type": "media",
            "data": base64.standard_b64encode(data).decode("ascii"),
            "mime_type": mime,
        }

    model = create_model(model_name, temperature=0.0, max_output_tokens=8192)

    message = HumanMessage(
        content=[
            {
                # Verbatim, because this becomes a searchable document:
                # a summary would make the corpus answer questions the
                # source never did.
                "type": "text",
                "text": (
                    "Transcribe this recording verbatim. Include speaker "
                    "labels where you can tell them apart, and timestamps "
                    "at natural section boundaries. Output only the "
                    "transcript."
                ),
            },
            media_block,
        ]
    )

    try:
        result = model.invoke([message])
    except Exception as exc:
        logger.warning("Transcription failed for %s: %s", filename, exc)
        raise TranscriptionUnavailable(
            f"{filename}: could not be transcribed ({type(exc).__name__})"
        ) from exc

    content = result.content
    if isinstance(content, list):
        content = "\n".join(b.get("text", "") for b in content if isinstance(b, dict))
    text = str(content or "").strip()

    if not text:
        raise TranscriptionUnavailable(
            f"{filename}: the recording produced no transcript — it may be "
            "silent or have no speech"
        )
    return text
