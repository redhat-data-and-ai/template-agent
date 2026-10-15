"""Attachment limits, from `config/agent/runtime/agent.yaml`.

Every value here is a refusal threshold, so the defaults matter more
than usual: a template that ships generous limits is a template whose
forks inherit them without deciding.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ExtractionConfig(BaseModel):
    """Which parser reads an uploaded document.

    `builtin` — pypdf, python-docx, openpyxl, python-pptx — is the
    default and the only one with no operational cost. The others are
    HTTP clients for a service somebody else runs; see `engines.py` for
    why nothing here is ever hosted by this template.
    """

    engine: Literal["builtin", "docling", "tika"] = "builtin"

    # Base URL of the service. Empty with a non-builtin engine is a
    # misconfiguration, caught by `probe()` rather than at upload time.
    url: str = ""
    api_key: str = ""

    # Passed to the engine verbatim. Theirs to validate, not ours.
    params: dict[str, Any] = Field(default_factory=dict)

    # A layout model on a long scanned PDF is slow; an extractor that
    # has wedged is slower. Bounded either way — an upload that never
    # returns is worse than one that degrades.
    timeout_seconds: float = Field(default=120.0, gt=0, le=600)

    # Above this the file is parsed locally regardless. Shipping 50 MB
    # to a shared extractor to read its first page is a bad trade, and
    # it is how one upload starves everyone else's.
    max_bytes: int = Field(default=20 * 1024 * 1024, ge=1)

    # Only these go to the remote engine. The omissions are the point:
    # csv, json, md, txt and xlsx are parsed exactly right locally, and
    # a network round trip cannot improve on exactly right.
    suffixes: list[str] = Field(
        default_factory=lambda: [
            "pdf",
            "docx",
            "doc",
            "pptx",
            "ppt",
            "odt",
            "odp",
            "rtf",
            "epub",
            "html",
            "htm",
            "png",
            "jpg",
            "jpeg",
            "tiff",
            "tif",
            "bmp",
            "webp",
        ]
    )

    # Fall back to the built-in parser when the engine fails. Turning
    # this off makes the extractor a hard dependency of uploading,
    # which is a choice a deployment is allowed to make and should not
    # make by accident.
    fallback: bool = True

    @field_validator("url")
    @classmethod
    def _http_only(cls, value: str) -> str:
        if value and not value.startswith(("http://", "https://")):
            raise ValueError("extraction.url must start with http:// or https://")
        return value

    def handles(self, filename: str, size: int) -> bool:
        """True when this file should go to the remote engine."""
        if self.engine == "builtin" or not self.url:
            return False
        if size > self.max_bytes:
            return False
        suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        return suffix in {s.lower().lstrip(".") for s in self.suffixes}


class AttachmentsConfig(BaseModel):
    """What a user may attach to a thread."""

    enabled: bool = True

    # Per file. 50 MB covers a long PDF, a large CSV, or a short video
    # clip. Video is the reason it is not smaller: a few seconds of
    # 1080p is tens of megabytes, and 10 MB rejected every real
    # recording.
    #
    # It is not larger because binary is stored base64 — a 50 MB file
    # is ~67 MB in the checkpoint, and reading it spends that much of
    # the model's context.
    max_file_bytes: int = Field(default=50 * 1024 * 1024, ge=1, le=500 * 1024 * 1024)

    # Per thread, cumulative. Without this, a user can attach 10 MB a
    # hundred times and the thread's state becomes unloadable.
    max_thread_bytes: int = Field(default=200 * 1024 * 1024, ge=1)

    max_files_per_request: int = Field(default=10, ge=1, le=50)

    # Extensions, lower-case, no dot. **Empty means every format is
    # accepted**, which is the default.
    #
    # The allowlist was doing less work than it looked like it was.
    # Nothing here is ever executed: text is extracted, binary is
    # passed through as base64, and anything unrecognised is stored as
    # an inert placeholder. What actually protects this endpoint is
    # filename sanitisation, the size caps above, and the fact that
    # uploads are data rather than code.
    #
    # Set a list to restrict — e.g. a deployment that only wants
    # documents can list them, and everything else gets a 415.
    allowed_extensions: list[str] = Field(default_factory=list)

    # Where attachments appear in the agent's filesystem. Must stay
    # inside a route the agent can read — see `filesystem.routes`.
    mount_path: str = "/uploads/"

    # Characters of extracted text injected into the conversation per
    # file. Beyond this the model is reading the file, not the message,
    # and the context cost stops being worth it.
    max_inline_chars: int = Field(default=20_000, ge=100)

    def is_allowed(self, filename: str) -> bool:
        """True unless a restrictive allowlist is configured."""
        if not self.allowed_extensions:
            return True
        suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        return suffix in {e.lower().lstrip(".") for e in self.allowed_extensions}
