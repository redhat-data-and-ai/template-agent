"""Split a document into retrievable pieces.

Chunking is the part of RAG that quietly decides answer quality. Two
rules do most of the work: break on structure rather than character
count where structure exists, and overlap, so a fact that straddles a
boundary survives in at least one chunk whole.
"""

from __future__ import annotations

import re

# Prefer breaking here, strongest first. Paragraphs keep an idea
# together; sentences keep a clause together; a space is the last
# resort before cutting mid-word.
_BOUNDARIES = ("\n\n", "\n", ". ", " ")


def chunk_text(text: str, *, size: int = 1200, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks of roughly ``size`` characters.

    Characters rather than tokens: tokenising needs the model's
    tokeniser, which differs per provider and would make chunking
    depend on which model is configured. For retrieval the imprecision
    does not matter — chunks only have to be small enough to be
    specific and large enough to be self-contained.
    """
    text = text.strip()
    if not text:
        return []
    # Short-circuit before validating overlap: text that fits in one
    # chunk never uses the overlap, and rejecting it there would make a
    # perfectly valid call fail on an irrelevant argument.
    if len(text) <= size:
        return [text]

    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))

        if end < len(text):
            # Look for a boundary in the last quarter of the window, so
            # a break never shortens a chunk by more than 25%.
            window_start = max(start, end - size // 4)
            for boundary in _BOUNDARIES:
                found = text.rfind(boundary, window_start, end)
                if found > start:
                    end = found + len(boundary)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        # Overlap backwards from the new start, never past the previous
        # one — without the guard a tiny chunk can make start go
        # backwards and loop forever.
        start = max(start + 1, end - overlap)

    return chunks


_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def chunk_markdown(text: str, *, size: int = 1200, overlap: int = 200) -> list[str]:
    """Chunk markdown along headings, then by size within each section.

    A heading is the author telling you where one idea stops. Ignoring
    that and cutting every 1200 characters is how a retrieved chunk ends
    up half about one thing and half about another.
    """
    matches = list(_HEADING.finditer(text))
    if not matches:
        return chunk_text(text, size=size, overlap=overlap)

    sections: list[str] = []
    if matches[0].start() > 0:
        sections.append(text[: matches[0].start()])

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append(text[match.start() : end])

    chunks: list[str] = []
    for section in sections:
        chunks.extend(chunk_text(section, size=size, overlap=overlap))
    return chunks
