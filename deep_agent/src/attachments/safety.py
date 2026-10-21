"""Filename and content checks for uploaded files.

An upload endpoint is the classic path-traversal target: the filename
is attacker-controlled and ends up concatenated into a path. Everything
here assumes the name is hostile.
"""

from __future__ import annotations

import re
import unicodedata

from deep_agent.src.settings import settings

# Anything outside this set is replaced. Deliberately narrow — a
# filename only has to be readable, not expressive.
_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]")
_COLLAPSE = re.compile(r"_{2,}")

# Windows reserves these regardless of extension, and a fork may well
# mount the same volume from a Windows box.
_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}

MAX_NAME_LENGTH = settings.ATTACHMENT_MAX_NAME_LENGTH


def sanitise_filename(raw: str) -> str:
    """Reduce a user-supplied filename to something safe to join.

    Strips directory components rather than escaping them: a name is a
    name, and anything that looks like a path is a caller trying to
    write somewhere it should not.
    """
    # Normalise first: `.․/` and other look-alikes normalise into
    # characters the later checks actually catch.
    name = unicodedata.normalize("NFKC", raw or "")

    # Take the last component under either separator, so `../../etc/passwd`
    # and `..\\..\\windows` both reduce to the final element.
    name = name.replace("\\", "/").split("/")[-1]

    # Null bytes truncate paths in C APIs underneath Python.
    name = name.replace("\x00", "")

    name = _SAFE_CHARS.sub("_", name)
    name = _COLLAPSE.sub("_", name).strip("._-")

    if not name:
        return "upload"

    stem, _, suffix = name.rpartition(".")
    if stem and stem.lower() in _RESERVED:
        stem = f"{stem}_file"
        name = f"{stem}.{suffix}" if suffix else stem
    elif not stem and name.lower() in _RESERVED:
        name = f"{name}_file"

    if len(name) > MAX_NAME_LENGTH:
        stem, _, suffix = name.rpartition(".")
        keep = MAX_NAME_LENGTH - len(suffix) - 1 if suffix else MAX_NAME_LENGTH
        name = f"{stem[:keep]}.{suffix}" if suffix else name[:MAX_NAME_LENGTH]

    return name


def unique_name(name: str, taken: set[str]) -> str:
    """Avoid clobbering an existing attachment with the same name.

    Two files called `report.pdf` in one thread is normal; silently
    replacing the first with the second is not.
    """
    if name not in taken:
        return name
    stem, _, suffix = name.rpartition(".")
    if not stem:
        stem, suffix = name, ""
    for n in range(2, 1000):
        candidate = f"{stem}-{n}.{suffix}" if suffix else f"{stem}-{n}"
        if candidate not in taken:
            return candidate
    raise ValueError("too many files with the same name")
