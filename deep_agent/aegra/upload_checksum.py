"""Parsing and comparison for ``/upload`` checksums.

A caller declares what they are sending, and what gets verified follows the
payload type:

* An **archive** is verified as transmitted. ``checksum`` is the sha256 of
  the ``.zip``/``.tar`` file itself, exactly what ``sha256sum data.zip``
  prints, and is compared before the payload is unpacked. Members inside the
  archive are not digested.
* **Individually uploaded files** are verified one by one: ``checksum`` for a
  single file, or ``checksums`` -- a JSON object of filename to digest --
  for several. :func:`compare_manifest` runs against the staged files before
  publishing, so a payload that does not match never reaches the live tree.

Both are caught before anything becomes visible to the agent. What a digest
proves either way is that the bytes received are the bytes sent: it catches
truncation, corruption and tampering in transit, not a caller who assembled
the wrong payload in the first place and digested that.

One property worth knowing about the archive form: an archive's digest
changes for reasons that are not its contents -- a member's mtime, the
compression level, the packing tool -- and a ``.tar.gz`` differs on every
run, since gzip writes the current time into its header. So a caller must
digest the exact artifact they are about to send, not an equivalent one
rebuilt from the same files.

Kept separate from ``upload_routes`` because these are pure functions over
caller input, with edge cases worth testing on their own.
"""

from __future__ import annotations

import json
from pathlib import PurePosixPath

# Algorithm name -> required hex length. A dict of one: the "<algo>:" prefix is
# parsed rather than assumed, so adding an algorithm later is an entry here
# instead of a reshape of the parser, and existing callers keep working.
_ALGORITHMS = {"sha256": 64}

# What a bare, unprefixed digest is taken to be. sha256 is what every other
# digest in this codebase uses.
_DEFAULT_ALGORITHM = "sha256"

_HEX = frozenset("0123456789abcdef")

# Caller-supplied paths are echoed back in rejection messages. They are the
# caller's own data, so there is nothing to disclose, but an unbounded path
# would turn one bad entry into an unreadable error.
_MAX_PATH_IN_ERROR = 120


class ChecksumInvalid(Exception):
    """A checksum field is malformed. Surfaces as a 400 with the reason."""


def _short(path: str) -> str:
    """Truncate a path for inclusion in an error message."""
    if len(path) <= _MAX_PATH_IN_ERROR:
        return path
    return f"{path[:_MAX_PATH_IN_ERROR]}…"


def normalize_path(name: str) -> str:
    """Return the comparison form of a manifest key.

    Collapses spellings that address the same file: ``./docs/a.md``,
    ``docs//a.md`` and ``docs/a.md`` are one path. This matters in practice
    because ``find . -type f`` prints the first spelling while ``zipfile``
    reports the last, and a caller should not have to reconcile the two.

    Whitespace is deliberately **not** stripped. A leading or trailing space
    is a legal part of a filename, so `` a.md`` and ``a.md`` name two
    different files; folding them together would let one entry's digest
    stand in for another's and publish a file nobody declared.
    """
    return str(PurePosixPath(name))


def parse_digest(raw: str) -> str:
    """Validate one digest and return it as lowercase hex.

    Accepts ``sha256:<hex>`` or a bare hex string, in any case.

    Raises:
        ChecksumInvalid: If the algorithm is unsupported, or the value is not
            hex of the length that algorithm produces.
    """
    value = raw.strip().lower()
    algorithm = _DEFAULT_ALGORITHM

    if ":" in value:
        algorithm, _, value = value.partition(":")
        if algorithm not in _ALGORITHMS:
            supported = ", ".join(sorted(_ALGORITHMS))
            raise ChecksumInvalid(
                f"unsupported checksum algorithm '{_short(algorithm)}': "
                f"expected {supported}"
            )

    width = _ALGORITHMS[algorithm]
    if len(value) != width or not set(value) <= _HEX:
        raise ChecksumInvalid(
            f"malformed {algorithm} digest: expected {width} hex characters"
        )
    return value


def parse_manifest(raw: str, max_entries: int) -> dict[str, str]:
    """Parse the ``checksums`` field into normalized path -> hex digest.

    Args:
        raw: The raw JSON object sent by the caller.
        max_entries: Upper bound on entries, mirroring the payload's own
            entry cap so a manifest cannot be the cheaper way to exhaust
            memory.

    Returns:
        Mapping of normalized path to lowercase hex digest.

    Raises:
        ChecksumInvalid: If the field is not a non-empty JSON object of
            string digests with unique, non-empty paths.
    """
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ChecksumInvalid(f"checksums is not valid JSON: {exc.msg}") from None
    except (RecursionError, ValueError):
        # json.loads has two failure modes that are NOT JSONDecodeError:
        # deeply nested input ("[" * 100000, which fits comfortably in a
        # text part) blows the recursion limit, and an over-long integer
        # literal trips CPython's int/str conversion guard with a bare
        # ValueError. RecursionError is not a ValueError at all, so without
        # this it escapes the route entirely as a 500; the ValueError lands
        # in the route's multipart handler and is misreported as a
        # malformed body. Both are a bad `checksums` field, so both belong
        # here -- a 400 naming the real field, audited as a rejection.
        # JSONDecodeError is itself a ValueError subclass, hence the
        # ordering above.
        raise ChecksumInvalid("checksums is not valid JSON") from None

    if not isinstance(loaded, dict):
        raise ChecksumInvalid("checksums must be a JSON object of path to digest")
    if not loaded:
        # An empty manifest asserts nothing. Taking it as "verified" would
        # hand back a 200 that means the opposite of what the caller wanted.
        raise ChecksumInvalid("checksums is empty")
    if len(loaded) > max_entries:
        raise ChecksumInvalid(f"checksums exceeds {max_entries} entries")

    manifest: dict[str, str] = {}
    for path, digest in loaded.items():
        if not isinstance(digest, str):
            raise ChecksumInvalid(f"checksum for '{_short(path)}' is not a string")

        key = normalize_path(path)
        if not key or key == ".":
            raise ChecksumInvalid("checksums contains an empty path")
        # JSON itself collapses literal duplicate keys, but two spellings of
        # one path survive until normalization and would silently drop an
        # assertion the caller made.
        if key in manifest:
            raise ChecksumInvalid(f"duplicate path in checksums: {_short(key)}")

        try:
            manifest[key] = parse_digest(digest)
        except ChecksumInvalid as exc:
            raise ChecksumInvalid(f"{_short(path)}: {exc}") from None

    return manifest


def compare_manifest(
    expected: dict[str, str],
    actual: dict[str, str],
    filtered: set[str] | None = None,
) -> str | None:
    """Compare a parsed manifest against what the payload actually produced.

    An exact set match in both directions: a declared file that did not land
    and a file that landed undeclared are both failures. The first is the
    truncated-archive case this feature exists for; the second means the
    caller shipped something they did not account for.

    Paths are checked in sorted order so the same payload always reports the
    same reason, rather than whichever entry iteration happened to reach
    first.

    Args:
        expected: Normalized path -> digest, from the caller's manifest.
        actual: Path -> digest for what actually landed in staging.
        filtered: Paths dropped by the extension allowlist. Named separately
            so a declared file that the *server* filtered is reported as
            such, instead of as missing -- which would send the caller
            hunting for a truncated upload that never happened.

    Returns:
        A human-readable reason for the first mismatch, or None if the
        payload matches the manifest exactly.
    """
    filtered = filtered or set()

    for path in sorted(expected):
        if path in filtered:
            return (
                f"{_short(path)} is listed in checksums but its extension is "
                f"not in UPLOAD_ALLOWED_EXTENSIONS, so it was not published"
            )
        if path not in actual:
            return f"{_short(path)} is listed in checksums but was not published"

    for path in sorted(actual):
        if path not in expected:
            return f"{_short(path)} was published but is not listed in checksums"

    for path in sorted(expected):
        if expected[path] != actual[path]:
            return (
                f"{_short(path)} does not match: expected sha256:{expected[path]}, "
                f"got sha256:{actual[path]}"
            )

    return None
