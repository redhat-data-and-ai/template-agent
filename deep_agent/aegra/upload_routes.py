"""Archive and file ingestion into the agent's data directory.

Provides ``POST /upload``, which accepts either a single ``.zip``/``.tar``
archive or one or more loose files and publishes them under ``DATA_PATH`` so
the agent can read them back through the existing read-only filesystem tool
sandbox (the ``/data/`` route in ``runtime/agent.yaml``).

Mounted from ``http_app.py``. Gated by ``UPLOAD_ENABLED`` (404 when off) and
``check_ldap_role(developer_only=True)``.

Semantics follow the payload type: an archive is a snapshot and **replaces**
the whole tree; loose files are **merged** into it. Both publish atomically —
the live tree is never modified unless the entire payload passes every guard.
See ``adrs/upload-data-pvc.md``.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import stat
import tarfile
import tempfile
import time
import uuid
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath
from typing import IO, Literal, TypeAlias
from urllib.parse import unquote

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel

from deep_agent.aegra.auth_helpers import check_ldap_role
from deep_agent.src.audit.context import bind_audit_context
from deep_agent.src.audit.emitter import emit_audit_event
from deep_agent.src.audit.events import AuditEventType
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

upload_router = APIRouter(tags=["upload"])

# Name of the live tree inside DATA_PATH. DATA_PATH is itself a mount point and
# cannot be renamed over, so the swap target has to be a regular directory one
# level in. Hidden from the agent by the `roots:` mapping in agent.yaml, so the
# agent-visible path stays /data/<exactly what the caller uploaded>.
LIVE_DIR = "current"

_STAGING_PREFIX = ".staging-"
_OLD_PREFIX = ".old-"
_LOCK_NAME = ".upload.lock"

# Rejected anywhere in a path: NUL, newline, and the rest of the C0 range.
_CONTROL_CHARS = frozenset(chr(c) for c in range(32)) | {chr(127)}

# 255 bytes is the per-component limit on ext4, XFS, APFS and NFS alike.
_MAX_NAME_COMPONENT = 255

# Group-writable, because the data outlives the process that wrote it. Some
# platforms assign the container's UID at runtime, so it can differ between
# restarts while the volume persists; a shared group keeps the tree manageable.
_DIR_MODE = 0o775
_FILE_MODE = 0o664

# 64KB: large enough that the per-chunk overhead is irrelevant, small enough
# that the size caps are enforced promptly on a hostile stream.
_CHUNK = 64 * 1024

# "ustar" sits at offset 257 of a tar header block.
_TAR_MAGIC_OFFSET = 257
_TAR_MAGIC_END = 262


# An archive is a snapshot and replaces the tree; loose files are merged into
# it. Inferred from the payload, never from the filename or content-type.
Mode: TypeAlias = Literal["replace", "merge"]


class UploadRejected(Exception):
    """A payload failed validation. Surfaces as a 400 with the given reason."""


class UploadResponse(BaseModel):
    """Result of a successful upload."""

    mode: Mode
    files_written: int
    files_skipped: int
    bytes_written: int


class DeleteResponse(BaseModel):
    """Result of clearing the data directory."""

    files_deleted: int
    bytes_deleted: int


@dataclass
class _Entry:
    """One payload item, normalized across zip entries, tar members and files.

    Attributes:
        name: Caller-supplied path, preserved verbatim if it passes the guards.
        is_dir: True for directory entries. Archives carry one per
            subdirectory; nothing is written for them and they are not
            reported as skipped files, since a caller does not think of them
            as files at all.
        is_regular: False for symlinks, devices and other non-regular
            members. These are skipped rather than written, but unlike
            directories they ARE reported, because a caller may well have
            expected them to land.
        open: Callable returning a binary stream of the entry's contents.
    """

    name: str
    is_dir: bool
    is_regular: bool
    open: Callable[[], IO[bytes]]


def _chmod_shared(path: Path, mode: int) -> None:
    """Make a path group-writable, ignoring failure.

    Uploaded data outlives the process that wrote it, and the container's
    UID is not guaranteed to be the same on the next start — some platforms
    assign it at runtime. Group-writable modes let a successor UID sharing
    the group still manage the tree rather than inheriting files it cannot
    replace. Best-effort: where the filesystem rejects chmod this is
    cosmetic, not correctness.
    """
    try:
        path.chmod(mode)
    except OSError as exc:  # pragma: no cover - filesystem dependent
        logger.debug("upload_chmod_skipped", path=str(path), error=str(exc))


def _mkdir_shared(path: Path) -> None:
    """Create a directory (and parents) group-writable."""
    path.mkdir(parents=True, exist_ok=True)
    _chmod_shared(path, _DIR_MODE)


def _link_or_copy(src: str, dst: str, *, follow_symlinks: bool = True) -> None:
    """Hardlink a file into staging, falling back to a real copy.

    Hardlinks make merge staging near-free: no data is duplicated and
    unchanged files keep sharing inodes. But Linux's
    ``fs.protected_hardlinks=1`` (the default) refuses a link to a file the
    caller neither owns nor can write, which can happen if the container's
    UID changed since the tree was written. Falling back to a copy keeps merge
    working, just slower, instead of failing the upload outright.
    """
    try:
        os.link(src, dst)
    except OSError as exc:  # pragma: no cover - requires a foreign-owned tree
        logger.warning("upload_hardlink_fallback_copy", error=str(exc))
        shutil.copy2(src, dst)


def _allowed_extensions() -> frozenset[str]:
    """Parse UPLOAD_ALLOWED_EXTENSIONS into a lowercase set of suffixes."""
    return frozenset(
        ext.strip().lower()
        for ext in settings.UPLOAD_ALLOWED_EXTENSIONS.split(",")
        if ext.strip()
    )


def _require_enabled() -> None:
    """Raise 404 if the upload feature is disabled."""
    if not settings.UPLOAD_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Upload feature is disabled"
        )


def _data_root() -> Path:
    """Return the writable data mount, creating it if absent."""
    root = Path(settings.DATA_PATH)
    _mkdir_shared(root)
    return root


# ── Payload normalization ─────────────────────────────────────────────


def _zip_entries(archive: zipfile.ZipFile) -> Iterator[_Entry]:
    """Yield normalized entries from a zip archive."""
    for info in archive.infolist():
        # zipfile has no symlink concept of its own — the bit lives in the
        # high half of external_attr. Without this check an unchecked extract
        # silently materializes a symlink.
        mode = info.external_attr >> 16
        is_dir = info.is_dir()
        yield _Entry(
            name=info.filename,
            is_dir=is_dir,
            is_regular=not is_dir and not stat.S_ISLNK(mode),
            open=partial(archive.open, info),
        )


def _tar_entries(archive: tarfile.TarFile) -> Iterator[_Entry]:
    """Yield normalized entries from a tar archive."""

    def _open(member: tarfile.TarInfo) -> IO[bytes]:
        fh = archive.extractfile(member)
        if fh is None:  # pragma: no cover - non-regular members are skipped
            raise UploadRejected(f"unreadable archive member: {member.name}")
        return fh

    for member in archive.getmembers():
        yield _Entry(
            name=member.name,
            is_dir=member.isdir(),
            is_regular=member.isreg(),
            open=partial(_open, member),
        )


def _loose_entries(files: list[UploadFile]) -> Iterator[_Entry]:
    """Yield normalized entries from directly posted files."""

    def _open(upload: UploadFile) -> IO[bytes]:
        stream: IO[bytes] = upload.file
        return stream

    for upload in files:
        yield _Entry(
            name=upload.filename or "",
            is_dir=False,
            is_regular=True,
            open=partial(_open, upload),
        )


# ── Guards ────────────────────────────────────────────────────────────


def _safe_destination(name: str, root: Path) -> Path:
    """Resolve an entry name inside root, rejecting traversal attempts.

    Applies to archive entries and to multipart filenames alike — a
    caller-controlled ``filename`` is the same attack by a different route.

    Raises:
        UploadRejected: If the name is empty, absolute, or escapes root.
    """
    if not name or name.endswith("/"):
        raise UploadRejected("entry has no filename")

    # Judge the DECODED name, not the literal one. Multipart transport
    # percent-encodes hostile characters before they reach us, so checking
    # the raw string alone would miss "%00" and "%2e%2e". Decoding first
    # catches both spellings with one rule, and a name whose decoded form is
    # dangerous is a name worth refusing however it arrived.
    decoded = unquote(name)

    if any(ch in decoded for ch in _CONTROL_CHARS):
        raise UploadRejected("filename contains control characters")

    # Backslashes never escape on POSIX -- they land as one literal, bizarre
    # filename -- but they signal a caller doing something wrong, and such a
    # tree is a traversal waiting to happen if it is ever read somewhere
    # that treats "\" as a separator.
    if "\\" in decoded:
        raise UploadRejected(f"backslash in path rejected: {name}")

    # Traversal in the decoded form, e.g. "%2e%2e/%2e%2e/etc".
    if decoded != name:
        d = PurePosixPath(decoded)
        if d.is_absolute() or ".." in d.parts:
            raise UploadRejected(f"percent-encoded path rejected: {name}")

    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts:
        raise UploadRejected(f"path traversal rejected: {name}")

    # Bound each component to what any common filesystem accepts, so an
    # over-long name is the caller's 400 rather than an ENAMETOOLONG 500
    # that would also disclose the internal staging path.
    for part in pure.parts:
        if len(part.encode()) > _MAX_NAME_COMPONENT:
            raise UploadRejected(
                f"path component exceeds {_MAX_NAME_COMPONENT} bytes: {part[:40]}…"
            )

    destination = (root / pure).resolve()
    if not destination.is_relative_to(root.resolve()):
        raise UploadRejected(f"path traversal rejected: {name}")
    return destination


def _write_entry(source: IO[bytes], destination: Path) -> int:
    """Stream one entry to disk, enforcing the per-file cap as it goes.

    The cap is checked against bytes actually written rather than any size
    declared in the archive header, which a crafted archive can understate.

    Returns:
        Number of bytes written.

    Raises:
        UploadRejected: If the file exceeds UPLOAD_MAX_FILE_BYTES.
    """
    _mkdir_shared(destination.parent)
    written = 0
    # Overwrite by replacing, never by mutating in place: staging may contain
    # hardlinks to the live tree, and writing through one would corrupt it.
    destination.unlink(missing_ok=True)
    with destination.open("wb") as out:
        while chunk := source.read(_CHUNK):
            written += len(chunk)
            if written > settings.UPLOAD_MAX_FILE_BYTES:
                raise UploadRejected(
                    f"file exceeds per-file cap of "
                    f"{settings.UPLOAD_MAX_FILE_BYTES} bytes: {destination.name}"
                )
            out.write(chunk)
    _chmod_shared(destination, _FILE_MODE)
    return written


def _tree_size(root: Path) -> int:
    """Return the total size in bytes of all files under root."""
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def _populate(
    entries: Iterator[_Entry],
    staging: Path,
    baseline: int,
    *,
    apply_allowlist: bool,
) -> tuple[int, int, int]:
    """Write permitted entries into staging, enforcing every per-entry guard.

    Args:
        entries: Normalized payload items.
        staging: Destination directory, already seeded for merge mode.
        baseline: Bytes already present in staging (non-zero when merging),
            counted against the cumulative cap so repeated merges cannot grow
            the tree past it.
        apply_allowlist: Whether to filter by UPLOAD_ALLOWED_EXTENSIONS.
            True for loose files, False for archives -- an archive is a
            snapshot the caller deliberately assembled, so silently dropping
            parts of it is worse than accepting what they chose to send. The
            safety guards (traversal, symlinks, per-file and total size,
            entry count) apply to both.

    Returns:
        Tuple of (files_written, files_skipped, bytes_written).

    Raises:
        UploadRejected: If any guard trips.
    """
    allowed = _allowed_extensions() if apply_allowlist else frozenset()
    written_files = skipped = 0
    written_bytes = 0
    seen = 0
    # Running size of the resulting tree. Replacing an existing file costs only
    # the difference, so a merge that overwrites in place doesn't count the old
    # copy and the new one against the cap.
    tree_bytes = baseline

    for entry in entries:
        seen += 1
        if seen > settings.UPLOAD_MAX_FILE_COUNT:
            raise UploadRejected(
                f"upload exceeds {settings.UPLOAD_MAX_FILE_COUNT} entries"
            )

        # Directory entries carry nothing to write -- the directories are
        # created implicitly as files land in them -- and a caller does not
        # count them as files, so they are not reported as skipped. An
        # archive of a modest repo holds hundreds of them, which would
        # otherwise swamp the skipped count and read as data loss.
        if entry.is_dir:
            continue

        # Symlinks and devices are skipped but DO count: unlike a directory,
        # the caller may have expected one of these to land.
        if not entry.is_regular:
            skipped += 1
            continue

        # For entries that WILL be written, traversal is checked before the
        # allowlist: a hostile path is rejected outright, not quietly skipped.
        destination = _safe_destination(entry.name, staging)

        if apply_allowlist and PurePosixPath(entry.name).suffix.lower() not in allowed:
            skipped += 1
            continue

        replaced = destination.stat().st_size if destination.is_file() else 0

        with entry.open() as source:
            size = _write_entry(source, destination)

        written_files += 1
        written_bytes += size
        tree_bytes += size - replaced
        if tree_bytes > settings.UPLOAD_MAX_EXTRACTED_BYTES:
            raise UploadRejected(
                f"total size exceeds {settings.UPLOAD_MAX_EXTRACTED_BYTES} bytes"
            )

    if written_files == 0:
        raise UploadRejected("payload contained no permitted files")

    return written_files, skipped, written_bytes


# ── Staging and publish ───────────────────────────────────────────────


def _publish(staging: Path, live: Path) -> None:
    """Swap staging into place as the live tree.

    POSIX rename() onto a non-empty directory fails with ENOTEMPTY, so this is
    move-aside rather than a single rename. Readers therefore see either the
    old tree or the new one, never a half-written directory.

    Raises:
        OSError: If the swap fails after the old tree was moved aside, in
            which case the old tree is restored first.
    """
    old = live.parent / f"{_OLD_PREFIX}{uuid.uuid4().hex}"
    moved = False
    if live.exists():
        os.rename(live, old)
        moved = True
    try:
        os.rename(staging, live)
    except OSError:
        # Two separate renames: without this the agent is left with no data.
        if moved:
            os.rename(old, live)
        raise
    if moved:
        shutil.rmtree(old, ignore_errors=True)


class UploadBusy(Exception):
    """Another upload holds the lock. Surfaces as a 409."""


def _sweep_orphans(data_root: Path) -> None:
    """Delete staging/old directories abandoned by a crashed or raced upload.

    Runs under the lock, so any such directory belongs to a dead request and
    is safe to remove. Without this they accumulate on the data volume
    forever — "the caller owns data lifecycle" covers the data, not the
    feature's own scratch.
    """
    for path in data_root.iterdir():
        if path.name.startswith((_STAGING_PREFIX, _OLD_PREFIX)) and path.is_dir():
            logger.warning("upload_orphan_swept", path=path.name)
            shutil.rmtree(path, ignore_errors=True)


@contextmanager
def _upload_lock(data_root: Path) -> Iterator[None]:
    """Serialize uploads across every process sharing the data volume.

    ``O_CREAT | O_EXCL`` is atomic on POSIX and on networked filesystems, so
    the lock holds across processes and hosts — an in-process lock would
    not, since replicas of this service all mount the same volume. Without it, two publishers race: one can fail the swap
    with ENOTEMPTY and leak an orphaned tree, and concurrent merges can
    silently lose updates.

    A lock older than UPLOAD_LOCK_TIMEOUT_SECONDS is treated as stale and
    broken, so a process killed mid-upload cannot wedge ingestion forever.

    Raises:
        UploadBusy: If another upload currently holds the lock.
    """
    lock = data_root / _LOCK_NAME
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        try:
            age = time.time() - lock.stat().st_mtime
        except FileNotFoundError:
            # Released between our attempt and the stat — caller can retry.
            raise UploadBusy("another upload is in progress") from None
        if age < settings.UPLOAD_LOCK_TIMEOUT_SECONDS:
            raise UploadBusy("another upload is in progress") from None

        # Steal the stale lock by RENAMING it aside, not by unlinking it.
        # Unlinking by path is unsafe: if two holders both see the same
        # stale lock, the second one's unlink deletes the *fresh* lock the
        # first just created, and both go on to publish concurrently.
        # rename() is atomic and names a specific directory entry, so of N
        # racing stealers exactly one succeeds and the rest get ENOENT.
        stolen = data_root / f"{_LOCK_NAME}.stale-{uuid.uuid4().hex}"
        try:
            os.rename(lock, stolen)
        except OSError:
            # Someone else stole it first; they now hold it.
            raise UploadBusy("another upload is in progress") from None
        logger.warning("upload_lock_stale_broken", age_seconds=int(age))
        Path(stolen).unlink(missing_ok=True)
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            # Stole the stale entry but lost the race to recreate it.
            raise UploadBusy("another upload is in progress") from None

    # Identify *our* lock by inode so release cannot delete a successor's.
    # Without this, a release after our lock was stolen would unlink the
    # new holder's file and let a third upload in alongside them.
    try:
        our_inode = os.fstat(fd).st_ino
        os.write(fd, f"{os.getpid()}\n".encode())
    finally:
        os.close(fd)

    try:
        yield
    finally:
        try:
            if lock.stat().st_ino == our_inode:
                lock.unlink(missing_ok=True)
            else:
                logger.warning("upload_lock_not_ours_on_release")
        except FileNotFoundError:
            pass


def _ingest(
    payload: Path | list[UploadFile], mode: Mode, data_root: Path
) -> UploadResponse:
    """Stage, validate and publish a payload. Runs in a worker thread.

    Args:
        payload: Path to a buffered archive, or the list of loose uploads.
        mode: Either ``replace`` (archive) or ``merge`` (loose files).
        data_root: The writable mount containing the live tree.

    Returns:
        Counts describing what was published.

    Raises:
        UploadRejected: If the payload fails any guard.
        UploadBusy: If another upload is already in progress.
    """
    live = data_root / LIVE_DIR
    staging = data_root / f"{_STAGING_PREFIX}{uuid.uuid4().hex}"

    # Held for the whole stage-and-swap. Without it two publishers race: one
    # can fail the swap with ENOTEMPTY and leak an orphaned tree, and
    # concurrent merges can silently lose each other's files.
    with _upload_lock(data_root):
        try:
            _sweep_orphans(data_root)
            if mode == "merge" and live.is_dir():
                # Hardlink copy: no file data is duplicated and unchanged
                # files keep sharing inodes, so this costs milliseconds at
                # 30MB/5000 files. _write_entry unlinks before writing, so
                # overwriting a merged entry breaks the link instead of
                # touching the live tree.
                shutil.copytree(live, staging, copy_function=_link_or_copy)
                baseline = _tree_size(staging)
            else:
                _mkdir_shared(staging)
                baseline = 0

            # The extension allowlist applies to loose files only. An archive
            # is a snapshot the caller assembled on purpose, so it is taken
            # whole; filtering it would silently deliver less than they sent.
            if isinstance(payload, list):
                written, skipped, written_bytes = _populate(
                    _loose_entries(payload),
                    staging,
                    baseline,
                    apply_allowlist=True,
                )
            elif zipfile.is_zipfile(payload):
                with zipfile.ZipFile(payload) as archive:
                    written, skipped, written_bytes = _populate(
                        _zip_entries(archive),
                        staging,
                        baseline,
                        apply_allowlist=False,
                    )
            else:
                with tarfile.open(payload) as archive:
                    written, skipped, written_bytes = _populate(
                        _tar_entries(archive),
                        staging,
                        baseline,
                        apply_allowlist=False,
                    )

            _publish(staging, live)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    return UploadResponse(
        mode=mode,
        files_written=written,
        files_skipped=skipped,
        bytes_written=written_bytes,
    )


def _clear(data_root: Path) -> DeleteResponse:
    """Remove the live tree. Runs in a worker thread.

    Takes the same lock and uses the same move-aside as a publish, so a
    reader mid-conversation sees the complete old tree or nothing, never a
    directory being emptied underneath it.

    Raises:
        UploadBusy: If an upload is in progress.
    """
    live = data_root / LIVE_DIR

    with _upload_lock(data_root):
        _sweep_orphans(data_root)
        if not live.is_dir():
            return DeleteResponse(files_deleted=0, bytes_deleted=0)

        files = [p for p in live.rglob("*") if p.is_file()]
        count = len(files)
        size = sum(p.stat().st_size for p in files)

        # Rename out of the way first so the tree disappears atomically;
        # the (slower) recursive delete then happens out of band.
        condemned = data_root / f"{_OLD_PREFIX}{uuid.uuid4().hex}"
        os.rename(live, condemned)
        shutil.rmtree(condemned, ignore_errors=True)

    return DeleteResponse(files_deleted=count, bytes_deleted=size)


async def _looks_like_archive(upload: UploadFile) -> bool:
    """Detect a zip/tar by magic bytes without consuming the stream.

    Used only to reject an archive sent alongside loose files. The
    authoritative check for the single-file case is still
    ``zipfile.is_zipfile`` / ``tarfile.is_tarfile`` on the buffered payload.
    """
    head: bytes = await upload.read(_TAR_MAGIC_END)
    await upload.seek(0)
    # zip (incl. empty/spanned), gzip, bzip2 and xz — the last three being
    # the usual compressed-tar wrappers.
    if head.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
        return True
    if head.startswith((b"\x1f\x8b", b"BZh", b"\xfd7zXZ")):
        return True
    # Uncompressed tar: "ustar" lives at offset 257 in the first header block.
    return head[_TAR_MAGIC_OFFSET:_TAR_MAGIC_END] == b"ustar"


async def _buffer(upload: UploadFile, destination: Path) -> None:
    """Stream an upload to disk, enforcing the archive size cap.

    Content-Length is not trusted, and the body is never held in memory.

    Raises:
        HTTPException: 413 if the body exceeds UPLOAD_MAX_ARCHIVE_BYTES.
    """
    total = 0
    with destination.open("wb") as out:
        while chunk := await upload.read(_CHUNK):
            total += len(chunk)
            if total > settings.UPLOAD_MAX_ARCHIVE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=(
                        f"Upload exceeds maximum size of "
                        f"{settings.UPLOAD_MAX_ARCHIVE_BYTES} bytes"
                    ),
                )
            out.write(chunk)
    await upload.seek(0)


# ── Route ─────────────────────────────────────────────────────────────


@upload_router.post("/upload", response_model=UploadResponse)
async def upload(
    request: Request, files: list[UploadFile] = File(...)
) -> UploadResponse:
    """Publish an archive or a set of files to the agent's data directory.

    A single ``.zip``/``.tar`` replaces the whole tree; loose files are merged
    into it. Caller-supplied names are preserved verbatim.

    Args:
        request: The incoming request, used for authorization.
        files: One or more multipart file fields.

    Returns:
        Counts describing what was published.

    Raises:
        HTTPException: 400 rejected payload, 401/403 unauthorized,
            404 feature disabled, 413 payload too large.
    """
    _require_enabled()
    user_id = await check_ldap_role(request, developer_only=True)
    # Populates the envelope's top-level user/trace_id. HTTP routes aren't
    # covered by AuditMiddleware, which binds this for agent runs.
    bind_audit_context(user=user_id)

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    data_root = _data_root()
    scratch = Path(tempfile.mkdtemp(prefix="upload-"))
    try:
        payload: Path | list[UploadFile]
        mode: Mode
        if len(files) == 1:
            buffered = scratch / "payload"
            await _buffer(files[0], buffered)
            # Mode is decided by content, never by filename or content-type.
            if zipfile.is_zipfile(buffered) or tarfile.is_tarfile(buffered):
                payload, mode = buffered, "replace"
            else:
                name = (files[0].filename or "").lower()
                if name.endswith((".zip", ".tar")):
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "File is named as an archive but is not a valid zip or tar"
                        ),
                    )
                payload, mode = files, "merge"
        else:
            # An archive is a snapshot and replaces the tree; loose files are
            # merged into it. Mixing them in one request has no coherent
            # meaning, and silently skipping the archive (it isn't on the
            # extension allowlist) would look like a successful upload.
            for candidate in files:
                if await _looks_like_archive(candidate):
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Archive '{candidate.filename}' must be uploaded on "
                            f"its own, not alongside other files"
                        ),
                    )
            payload, mode = files, "merge"

        try:
            result = await asyncio.to_thread(_ingest, payload, mode, data_root)
        except UploadRejected as exc:
            logger.warning("upload_rejected", user_id=user_id, reason=str(exc))
            # Audited too: a rejected push onto shared storage is exactly the
            # kind of event worth seeing a pattern of.
            emit_audit_event(
                AuditEventType.DATA_UPLOAD_REJECTED,
                user=user_id,
                mode=mode,
                reason=str(exc),
            )
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except UploadBusy as exc:
            logger.info("upload_busy", user_id=user_id)
            raise HTTPException(status_code=409, detail=str(exc)) from None
        except OSError as exc:
            # Staging is cleaned up and the previous tree restored by _ingest /
            # _publish before this point; the upload simply didn't land.
            # The OSError text carries absolute staging paths; log it, but
            # do not hand internal filesystem layout back to the caller.
            logger.exception("upload_publish_failed", user_id=user_id, error=str(exc))
            raise HTTPException(
                status_code=500, detail="Failed to publish upload"
            ) from None
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    logger.info(
        "upload_published",
        user_id=user_id,
        mode=result.mode,
        files_written=result.files_written,
        files_skipped=result.files_skipped,
        bytes_written=result.bytes_written,
    )
    emit_audit_event(
        AuditEventType.DATA_UPLOAD,
        user=user_id,
        mode=result.mode,
        files_written=result.files_written,
        files_skipped=result.files_skipped,
        bytes_written=result.bytes_written,
    )
    return result


@upload_router.delete("/upload", response_model=DeleteResponse)
async def clear_uploaded_data(request: Request) -> DeleteResponse:
    """Delete everything previously published to the agent's data directory.

    Gated identically to the upload itself — ``UPLOAD_ENABLED`` plus a
    developer LDAP role — since clearing the tree is the more destructive of
    the two operations.

    Args:
        request: The incoming request, used for authorization.

    Returns:
        Counts describing what was removed. Clearing an already-empty tree
        is a no-op 200, so repeated calls are idempotent.

    Raises:
        HTTPException: 401/403 unauthorized, 404 feature disabled,
            409 an upload is in progress, 500 the delete failed.
    """
    _require_enabled()
    user_id = await check_ldap_role(request, developer_only=True)
    # Populates the envelope's top-level user/trace_id. HTTP routes aren't
    # covered by AuditMiddleware, which binds this for agent runs.
    bind_audit_context(user=user_id)

    data_root = _data_root()
    try:
        result = await asyncio.to_thread(_clear, data_root)
    except UploadBusy as exc:
        logger.info("upload_busy", user_id=user_id)
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except OSError as exc:
        logger.exception("upload_clear_failed", user_id=user_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to clear data") from None

    logger.warning(
        "upload_data_cleared",
        user_id=user_id,
        files_deleted=result.files_deleted,
        bytes_deleted=result.bytes_deleted,
    )
    emit_audit_event(
        AuditEventType.DATA_DELETE,
        user=user_id,
        files_deleted=result.files_deleted,
        bytes_deleted=result.bytes_deleted,
    )
    return result
