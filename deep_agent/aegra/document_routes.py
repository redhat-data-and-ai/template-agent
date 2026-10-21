"""Documents on a mounted volume.

The agent reads these through its filesystem tools at `/documents/`;
this module is how they get there. Writes happen here, over HTTP,
from a person — the model's route is `filesystem_readonly` and the
permission rules deny write on `documents/**`. Human writes, model
reads, and the two never share a path.

Every route is **developer-only**:
`authenticated_user_id(..., reject_anonymous=True, developer_only=True)`,
the same privileged LDAP role that gates `/analytics/*` and
`/admin/*`.

Authentication alone was not enough. The volume is one shared 500 MB
space with no per-user partitioning — every authenticated person
would be able to read, overwrite and delete everybody else's data,
and the agent reads whatever is on it. Write access to `/documents/`
is therefore write access to what the model sees, which is an
operator capability rather than an end-user one.

The volume is small and finite, which is the whole reason this is an
API rather than a `kubectl cp`. A PVC that fills does not fail only
the upload that filled it: every later write fails too, including
ones from unrelated code, with an ENOSPC that surfaces somewhere far
from the cause. So capacity is checked *before* the bytes are
written, and reported on every read so it is visible before it is
urgent.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi import File as FileParam
from pydantic import BaseModel

from deep_agent.aegra.auth_helpers import authenticated_user_id
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

router = APIRouter(prefix=settings.DOCUMENTS_ROUTE_PREFIX, tags=["documents"])

# One writer at a time.
#
# Every write computes its headroom from a `_usage(root)` snapshot.
# Two concurrent uploads each read the same snapshot, each decide
# they fit, and together they overfill the volume — which is the one
# failure this endpoint's whole budget arithmetic exists to prevent.
# Per-process, so it holds for the single writer OpenShift actually
# gives this ReadWriteOnce volume.
_write_lock = asyncio.Lock()

# Matches the PVC in deployment/components/documents-pvc. Kept as a
# setting rather than read from the filesystem: `statvfs` reports the
# *volume's* size, which in Kind is the node disk, so it would happily
# accept 40 GB into a claim requesting 500Mi and fail at some later
# unrelated write.
_DEFAULT_CAPACITY_BYTES = settings.DOCUMENTS_CAPACITY_BYTES

# Warn well before refusing. A limit that is only mentioned at the
# moment it blocks you is a surprise, not a budget.
_WARN_AT = settings.DOCUMENTS_CAPACITY_WARN_FRACTION

_READABLE_SUFFIXES = {
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".parquet",
    ".xlsx",
    ".txt",
    ".md",
}


def _root() -> Path:
    return Path(os.environ.get("DOCUMENTS_PATH") or "/app/data/documents")


def _capacity_bytes() -> int:
    raw = os.environ.get("DOCUMENTS_MAX_BYTES")
    if raw:
        try:
            return int(raw)
        except ValueError:
            logger.warning("Invalid DOCUMENTS_MAX_BYTES %r — using default", raw)
    return _DEFAULT_CAPACITY_BYTES


def _used_bytes(root: Path) -> int:
    """Bytes currently on the volume.

    Walked rather than cached. The volume is small, the walk is cheap,
    and a cached figure that drifts below the truth is the one number
    that must not be wrong here.
    """
    total = 0
    if not root.is_dir():
        return 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def _usage(root: Path) -> dict[str, Any]:
    used = _used_bytes(root)
    capacity = _capacity_bytes()
    fraction = used / capacity if capacity else 0.0
    return {
        "used_bytes": used,
        "capacity_bytes": capacity,
        "percent_used": round(fraction * 100, 1),
        # A flag rather than a threshold the caller has to re-derive,
        # so the UI and the API agree on when to say something.
        "warning": fraction >= _WARN_AT,
    }


def _safe_target(root: Path, filename: str) -> Path:
    """Resolve *filename* under *root*, refusing anything that escapes.

    The name comes from a multipart upload, so it is attacker-chosen:
    `../../app/config/agent/PROMPT.md` would otherwise rewrite the
    agent's instructions through a dataset upload.

    Subdirectories are kept rather than flattened. Reducing the name
    to its last component meant `PUT /documents/sub/file.csv`
    answered 200 and wrote `file.csv` in the root, so the caller was
    told their path existed when it did not. Containment is checked
    after resolution, which is what actually stops the traversal.
    """
    candidate = (root / filename).resolve()
    if not candidate.is_relative_to(root.resolve()) or candidate == root.resolve():
        raise HTTPException(status_code=400, detail="Invalid filename")
    return candidate


@router.get("")
async def list_documents(request: Request) -> dict[str, Any]:
    """Files on the volume, and how full it is."""
    await authenticated_user_id(request, reject_anonymous=True, developer_only=True)
    root = _root()
    files: list[dict[str, Any]] = []
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            files.append(
                {
                    "path": str(path.relative_to(root)),
                    "bytes": stat.st_size,
                    "modified": stat.st_mtime,
                }
            )
    return {"documents": files, "usage": _usage(root)}


@router.post("")
async def upload_document(
    request: Request,
    files: list[UploadFile] = FileParam(...),
) -> dict[str, Any]:
    """Add files to the volume, refusing anything that would overfill it."""
    await authenticated_user_id(request, reject_anonymous=True, developer_only=True)
    root = _root()
    root.mkdir(parents=True, exist_ok=True)

    async with _write_lock:
        return await _store_uploads(root, files)


async def _store_uploads(root: Path, files: list[UploadFile]) -> dict[str, Any]:
    usage = _usage(root)
    remaining = usage["capacity_bytes"] - usage["used_bytes"]

    # Pre-flight on the declared sizes. `UploadFile.size` is what the
    # client said, so it is checked again below against the bytes
    # actually written — a declared size is a hint, not a guarantee.
    declared = sum(f.size or 0 for f in files)
    if declared > remaining:
        raise HTTPException(
            status_code=507,
            detail=(
                f"Not enough space: {_mb(declared)} requested, "
                f"{_mb(remaining)} free of {_mb(usage['capacity_bytes'])}. "
                "Delete a document first."
            ),
        )

    written: list[dict[str, Any]] = []
    budget = remaining
    for upload in files:
        target = _safe_target(root, upload.filename or "unnamed")
        if target.suffix.lower() not in _READABLE_SUFFIXES:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"{target.name}: unsupported type. Allowed: "
                    + ", ".join(sorted(_READABLE_SUFFIXES))
                ),
            )

        # Written beside the target and renamed on success. Writing
        # in place meant an oversize or failed re-upload deleted the
        # file that was already there: the cleanup below unlinks
        # `target`, which for a re-upload is the good copy.
        # `sub/new.csv` is a valid name and its directory need not
        # exist yet; without this both the staging write and the
        # rename raise and the upload answers 500.
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(f".{target.name}.part")
        size = 0
        try:
            with staging.open("wb") as sink:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    # Enforced per chunk, so a client that understated
                    # its size cannot walk past the limit. Streaming
                    # the whole file first and checking afterwards
                    # would already have filled the volume.
                    if size > budget:
                        sink.close()
                        staging.unlink(missing_ok=True)
                        raise HTTPException(
                            status_code=507,
                            detail=(
                                f"{target.name} exceeds the remaining "
                                f"{_mb(budget)} on the documents volume."
                            ),
                        )
                    sink.write(chunk)
        except HTTPException:
            staging.unlink(missing_ok=True)
            raise
        except OSError as exc:
            staging.unlink(missing_ok=True)
            logger.error("Could not write dataset %s: %s", target, exc)
            raise HTTPException(
                status_code=500, detail=f"Could not write {target.name}"
            ) from exc

        # Only now does the previous copy go, and atomically.
        staging.replace(target)
        budget -= size
        written.append({"path": target.name, "bytes": size})

    return {"uploaded": written, "usage": _usage(root)}


class DocumentBody(BaseModel):
    """Text content for a created or replaced document file."""

    content: str


# Big enough for a data dictionary or a config, small enough that a
# browser can render it. Anything larger is a file to download, not
# to edit in a text box.
_MAX_EDITABLE_BYTES = settings.DOCUMENTS_MAX_EDITABLE_BYTES


@router.get("/{name:path}")
async def read_document(name: str, request: Request) -> dict[str, Any]:
    """One file's content, for viewing or editing."""
    await authenticated_user_id(request, reject_anonymous=True, developer_only=True)
    root = _root()
    target = _resolve_existing(root, name)
    if target.is_dir():
        raise HTTPException(status_code=400, detail=f"{name} is a directory")

    # One byte past the cap, not the whole file: this is bounded by
    # the documents volume, not by anything sensible, so reading it
    # all to slice the first megabyte is how a large upload becomes
    # an out-of-memory kill.
    with target.open("rb") as handle:
        raw = handle.read(_MAX_EDITABLE_BYTES + 1)
    truncated = len(raw) > _MAX_EDITABLE_BYTES
    return {
        "path": name,
        "bytes": target.stat().st_size,
        # errors="replace" rather than refusing: a CSV with one bad
        # byte should still be readable enough to find the bad byte.
        "content": raw[:_MAX_EDITABLE_BYTES].decode("utf-8", errors="replace"),
        "truncated": truncated,
        # A truncated file must not be saved back — that would write
        # the prefix over the whole file and silently lose the rest.
        "editable": not truncated,
    }


@router.put("/{name:path}")
async def write_document(
    name: str, body: DocumentBody, request: Request
) -> dict[str, Any]:
    """Create or replace a text document file."""
    await authenticated_user_id(request, reject_anonymous=True, developer_only=True)
    root = _root()
    root.mkdir(parents=True, exist_ok=True)
    async with _write_lock:
        return _write_document(root, name, body)


def _write_document(root: Path, name: str, body: DocumentBody) -> dict[str, Any]:
    target = _safe_target(root, name)
    if target.suffix.lower() not in _READABLE_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"{target.name}: unsupported type. Allowed: "
                + ", ".join(sorted(_READABLE_SUFFIXES))
            ),
        )

    incoming = body.content.encode("utf-8")
    usage = _usage(root)
    # The bytes this file already occupies are about to be freed, so
    # a same-size edit of a file on a nearly full volume must not be
    # refused for needing space it is already using.
    existing = target.stat().st_size if target.is_file() else 0
    remaining = usage["capacity_bytes"] - usage["used_bytes"] + existing
    if len(incoming) > remaining:
        raise HTTPException(
            status_code=507,
            detail=(
                f"Not enough space: {_mb(len(incoming))} needed, "
                f"{_mb(remaining)} available."
            ),
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(incoming)
    return {"path": target.name, "bytes": len(incoming), "usage": _usage(root)}


def _resolve_existing(root: Path, name: str) -> Path:
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()) or not target.exists():
        raise HTTPException(status_code=404, detail=f"No such document: {name}")
    # `.` and `` both resolve to the root, which is inside itself and
    # does exist, so `DELETE /documents/.` passed every check and
    # `shutil.rmtree` took the whole volume.
    if target == root.resolve():
        raise HTTPException(status_code=400, detail="Refusing to act on the root")
    return target


@router.delete("/{name:path}")
async def delete_document(name: str, request: Request) -> dict[str, Any]:
    """Remove one document file or directory."""
    await authenticated_user_id(request, reject_anonymous=True, developer_only=True)
    root = _root()
    target = _resolve_existing(root, name)
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    return {"deleted": name, "usage": _usage(root)}


def _mb(value: int) -> str:
    return f"{value / (1024 * 1024):.1f} MB"
