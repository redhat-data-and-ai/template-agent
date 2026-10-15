"""Upload files to a thread.

Files are written into the agent's own virtual filesystem under the
configured mount path, so `read_file`, `grep` and `glob` reach them with
no new tool and no new permission surface. The alternative — a bespoke
attachment tool — would have meant a second filesystem with its own
authorisation story.

Uploads are the classic path-traversal and resource-exhaustion target,
so every limit here is enforced server-side regardless of what the
client claims: the declared size is ignored in favour of the bytes
actually read, and the filename is treated as hostile.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from deep_agent.aegra.auth_helpers import authenticated_user_id
from deep_agent.src.attachments.config import AttachmentsConfig, ExtractionConfig
from deep_agent.src.attachments.extract import (
    describe,
    extract_best,
    needs_base64,
    suffix_of,
)
from deep_agent.src.attachments.safety import sanitise_filename, unique_name
from deep_agent.utils.pylogger import get_python_logger


def suffix_or_file(name: str) -> str:
    """File extension, or `binary` when there is none to report."""
    return suffix_of(name) or "binary"


def _orchestrator_model() -> str:
    """Model used for transcription, from the agent's own config."""
    try:
        from deep_agent.src.agent.config import agent_config

        configured = (agent_config.get_orchestrator_config() or {}).get("model")
        if isinstance(configured, dict):
            return str(configured.get("name") or "gemini-3.5-flash-lite")
        return str(configured or "gemini-3.5-flash-lite")
    except Exception:
        return "gemini-3.5-flash-lite"


logger = get_python_logger()

router = APIRouter(tags=["attachments"])


def _config() -> AttachmentsConfig:
    """Attachment limits from agent.yaml, defaults when unset."""
    try:
        from deep_agent.src.agent.config import agent_config

        raw = (
            (agent_config.get_raw_section("attachments") or {})
            if hasattr(agent_config, "get_raw_section")
            else {}
        )
        return AttachmentsConfig(**raw)
    except Exception:
        # A broken attachments block must not take the endpoint down
        # with it; the defaults are the conservative ones.
        logger.warning("Could not read attachments config — using defaults")
        return AttachmentsConfig()


def _extraction_config() -> ExtractionConfig:
    """Document-extraction engine from agent.yaml. Defaults to built-in."""
    try:
        from deep_agent.src.agent.config import agent_config

        raw = (
            (agent_config.get_raw_section("extraction") or {})
            if hasattr(agent_config, "get_raw_section")
            else {}
        )
        return ExtractionConfig(**raw)
    except Exception:
        # Same reasoning as above, and more strongly: a typo in an
        # optional quality setting must not stop anyone uploading.
        logger.warning("Could not read extraction config — using built-in parsers")
        return ExtractionConfig()


# How long to wait for an in-flight run before giving up on the upload.
# Long enough for a normal reply, short enough that a wedged run does
# not hang the request forever.
_RUN_WAIT_SECONDS = 90
_RUN_POLL_SECONDS = 1.0
_ACTIVE_RUN_STATUSES = {"pending", "running"}


async def _await_idle_thread(client: Any, thread_id: str) -> None:
    """Block until no run is executing on this thread.

    Attachments are written as a thread-state update, and a run that
    was already executing checkpoints the state it loaded *before* that
    write — silently discarding the file. The upload returns 200, the
    chip appears, and the file is gone by the time the user asks about
    it.

    Waiting is the fix rather than retrying after, because the run that
    clobbers it may be the very run that was going to read it.
    """
    import asyncio
    import time

    deadline = time.monotonic() + _RUN_WAIT_SECONDS
    while time.monotonic() < deadline:
        try:
            runs = await client.get(f"/threads/{thread_id}/runs")
        except Exception:
            return  # Cannot tell; proceed rather than block the upload.
        if runs.status_code != 200:
            return
        body = runs.json()
        items = body if isinstance(body, list) else (body or {}).get("runs") or []
        if not any((r or {}).get("status") in _ACTIVE_RUN_STATUSES for r in items):
            return
        await asyncio.sleep(_RUN_POLL_SECONDS)

    logger.warning(
        "Thread %s still has an active run after %ss — writing anyway",
        thread_id,
        _RUN_WAIT_SECONDS,
    )


async def _read_bounded(upload: UploadFile, limit: int) -> bytes:
    """Read at most ``limit`` bytes, then one more to detect overflow.

    Trusting `content-length` would let a client declare 1 KB and send a
    gigabyte. Reading in chunks bounds memory regardless of what the
    header says.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(64 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail=f"{upload.filename}: exceeds {limit} bytes",
            )
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("/threads/{thread_id}/attachments")
async def upload_attachments(
    thread_id: str,
    request: Request,
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    """Attach one or more files to a thread."""
    user_id = await authenticated_user_id(request, reject_anonymous=True)
    config = _config()
    extraction = _extraction_config()

    if not config.enabled:
        raise HTTPException(status_code=404, detail="Attachments are disabled")

    if len(files) > config.max_files_per_request:
        raise HTTPException(
            status_code=400,
            detail=f"At most {config.max_files_per_request} files per request",
        )

    import httpx

    from deep_agent.src.settings import settings

    headers = {}
    auth = request.headers.get("Authorization")
    if auth:
        # The caller's own credentials, so reading and writing the
        # thread is authorised as them rather than as the service.
        headers["Authorization"] = auth

    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{settings.AGENT_PORT}",
        headers=headers,
        timeout=60,
    ) as client:
        state = await client.get(f"/threads/{thread_id}/state")

        if state.status_code == 404:
            # The UI creates a chat client-side and only materialises
            # the thread on the first message — which is precisely when
            # someone attaches a file. Create it rather than refusing:
            # the id is the user's own chat, and a 404 here reads as
            # "attachments are broken".
            created = await client.post(
                "/threads",
                json={
                    "thread_id": thread_id,
                    "metadata": {"graph_id": "agent", "assistant_id": "agent"},
                    "if_exists": "do_nothing",
                },
            )
            if created.status_code not in (200, 201, 409):
                logger.error(
                    "Could not create thread %s for attachment: %s",
                    thread_id,
                    created.text[:200],
                )
                raise HTTPException(status_code=502, detail="Could not create thread")
            state = await client.get(f"/threads/{thread_id}/state")

        if state.status_code != 200:
            raise HTTPException(status_code=502, detail="Could not read thread state")

        await _await_idle_thread(client, thread_id)
        # Re-read after waiting: the run we waited for may have written
        # files of its own, and merging against a stale snapshot would
        # drop them.
        refreshed = await client.get(f"/threads/{thread_id}/state")
        if refreshed.status_code == 200:
            state = refreshed

        payload = state.json() or {}
        values = payload.get("values") or {}
        # A thread created by the UI carries no graph association, and a
        # state update without one fails with "has no associated graph.
        # Cannot update state." — which surfaces as a bare 502 that says
        # nothing about the real cause.
        assistant_id = ((payload.get("metadata") or {}).get("assistant_id")) or "agent"
        existing: dict[str, Any] = dict(values.get("files") or {})

        def _size(entry: Any) -> int:
            if isinstance(entry, dict):
                return len(str(entry.get("content", "")))
            return len(str(entry))

        used = sum(_size(v) for v in existing.values())
        taken = {k.rsplit("/", 1)[-1] for k in existing}

        stored: list[dict[str, Any]] = []
        notes: list[str] = []

        for upload in files:
            raw_name = upload.filename or "upload"
            if not config.is_allowed(raw_name):
                raise HTTPException(
                    status_code=415,
                    detail=f"{raw_name}: file type not allowed",
                )

            data = await _read_bounded(upload, config.max_file_bytes)

            used += len(data)
            if used > config.max_thread_bytes:
                raise HTTPException(
                    status_code=413,
                    detail="Thread attachment quota exceeded",
                )

            name = unique_name(sanitise_filename(raw_name), taken)
            taken.add(name)

            result = await extract_best(name, data, config.max_inline_chars, extraction)
            text, complete = result.text, result.complete
            engine, degraded = result.engine, result.degraded
            path = f"{config.mount_path.rstrip('/')}/{name}"

            # Media the provider cannot take inline. The upload would
            # succeed and the *read* would fail, so the user learns
            # about it one question later with an error naming neither
            # the file nor the size.
            from deep_agent.src.knowledge.transcribe import (
                MAX_INLINE_BYTES,
                is_media,
            )

            if is_media(name) and len(data) > MAX_INLINE_BYTES:
                # Too large to inline, but not too large to use: it is
                # transcribed instead of watched, and the transcript is
                # what goes into the conversation. Cheaper in context
                # than the video would have been, and searchable.
                from deep_agent.src.knowledge.transcribe import (
                    TranscriptionUnavailable,
                    transcribe,
                )

                try:
                    text = transcribe(name, data, _orchestrator_model())
                    complete = True
                    engine, degraded = "transcription", ""
                    logger.info(
                        "Transcribed oversized media %s into %d characters",
                        name,
                        len(text),
                    )
                except TranscriptionUnavailable as exc:
                    raise HTTPException(status_code=413, detail=str(exc)) from None

            # deepagents' `files` channel holds FileData
            # (`{content, encoding}`), not bare strings. Writing a
            # string there fails at read time with "string indices must
            # be integers", which names neither the file nor the cause.
            if text:
                # Anything we could read: store the text.
                existing[path] = {"content": text, "encoding": "utf-8"}
            elif needs_base64(name):
                # PDFs, images and slides are delivered to the model as
                # multimodal blocks, which base64-decode the content
                # whatever `encoding` claims. Storing a placeholder
                # here breaks the *next* model call, not the upload —
                # with "Incorrect padding", naming neither the file nor
                # the attachment that caused it.
                #
                # Storing the real bytes also means the model can see
                # the page or the picture, rather than being told one
                # exists.
                import base64

                existing[path] = {
                    "content": base64.standard_b64encode(data).decode("ascii"),
                    "encoding": "base64",
                }
            else:
                # An unrecognised binary. deepagents classifies unknown
                # extensions as text, so a placeholder is safe here —
                # and far better than base64 of an archive the model
                # cannot interpret anyway. The file is recorded; it is
                # simply not readable.
                existing[path] = {
                    "content": (
                        f"[{suffix_or_file(name)} file, {len(data)} bytes — "
                        "stored but not readable as text]"
                    ),
                    "encoding": "utf-8",
                }

            stored.append(
                {
                    "filename": name,
                    "path": path,
                    "bytes": len(data),
                    "text_extracted": bool(text),
                    "truncated": not complete,
                    "engine": engine,
                    "degraded": degraded or None,
                }
            )
            notes.append(
                describe(
                    name,
                    len(data),
                    text,
                    complete,
                    engine=engine,
                    degraded=degraded,
                )
            )

        update = await client.post(
            f"/threads/{thread_id}/state",
            json={"values": {"files": existing}, "assistant_id": assistant_id},
        )
        if update.status_code == 400 and "no associated graph" in update.text:
            # A thread only acquires a graph when it first runs, so
            # attaching before the first message fails. PATCH the
            # association on and retry once, rather than failing an
            # upload the user has already waited for.
            #
            # Not POST /threads with if_exists=do_nothing: that is a
            # no-op on an existing thread, so it left the metadata
            # exactly as broken as it found it.
            await client.patch(
                f"/threads/{thread_id}",
                json={"metadata": {"graph_id": "agent", "assistant_id": assistant_id}},
            )
            update = await client.post(
                f"/threads/{thread_id}/state",
                json={"values": {"files": existing}, "assistant_id": assistant_id},
            )
        if update.status_code not in (200, 201):
            logger.error("Attachment state update failed: %s", update.text[:200])
            raise HTTPException(status_code=502, detail="Could not store attachments")

    logger.info(
        "Stored %d attachment(s) on thread %s for %s", len(stored), thread_id, user_id
    )
    return {"attachments": stored, "summary": "\n".join(notes)}


@router.get("/attachments/extraction")
async def extraction_status(request: Request) -> dict[str, Any]:
    """Which parser is in use, and can it actually be reached?

    A misconfigured extractor is otherwise discovered at upload time,
    in front of a user, in a message about their file. This turns it
    into something a readiness check or an operator can ask about.
    """
    await authenticated_user_id(request, reject_anonymous=True)

    from deep_agent.src.attachments.engines import probe

    extraction = _extraction_config()
    ok, detail = await probe(extraction)
    return {
        "engine": extraction.engine,
        "reachable": ok,
        "detail": detail,
        "fallback": extraction.fallback,
        "handles": sorted(extraction.suffixes)
        if extraction.engine != "builtin"
        else [],
        "max_bytes": extraction.max_bytes,
    }


@router.get("/threads/{thread_id}/attachments")
async def list_attachments(thread_id: str, request: Request) -> dict[str, Any]:
    """What is already attached to this thread."""
    await authenticated_user_id(request, reject_anonymous=True)
    config = _config()

    import httpx

    from deep_agent.src.settings import settings

    headers = {}
    auth = request.headers.get("Authorization")
    if auth:
        headers["Authorization"] = auth

    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{settings.AGENT_PORT}", headers=headers, timeout=30
    ) as client:
        state = await client.get(f"/threads/{thread_id}/state")
        if state.status_code != 200:
            raise HTTPException(status_code=404, detail="Thread not found")

        files = ((state.json() or {}).get("values") or {}).get("files") or {}

    prefix = config.mount_path.rstrip("/") + "/"
    return {
        "attachments": [
            {
                "filename": path[len(prefix) :],
                "path": path,
                "chars": len(
                    str(body.get("content", "")) if isinstance(body, dict) else body
                ),
            }
            for path, body in files.items()
            if path.startswith(prefix)
        ]
    }
