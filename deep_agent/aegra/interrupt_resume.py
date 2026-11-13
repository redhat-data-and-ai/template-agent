"""Checkpoint-aware resume adapter for the pinned Aegra 0.10.8 runtime.

A failed resume can leave SQL thread status ``error`` while checkpoint
interrupts remain. Validate before scheduling, and reconcile that specific case
in the run-creation transaction. Every resume must be keyed by interrupt ID,
including single-interrupt responses. Remove when Aegra supplies this behavior.
"""

from __future__ import annotations

import asyncio
import functools
import re
import sys
from importlib.metadata import version
from typing import Any

from fastapi import HTTPException
from sqlalchemy import select

_RESUME_SETTLE_ATTEMPTS = 10
_RESUME_SETTLE_INTERVAL_SECONDS = 0.1


def _pending_ids(snapshot: Any) -> set[str]:
    return {
        item.id
        for task in snapshot.tasks
        for item in task.interrupts
        if getattr(item, "id", None)
    }


async def validate_checkpoint_resume(
    session: Any, thread_id: str, request: Any, user: Any
) -> None:
    """Serialize resumes and reject stale/ambiguous responses before execution."""
    from aegra_api.core.orm import Run, Thread
    from aegra_api.services.langgraph_service import (
        create_thread_config,
        get_langgraph_service,
    )

    if not request.command or "resume" not in request.command:
        return

    response = request.command["resume"]
    # An unkeyed response has no stable target, even when only one interrupt
    # is pending. Reject it before waiting.
    if not (
        isinstance(response, dict)
        and response
        and all(
            isinstance(key, str) and re.fullmatch(r"[0-9a-f]{32}", key)
            for key in response
        )
    ):
        raise HTTPException(400, "Specify interrupt IDs for every approval response")

    thread_stmt = (
        select(Thread)
        .where(Thread.thread_id == thread_id, Thread.user_id == user.identity)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    active_stmt = (
        select(Run.run_id)
        .where(
            Run.thread_id == thread_id,
            Run.user_id == user.identity,
            Run.status.in_(("pending", "running")),
        )
        .limit(1)
    )
    for attempt in range(_RESUME_SETTLE_ATTEMPTS + 1):
        thread = await session.scalar(thread_stmt)
        if thread is None:
            raise HTTPException(404, f"Thread '{thread_id}' not found")
        active = await session.scalar(active_stmt)
        if not active and thread.status != "busy":
            break

        # Interrupt SSE can arrive before finalize_run commits.
        # Never sleep holding its thread row lock.
        await session.rollback()
        if attempt == _RESUME_SETTLE_ATTEMPTS:
            if active:
                raise HTTPException(409, "An active run already exists for this thread")
            raise HTTPException(409, "Thread is not ready for an approval response")
        await asyncio.sleep(_RESUME_SETTLE_INTERVAL_SECONDS)

    graph_id = (thread.metadata_json or {}).get("graph_id")
    if not graph_id:
        raise HTTPException(400, "Cannot resume: thread has no checkpoint graph")
    if request.checkpoint or request.checkpoint_id:
        raise HTTPException(400, "Resume approvals against the current checkpoint")
    config = create_thread_config(thread_id, user)
    service = get_langgraph_service()
    async with service.get_graph(
        graph_id, config=config, access_context="threads.read", user=user
    ) as graph:
        snapshot = await graph.aget_state(config)
    pending = _pending_ids(snapshot) if snapshot else set()
    if not pending:
        raise HTTPException(400, "Cannot resume: no pending checkpoint interrupts")

    if not set(response).issubset(pending):
        raise HTTPException(409, "Approval interrupt is no longer pending")

    if thread.status not in ("interrupted", "error"):
        raise HTTPException(409, "Thread is not ready for an approval response")
    if thread.status == "error":
        thread.status = "interrupted"
        await session.flush()


def install_resume_guard() -> None:
    """Install once at custom-app load, including already imported call sites."""
    from aegra_api.services import run_preparation

    original = run_preparation._prepare_run
    if getattr(original, "_checkpoint_resume_guard", False):
        return
    if version("aegra-api") != "0.10.8":
        raise RuntimeError(
            "Revalidate the checkpoint resume adapter before upgrading aegra-api"
        )

    @functools.wraps(original)
    async def guarded(
        session: Any, thread_id: str, request: Any, user: Any, **kwargs: Any
    ) -> Any:
        try:
            await validate_checkpoint_resume(session, thread_id, request, user)
            return await original(session, thread_id, request, user, **kwargs)
        except Exception:
            await session.rollback()
            raise

    guarded._checkpoint_resume_guard = True  # type: ignore[attr-defined]
    run_preparation._prepare_run = guarded
    # The 0.10.8 call sites importing by value. Later imports see the wrapper.
    for name in (
        "aegra_api.api.runs",
        "aegra_api.api.crons",
        "aegra_api.services.cron_scheduler",
        "aegra_api.services.event_streaming.commands",
    ):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "_prepare_run", None) is original:
            setattr(module, "_prepare_run", guarded)
