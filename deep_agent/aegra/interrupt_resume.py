"""Checkpoint-aware resume adapter for the pinned Aegra 0.10.8 runtime.

A failed resume can leave SQL thread status ``error`` while checkpoint
interrupts remain. Validate before scheduling, and reconcile that specific case
in the run-creation transaction. Remove when Aegra supplies this behavior.
"""

from __future__ import annotations

import functools
import re
import sys
from importlib.metadata import version
from typing import Any

from fastapi import HTTPException
from sqlalchemy import select


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

    # Held until _prepare_run commits the new run and busy status. Another
    # browser/process then sees the pending run and cannot double-submit.
    thread = await session.scalar(
        select(Thread)
        .where(Thread.thread_id == thread_id, Thread.user_id == user.identity)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if thread is None:
        raise HTTPException(404, f"Thread '{thread_id}' not found")
    active = await session.scalar(
        select(Run.run_id)
        .where(
            Run.thread_id == thread_id,
            Run.user_id == user.identity,
            Run.status.in_(("pending", "running")),
        )
        .limit(1)
    )
    if active:
        raise HTTPException(409, "An active run already exists for this thread")

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

    response = request.command["resume"]
    # Recognize a stale ID map too, so it cannot fall through as an unkeyed
    # response to a single interrupt. IDs in LangGraph 1.2 are 32 hex digits.
    keyed = (
        isinstance(response, dict)
        and bool(response)
        and all(
            isinstance(key, str) and re.fullmatch(r"[0-9a-f]{32}", key)
            for key in response
        )
    )
    if keyed:
        if not set(response).issubset(pending):
            raise HTTPException(409, "Approval interrupt is no longer pending")
    elif len(pending) > 1:
        raise HTTPException(
            400, "Specify interrupt IDs when resuming multiple pending interrupts"
        )
    elif thread.status != "interrupted":
        raise HTTPException(
            400, "Specify the pending interrupt ID to recover this thread"
        )

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
