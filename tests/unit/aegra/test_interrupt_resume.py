"""Validate real checkpoint interrupts with isolated SQL I/O."""

import operator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import Annotated, TypedDict

from deep_agent.aegra.interrupt_resume import validate_checkpoint_resume


class State(TypedDict):
    answers: Annotated[list, operator.add]


@pytest.fixture
async def checkpoint(monkeypatch):
    from aegra_api.models import User
    from aegra_api.services import langgraph_service

    def a(state):
        return {"answers": [("a", interrupt("Approve A"))]}

    def b(state):
        return {"answers": [("b", interrupt("Approve B"))]}

    builder = StateGraph(State)
    for name, node in (("a", a), ("b", b)):
        builder.add_node(name, node)
        builder.add_edge(START, name)
        builder.add_edge(name, END)
    graph = builder.compile(checkpointer=InMemorySaver())
    user = User(identity="owner")
    config = langgraph_service.create_thread_config("thread", user)
    await graph.ainvoke({"answers": []}, config)
    snapshot = await graph.aget_state(config)
    ids = [i.id for task in snapshot.tasks for i in task.interrupts]

    @asynccontextmanager
    async def get_graph(*args, **kwargs):
        yield graph

    service = MagicMock()
    service.get_graph = get_graph
    monkeypatch.setattr(langgraph_service, "get_langgraph_service", lambda: service)
    thread = SimpleNamespace(status="interrupted", metadata_json={"graph_id": "agent"})
    session = SimpleNamespace(
        scalar=AsyncMock(side_effect=[thread, None]),
        flush=AsyncMock(),
        rollback=AsyncMock(),
    )
    return graph, config, ids, user, thread, session


def request(response):
    return SimpleNamespace(
        command={"resume": response}, checkpoint=None, checkpoint_id=None
    )


@pytest.mark.parametrize("order", [("a", "b"), ("b", "a")])
async def test_nested_parallel_approvals_do_not_repeat_completed_effects(order):
    effects = []

    def node(name):
        def approve(state):
            interrupt(f"Approve {name}")
            effects.append(name)
            return {"answers": [name]}

        return approve

    child = StateGraph(State)
    for name in ("a", "b"):
        child.add_node(name, node(name))
        child.add_edge(START, name)
        child.add_edge(name, END)
    parent = StateGraph(State)
    parent.add_node("child", child.compile())
    parent.add_edge(START, "child")
    parent.add_edge("child", END)
    graph = parent.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "nested-approvals"}}
    await graph.ainvoke({"answers": []}, config)

    for name in order:
        snapshot = await graph.aget_state(config)
        pending = next(i for i in snapshot.interrupts if i.value == f"Approve {name}")
        await graph.ainvoke(Command(resume={pending.id: "approved"}), config)

    assert effects == list(order)
    assert not (await graph.aget_state(config)).interrupts


async def test_rejects_unkeyed_multiple_without_changing_thread(checkpoint):
    graph, config, _, user, thread, session = checkpoint
    with pytest.raises(HTTPException, match="Specify interrupt IDs") as error:
        await validate_checkpoint_resume(session, "thread", request("approved"), user)
    assert error.value.status_code == 400
    assert thread.status == "interrupted"
    assert len((await graph.aget_state(config)).tasks) == 2
    session.flush.assert_not_awaited()


async def test_keyed_partial_resume_preserves_other_interrupt(checkpoint):
    graph, config, ids, user, thread, session = checkpoint
    response = {ids[0]: "approved"}
    await validate_checkpoint_resume(session, "thread", request(response), user)
    await graph.ainvoke(Command(resume=response), config)
    pending = [
        i.id for t in (await graph.aget_state(config)).tasks for i in t.interrupts
    ]
    assert pending == [ids[1]]
    session.scalar.side_effect = [thread, None]
    await validate_checkpoint_resume(
        session, "thread", request({ids[1]: "rejected"}), user
    )
    result = await graph.ainvoke(Command(resume={ids[1]: "rejected"}), config)
    assert sorted(tuple(answer) for answer in result["answers"]) == [
        ("a", "approved"),
        ("b", "rejected"),
    ]
    assert not (await graph.aget_state(config)).tasks


async def test_recovers_error_only_with_matching_checkpoint_ids(checkpoint):
    graph, config, ids, user, thread, session = checkpoint
    with pytest.raises(RuntimeError, match="multiple pending interrupts"):
        await graph.ainvoke(Command(resume="approved"), config)
    thread.status = "error"
    response = {key: "approved" for key in ids}
    await validate_checkpoint_resume(session, "thread", request(response), user)
    assert thread.status == "interrupted"
    session.flush.assert_awaited_once()
    result = await graph.ainvoke(Command(resume=response), config)
    assert len(result["answers"]) == 2
    assert not (await graph.aget_state(config)).tasks


async def test_rejects_stale_id_without_reconciling_error(checkpoint):
    _, _, _, user, thread, session = checkpoint
    thread.status = "error"
    with pytest.raises(HTTPException) as error:
        await validate_checkpoint_resume(
            session, "thread", request({"f" * 32: "approved"}), user
        )
    assert error.value.status_code == 409
    assert thread.status == "error"
    session.flush.assert_not_awaited()


async def test_rejects_active_run_and_locks_owned_thread(checkpoint, monkeypatch):
    from deep_agent.aegra import interrupt_resume

    _, _, ids, user, thread, session = checkpoint
    sleep = AsyncMock()
    monkeypatch.setattr(interrupt_resume, "asyncio", SimpleNamespace(sleep=sleep))
    session.scalar.side_effect = [thread, "running-run"] * 11
    with pytest.raises(HTTPException) as error:
        await validate_checkpoint_resume(
            session, "thread", request({ids[0]: "approved"}), user
        )
    assert error.value.status_code == 409
    sql = str(session.scalar.call_args_list[0].args[0])
    assert "FOR UPDATE" in sql
    assert "user_id" in sql
    assert sleep.await_count == 10
    assert session.rollback.await_count == 11
    session.flush.assert_not_awaited()


async def test_waits_for_active_run_to_finalize_before_accepting(
    checkpoint, monkeypatch
):
    from deep_agent.aegra import interrupt_resume

    _, _, ids, user, thread, session = checkpoint
    busy = SimpleNamespace(status="busy", metadata_json={"graph_id": "agent"})
    session.scalar.side_effect = [busy, "finishing-run", thread, None]

    async def settle(_interval):
        # The executor must be able to acquire the row lock while we wait.
        session.rollback.assert_awaited_once()

    sleep = AsyncMock(side_effect=settle)
    monkeypatch.setattr(interrupt_resume, "asyncio", SimpleNamespace(sleep=sleep))
    await validate_checkpoint_resume(
        session, "thread", request({ids[0]: "approved"}), user
    )
    sleep.assert_awaited_once_with(0.1)
    assert session.scalar.await_count == 4
    # Readiness is checked again under the scheduling transaction's row lock.
    assert "FOR UPDATE" in str(session.scalar.call_args_list[2].args[0])
    session.flush.assert_not_awaited()


async def test_waits_for_busy_thread_even_without_visible_active_run(
    checkpoint, monkeypatch
):
    from deep_agent.aegra import interrupt_resume

    _, _, ids, user, thread, session = checkpoint
    busy = SimpleNamespace(status="busy", metadata_json={"graph_id": "agent"})
    session.scalar.side_effect = [busy, None, thread, None]
    sleep = AsyncMock()
    monkeypatch.setattr(interrupt_resume, "asyncio", SimpleNamespace(sleep=sleep))
    await validate_checkpoint_resume(
        session, "thread", request({ids[0]: "approved"}), user
    )
    sleep.assert_awaited_once_with(0.1)
    session.rollback.assert_awaited_once()


async def test_busy_thread_that_never_settles_is_rejected(checkpoint, monkeypatch):
    from deep_agent.aegra import interrupt_resume

    _, _, ids, user, thread, session = checkpoint
    thread.status = "busy"
    session.scalar.side_effect = [thread, None] * 11
    sleep = AsyncMock()
    monkeypatch.setattr(interrupt_resume, "asyncio", SimpleNamespace(sleep=sleep))
    with pytest.raises(HTTPException, match="not ready") as error:
        await validate_checkpoint_resume(
            session, "thread", request({ids[0]: "approved"}), user
        )
    assert error.value.status_code == 409
    assert sleep.await_count == 10
    assert session.rollback.await_count == 11
    session.flush.assert_not_awaited()


async def test_interrupt_consumed_while_waiting_is_still_rejected(
    checkpoint, monkeypatch
):
    from deep_agent.aegra import interrupt_resume

    graph, config, ids, user, thread, session = checkpoint
    busy = SimpleNamespace(status="busy", metadata_json={"graph_id": "agent"})
    session.scalar.side_effect = [busy, "competing-run", thread, None]

    async def competing_resume(_interval):
        session.rollback.assert_awaited_once()
        await graph.ainvoke(Command(resume={ids[0]: "approved"}), config)

    sleep = AsyncMock(side_effect=competing_resume)
    monkeypatch.setattr(interrupt_resume, "asyncio", SimpleNamespace(sleep=sleep))
    with pytest.raises(HTTPException, match="no longer pending") as error:
        await validate_checkpoint_resume(
            session, "thread", request({ids[0]: "approved"}), user
        )
    assert error.value.status_code == 409
    sleep.assert_awaited_once_with(0.1)
    session.flush.assert_not_awaited()


async def test_rejects_missing_thread(checkpoint):
    _, _, ids, user, _, session = checkpoint
    session.scalar.side_effect = [None]
    with pytest.raises(HTTPException) as error:
        await validate_checkpoint_resume(
            session, "thread", request({ids[0]: "approved"}), user
        )
    assert error.value.status_code == 404


async def test_no_pending_interrupts_cannot_recover_error(checkpoint):
    graph, config, ids, user, thread, session = checkpoint
    await graph.ainvoke(Command(resume={key: "approved" for key in ids}), config)
    thread.status = "error"
    with pytest.raises(HTTPException, match="no pending checkpoint interrupts"):
        await validate_checkpoint_resume(
            session, "thread", request({ids[0]: "approved"}), user
        )
    assert thread.status == "error"


async def test_non_resume_does_not_touch_database():
    session = SimpleNamespace(scalar=AsyncMock())
    await validate_checkpoint_resume(
        session, "thread", SimpleNamespace(command=None), None
    )
    session.scalar.assert_not_awaited()


async def test_adapter_wraps_existing_call_sites_and_rolls_back_rejected_requests(
    monkeypatch,
):
    import sys

    from aegra_api.services import run_preparation

    from deep_agent.aegra import interrupt_resume

    async def original(*args, **kwargs):
        return "scheduled"

    caller = SimpleNamespace(_prepare_run=original)
    monkeypatch.setattr(run_preparation, "_prepare_run", original)
    monkeypatch.setitem(sys.modules, "aegra_api.api.runs", caller)
    monkeypatch.setattr(interrupt_resume, "version", lambda _: "0.10.8")
    validation = AsyncMock()
    monkeypatch.setattr(interrupt_resume, "validate_checkpoint_resume", validation)
    interrupt_resume.install_resume_guard()
    guarded = run_preparation._prepare_run
    assert caller._prepare_run is guarded
    interrupt_resume.install_resume_guard()
    assert run_preparation._prepare_run is guarded
    session = SimpleNamespace(rollback=AsyncMock())
    assert (
        await guarded(session, "thread", None, None, initial_status="pending")
        == "scheduled"
    )
    session.rollback.assert_not_awaited()
    validation.side_effect = HTTPException(409, "stale")
    with pytest.raises(HTTPException):
        await guarded(session, "thread", None, None, initial_status="pending")
    session.rollback.assert_awaited_once()


def test_dependency_upgrade_requires_revalidation(monkeypatch):
    from aegra_api.services import run_preparation

    from deep_agent.aegra import interrupt_resume

    async def original(*args, **kwargs):
        pass

    monkeypatch.setattr(run_preparation, "_prepare_run", original)
    monkeypatch.setattr(interrupt_resume, "version", lambda _: "different-version")
    with pytest.raises(RuntimeError, match="Revalidate"):
        interrupt_resume.install_resume_guard()


@pytest.mark.parametrize(
    "response",
    [
        "approved",
        {"decisions": [{"type": "approve"}]},
        None,
        {},
        {"not-an-interrupt-id": "approved"},
    ],
)
async def test_single_pending_interrupt_still_requires_explicit_id(
    checkpoint, response
):
    graph, config, ids, user, _, session = checkpoint
    await graph.ainvoke(Command(resume={ids[0]: "approved"}), config)
    with pytest.raises(HTTPException, match="Specify interrupt IDs") as error:
        await validate_checkpoint_resume(session, "thread", request(response), user)
    assert error.value.status_code == 400
    session.scalar.assert_not_awaited()
    session.flush.assert_not_awaited()
    pending = [
        i.id for task in (await graph.aget_state(config)).tasks for i in task.interrupts
    ]
    assert pending == [ids[1]]


async def test_unkeyed_retry_cannot_follow_active_run_into_later_interrupt(
    checkpoint, monkeypatch
):
    from deep_agent.aegra import interrupt_resume

    graph, config, ids, user, thread, session = checkpoint
    busy = SimpleNamespace(status="busy", metadata_json={"graph_id": "agent"})
    session.scalar.side_effect = [busy, "run-A", thread, None]

    async def finish_run_a(_interval):
        await graph.ainvoke(Command(resume={ids[0]: "approved"}), config)

    sleep = AsyncMock(side_effect=finish_run_a)
    monkeypatch.setattr(interrupt_resume, "asyncio", SimpleNamespace(sleep=sleep))
    with pytest.raises(HTTPException, match="Specify interrupt IDs") as error:
        await validate_checkpoint_resume(
            session, "thread", request({"decisions": [{"type": "approve"}]}), user
        )
    assert error.value.status_code == 400
    sleep.assert_not_awaited()
    session.scalar.assert_not_awaited()
