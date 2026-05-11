"""Tests for a2a/executor.py -- TemplateAgentExecutor."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from template_agent.src.a2a.executor import ACCESS_TOKEN_STATE_KEY, TemplateAgentExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_context(
    *,
    task_id: str = "task-1",
    context_id: str = "ctx-1",
    user_input: str | None = "Hello",
    accepted_output_modes: list[str] | None = None,
    current_task=None,
    access_token: str | None = "tok",
) -> MagicMock:
    """Build a minimal RequestContext mock."""
    from a2a.types import Message, Part, Role

    ctx = MagicMock()
    ctx.task_id = task_id
    ctx.context_id = context_id
    ctx.current_task = current_task
    ctx.get_user_input.return_value = user_input

    if accepted_output_modes is not None:
        ctx.configuration = MagicMock()
        ctx.configuration.accepted_output_modes = accepted_output_modes
    else:
        ctx.configuration = None

    ctx.call_context = MagicMock()
    ctx.call_context.state = {ACCESS_TOKEN_STATE_KEY: access_token}

    ctx.message = Message(
        message_id="msg-1",
        role=Role.ROLE_USER,
        parts=[Part(text=user_input or "")],
    )
    return ctx


def _make_event_queue() -> MagicMock:
    """Build an async EventQueue mock."""
    eq = MagicMock()
    eq.enqueue_event = AsyncMock()
    return eq


# ---------------------------------------------------------------------------
# Output mode compatibility
# ---------------------------------------------------------------------------
class TestOutputModeCheck:
    def test_compatible_when_no_preference(self):
        executor = TemplateAgentExecutor(supported_output_modes=["text/plain"])
        ctx = _make_context(accepted_output_modes=None)
        assert executor._is_output_mode_incompatible(ctx) is False

    def test_compatible_when_empty_list(self):
        executor = TemplateAgentExecutor(supported_output_modes=["text/plain"])
        ctx = _make_context(accepted_output_modes=[])
        assert executor._is_output_mode_incompatible(ctx) is False

    def test_compatible_when_overlap(self):
        executor = TemplateAgentExecutor(
            supported_output_modes=["text/plain", "application/json"]
        )
        ctx = _make_context(accepted_output_modes=["text/plain", "text/html"])
        assert executor._is_output_mode_incompatible(ctx) is False

    def test_incompatible_when_no_overlap(self):
        executor = TemplateAgentExecutor(supported_output_modes=["text/plain"])
        ctx = _make_context(accepted_output_modes=["image/png"])
        assert executor._is_output_mode_incompatible(ctx) is True


# ---------------------------------------------------------------------------
# execute() - entry point
# ---------------------------------------------------------------------------
class TestExecute:
    @pytest.mark.asyncio
    async def test_rejects_unsupported_output_modes(self):
        from a2a.types import ContentTypeNotSupportedError

        executor = TemplateAgentExecutor(supported_output_modes=["text/plain"])
        ctx = _make_context(accepted_output_modes=["image/png"])
        eq = _make_event_queue()

        with pytest.raises(ContentTypeNotSupportedError):
            await executor.execute(ctx, eq)

    @pytest.mark.asyncio
    async def test_tracks_and_cleans_running_task(self):
        executor = TemplateAgentExecutor(supported_output_modes=["text/plain"])

        with patch.object(
            executor, "_execute_inner", new_callable=AsyncMock
        ) as mock_inner:
            ctx = _make_context()
            eq = _make_event_queue()
            await executor.execute(ctx, eq)

            mock_inner.assert_awaited_once()
            assert "task-1" not in executor._running_tasks

    @pytest.mark.asyncio
    async def test_cleans_task_on_cancel(self):
        executor = TemplateAgentExecutor(supported_output_modes=["text/plain"])

        async def _inner_cancel(*_args, **_kwargs):
            raise asyncio.CancelledError()

        with patch.object(executor, "_execute_inner", side_effect=_inner_cancel):
            ctx = _make_context()
            eq = _make_event_queue()
            with pytest.raises(asyncio.CancelledError):
                await executor.execute(ctx, eq)

            assert "task-1" not in executor._running_tasks


# ---------------------------------------------------------------------------
# _execute_inner() - core logic
# ---------------------------------------------------------------------------
class TestExecuteInner:
    @pytest.mark.asyncio
    async def test_fails_when_no_user_input(self):
        executor = TemplateAgentExecutor(supported_output_modes=["text/plain"])
        ctx = _make_context(user_input=None)
        eq = _make_event_queue()

        with patch(
            "template_agent.src.a2a.executor.TaskUpdater"
        ) as MockUpdater:
            mock_updater = MagicMock()
            mock_updater.new_agent_message = MagicMock(return_value="msg")
            mock_updater.failed = AsyncMock()
            MockUpdater.return_value = mock_updater

            await executor._execute_inner(ctx, eq)

            mock_updater.failed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_creates_new_task_when_no_current(self):
        executor = TemplateAgentExecutor(supported_output_modes=["text/plain"])
        ctx = _make_context(current_task=None)
        eq = _make_event_queue()

        with patch(
            "template_agent.src.a2a.executor.TaskUpdater"
        ) as MockUpdater, patch.object(
            executor, "_run_agent_streaming", new_callable=AsyncMock
        ):
            mock_updater = MagicMock()
            mock_updater.new_agent_message = MagicMock(return_value="msg")
            mock_updater.start_work = AsyncMock()
            mock_updater.complete = AsyncMock()
            MockUpdater.return_value = mock_updater

            await executor._execute_inner(ctx, eq)

            eq.enqueue_event.assert_awaited_once()
            event_arg = eq.enqueue_event.call_args[0][0]
            assert event_arg.id == "task-1"

    @pytest.mark.asyncio
    async def test_uses_current_task_when_provided(self):
        executor = TemplateAgentExecutor(supported_output_modes=["text/plain"])
        existing_task = MagicMock()
        existing_task.id = "task-existing"
        ctx = _make_context(current_task=existing_task)
        eq = _make_event_queue()

        with patch(
            "template_agent.src.a2a.executor.TaskUpdater"
        ) as MockUpdater, patch.object(
            executor, "_run_agent_streaming", new_callable=AsyncMock
        ):
            mock_updater = MagicMock()
            mock_updater.new_agent_message = MagicMock(return_value="msg")
            mock_updater.start_work = AsyncMock()
            mock_updater.complete = AsyncMock()
            MockUpdater.return_value = mock_updater

            await executor._execute_inner(ctx, eq)

            event_arg = eq.enqueue_event.call_args[0][0]
            assert event_arg is existing_task

    @pytest.mark.asyncio
    async def test_agent_error_calls_failed(self):
        executor = TemplateAgentExecutor(supported_output_modes=["text/plain"])
        ctx = _make_context()
        eq = _make_event_queue()

        with patch(
            "template_agent.src.a2a.executor.TaskUpdater"
        ) as MockUpdater, patch.object(
            executor,
            "_run_agent_streaming",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ):
            mock_updater = MagicMock()
            mock_updater.new_agent_message = MagicMock(return_value="msg")
            mock_updater.start_work = AsyncMock()
            mock_updater.failed = AsyncMock()
            mock_updater.complete = AsyncMock()
            MockUpdater.return_value = mock_updater

            await executor._execute_inner(ctx, eq)

            mock_updater.failed.assert_awaited_once()
            mock_updater.complete.assert_not_awaited()


# ---------------------------------------------------------------------------
# cancel()
# ---------------------------------------------------------------------------
class TestCancel:
    @pytest.mark.asyncio
    async def test_cancel_running_task(self):
        executor = TemplateAgentExecutor(supported_output_modes=[])

        mock_asyncio_task = MagicMock()
        mock_asyncio_task.done.return_value = False
        executor._running_tasks["task-1"] = mock_asyncio_task

        ctx = _make_context(task_id="task-1")
        eq = _make_event_queue()

        with patch("template_agent.src.a2a.executor.TaskUpdater") as MockUpdater:
            mock_updater = MagicMock()
            mock_updater.cancel = AsyncMock()
            MockUpdater.return_value = mock_updater

            await executor.cancel(ctx, eq)

        mock_asyncio_task.cancel.assert_called_once()
        assert "task-1" not in executor._running_tasks

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_task_is_safe(self):
        executor = TemplateAgentExecutor(supported_output_modes=[])
        ctx = _make_context(task_id="no-such-task")
        eq = _make_event_queue()

        with patch("template_agent.src.a2a.executor.TaskUpdater") as MockUpdater:
            mock_updater = MagicMock()
            mock_updater.cancel = AsyncMock()
            MockUpdater.return_value = mock_updater

            await executor.cancel(ctx, eq)

        mock_updater.cancel.assert_awaited_once()
