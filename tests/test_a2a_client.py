"""Tests for a2a/client.py -- A2A client, credential forwarding, interceptors."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from template_agent.src.a2a.client import (
    CallingAgentInterceptor,
    ForwardingCredentialService,
    send_to_downstream_agent,
)


# ---------------------------------------------------------------------------
# ForwardingCredentialService
# ---------------------------------------------------------------------------
class TestForwardingCredentialService:
    @pytest.mark.asyncio
    async def test_returns_stored_token(self):
        svc = ForwardingCredentialService("my-token")
        result = await svc.get_credentials("bearer", None)
        assert result == "my-token"

    @pytest.mark.asyncio
    async def test_returns_token_regardless_of_scheme(self):
        svc = ForwardingCredentialService("tok")
        assert await svc.get_credentials("api-key", MagicMock()) == "tok"

    @pytest.mark.asyncio
    async def test_returns_token_with_none_context(self):
        svc = ForwardingCredentialService("tok")
        assert await svc.get_credentials("bearer", None) == "tok"


# ---------------------------------------------------------------------------
# CallingAgentInterceptor
# ---------------------------------------------------------------------------
class TestCallingAgentInterceptor:
    @pytest.mark.asyncio
    async def test_injects_agent_id_header(self):
        interceptor = CallingAgentInterceptor("MyAgent")
        args = MagicMock()
        args.context = MagicMock()
        args.context.service_parameters = {}

        await interceptor.before(args)

        assert args.context.service_parameters["X-Calling-Agent-ID"] == "MyAgent"

    @pytest.mark.asyncio
    async def test_creates_context_when_none(self):
        interceptor = CallingAgentInterceptor("MyAgent")
        args = MagicMock()
        args.context = None

        await interceptor.before(args)

        assert args.context is not None
        assert args.context.service_parameters["X-Calling-Agent-ID"] == "MyAgent"

    @pytest.mark.asyncio
    async def test_creates_service_params_when_none(self):
        interceptor = CallingAgentInterceptor("MyAgent")
        args = MagicMock()
        args.context = MagicMock()
        args.context.service_parameters = None

        await interceptor.before(args)

        assert args.context.service_parameters["X-Calling-Agent-ID"] == "MyAgent"

    @pytest.mark.asyncio
    async def test_after_is_noop(self):
        interceptor = CallingAgentInterceptor("MyAgent")
        args = MagicMock()
        await interceptor.after(args)


# ---------------------------------------------------------------------------
# send_to_downstream_agent
# ---------------------------------------------------------------------------
class TestSendToDownstreamAgent:
    @pytest.mark.asyncio
    async def test_collects_artifact_text(self):
        mock_artifact_event = MagicMock()
        mock_artifact_event.HasField = lambda field: field == "artifact_update"
        part = MagicMock()
        part.HasField = lambda field: field == "text"
        part.text = "Hello from downstream"
        mock_artifact_event.artifact_update.artifact.parts = [part]

        mock_status_event = MagicMock()
        mock_status_event.HasField = lambda field: field == "status_update"
        from a2a.types import TaskState

        mock_status_event.status_update.status.state = TaskState.TASK_STATE_COMPLETED

        async def _mock_send_message(_req):
            for ev in [mock_artifact_event, mock_status_event]:
                yield ev

        with patch(
            "template_agent.src.a2a.client.create_a2a_client",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_client = MagicMock()
            mock_client.send_message = _mock_send_message
            mock_create.return_value = mock_client

            result = await send_to_downstream_agent(
                "http://downstream:8080",
                "Tell me a joke",
                access_token="tok",
            )

        assert "Hello from downstream" in result

    @pytest.mark.asyncio
    async def test_raises_on_failed_status(self):
        from a2a.types import TaskState

        mock_status_event = MagicMock()
        mock_status_event.HasField = lambda field: field == "status_update"
        mock_status_event.status_update.status.state = TaskState.TASK_STATE_FAILED
        mock_status_event.status_update.status.HasField = lambda f: f == "message"
        part = MagicMock()
        part.HasField = lambda field: field == "text"
        part.text = "Downstream error"
        mock_status_event.status_update.status.message.parts = [part]

        async def _mock_send_message(_req):
            yield mock_status_event

        with patch(
            "template_agent.src.a2a.client.create_a2a_client",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_client = MagicMock()
            mock_client.send_message = _mock_send_message
            mock_create.return_value = mock_client

            with pytest.raises(RuntimeError, match="Downstream agent failed"):
                await send_to_downstream_agent(
                    "http://downstream:8080", "Hi"
                )

    @pytest.mark.asyncio
    async def test_returns_fallback_when_no_artifacts(self):
        from a2a.types import TaskState

        mock_status_event = MagicMock()
        mock_status_event.HasField = lambda field: field == "status_update"
        mock_status_event.status_update.status.state = TaskState.TASK_STATE_COMPLETED

        async def _mock_send_message(_req):
            yield mock_status_event

        with patch(
            "template_agent.src.a2a.client.create_a2a_client",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_client = MagicMock()
            mock_client.send_message = _mock_send_message
            mock_create.return_value = mock_client

            result = await send_to_downstream_agent(
                "http://downstream:8080", "Hi"
            )

        assert result == "No response from downstream agent."

    @pytest.mark.asyncio
    async def test_sets_context_id_on_message(self):
        from a2a.types import TaskState

        mock_status_event = MagicMock()
        mock_status_event.HasField = lambda field: field == "status_update"
        mock_status_event.status_update.status.state = TaskState.TASK_STATE_COMPLETED

        async def _mock_send_message(_req):
            yield mock_status_event

        captured_request = None

        async def _mock_create(url, token=None):
            mock_client = MagicMock()

            async def _send(req):
                nonlocal captured_request
                captured_request = req
                async for ev in _mock_send_message(req):
                    yield ev

            mock_client.send_message = _send
            return mock_client

        with patch(
            "template_agent.src.a2a.client.create_a2a_client",
            side_effect=_mock_create,
        ):
            await send_to_downstream_agent(
                "http://downstream:8080",
                "Hi",
                context_id="ctx-123",
            )

        assert captured_request is not None
        assert captured_request.message.context_id == "ctx-123"

    @pytest.mark.asyncio
    async def test_raises_on_failed_task_event(self):
        from a2a.types import TaskState, TaskStatus

        mock_task_event = MagicMock()
        mock_task_event.HasField = lambda field: field == "task"
        mock_task_event.task.status.state = TaskState.TASK_STATE_FAILED

        async def _mock_send_message(_req):
            yield mock_task_event

        with patch(
            "template_agent.src.a2a.client.create_a2a_client",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_client = MagicMock()
            mock_client.send_message = _mock_send_message
            mock_create.return_value = mock_client

            with pytest.raises(RuntimeError, match="task failed"):
                await send_to_downstream_agent(
                    "http://downstream:8080", "Hi"
                )
