"""Integration tests for A2A upstream and downstream header propagation.

These tests exercise the real A2A Starlette app (middleware, SDK handler,
executor, delegation) end-to-end using Starlette TestClient -- no containers
or live servers required.

The AgentManager is mocked to avoid needing LLM/MCP infrastructure.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, patch

import pytest
from starlette.testclient import TestClient

from a2a.types import TaskState

from template_agent.src.a2a.app import build_a2a_starlette_app
from template_agent.src.a2a.models import A2ATargetAgent
from template_agent.src.a2a.registry import A2AAgentRegistry
from template_agent.src.settings import Settings


def _cfg(**overrides) -> Settings:
    defaults = {
        "USE_INMEMORY_SAVER": True,
        "A2A_ENABLED": True,
        "A2A_AUTH_REQUIRED": False,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _jsonrpc_request(text: str = "hello", method: str = "message/send") -> dict:
    return {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": method,
        "params": {
            "message": {
                "role": "user",
                "messageId": str(uuid.uuid4()),
                "parts": [{"kind": "text", "text": text}],
                "metadata": {
                    "user_id": "integration-tester",
                    "session_id": "session-001",
                },
            },
        },
    }


def _mock_stream_response(reply_text: str = "mocked reply"):
    """Return an async generator that yields a single AI message event."""

    async def _stream(request):
        yield {
            "type": "message",
            "content": {"type": "ai", "content": reply_text},
        }

    return _stream


def _mock_streaming_response(tokens: list[str], final_text: str):
    """Return an async generator that yields token chunks then a final AI message."""

    async def _stream(request):
        for tok in tokens:
            yield {"type": "token", "content": tok}
        yield {
            "type": "message",
            "content": {"type": "ai", "content": final_text},
        }

    return _stream


# ---------------------------------------------------------------------------
# Upstream tests: an external caller sends a request to the template-agent
# ---------------------------------------------------------------------------


class TestUpstreamInbound:
    """Verify that the A2A app correctly receives and processes headers
    from an upstream caller (simulating what dummy-upstream-agent.py does).
    """

    def _build_app_and_client(self, **cfg_overrides):
        cfg = _cfg(**cfg_overrides)
        app = build_a2a_starlette_app(cfg)
        return TestClient(app, raise_server_exceptions=False)

    def test_upstream_message_returns_200(self):
        """A valid message/send request gets a successful JSON-RPC response."""
        client = self._build_app_and_client()
        with patch(
            "template_agent.src.core.a2a_executor.AgentManager"
        ) as MockManager:
            instance = MockManager.return_value
            instance.stream_response = _mock_stream_response("test reply")
            resp = client.post(
                "/",
                json=_jsonrpc_request("What is 2+2?"),
                headers={
                    "Authorization": "Bearer upstream-token",
                    "X-Calling-Agent-ID": "upstream-agent-1",
                    "X-Correlation-ID": "corr-integration-001",
                },
            )

        assert resp.status_code == 200
        body = resp.json()
        assert "result" in body
        result = body["result"]
        artifacts = result.get("artifacts", [])
        assert any(
            p.get("text") == "test reply"
            for a in artifacts
            for p in a.get("parts", [])
            if p.get("kind") == "text"
        )

    def test_upstream_token_reaches_executor(self):
        """Bearer token from upstream is extracted by middleware and passed to AgentManager."""
        client = self._build_app_and_client()
        with patch(
            "template_agent.src.core.a2a_executor.AgentManager"
        ) as MockManager:
            instance = MockManager.return_value
            instance.stream_response = _mock_stream_response()
            client.post(
                "/",
                json=_jsonrpc_request(),
                headers={"Authorization": "Bearer my-secret-token"},
            )
            MockManager.assert_called_once()
            call_kwargs = MockManager.call_args
            assert call_kwargs.kwargs.get("redhat_sso_token") == "my-secret-token"

    def test_upstream_correlation_id_logged(self):
        """X-Correlation-ID from upstream reaches the executor via ContextVar."""
        captured_ctx = {}
        original_execute = None

        client = self._build_app_and_client()

        with patch(
            "template_agent.src.core.a2a_executor.AgentManager"
        ) as MockManager:
            instance = MockManager.return_value
            instance.stream_response = _mock_stream_response()

            from template_agent.src.a2a.context import a2a_request_ctx
            from template_agent.src.core.a2a_executor import TemplateAgentA2AExecutor

            original_execute = TemplateAgentA2AExecutor.execute

            async def _spy_execute(self_exec, context, event_queue):
                ctx = a2a_request_ctx.get()
                captured_ctx["correlation_id"] = ctx.correlation_id
                captured_ctx["calling_agent_id"] = ctx.calling_agent_id
                captured_ctx["access_token"] = ctx.access_token
                return await original_execute(self_exec, context, event_queue)

            with patch.object(TemplateAgentA2AExecutor, "execute", _spy_execute):
                client.post(
                    "/",
                    json=_jsonrpc_request(),
                    headers={
                        "Authorization": "Bearer tok-123",
                        "X-Calling-Agent-ID": "caller-agent",
                        "X-Correlation-ID": "corr-abc-xyz",
                    },
                )

        assert captured_ctx["correlation_id"] == "corr-abc-xyz"
        assert captured_ctx["calling_agent_id"] == "caller-agent"
        assert captured_ctx["access_token"] == "tok-123"

    def test_upstream_auto_generates_correlation_id(self):
        """When no X-Correlation-ID is sent, middleware generates one."""
        captured_ctx = {}
        client = self._build_app_and_client()

        with patch(
            "template_agent.src.core.a2a_executor.AgentManager"
        ) as MockManager:
            instance = MockManager.return_value
            instance.stream_response = _mock_stream_response()

            from template_agent.src.a2a.context import a2a_request_ctx
            from template_agent.src.core.a2a_executor import TemplateAgentA2AExecutor

            original_execute = TemplateAgentA2AExecutor.execute

            async def _spy(self_exec, context, event_queue):
                ctx = a2a_request_ctx.get()
                captured_ctx["correlation_id"] = ctx.correlation_id
                return await original_execute(self_exec, context, event_queue)

            with patch.object(TemplateAgentA2AExecutor, "execute", _spy):
                client.post("/", json=_jsonrpc_request())

        assert captured_ctx["correlation_id"] is not None
        uuid.UUID(captured_ctx["correlation_id"])

    def test_upstream_auth_rejected_when_required(self):
        """With A2A_AUTH_REQUIRED=True, missing token returns 401."""
        client = self._build_app_and_client(A2A_AUTH_REQUIRED=True)
        resp = client.post("/", json=_jsonrpc_request())
        assert resp.status_code == 401

    def test_upstream_agent_card_unauthenticated(self):
        """Agent card is accessible without auth even when auth is required."""
        client = self._build_app_and_client(A2A_AUTH_REQUIRED=True)
        resp = client.get("/.well-known/agent-card.json")
        assert resp.status_code == 200
        card = resp.json()
        assert card["name"] == "Template Agent"

    def test_upstream_prompt_reaches_executor(self):
        """The user's text from the A2A message reaches the AgentManager stream."""
        client = self._build_app_and_client()
        with patch(
            "template_agent.src.core.a2a_executor.AgentManager"
        ) as MockManager:
            instance = MockManager.return_value

            captured_requests = []

            async def _capture_stream(req):
                captured_requests.append(req)
                async for evt in _mock_stream_response("captured")(req):
                    yield evt

            instance.stream_response = _capture_stream
            client.post("/", json=_jsonrpc_request("hello world"))

        assert len(captured_requests) == 1
        assert captured_requests[0].message == "hello world"


# ---------------------------------------------------------------------------
# Downstream tests: template-agent delegates to a downstream echo agent
# ---------------------------------------------------------------------------


class TestDownstreamDelegation:
    """Verify that when the executor delegates to a downstream agent,
    all headers (Authorization, X-Calling-Agent-ID, X-Correlation-ID)
    are correctly propagated (simulating what echo-a2a-agent.py captures).
    """

    @pytest.fixture
    def echo_registry(self, monkeypatch):
        """Inject a registry with a fake echo downstream agent."""
        import template_agent.src.a2a.registry as reg_mod

        registry = A2AAgentRegistry.__new__(A2AAgentRegistry)
        registry._agents = {
            "echo": A2ATargetAgent(
                agent_id="echo",
                base_url="http://echo-agent:9090",
                description="Echo agent for header testing",
                skills=["echo-headers"],
            )
        }
        registry._client = AsyncMock()
        monkeypatch.setattr(reg_mod, "_registry", registry)
        return registry

    def _intercept_downstream_post(self):
        """Create a mock httpx client that captures outbound requests and
        returns a valid A2A echo response with the captured headers.
        """
        from unittest.mock import MagicMock

        captured = {}

        mock_client = AsyncMock()

        async def _fake_post(url, *, json=None, headers=None, **kwargs):
            captured["url"] = url
            captured["headers"] = dict(headers) if headers else {}
            captured["payload"] = json

            user_text = ""
            if json:
                parts = json.get("params", {}).get("message", {}).get("parts", [])
                user_text = parts[0].get("text", "") if parts else ""

            headers_echo = "\n".join(
                f"  {k}: {v}" for k, v in sorted(captured["headers"].items())
            )

            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {
                "jsonrpc": "2.0",
                "id": json.get("id", "1") if json else "1",
                "result": {
                    "artifacts": [
                        {
                            "parts": [
                                {
                                    "kind": "text",
                                    "text": (
                                        f"Echo: '{user_text}'\n"
                                        f"Headers:\n{headers_echo}"
                                    ),
                                }
                            ]
                        }
                    ],
                    "status": {"state": "completed"},
                },
            }
            return resp

        mock_client.post = _fake_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        return mock_client, captured

    async def test_delegation_forwards_bearer_token(self, echo_registry):
        """Authorization header from the upstream request is forwarded to the downstream agent."""
        from template_agent.src.a2a.context import A2ARequestContext, a2a_request_ctx
        from template_agent.src.a2a.delegation import delegate_to_a2a_agent

        mock_client, captured = self._intercept_downstream_post()

        ctx = A2ARequestContext(
            access_token="upstream-bearer-xyz",
            calling_agent_id="upstream-agent",
            correlation_id="corr-downstream-001",
        )
        token = a2a_request_ctx.set(ctx)
        try:
            with patch(
                "template_agent.src.a2a.delegation.httpx.AsyncClient",
                return_value=mock_client,
            ):
                result = await delegate_to_a2a_agent("echo", "hello downstream")
        finally:
            a2a_request_ctx.reset(token)

        assert captured["headers"]["Authorization"] == "Bearer upstream-bearer-xyz"
        assert "Echo: 'hello downstream'" in result

    async def test_delegation_forwards_correlation_id(self, echo_registry):
        """X-Correlation-ID from the upstream request is forwarded."""
        from template_agent.src.a2a.context import A2ARequestContext, a2a_request_ctx
        from template_agent.src.a2a.delegation import delegate_to_a2a_agent

        mock_client, captured = self._intercept_downstream_post()

        ctx = A2ARequestContext(correlation_id="corr-end-to-end-test")
        token = a2a_request_ctx.set(ctx)
        try:
            with patch(
                "template_agent.src.a2a.delegation.httpx.AsyncClient",
                return_value=mock_client,
            ):
                await delegate_to_a2a_agent("echo", "test")
        finally:
            a2a_request_ctx.reset(token)

        assert captured["headers"]["X-Correlation-ID"] == "corr-end-to-end-test"

    async def test_delegation_sets_own_agent_id(self, echo_registry):
        """X-Calling-Agent-ID sent downstream is the template-agent's own ID,
        not the upstream caller's ID.
        """
        from template_agent.src.a2a.context import A2ARequestContext, a2a_request_ctx
        from template_agent.src.a2a.delegation import delegate_to_a2a_agent

        mock_client, captured = self._intercept_downstream_post()

        ctx = A2ARequestContext(calling_agent_id="some-other-upstream")
        token = a2a_request_ctx.set(ctx)
        try:
            with patch(
                "template_agent.src.a2a.delegation.httpx.AsyncClient",
                return_value=mock_client,
            ):
                await delegate_to_a2a_agent("echo", "test")
        finally:
            a2a_request_ctx.reset(token)

        assert captured["headers"]["X-Calling-Agent-ID"] == "template-agent"

    async def test_delegation_omits_auth_when_no_token(self, echo_registry):
        """When no bearer token is available, no Authorization header is sent downstream."""
        from template_agent.src.a2a.context import A2ARequestContext, a2a_request_ctx
        from template_agent.src.a2a.delegation import delegate_to_a2a_agent

        mock_client, captured = self._intercept_downstream_post()

        ctx = A2ARequestContext()
        token = a2a_request_ctx.set(ctx)
        try:
            with patch(
                "template_agent.src.a2a.delegation.httpx.AsyncClient",
                return_value=mock_client,
            ):
                await delegate_to_a2a_agent("echo", "test")
        finally:
            a2a_request_ctx.reset(token)

        assert "Authorization" not in captured["headers"]

    async def test_full_chain_upstream_to_downstream(self, echo_registry):
        """End-to-end: upstream headers flow through middleware -> context ->
        delegation to the downstream agent.
        """
        from template_agent.src.a2a.context import a2a_request_ctx
        from template_agent.src.a2a.delegation import delegate_to_a2a_agent

        mock_client, captured = self._intercept_downstream_post()

        cfg = _cfg()
        app = build_a2a_starlette_app(cfg)
        client = TestClient(app, raise_server_exceptions=False)

        async def _executor_that_delegates(self_exec, context, event_queue):
            from a2a.helpers import new_text_message

            with patch(
                "template_agent.src.a2a.delegation.httpx.AsyncClient",
                return_value=mock_client,
            ):
                result = await delegate_to_a2a_agent(
                    "echo",
                    context.get_user_input(),
                    user_id="integration-tester",
                )

            await event_queue.enqueue_event(
                new_text_message(
                    result,
                    task_id=context.task_id,
                    context_id=context.context_id,
                )
            )

        from template_agent.src.core.a2a_executor import TemplateAgentA2AExecutor

        with patch.object(TemplateAgentA2AExecutor, "execute", _executor_that_delegates):
            resp = client.post(
                "/",
                json=_jsonrpc_request("hello from upstream"),
                headers={
                    "Authorization": "Bearer e2e-token-abc",
                    "X-Calling-Agent-ID": "e2e-upstream",
                    "X-Correlation-ID": "corr-e2e-full-chain",
                },
            )

        assert resp.status_code == 200

        assert captured["headers"]["Authorization"] == "Bearer e2e-token-abc"
        assert captured["headers"]["X-Calling-Agent-ID"] == "template-agent"
        assert captured["headers"]["X-Correlation-ID"] == "corr-e2e-full-chain"

        body = resp.json()
        result_text = body["result"]["parts"][0]["text"]
        assert "Echo: 'hello from upstream'" in result_text
        assert "e2e-token-abc" in result_text
        assert "corr-e2e-full-chain" in result_text


# ---------------------------------------------------------------------------
# Streaming tests: verify the executor yields incremental events
# ---------------------------------------------------------------------------


class TestA2AStreaming:
    """Verify A2A streaming support: agent card advertises streaming,
    executor pushes incremental token events, and final message is intact.
    """

    def _build_client(self, **cfg_overrides):
        cfg = _cfg(**cfg_overrides)
        app = build_a2a_starlette_app(cfg)
        return TestClient(app, raise_server_exceptions=False)

    def test_agent_card_advertises_streaming(self):
        client = self._build_client()
        resp = client.get("/.well-known/agent-card.json")
        assert resp.status_code == 200
        card = resp.json()
        assert card["capabilities"]["streaming"] is True

    def test_non_streaming_still_works_with_tokens(self):
        """SendMessage (non-streaming) returns final text even when executor
        pushes incremental events internally.
        """
        client = self._build_client()
        with patch(
            "template_agent.src.core.a2a_executor.AgentManager"
        ) as MockManager:
            instance = MockManager.return_value
            instance.stream_response = _mock_streaming_response(
                tokens=["The ", "answer ", "is ", "42"],
                final_text="The answer is 42",
            )
            resp = client.post("/", json=_jsonrpc_request("What is the answer?"))

        assert resp.status_code == 200
        body = resp.json()
        assert "result" in body
        result = body["result"]
        artifacts = result.get("artifacts", [])
        texts = [
            p["text"]
            for a in artifacts
            for p in a.get("parts", [])
            if p.get("kind") == "text"
        ]
        assert any("The answer is 42" in t for t in texts)

    async def test_executor_enqueues_token_events(self):
        """Verify the executor pushes Task + TaskArtifactUpdateEvent for each token chunk."""
        from unittest.mock import MagicMock

        from a2a.types import Task, TaskArtifactUpdateEvent, TaskStatusUpdateEvent

        from template_agent.src.core.a2a_executor import TemplateAgentA2AExecutor

        enqueued_events = []

        class SpyEventQueue:
            async def enqueue_event(self, event):
                enqueued_events.append(event)

        executor = TemplateAgentA2AExecutor()
        mock_context = MagicMock()
        mock_context.get_user_input.return_value = "hello"
        mock_context.task_id = "task-1"
        mock_context.context_id = "ctx-1"
        mock_context.metadata = {}

        with patch(
            "template_agent.src.core.a2a_executor.AgentManager"
        ) as MockManager:
            instance = MockManager.return_value
            instance.stream_response = _mock_streaming_response(
                tokens=["Hello", " world"],
                final_text="Hello world",
            )
            await executor.execute(mock_context, SpyEventQueue())

        task_events = [e for e in enqueued_events if isinstance(e, Task)]
        assert len(task_events) == 1
        assert task_events[0].status.state == TaskState.TASK_STATE_WORKING

        artifact_events = [
            e for e in enqueued_events if isinstance(e, TaskArtifactUpdateEvent)
        ]
        token_chunks = [e for e in artifact_events if e.append]
        assert len(token_chunks) == 2

        final_artifacts = [e for e in artifact_events if e.last_chunk]
        assert len(final_artifacts) == 1
        assert "Hello world" in final_artifacts[0].artifact.parts[0].text

        status_events = [
            e for e in enqueued_events if isinstance(e, TaskStatusUpdateEvent)
        ]
        completed = [
            e
            for e in status_events
            if e.status.state == TaskState.TASK_STATE_COMPLETED
        ]
        assert len(completed) == 1

    async def test_executor_handles_error_during_streaming(self):
        """Verify error events produce a FAILED status update."""
        from unittest.mock import MagicMock

        from a2a.types import TaskStatusUpdateEvent

        from template_agent.src.core.a2a_executor import TemplateAgentA2AExecutor

        enqueued_events = []

        class SpyEventQueue:
            async def enqueue_event(self, event):
                enqueued_events.append(event)

        async def _error_stream(request):
            yield {"type": "token", "content": "partial "}
            yield {
                "type": "error",
                "content": {"message": "something went wrong"},
            }

        executor = TemplateAgentA2AExecutor()
        mock_context = MagicMock()
        mock_context.get_user_input.return_value = "fail please"
        mock_context.task_id = "task-err"
        mock_context.context_id = "ctx-err"
        mock_context.metadata = {}

        with patch(
            "template_agent.src.core.a2a_executor.AgentManager"
        ) as MockManager:
            instance = MockManager.return_value
            instance.stream_response = _error_stream
            await executor.execute(mock_context, SpyEventQueue())

        status_events = [
            e for e in enqueued_events if isinstance(e, TaskStatusUpdateEvent)
        ]
        failed = [
            e
            for e in status_events
            if e.status.state == TaskState.TASK_STATE_FAILED
        ]
        assert len(failed) == 1
        assert "something went wrong" in failed[0].status.message.parts[0].text
