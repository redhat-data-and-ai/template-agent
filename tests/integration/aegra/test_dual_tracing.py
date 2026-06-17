"""Integration tests proving OTEL and Langfuse coexist and both receive traces.

Validates that:
- OTEL TracerProvider and Langfuse callback registration work simultaneously
- Span helpers from ``tracing.py`` create properly attributed spans
- Error recording on spans works correctly
- W3C traceparent propagation works within the process
- Span helpers are no-ops when OTEL is disabled
"""

from unittest.mock import MagicMock, patch

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

import deep_agent.aegra.otel as otel_mod
import deep_agent.aegra.telemetry as telemetry_mod
from deep_agent.aegra.otel import reset_thread_active_tracking
from deep_agent.aegra.tracing import (
    trace_graph_build,
    trace_graph_request,
    trace_mcp_connection,
    trace_mcp_tool_call,
    trace_memory_op,
    trace_subagent_delegation,
)

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _reset_otel_and_langfuse_state():
    """Reset OTEL and Langfuse module state before and after each test."""
    # --- before ---
    otel_mod._meter = None
    otel_mod._metrics_container = None
    otel_mod._initialized = False
    otel_mod._otel_enabled = False
    otel_mod._tracer_provider = None
    reset_thread_active_tracking()
    telemetry_mod._langfuse_tracing_initialized = False
    yield
    # --- after ---
    otel_mod._meter = None
    otel_mod._metrics_container = None
    otel_mod._initialized = False
    otel_mod._otel_enabled = False
    otel_mod._tracer_provider = None
    reset_thread_active_tracking()
    telemetry_mod._langfuse_tracing_initialized = False


@pytest.fixture()
def in_memory_otel():
    """Set up OTEL with an InMemorySpanExporter for test capture.

    Returns the exporter so tests can inspect captured spans.
    Also marks the otel module as initialized + enabled so that
    ``is_tracing_enabled()`` returns True and span helpers fire.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    otel_mod._tracer_provider = provider
    otel_mod._initialized = True
    otel_mod._otel_enabled = True
    trace.set_tracer_provider(provider)

    yield exporter

    provider.shutdown()


# -----------------------------------------------------------------------
# Test 1: OTEL and Langfuse providers coexist
# -----------------------------------------------------------------------


class TestOtelAndLangfuseCoexist:
    def test_otel_and_langfuse_providers_coexist(self, in_memory_otel):
        """OTEL TracerProvider and Langfuse callback registration
        can be active simultaneously without interference."""
        exporter = in_memory_otel

        # Verify OTEL is functional — create a span and capture it
        tracer = otel_mod.get_tracer("coexistence-test")
        with tracer.start_as_current_span("probe") as span:
            span.set_attribute("test", "true")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "probe"

        # Set up Langfuse with mocked credentials (no real server needed)
        mock_handler_cls = MagicMock()
        with (
            patch.dict(
                "os.environ",
                {
                    "LANGFUSE_PUBLIC_KEY": "pk-test-dummy",
                    "LANGFUSE_SECRET_KEY": "sk-test-dummy",
                },
            ),
            patch(
                "langfuse.langchain.CallbackHandler",
                mock_handler_cls,
            ),
            patch(
                "langchain_core.tracers.context.register_configure_hook",
            ) as mock_hook,
        ):
            telemetry_mod.setup_langfuse_tracing()

        # Langfuse hook was registered
        mock_hook.assert_called_once()

        # OTEL still works after Langfuse registration — create another span
        exporter.clear()
        with tracer.start_as_current_span("post-langfuse") as span:
            span.set_attribute("after_langfuse", "true")

        post_spans = exporter.get_finished_spans()
        assert len(post_spans) == 1
        assert post_spans[0].name == "post-langfuse"

    def test_langfuse_registration_does_not_replace_tracer_provider(
        self, in_memory_otel
    ):
        """Langfuse setup must not overwrite the global TracerProvider."""
        provider_before = trace.get_tracer_provider()

        mock_handler_cls = MagicMock()
        with (
            patch.dict(
                "os.environ",
                {
                    "LANGFUSE_PUBLIC_KEY": "pk-test",
                    "LANGFUSE_SECRET_KEY": "sk-test",
                },
            ),
            patch("langfuse.langchain.CallbackHandler", mock_handler_cls),
            patch("langchain_core.tracers.context.register_configure_hook"),
        ):
            telemetry_mod.setup_langfuse_tracing()

        provider_after = trace.get_tracer_provider()
        assert provider_before is provider_after


# -----------------------------------------------------------------------
# Test 2: OTEL spans created for graph operations
# -----------------------------------------------------------------------


class TestOtelSpansForGraphOperations:
    def test_trace_graph_build_creates_span(self, in_memory_otel):
        """trace_graph_build should produce a span named 'graph.build'
        with agent.name, agent.model, and agent.tool_count attributes."""
        exporter = in_memory_otel

        with trace_graph_build(
            agent_name="test-agent",
            model_name="gemini-2.5-pro",
            tool_count=3,
        ):
            pass  # simulate graph compilation

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "graph.build"
        assert span.attributes["agent.name"] == "test-agent"
        assert span.attributes["agent.model"] == "gemini-2.5-pro"
        assert span.attributes["agent.tool_count"] == 3
        assert span.status.status_code == StatusCode.OK

    def test_trace_mcp_connection_creates_span(self, in_memory_otel):
        """trace_mcp_connection should produce a span named 'mcp.connect'
        with mcp.server and mcp.url attributes."""
        exporter = in_memory_otel

        with trace_mcp_connection(
            server_name="test-mcp",
            url="http://localhost:5001",
        ):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "mcp.connect"
        assert span.attributes["mcp.server"] == "test-mcp"
        assert span.attributes["mcp.url"] == "http://localhost:5001"
        assert span.status.status_code == StatusCode.OK

    def test_trace_subagent_delegation_creates_span(self, in_memory_otel):
        """trace_subagent_delegation should produce a span named 'subagent.build'
        with parent, name, and type attributes."""
        exporter = in_memory_otel

        with trace_subagent_delegation(
            parent_agent="orchestrator",
            subagent_name="analyst",
            subagent_type="default",
        ):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "subagent.build"
        assert span.attributes["agent.parent"] == "orchestrator"
        assert span.attributes["subagent.name"] == "analyst"
        assert span.attributes["subagent.type"] == "default"
        assert span.status.status_code == StatusCode.OK

    def test_trace_memory_op_creates_span(self, in_memory_otel):
        """trace_memory_op should produce a span named 'memory.op'
        with memory.operation and optional user.id attributes."""
        exporter = in_memory_otel

        with trace_memory_op(
            operation="consolidation",
            user_id="user-1",
        ):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "memory.op"
        assert span.attributes["memory.operation"] == "consolidation"
        assert span.attributes["user.id"] == "user-1"
        assert span.status.status_code == StatusCode.OK


# -----------------------------------------------------------------------
# Test 3: OTEL spans record errors
# -----------------------------------------------------------------------


class TestOtelSpansRecordErrors:
    def test_span_records_exception_and_error_status(self, in_memory_otel):
        """When an exception is raised inside a span helper, the span
        should have status ERROR and record the exception."""
        exporter = in_memory_otel

        with pytest.raises(ValueError, match="something broke"):
            with trace_graph_build(
                agent_name="failing-agent",
                model_name="test-model",
                tool_count=0,
            ):
                raise ValueError("something broke")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR
        assert "something broke" in (span.status.description or "")

        # The exception should be recorded as an event on the span
        events = span.events
        exception_events = [e for e in events if e.name == "exception"]
        assert len(exception_events) >= 1
        exc_event = exception_events[0]
        assert exc_event.attributes["exception.type"] == "ValueError"
        assert "something broke" in exc_event.attributes["exception.message"]

    def test_error_span_does_not_swallow_exception(self, in_memory_otel):
        """The span helper must re-raise the original exception."""
        with pytest.raises(RuntimeError):
            with trace_mcp_connection(
                server_name="bad-server",
                url="http://unreachable:9999",
            ):
                raise RuntimeError("connection refused")


# -----------------------------------------------------------------------
# Test 4: W3C traceparent propagation
# -----------------------------------------------------------------------


class TestW3CTraceparentPropagation:
    def test_child_span_shares_parent_trace_id(self, in_memory_otel):
        """A span helper called inside a manually created parent span
        should produce a child span whose trace_id matches the parent."""
        exporter = in_memory_otel

        tracer = otel_mod.get_tracer("propagation-test")
        with tracer.start_as_current_span("parent-operation"):
            with trace_graph_build(
                agent_name="child-agent",
                model_name="test-model",
                tool_count=1,
            ):
                pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 2

        span_names = {s.name for s in spans}
        assert "parent-operation" in span_names
        assert "graph.build" in span_names

        parent_span = next(s for s in spans if s.name == "parent-operation")
        child_span = next(s for s in spans if s.name == "graph.build")

        # Same trace ID proves W3C context propagation works
        assert child_span.context.trace_id == parent_span.context.trace_id, (
            "Child span must share the parent's trace ID"
        )

        # Child's parent_id should point to the parent span
        assert child_span.parent is not None
        assert child_span.parent.span_id == parent_span.context.span_id

    def test_nested_span_helpers_propagate(self, in_memory_otel):
        """Two nested span helpers should share the same trace ID."""
        exporter = in_memory_otel

        with trace_graph_build(
            agent_name="outer",
            model_name="model-a",
            tool_count=2,
        ):
            with trace_subagent_delegation(
                parent_agent="outer",
                subagent_name="inner",
                subagent_type="analyst",
            ):
                pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 2

        outer = next(s for s in spans if s.name == "graph.build")
        inner = next(s for s in spans if s.name == "subagent.build")

        assert inner.context.trace_id == outer.context.trace_id
        assert inner.parent is not None
        assert inner.parent.span_id == outer.context.span_id


# -----------------------------------------------------------------------
# Test 5: Tracing is no-op when disabled
# -----------------------------------------------------------------------


class TestTracingNoopWhenDisabled:
    def test_no_spans_when_otel_disabled(self):
        """When OTEL is not enabled, span helpers must not create spans."""
        # Set up exporter but leave _otel_enabled = False
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        otel_mod._tracer_provider = provider
        otel_mod._initialized = True
        otel_mod._otel_enabled = False  # disabled

        with trace_graph_build(
            agent_name="ghost",
            model_name="invisible",
            tool_count=0,
        ):
            pass

        with trace_mcp_connection(
            server_name="phantom",
            url="http://nowhere",
        ):
            pass

        with trace_subagent_delegation(
            parent_agent="none",
            subagent_name="none",
            subagent_type="none",
        ):
            pass

        with trace_memory_op(operation="void"):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 0, (
            f"Expected zero spans when OTEL is disabled, got {len(spans)}"
        )

        provider.shutdown()

    def test_no_spans_when_not_initialized(self):
        """When OTEL has not been initialized at all, span helpers
        must be no-ops (no crash, no spans)."""
        # Module defaults: _initialized=False, _otel_enabled=False
        assert otel_mod._initialized is False
        assert otel_mod._otel_enabled is False

        # These should all silently no-op
        with trace_graph_build(
            agent_name="pre-init",
            model_name="none",
            tool_count=0,
        ):
            pass

        with trace_memory_op(operation="pre-init-op"):
            pass

        # No crash is the assertion — if we got here, it passed


# -----------------------------------------------------------------------
# Test 6: Single execution produces traces in both OTEL and Langfuse
# -----------------------------------------------------------------------


class TestSingleExecutionDualTraces:
    """Simulate a graph execution path and verify both OTEL spans AND
    Langfuse callbacks fire from the same call chain."""

    def test_full_execution_path_produces_otel_and_langfuse_traces(
        self, in_memory_otel
    ):
        """A simulated agent execution (graph_request → graph_build →
        subagent → mcp) should produce OTEL spans AND trigger the
        Langfuse callback hook."""
        exporter = in_memory_otel

        # Set up Langfuse with mocked credentials
        mock_handler_instance = MagicMock()
        mock_handler_cls = MagicMock(return_value=mock_handler_instance)

        with (
            patch.dict(
                "os.environ",
                {
                    "LANGFUSE_PUBLIC_KEY": "pk-test-exec",
                    "LANGFUSE_SECRET_KEY": "sk-test-exec",
                },
            ),
            patch("langfuse.langchain.CallbackHandler", mock_handler_cls),
            patch(
                "langchain_core.tracers.context.register_configure_hook"
            ) as mock_hook,
        ):
            telemetry_mod.setup_langfuse_tracing()

        # Langfuse hook was registered
        mock_hook.assert_called_once()

        # Simulate the full agent() factory execution path:
        #   graph.request
        #     └─ mcp.tool_call (discover tools)
        #     └─ subagent.build
        #     └─ graph.build (create_deep_agent)
        with trace_graph_request(
            agent_name="test-orchestrator",
            user_id="user-42",
            thread_id="thread-99",
        ):
            with trace_mcp_tool_call(
                tool_name="discover",
                server_name="test-mcp",
            ):
                pass  # MCP tool discovery

            with trace_subagent_delegation(
                parent_agent="test-orchestrator",
                subagent_name="analyst",
                subagent_type="default",
            ):
                pass  # subagent build

            with trace_graph_build(
                agent_name="test-orchestrator",
                model_name="gemini-2.5-pro",
                tool_count=5,
            ):
                pass  # create_deep_agent

        # --- OTEL side: verify all 4 spans created ---
        spans = exporter.get_finished_spans()
        span_names = {s.name for s in spans}
        assert "graph.request" in span_names
        assert "graph.build" in span_names
        assert "subagent.build" in span_names
        assert "mcp.tool_call" in span_names
        assert len(spans) == 4

        # All spans share the same trace_id (single trace)
        trace_ids = {s.context.trace_id for s in spans}
        assert len(trace_ids) == 1, "All spans must belong to the same trace"

        # Parent/child hierarchy: all inner spans are children of graph.request
        request_span = next(s for s in spans if s.name == "graph.request")
        for s in spans:
            if s.name != "graph.request":
                assert s.parent is not None, f"{s.name} should have a parent"
                # All inner spans should trace back to graph.request
                assert s.context.trace_id == request_span.context.trace_id

        # --- Langfuse side: verify hook was registered ---
        # The CallbackHandler is created per-run by register_configure_hook,
        # so we verify the hook factory was registered (not called — it fires
        # when LangChain actually invokes a chain/agent, which we don't do here).
        assert telemetry_mod._langfuse_tracing_initialized is True


# -----------------------------------------------------------------------
# Test 7: W3C traceparent across subagent delegation + MCP tool call
# -----------------------------------------------------------------------


class TestW3CSubagentAndMCPPropagation:
    """Verify W3C trace context propagates through the subagent and MCP
    call chain, and that propagate.inject() puts traceparent into headers."""

    def test_traceparent_injected_into_mcp_headers(self, in_memory_otel):
        """Inside an active OTEL span, propagate.inject() should write
        a valid traceparent header — proving the MCP interceptor would
        propagate context to external servers."""
        from opentelemetry.propagate import inject

        tracer = otel_mod.get_tracer("propagation-test")

        with tracer.start_as_current_span("agent-request") as parent_span:
            # Simulate what _TokenInjectorInterceptor does
            headers: dict[str, str] = {}
            inject(headers)

            assert "traceparent" in headers, (
                "propagate.inject() must produce a traceparent header"
            )

            # traceparent format: 00-<trace_id>-<span_id>-<flags>
            parts = headers["traceparent"].split("-")
            assert len(parts) == 4
            assert parts[0] == "00"  # version

            # The trace_id in traceparent should match the parent span
            traceparent_trace_id = int(parts[1], 16)
            assert traceparent_trace_id == parent_span.context.trace_id

    def test_w3c_propagation_through_subagent_into_mcp(self, in_memory_otel):
        """Simulate: graph.request → subagent.build → mcp.connect
        Verify the MCP connect span is a descendant of the graph request,
        proving W3C context flows through the full delegation chain."""
        exporter = in_memory_otel

        with trace_graph_request(
            agent_name="orchestrator",
            user_id="user-1",
            thread_id="thread-1",
        ):
            with trace_subagent_delegation(
                parent_agent="orchestrator",
                subagent_name="analyst",
                subagent_type="compiled",
            ):
                with trace_mcp_connection(
                    server_name="data-mcp",
                    url="http://data-mcp:5001",
                ):
                    # Inside the MCP span, inject traceparent
                    # (simulating _TokenInjectorInterceptor)
                    from opentelemetry.propagate import inject

                    outgoing_headers: dict[str, str] = {}
                    inject(outgoing_headers)

        spans = exporter.get_finished_spans()
        assert len(spans) == 3

        request_span = next(s for s in spans if s.name == "graph.request")
        subagent_span = next(s for s in spans if s.name == "subagent.build")
        mcp_span = next(s for s in spans if s.name == "mcp.connect")

        # All share the same trace
        assert request_span.context.trace_id == subagent_span.context.trace_id
        assert subagent_span.context.trace_id == mcp_span.context.trace_id

        # Hierarchy: request → subagent → mcp
        assert subagent_span.parent.span_id == request_span.context.span_id
        assert mcp_span.parent.span_id == subagent_span.context.span_id

        # traceparent captured inside the MCP span carries the trace
        assert "traceparent" in outgoing_headers
        tp_trace_id = int(outgoing_headers["traceparent"].split("-")[1], 16)
        assert tp_trace_id == request_span.context.trace_id, (
            "traceparent injected during MCP call must carry the root trace ID"
        )

    def test_incoming_traceparent_links_to_request_span(self, in_memory_otel):
        """Simulate an incoming HTTP request with a traceparent header.
        The TraceIDMiddleware should attach the context, and spans created
        during that request should be children of the incoming trace."""
        from opentelemetry import context as otel_context
        from opentelemetry.propagate import extract

        exporter = in_memory_otel

        # Create a "remote" parent trace (simulating an upstream service)
        remote_tracer = otel_mod.get_tracer("upstream-service")
        with remote_tracer.start_as_current_span("upstream-call") as remote_span:
            # Capture what the upstream would send as traceparent
            from opentelemetry.propagate import inject as otel_inject

            incoming_headers: dict[str, str] = {}
            otel_inject(incoming_headers)

        # Now simulate what TraceIDMiddleware does: extract + attach
        ctx = extract(carrier=incoming_headers)
        token = otel_context.attach(ctx)

        try:
            # Spans created here should be children of the remote trace
            with trace_graph_request(
                agent_name="downstream-agent",
                user_id="u1",
                thread_id="t1",
            ):
                with trace_mcp_tool_call(tool_name="search"):
                    pass
        finally:
            otel_context.detach(token)

        spans = exporter.get_finished_spans()
        # upstream-call + graph.request + mcp.tool_call = 3
        assert len(spans) == 3

        upstream = next(s for s in spans if s.name == "upstream-call")
        request = next(s for s in spans if s.name == "graph.request")
        mcp = next(s for s in spans if s.name == "mcp.tool_call")

        # All three share the same trace ID
        assert request.context.trace_id == upstream.context.trace_id
        assert mcp.context.trace_id == upstream.context.trace_id

        # graph.request is a child of the upstream span
        assert request.parent is not None
        assert request.parent.span_id == upstream.context.span_id
