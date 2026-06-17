"""Span helpers for OTEL distributed tracing.

Provides context-manager wrappers around ``get_tracer()`` for the main
execution paths: graph builds, per-request graph invocations, subagent
delegation, MCP connections/tool calls, and background memory operations.

All helpers are **zero-cost** when OTEL is disabled — they check
``is_tracing_enabled()`` and become a plain ``yield`` (no span created,
no OTEL imports beyond the guard).

Usage::

    from deep_agent.aegra.tracing import trace_graph_request

    with trace_graph_request(agent_name="orchestrator", user_id="u1", thread_id="t1"):
        compiled = create_deep_agent(...)

Exceptions are recorded on the span and re-raised; they are never swallowed.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    pass

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


def _tracing_enabled() -> bool:
    """Check whether OTEL tracing is initialised and enabled.

    Lazily imports from ``otel`` to avoid import-time side effects.
    """
    try:
        from deep_agent.aegra.otel import is_tracing_enabled

        return is_tracing_enabled()
    except ImportError:
        return False


def _get_tracer() -> Any:
    """Return an OTEL Tracer, or ``None`` if unavailable."""
    try:
        from deep_agent.aegra.otel import get_tracer

        return get_tracer("template-agent")
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Span helpers
# ---------------------------------------------------------------------------


@contextmanager
def trace_graph_build(
    agent_name: str,
    model_name: str,
    tool_count: int,
) -> Iterator[None]:
    """Span around ``create_deep_agent()`` — the graph compilation step.

    Attributes:
        agent.name, agent.model, agent.tool_count
    """
    if not _tracing_enabled():
        yield
        return

    tracer = _get_tracer()
    if tracer is None:
        yield
        return

    try:
        from opentelemetry.trace import StatusCode
    except ImportError:
        yield
        return

    with tracer.start_as_current_span(
        "graph.build",
        attributes={
            "agent.name": agent_name,
            "agent.model": model_name,
            "agent.tool_count": tool_count,
        },
    ) as span:
        try:
            yield
            span.set_status(StatusCode.OK)
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise


@contextmanager
def trace_graph_request(
    agent_name: str,
    user_id: str | None,
    thread_id: str | None,
) -> Iterator[None]:
    """Span around the per-request ``agent()`` factory.

    Attributes:
        agent.name, user.id, thread.id
    """
    if not _tracing_enabled():
        yield
        return

    tracer = _get_tracer()
    if tracer is None:
        yield
        return

    try:
        from opentelemetry.trace import StatusCode
    except ImportError:
        yield
        return

    attrs: dict[str, str | int] = {"agent.name": agent_name}
    if user_id:
        attrs["user.id"] = user_id
    if thread_id:
        attrs["thread.id"] = thread_id

    with tracer.start_as_current_span("graph.request", attributes=attrs) as span:
        try:
            yield
            span.set_status(StatusCode.OK)
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise


@contextmanager
def trace_subagent_delegation(
    parent_agent: str,
    subagent_name: str,
    subagent_type: str,
) -> Iterator[None]:
    """Span around building a single subagent.

    Attributes:
        agent.parent, subagent.name, subagent.type
    """
    if not _tracing_enabled():
        yield
        return

    tracer = _get_tracer()
    if tracer is None:
        yield
        return

    try:
        from opentelemetry.trace import StatusCode
    except ImportError:
        yield
        return

    with tracer.start_as_current_span(
        "subagent.build",
        attributes={
            "agent.parent": parent_agent,
            "subagent.name": subagent_name,
            "subagent.type": subagent_type,
        },
    ) as span:
        try:
            yield
            span.set_status(StatusCode.OK)
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise


@contextmanager
def trace_mcp_connection(
    server_name: str,
    url: str,
) -> Iterator[None]:
    """Span around connecting to a single MCP server.

    Attributes:
        mcp.server, mcp.url
    """
    if not _tracing_enabled():
        yield
        return

    tracer = _get_tracer()
    if tracer is None:
        yield
        return

    try:
        from opentelemetry.trace import StatusCode
    except ImportError:
        yield
        return

    with tracer.start_as_current_span(
        "mcp.connect",
        attributes={
            "mcp.server": server_name,
            "mcp.url": url,
        },
    ) as span:
        try:
            yield
            span.set_status(StatusCode.OK)
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise


@contextmanager
def trace_mcp_tool_call(
    tool_name: str,
    server_name: str | None = None,
) -> Iterator[None]:
    """Span around an individual MCP tool invocation.

    Attributes:
        mcp.tool, mcp.server (optional)
    """
    if not _tracing_enabled():
        yield
        return

    tracer = _get_tracer()
    if tracer is None:
        yield
        return

    try:
        from opentelemetry.trace import StatusCode
    except ImportError:
        yield
        return

    attrs: dict[str, str] = {"mcp.tool": tool_name}
    if server_name:
        attrs["mcp.server"] = server_name

    with tracer.start_as_current_span("mcp.tool_call", attributes=attrs) as span:
        try:
            yield
            span.set_status(StatusCode.OK)
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise


@contextmanager
def trace_memory_op(
    operation: str,
    user_id: str | None = None,
) -> Iterator[None]:
    """Span around a background memory operation.

    Attributes:
        memory.operation, user.id (optional)
    """
    if not _tracing_enabled():
        yield
        return

    tracer = _get_tracer()
    if tracer is None:
        yield
        return

    try:
        from opentelemetry.trace import StatusCode
    except ImportError:
        yield
        return

    attrs: dict[str, str] = {"memory.operation": operation}
    if user_id:
        attrs["user.id"] = user_id

    with tracer.start_as_current_span("memory.op", attributes=attrs) as span:
        try:
            yield
            span.set_status(StatusCode.OK)
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise
