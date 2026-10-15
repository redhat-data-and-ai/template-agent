"""Contract tests for the routes `platform/template-ui` depends on.

The two repos ship as a pair but build and deploy separately, so nothing
else fails when the agent's route surface moves. That is not theoretical:
the UI carried a `/v1/stream` caller for months against an agent that has
never served that path. It went unnoticed because no test spanned the
boundary and the live path had quietly moved to `proxy.router.ts`.

Paths here were read from `template-ui/src/server/router/proxy.router.ts`.
If one of these fails, either restore the route or update the UI proxy in
the same change.
"""

from __future__ import annotations

import pytest
from aegra_api.main import create_app
from fastapi.routing import APIRoute

# (path, method) pairs the UI's Fastify proxy calls on the agent.
REQUIRED_ROUTES: list[tuple[str, str]] = [
    # Conversation lifecycle.
    ("/threads", "POST"),
    ("/threads/{thread_id}/runs/stream", "POST"),
    ("/threads/{thread_id}/state", "GET"),
    # Liveness, used by the proxy's upstream check.
    ("/health", "GET"),
    # Custom agent routes the UI calls directly.
    ("/feedback", "POST"),
    ("/mcp/{mcp_name}/connect", "POST"),
    ("/mcp/{mcp_name}/resources/list", "POST"),
    ("/mcp/{mcp_name}/tools/list", "POST"),
    # Operator introspection of the resolved approval policy.
    ("/admin/config", "GET"),
]


@pytest.fixture(scope="module")
def route_table() -> set[tuple[str, str]]:
    """Every (path, method) pair the agent actually serves.

    Aegra registers core routers as nested router objects and recent
    FastAPI exposes the wrapped router as `original_router`, so the table
    has to be walked recursively rather than read off `app.routes`.
    """
    table: set[tuple[str, str]] = set()

    def walk(routes: list) -> None:
        for route in routes:
            if isinstance(route, APIRoute):
                for method in route.methods:
                    table.add((route.path, method))
                continue
            nested = getattr(route, "original_router", None)
            if nested is not None:
                walk(list(nested.routes))
            elif getattr(route, "routes", None):
                walk(list(route.routes))

    walk(list(create_app().routes))
    return table


@pytest.mark.parametrize(("path", "method"), REQUIRED_ROUTES)
def test_route_required_by_ui_exists(
    route_table: set[tuple[str, str]], path: str, method: str
) -> None:
    assert (path, method) in route_table, (
        f"template-ui's proxy calls {method} {path}, which this service no "
        f"longer serves. Update template-ui/src/server/router/proxy.router.ts "
        f"in the same change, or restore the route."
    )


def test_run_create_accepts_a_resume_command() -> None:
    """HITL resume depends on `command` surviving on the run payload.

    Without it an interrupted run cannot be continued and every approval
    prompt in the UI is a dead end.
    """
    from aegra_api.models.runs import RunCreate

    run = RunCreate(assistant_id="agent", command={"resume": [{"type": "accept"}]})

    assert run.command == {"resume": [{"type": "accept"}]}
    # Resuming supplies no new input; the graph picks up from its checkpoint.
    assert run.input is None


def test_no_v1_stream_route_is_reintroduced(
    route_table: set[tuple[str, str]],
) -> None:
    """Guards the specific drift that went unnoticed for months.

    A `/v1/stream` here would mean someone re-added the legacy shape
    instead of using the Agent Protocol thread/run endpoints.
    """
    assert ("/v1/stream", "POST") not in route_table
