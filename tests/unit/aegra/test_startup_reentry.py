"""Startup must not call itself.

`_warm_graph` builds the graph, `agent()` ensures startup has run, and
the "has it run" flag used to be set only after the warm-up returned.
So warming re-entered startup, which warmed again. A live pod logged
224 startups per boot, each repeating the database init, the
interrupted-run resume, the cache warm and another graph build.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from deep_agent.aegra import startup as st


@pytest.fixture(autouse=True)
def _reset_flag():
    original = st._startup_complete
    st._startup_complete = False
    yield
    st._startup_complete = original


def _stub_everything(warm_graph):
    """Every step a no-op except the graph warm-up under test."""
    return [
        patch.object(st, "_validate_config", AsyncMock(return_value="ok")),
        patch.object(st, "_init_ldap", lambda: "ok"),
        patch.object(st, "_init_aegra_db", AsyncMock(return_value="ok")),
        patch.object(st, "_ensure_database", AsyncMock(return_value="ok")),
        patch.object(st, "_check_mcp_encryption_key", lambda: None),
        patch.object(st, "_resume_interrupted_runs", AsyncMock(return_value="ok")),
        patch.object(st, "_setup_mcp_apps_capability", lambda: "ok"),
        patch.object(st, "_warm_caches", AsyncMock(return_value="ok")),
        patch.object(st, "_setup_otel", lambda: "ok"),
        patch.object(st, "_setup_telemetry", lambda: "ok"),
        patch.object(st, "_warmup_graph_imports", lambda: "ok"),
        patch.object(st, "_start_inflight_metrics", lambda: "ok"),
        patch.object(st, "_upgrade_signal_handlers", lambda: None),
        patch.object(st, "_warm_graph", warm_graph),
    ]


class TestReentry:
    async def test_warming_the_graph_does_not_restart_startup(self):
        """The flag is set before the warm-up, so re-entry is a no-op."""
        calls: list[str] = []

        async def warm_graph():
            calls.append("warm")
            # Exactly what `agent()` does.
            result = await st.run_startup()
            calls.append(result.get("status", "ran-again"))
            return "ok"

        patches = _stub_everything(warm_graph)
        for p in patches:
            p.start()
        try:
            await st.run_startup()
        finally:
            for p in patches:
                p.stop()

        assert calls == ["warm", "already_complete"]

    async def test_the_expensive_steps_run_once(self):
        """Resume and database init must not repeat per nesting level."""
        resume = AsyncMock(return_value="ok")
        db = AsyncMock(return_value="ok")

        async def warm_graph():
            await st.run_startup()
            return "ok"

        patches = _stub_everything(warm_graph)
        patches = [p for p in patches if True]
        for p in patches:
            p.start()
        try:
            with (
                patch.object(st, "_resume_interrupted_runs", resume),
                patch.object(st, "_ensure_database", db),
            ):
                await st.run_startup()
        finally:
            for p in patches:
                p.stop()

        assert resume.await_count == 1
        assert db.await_count == 1
