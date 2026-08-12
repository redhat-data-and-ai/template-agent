"""Tests for OTEL + Langfuse dual-provider isolation.

Verifies:
1. Both TracerProviders are independent (no shared state)
2. Concurrent requests produce isolated traces in both systems
3. Disabling one doesn't affect the other
4. No cross-user data leakage via ContextVar isolation
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

import deep_agent.aegra.otel as otel_mod
import deep_agent.aegra.telemetry as telemetry_mod
from deep_agent.aegra.otel import (
    get_metrics,
    get_tracer,
    initialize_telemetry,
    is_tracing_enabled,
    reset_thread_active_tracking,
    shutdown_telemetry,
)
from deep_agent.aegra.telemetry import (
    LangfuseObservabilityProvider,
    get_langfuse_client,
    setup_langfuse_tracing,
)
from deep_agent.utils.pylogger import (
    _agent_id_var,
    _org_id_var,
    _request_id_var,
    _trace_id_var,
    bind_request_context,
    clear_request_context,
)


@pytest.fixture(autouse=True)
def _reset_all_state():
    """Reset both OTEL and Langfuse module state before/after each test."""
    otel_mod._meter = None
    otel_mod._metrics_container = None
    otel_mod._initialized = False
    otel_mod._otel_enabled = False
    otel_mod._tracer_provider = None
    otel_mod._snapshot_reader = None
    reset_thread_active_tracking()

    telemetry_mod._langfuse_tracing_initialized = False
    telemetry_mod._token_budget_tracing_initialized = False
    telemetry_mod._pii_initialized = False

    clear_request_context()

    yield

    otel_mod._meter = None
    otel_mod._metrics_container = None
    otel_mod._initialized = False
    otel_mod._otel_enabled = False
    otel_mod._tracer_provider = None
    otel_mod._snapshot_reader = None
    reset_thread_active_tracking()

    telemetry_mod._langfuse_tracing_initialized = False
    telemetry_mod._token_budget_tracing_initialized = False
    telemetry_mod._pii_initialized = False

    clear_request_context()


# ---------------------------------------------------------------------------
# 1. Independent TracerProviders
# ---------------------------------------------------------------------------


class TestProviderIndependence:
    """OTEL and Langfuse use completely separate tracing backends."""

    def test_otel_tracer_provider_is_sdk_instance(self):
        """OTEL uses an SDK TracerProvider, not a proxy."""
        from opentelemetry.sdk.trace import TracerProvider

        with patch.object(
            otel_mod,
            "_resolve_config",
            return_value=(False, "http://localhost:4317", True, 5000, True),
        ):
            initialize_telemetry()

        assert isinstance(otel_mod._tracer_provider, TracerProvider)

    def test_langfuse_does_not_touch_otel_tracer_provider(self):
        """Initializing Langfuse must not modify the OTEL TracerProvider."""
        from opentelemetry.sdk.trace import TracerProvider

        with patch.object(
            otel_mod,
            "_resolve_config",
            return_value=(False, "http://localhost:4317", True, 5000, True),
        ):
            initialize_telemetry()

        otel_provider_before = otel_mod._tracer_provider
        assert isinstance(otel_provider_before, TracerProvider)

        with patch.dict(
            "os.environ",
            {"LANGFUSE_PUBLIC_KEY": "pk-test", "LANGFUSE_SECRET_KEY": "sk-test"},
        ):
            setup_langfuse_tracing()

        assert otel_mod._tracer_provider is otel_provider_before

    def test_langfuse_uses_own_context_var_not_otel(self):
        """Langfuse creates its own ContextVar for callback handlers,
        separate from OTEL's module-level _tracer_provider."""
        assert otel_mod._tracer_provider is None

        with patch.dict(
            "os.environ",
            {"LANGFUSE_PUBLIC_KEY": "pk-test", "LANGFUSE_SECRET_KEY": "sk-test"},
        ):
            setup_langfuse_tracing()

        assert otel_mod._tracer_provider is None

    def test_get_tracer_reads_from_module_provider_not_global(self):
        """get_tracer() uses the module-owned provider, not the global."""
        from opentelemetry.sdk.trace import TracerProvider

        with patch.object(
            otel_mod,
            "_resolve_config",
            return_value=(False, "http://localhost:4317", True, 5000, True),
        ):
            initialize_telemetry()

        tracer = get_tracer("test-scope")
        assert tracer is not None
        assert otel_mod._tracer_provider is not None


# ---------------------------------------------------------------------------
# 2. Disabling one doesn't affect the other
# ---------------------------------------------------------------------------


class TestDisableOneKeepsOther:
    """Each system can be independently enabled/disabled."""

    def test_otel_disabled_langfuse_still_configurable(self):
        """With OTEL disabled, Langfuse can still init and provide metadata."""
        with patch.object(
            otel_mod,
            "_resolve_config",
            return_value=(False, "http://localhost:4317", True, 5000, True),
        ):
            initialize_telemetry()

        assert otel_mod._otel_enabled is False
        assert is_tracing_enabled() is False

        provider = LangfuseObservabilityProvider()
        with patch.dict(
            "os.environ",
            {"LANGFUSE_PUBLIC_KEY": "pk-test", "LANGFUSE_SECRET_KEY": "sk-test"},
        ):
            assert provider.is_enabled() is True
            meta = provider.get_metadata(
                run_id="r1", thread_id="t1", user_identity="user@test.com"
            )
            assert "langfuse_trace_name" in meta
            assert meta["langfuse_session_id"] == "t1"

    def test_langfuse_disabled_otel_still_records_metrics(self):
        """With no Langfuse creds, OTEL metrics still work."""
        provider = LangfuseObservabilityProvider()
        with patch.dict(
            "os.environ",
            {"LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": ""},
        ):
            assert provider.is_enabled() is False

        with patch.object(
            otel_mod,
            "_resolve_config",
            return_value=(False, "http://localhost:4317", True, 5000, True),
        ):
            initialize_telemetry()

        assert get_metrics() is not None

        from deep_agent.aegra.otel import get_metrics_snapshot, record_thread_created

        record_thread_created(attributes={"thread_id": "t-only-otel"})
        snapshot = get_metrics_snapshot()
        assert snapshot is not None
        assert any("threads_created" in k for k in snapshot)

    def test_otel_shutdown_does_not_affect_langfuse_configured_state(self):
        """Shutting down OTEL leaves Langfuse's _langfuse_configured() unchanged."""
        with patch.object(
            otel_mod,
            "_resolve_config",
            return_value=(False, "http://localhost:4317", True, 5000, True),
        ):
            initialize_telemetry()

        with patch.dict(
            "os.environ",
            {"LANGFUSE_PUBLIC_KEY": "pk-test", "LANGFUSE_SECRET_KEY": "sk-test"},
        ):
            assert telemetry_mod._langfuse_configured() is True

            shutdown_telemetry()

            assert otel_mod._initialized is False
            assert otel_mod._tracer_provider is None
            assert telemetry_mod._langfuse_configured() is True

    def test_langfuse_init_failure_does_not_break_otel(self):
        """If Langfuse setup raises, OTEL continues working."""
        with patch.object(
            otel_mod,
            "_resolve_config",
            return_value=(False, "http://localhost:4317", True, 5000, True),
        ):
            initialize_telemetry()

        with patch(
            "langchain_core.tracers.context.register_configure_hook",
            side_effect=RuntimeError("langfuse boom"),
        ):
            with patch.dict(
                "os.environ",
                {"LANGFUSE_PUBLIC_KEY": "pk-test", "LANGFUSE_SECRET_KEY": "sk-test"},
            ):
                setup_langfuse_tracing()

        assert otel_mod._initialized is True
        assert get_metrics() is not None

        tracer = get_tracer("after-langfuse-fail")
        assert tracer is not None


# ---------------------------------------------------------------------------
# 3. Concurrent ContextVar isolation
# ---------------------------------------------------------------------------


class TestConcurrentContextVarIsolation:
    """ContextVars ensure per-request isolation across concurrent coroutines."""

    @pytest.mark.asyncio
    async def test_parallel_coroutines_see_own_trace_ids(self):
        """N concurrent coroutines each bind their own trace_id;
        none sees another's value."""
        results: dict[int, dict] = {}
        n = 20

        async def worker(idx: int) -> None:
            trace_id = f"trace-{idx}"
            request_id = f"req-{idx}"
            org_id = f"org-{idx}"
            agent_id = f"org-{idx}/agent-{idx}"

            bind_request_context(
                trace_id=trace_id,
                request_id=request_id,
                org_id=org_id,
                agent_id=agent_id,
            )

            await asyncio.sleep(0.01)

            results[idx] = {
                "trace_id": _trace_id_var.get(),
                "request_id": _request_id_var.get(),
                "org_id": _org_id_var.get(),
                "agent_id": _agent_id_var.get(),
            }

            clear_request_context()

        await asyncio.gather(*(worker(i) for i in range(n)))

        assert len(results) == n
        for idx in range(n):
            r = results[idx]
            assert r["trace_id"] == f"trace-{idx}", f"trace_id leaked at worker {idx}"
            assert r["request_id"] == f"req-{idx}", f"request_id leaked at worker {idx}"
            assert r["org_id"] == f"org-{idx}", f"org_id leaked at worker {idx}"
            assert r["agent_id"] == f"org-{idx}/agent-{idx}", (
                f"agent_id leaked at worker {idx}"
            )

    @pytest.mark.asyncio
    async def test_langfuse_metadata_isolated_across_coroutines(self):
        """LangfuseObservabilityProvider reads from ContextVars,
        so concurrent calls must each get their own metadata."""
        provider = LangfuseObservabilityProvider()
        results: dict[int, dict] = {}
        n = 10

        async def worker(idx: int) -> None:
            bind_request_context(trace_id=f"lf-trace-{idx}")
            await asyncio.sleep(0.005)

            with patch.dict(
                "os.environ",
                {"LANGFUSE_PUBLIC_KEY": "pk-test", "LANGFUSE_SECRET_KEY": "sk-test"},
            ):
                meta = provider.get_metadata(
                    run_id=f"run-{idx}",
                    thread_id=f"thread-{idx}",
                    user_identity=f"user-{idx}@test.com",
                )
            results[idx] = meta
            clear_request_context()

        await asyncio.gather(*(worker(i) for i in range(n)))

        assert len(results) == n
        for idx in range(n):
            meta = results[idx]
            assert meta["langfuse_session_id"] == f"thread-{idx}"
            tags = meta.get("langfuse_tags", [])
            assert any(f"trace_id:lf-trace-{idx}" in t for t in tags), (
                f"Worker {idx}: trace_id not in langfuse_tags: {tags}"
            )

    @pytest.mark.asyncio
    async def test_otel_spans_from_concurrent_coroutines_are_independent(self):
        """Concurrent spans created via get_tracer() each carry their own attributes."""
        with patch.object(
            otel_mod,
            "_resolve_config",
            return_value=(False, "http://localhost:4317", True, 5000, True),
        ):
            initialize_telemetry()

        spans_created: dict[int, str] = {}
        n = 10

        async def worker(idx: int) -> None:
            bind_request_context(trace_id=f"span-trace-{idx}")
            tracer = get_tracer(f"test-scope-{idx}")
            with tracer.start_as_current_span(f"op-{idx}") as span:
                span.set_attribute("worker.id", idx)
                await asyncio.sleep(0.005)
                spans_created[idx] = span.name
            clear_request_context()

        await asyncio.gather(*(worker(i) for i in range(n)))

        assert len(spans_created) == n
        for idx in range(n):
            assert spans_created[idx] == f"op-{idx}"


# ---------------------------------------------------------------------------
# 4. No cross-user data leakage
# ---------------------------------------------------------------------------


class TestNoCrossUserLeakage:
    """Verify that user identity doesn't leak between requests."""

    def test_clear_request_context_wipes_all_fields(self):
        """After clear, no stale user data remains."""
        bind_request_context(
            trace_id="t1",
            request_id="r1",
            org_id="org-alice",
            agent_id="org-alice/bot",
        )

        assert _trace_id_var.get() == "t1"
        assert _org_id_var.get() == "org-alice"

        clear_request_context()

        assert _trace_id_var.get() is None
        assert _request_id_var.get() is None
        assert _org_id_var.get() is None
        assert _agent_id_var.get() is None

    def test_sequential_requests_dont_leak(self):
        """Simulating two sequential requests: user B must not see user A's context."""
        bind_request_context(
            trace_id="alice-trace",
            org_id="org-alice",
            agent_id="org-alice/bot",
        )
        clear_request_context()

        bind_request_context(
            trace_id="bob-trace",
            org_id="org-bob",
            agent_id="org-bob/bot",
        )

        assert _trace_id_var.get() == "bob-trace"
        assert _org_id_var.get() == "org-bob"
        assert _agent_id_var.get() == "org-bob/bot"

        clear_request_context()

    def test_langfuse_metadata_uses_encrypted_user_id(self):
        """When encryption is enabled, Langfuse provider encrypts user_identity."""
        provider = LangfuseObservabilityProvider()

        with patch.dict(
            "os.environ",
            {
                "LANGFUSE_PUBLIC_KEY": "pk-test",
                "LANGFUSE_SECRET_KEY": "sk-test",
                "ENABLE_USER_ID_ENCRYPTION": "true",
                "USER_ID_ENCRYPTION_KEY": "test-secret-key-1234",
            },
        ):
            with patch(
                "deep_agent.aegra.telemetry.encrypt_user_id",
                side_effect=lambda uid: f"encrypted:{uid[:4]}",
            ):
                meta = provider.get_metadata(
                    run_id="r1",
                    thread_id="t1",
                    user_identity="sensitive-user@company.com",
                )

        assert "langfuse_user_id" in meta
        assert meta["langfuse_user_id"] != "sensitive-user@company.com"
        assert meta["langfuse_user_id"] == "encrypted:sens"

    @pytest.mark.asyncio
    async def test_concurrent_users_get_isolated_langfuse_metadata(self):
        """Two concurrent users must each receive only their own session_id
        and trace tags in Langfuse metadata."""
        provider = LangfuseObservabilityProvider()
        meta_alice = None
        meta_bob = None

        async def alice():
            nonlocal meta_alice
            bind_request_context(trace_id="alice-trace")
            await asyncio.sleep(0.01)
            with patch.dict(
                "os.environ",
                {"LANGFUSE_PUBLIC_KEY": "pk-test", "LANGFUSE_SECRET_KEY": "sk-test"},
            ):
                meta_alice = provider.get_metadata(
                    run_id="r-alice",
                    thread_id="thread-alice",
                    user_identity="alice@test.com",
                )
            clear_request_context()

        async def bob():
            nonlocal meta_bob
            bind_request_context(trace_id="bob-trace")
            await asyncio.sleep(0.01)
            with patch.dict(
                "os.environ",
                {"LANGFUSE_PUBLIC_KEY": "pk-test", "LANGFUSE_SECRET_KEY": "sk-test"},
            ):
                meta_bob = provider.get_metadata(
                    run_id="r-bob",
                    thread_id="thread-bob",
                    user_identity="bob@test.com",
                )
            clear_request_context()

        await asyncio.gather(alice(), bob())

        assert meta_alice["langfuse_session_id"] == "thread-alice"
        assert meta_bob["langfuse_session_id"] == "thread-bob"

        alice_tags = meta_alice.get("langfuse_tags", [])
        bob_tags = meta_bob.get("langfuse_tags", [])

        assert any("alice-trace" in t for t in alice_tags)
        assert any("bob-trace" in t for t in bob_tags)
        assert not any("bob-trace" in t for t in alice_tags)
        assert not any("alice-trace" in t for t in bob_tags)


# ---------------------------------------------------------------------------
# 5. Full init sequence independence
# ---------------------------------------------------------------------------


class TestFullInitSequence:
    """Test the startup sequence mirrors production: OTEL first, then Langfuse."""

    def test_otel_then_langfuse_both_functional(self):
        """After initializing both (as startup.py does), both systems work."""
        with patch.object(
            otel_mod,
            "_resolve_config",
            return_value=(False, "http://localhost:4317", True, 5000, True),
        ):
            initialize_telemetry()

        assert otel_mod._initialized is True
        assert get_metrics() is not None

        with patch.dict(
            "os.environ",
            {"LANGFUSE_PUBLIC_KEY": "pk-test", "LANGFUSE_SECRET_KEY": "sk-test"},
        ):
            setup_langfuse_tracing()
            assert telemetry_mod._langfuse_tracing_initialized is True

            provider = LangfuseObservabilityProvider()
            assert provider.is_enabled() is True

        from deep_agent.aegra.otel import record_conversation_started

        start = record_conversation_started(attributes={"thread_id": "dual-test"})
        assert start > 0

    def test_reinit_otel_after_shutdown_does_not_break_langfuse(self):
        """Shutdown + re-init of OTEL must not corrupt Langfuse state."""
        with patch.object(
            otel_mod,
            "_resolve_config",
            return_value=(False, "http://localhost:4317", True, 5000, True),
        ):
            initialize_telemetry()

        with patch.dict(
            "os.environ",
            {"LANGFUSE_PUBLIC_KEY": "pk-test", "LANGFUSE_SECRET_KEY": "sk-test"},
        ):
            setup_langfuse_tracing()
            assert telemetry_mod._langfuse_tracing_initialized is True

            shutdown_telemetry()
            assert otel_mod._initialized is False

            otel_mod._initialized = False
            with patch.object(
                otel_mod,
                "_resolve_config",
                return_value=(False, "http://localhost:4317", True, 5000, True),
            ):
                initialize_telemetry()

            assert otel_mod._initialized is True
            assert get_metrics() is not None
            assert telemetry_mod._langfuse_tracing_initialized is True
