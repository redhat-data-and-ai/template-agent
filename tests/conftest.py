"""Root test configuration and shared fixtures.

Ensures the project root is on sys.path so both ``deep_agent`` and
``aegra`` packages are importable in all test modules.

Provides:
- Mock LLM fixture (MR-58)
- Mock DB / Postgres fixtures (MR-57)
- Stream context fixture
- Settings override fixture
"""

import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ── Stream context fixture ───────────────────────────────────────


@pytest.fixture()
def stream_context():
    """Provide a standard StreamContext for unit tests."""
    from deep_agent.src.streaming import StreamContext

    return StreamContext(
        run_id="test_run_1",
        trace_id="test_trace_1",
        thread_id="test_thread_1",
        session_id="test_session_1",
        user_id="test_user",
        stream_tokens=True,
    )


# ── Mock LLM fixture (MR-58) ────────────────────────────────────


@pytest.fixture()
def mock_llm():
    """Return a MagicMock that behaves like a LangChain BaseChatModel.

    Supports both sync and async invocation paths.  The default
    response is a simple AIMessage; override ``mock_llm.invoke.return_value``
    in individual tests to customise.
    """
    from langchain_core.messages import AIMessage

    llm = MagicMock()
    default_response = AIMessage(content="mock llm response", id="mock_msg_1")

    llm.invoke.return_value = default_response
    llm.ainvoke = AsyncMock(return_value=default_response)
    llm.bind_tools.return_value = llm
    llm.with_structured_output.return_value = llm
    llm.model_name = "mock-model"

    return llm


# ── Mock DB / Postgres fixtures (MR-57) ─────────────────────────


@pytest.fixture()
def mock_db_uri() -> str:
    """Return a fake Postgres URI for unit tests (no real connection)."""
    return "postgresql://test:test@localhost:5432/testdb"


@pytest.fixture()
def mock_async_connection():
    """Return a mock ``psycopg.AsyncConnection`` context manager.

    Usage in tests::

        async with mock_async_connection as conn:
            conn.execute.return_value = cursor_mock
    """
    conn = AsyncMock()
    cursor = AsyncMock()
    cursor.fetchall = AsyncMock(return_value=[])
    cursor.fetchone = AsyncMock(return_value=None)
    cursor.rowcount = 0
    conn.execute = AsyncMock(return_value=cursor)
    conn.commit = AsyncMock()

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)

    conn._cursor = cursor
    conn._ctx = ctx
    return conn


@pytest.fixture()
def mock_checkpointer():
    """Return a mock async checkpointer (PostgresSaver-like)."""
    cp = AsyncMock()
    cp.setup = AsyncMock()
    cp.__aenter__ = AsyncMock(return_value=cp)
    cp.__aexit__ = AsyncMock(return_value=False)
    return cp


# ── Settings override fixture ────────────────────────────────────


@pytest.fixture()
def test_settings():
    """Return a Settings instance with safe test defaults.

    Patches POSTGRES_HOST to localhost so no accidental remote connections.
    """
    from deep_agent.src.settings import Settings

    return Settings(
        AGENT_HOST="127.0.0.1",
        AGENT_PORT=5099,
        POSTGRES_HOST="localhost",
        POSTGRES_PORT=5432,
        POSTGRES_USER="test",
        POSTGRES_PASSWORD="test",
        POSTGRES_DB="testdb",
        PYTHON_LOG_LEVEL="WARNING",
    )


# ── Agent fixtures ───────────────────────────────────────────────


@pytest.fixture()
def mock_agent_config():
    """Return a mock agent_config with a minimal orchestrator config."""
    config = MagicMock()
    config.get_orchestrator_config.return_value = {
        "name": "test-orchestrator",
        "model": "mock-model",
        "body": "You are a test agent.",
        "skill_paths": [],
        "tools": [],
    }
    config.resolve_tools.return_value = []
    return config


@pytest.fixture()
def mock_mcp_tools() -> list[Any]:
    """Return an empty list of MCP tools for unit tests."""
    return []


# ── Deterministic LLM transport (P1) ────────────────────────────


@pytest.fixture()
def llm_cassette(request):
    """Record/replay HTTP traffic for one test.

    Yields a factory that builds an `httpx.AsyncClient` bound to this
    test's cassette. In replay mode an unrecorded request raises rather
    than reaching the network.

        async def test_x(llm_cassette):
            async with llm_cassette() as client:
                ...
    """
    from tests.support.cassette import (
        CASSETTE_DIR,
        Cassette,
        CassetteTransport,
        cassette_mode,
    )

    mode = cassette_mode()
    cassette = Cassette(CASSETTE_DIR / f"{request.node.name}.json", mode)

    def _client(**kwargs):
        return httpx.AsyncClient(transport=CassetteTransport(cassette), **kwargs)

    yield _client

    if mode == "record":
        cassette.save()


# ── Controllable time (P2) ──────────────────────────────────────


@pytest.fixture()
def fake_clock(monkeypatch):
    """A clock the test advances by hand.

    Patches `time.time`, `time.monotonic`, `time.sleep` and
    `asyncio.sleep`, so time-dependent paths run instantly and
    deterministically.
    """
    from tests.support.clock import FakeClock, install

    clock = FakeClock()
    install(monkeypatch, clock)
    return clock


# ── Shared connection pool isolation ────────────────────────────


@pytest.fixture(autouse=True)
def _reset_shared_db_pool():
    """Stop a live pool leaking from one test into the next.

    ``deep_agent.src.db.pool`` memoises one pool per process, which is
    right in the agent and wrong under pytest: a test that opens it
    against a local Postgres leaves it there, and the next test to
    touch it gets a pool whose connections belong to an event loop
    that has already closed. The symptom is a hang or a loop-affinity
    error somewhere unrelated to the test that caused it.

    The reference is dropped rather than closed, for the same reason —
    closing needs the loop that opened it.
    """
    import deep_agent.src.db.pool as shared_pool

    shared_pool._pool = None
    yield
    shared_pool._pool = None


# ── Datastore isolation ─────────────────────────────────────────


@pytest.fixture(autouse=True)
def _no_real_datastores(request, monkeypatch):
    """Point unit tests at addresses nothing is listening on.

    Unit tests were reaching whatever Postgres and Redis happened to
    be running on the developer's machine. That is not a theoretical
    complaint: a test suite that passes locally because a datastore
    is up, and hangs for 29 minutes in CI because it is not, has
    already happened here once. The tests that hit a live Redis also
    contended with each other over real lock keys.

    Anything that genuinely needs a datastore should be an
    integration test, or should ask for the `real_datastores` marker
    and bring its own.
    """
    if request.node.get_closest_marker("real_datastores"):
        yield
        return

    # Port 1 is reserved and never bound, so a connection attempt
    # fails immediately rather than hanging on a timeout.
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")
    monkeypatch.setenv("POSTGRES_HOST", "127.0.0.1")
    monkeypatch.setenv("POSTGRES_PORT", "1")

    # The settings singleton is built at import time, so setting the
    # environment above does nothing for a module that already read
    # it — the shared pool kept the real database URI and this
    # fixture only looked like it was isolating the test.
    from deep_agent.src.settings import settings as app_settings

    monkeypatch.setattr(app_settings, "POSTGRES_HOST", "127.0.0.1")
    monkeypatch.setattr(app_settings, "POSTGRES_PORT", 1)
    monkeypatch.setattr(app_settings, "REDIS_URL", "redis://127.0.0.1:1/0")

    import deep_agent.aegra.redis as redis_mod

    monkeypatch.setattr(redis_mod, "REDIS_URL", "redis://127.0.0.1:1/0")
    # The clients are module-level singletons; a live one from an
    # earlier test would survive the env change.
    monkeypatch.setattr(redis_mod, "_client", None)
    monkeypatch.setattr(redis_mod, "_async_client", None)
    monkeypatch.setattr(redis_mod, "_async_client_loop", None)

    yield

    redis_mod._client = None
    redis_mod._async_client = None
    redis_mod._async_client_loop = None
