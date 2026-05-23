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
from unittest.mock import AsyncMock, MagicMock, patch

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
