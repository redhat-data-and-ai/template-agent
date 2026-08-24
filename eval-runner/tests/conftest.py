"""Shared pytest fixtures for eval-runner tests."""

from __future__ import annotations

from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def eval_dir(tmp_path: Path) -> str:
    """Return a temporary directory path for eval CRUD operations."""
    return str(tmp_path)


@pytest.fixture
def mock_psycopg2_conn() -> tuple[MagicMock, MagicMock]:
    """Return a (conn, cursor) pair with all context-manager dunder methods wired up."""
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)

    conn = MagicMock()
    conn.cursor.return_value = cursor
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)

    return conn, cursor


@pytest.fixture(autouse=True)
def reset_eval_postgres_state() -> Generator[None, None, None]:
    """Reset module-level state in eval_postgres between tests."""
    import eval_postgres

    eval_postgres._table_ensured = False
    yield
    eval_postgres._table_ensured = False
