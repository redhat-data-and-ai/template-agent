"""Shared fixtures for tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _minimal_env(monkeypatch):
    """Set minimal env vars so Settings() can be instantiated without .env."""
    monkeypatch.setenv("USE_INMEMORY_SAVER", "true")
    monkeypatch.setenv("A2A_ENABLED", "true")
