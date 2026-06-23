"""Unit tests for token budget callback."""

from __future__ import annotations

from deep_agent.src.token_budget.callback import (
    thread_id_from_metadata,
    user_id_from_metadata,
)


def test_thread_id_from_metadata_prefers_token_budget_key() -> None:
    assert thread_id_from_metadata({"token_budget_thread_id": "abc"}) == "abc"


def test_thread_id_from_metadata_falls_back_to_langfuse_session() -> None:
    assert thread_id_from_metadata({"langfuse_session_id": "xyz"}) == "xyz"


def test_thread_id_from_metadata_missing() -> None:
    assert thread_id_from_metadata({}) is None
    assert thread_id_from_metadata(None) is None


def test_user_id_from_metadata() -> None:
    assert user_id_from_metadata({"token_budget_user_id": "dev-user"}) == "dev-user"
    assert user_id_from_metadata({}) is None
    assert user_id_from_metadata(None) is None
