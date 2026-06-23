"""Unit tests for token budget service."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from deep_agent.src.token_budget.config import TokenBudgetConfig
from deep_agent.src.token_budget.service import (
    TokenUsageNotFoundError,
    TokenUsageUnavailableError,
    _mongo_repo,
    check_and_record,
    extract_tokens_from_message,
    get_thread_token_usage,
)


class _FakeMessage:
    def __init__(self, usage_metadata: dict | None = None) -> None:
        self.usage_metadata = usage_metadata
        self.response_metadata = {}


def test_extract_tokens_from_message_usage_metadata() -> None:
    msg = _FakeMessage({"input_tokens": 100, "output_tokens": 50})
    assert extract_tokens_from_message(msg) == (100, 50)


def test_extract_tokens_from_message_includes_reasoning_in_output() -> None:
    """Gemini visible output_tokens excludes reasoning; budget must include both."""
    msg = _FakeMessage(
        {
            "input_tokens": 8567,
            "output_tokens": 29,
            "total_tokens": 8737,
            "output_token_details": {"reasoning": 141},
        }
    )
    assert extract_tokens_from_message(msg) == (8567, 170)
    assert sum(extract_tokens_from_message(msg)) == 8737


def test_extract_tokens_from_message_zero_input_uses_total_minus_input() -> None:
    """Cached prompts may report input_tokens=0 while total_tokens is authoritative."""
    msg = _FakeMessage(
        {
            "input_tokens": 0,
            "output_tokens": 40,
            "total_tokens": 100,
        }
    )
    assert extract_tokens_from_message(msg) == (0, 100)


def test_extract_tokens_from_message_zero_input_without_total_uses_output() -> None:
    msg = _FakeMessage({"input_tokens": 0, "output_tokens": 50})
    assert extract_tokens_from_message(msg) == (0, 50)


@pytest.mark.asyncio
async def test_check_and_record_increments_mongo_and_daily_usage() -> None:
    config = TokenBudgetConfig(enabled=True)
    row = {
        "total_tokens": 150,
        "input_tokens": 100,
        "output_tokens": 50,
    }

    mock_repo = AsyncMock()
    mock_repo.increment_usage.return_value = row
    mock_repo.increment_daily_usage.return_value = {
        "user_id": "user-1",
        "total_tokens": 150,
        "date": "2026-06-23",
        "updated_at": datetime(2026, 6, 23, 12, 0, tzinfo=UTC),
    }

    mock_settings = MagicMock()
    mock_settings.MONGODB_URI = "mongodb://mongodb:27017"
    mock_settings.MONGODB_DB = "tokenusage"

    with (
        patch(
            "deep_agent.src.token_budget.service.agent_config.get_token_budget_config",
            return_value=config,
        ),
        patch(
            "deep_agent.src.token_budget.service.agent_config.get_name",
            return_value="health-assistant",
        ),
        patch("deep_agent.src.token_budget.service.settings", mock_settings),
        patch(
            "deep_agent.src.token_budget.service._mongo_repo",
            return_value=mock_repo,
        ),
        patch("deep_agent.src.token_budget.otel_emit.emit_token_usage") as emit_usage,
        patch("deep_agent.src.token_budget.otel_emit.emit_daily_token_usage") as emit_daily,
    ):
        await check_and_record("thread-1", 100, 50, user_id="user-1")

    mock_repo.increment_usage.assert_awaited_once_with(
        "thread-1",
        100,
        50,
        agent_name="health-assistant",
    )
    mock_repo.increment_daily_usage.assert_awaited_once_with("user-1", 150)
    emit_usage.assert_called_once_with(
        thread_id="thread-1",
        user_id="user-1",
        input_tokens=100,
        output_tokens=50,
        cumulative_total=150,
        cumulative_input=100,
        cumulative_output=50,
        timestamp=ANY,
    )
    emit_daily.assert_called_once_with(
        user_id="user-1",
        total_tokens=150,
        date="2026-06-23",
        timestamp=ANY,
    )


@pytest.mark.asyncio
async def test_check_and_record_skips_daily_without_user_id() -> None:
    config = TokenBudgetConfig(enabled=True)
    row = {
        "total_tokens": 150,
        "input_tokens": 100,
        "output_tokens": 50,
    }

    mock_repo = AsyncMock()
    mock_repo.increment_usage.return_value = row

    mock_settings = MagicMock()
    mock_settings.MONGODB_URI = "mongodb://mongodb:27017"
    mock_settings.MONGODB_DB = "tokenusage"

    with (
        patch(
            "deep_agent.src.token_budget.service.agent_config.get_token_budget_config",
            return_value=config,
        ),
        patch(
            "deep_agent.src.token_budget.service.agent_config.get_name",
            return_value="health-assistant",
        ),
        patch("deep_agent.src.token_budget.service.settings", mock_settings),
        patch(
            "deep_agent.src.token_budget.service._mongo_repo",
            return_value=mock_repo,
        ),
        patch("deep_agent.src.token_budget.otel_emit.emit_token_usage"),
        patch("deep_agent.src.token_budget.otel_emit.emit_daily_token_usage"),
    ):
        await check_and_record("thread-1", 100, 50)

    mock_repo.increment_daily_usage.assert_not_awaited()


def test_mongo_repo_returns_singleton() -> None:
    import deep_agent.src.token_budget.service as service_module

    service_module._mongo_repo_instance = None
    mock_settings = MagicMock()
    mock_settings.MONGODB_URI = "mongodb://mongodb:27017"
    mock_settings.MONGODB_DB = "tokenusage"

    with (
        patch("deep_agent.src.token_budget.service.settings", mock_settings),
        patch(
            "deep_agent.src.token_budget.mongo_repository.TokenUsageMongoRepository",
        ) as repo_cls,
    ):
        first = _mongo_repo()
        second = _mongo_repo()

    assert first is second
    repo_cls.assert_called_once_with(
        "mongodb://mongodb:27017",
        db_name="tokenusage",
    )
    service_module._mongo_repo_instance = None


@pytest.mark.asyncio
async def test_get_thread_token_usage_raises_when_not_configured() -> None:
    config = TokenBudgetConfig(enabled=False)
    mock_settings = MagicMock()
    mock_settings.MONGODB_URI = ""

    with (
        patch(
            "deep_agent.src.token_budget.service.agent_config.get_token_budget_config",
            return_value=config,
        ),
        patch("deep_agent.src.token_budget.service.settings", mock_settings),
    ):
        with pytest.raises(TokenUsageUnavailableError):
            await get_thread_token_usage("thread-1")


@pytest.mark.asyncio
async def test_get_thread_token_usage_raises_when_thread_missing() -> None:
    config = TokenBudgetConfig(enabled=True)
    mock_repo = AsyncMock()
    mock_repo.get_thread_usage.return_value = None
    mock_settings = MagicMock()
    mock_settings.MONGODB_URI = "mongodb://mongodb:27017"

    with (
        patch(
            "deep_agent.src.token_budget.service.agent_config.get_token_budget_config",
            return_value=config,
        ),
        patch("deep_agent.src.token_budget.service.settings", mock_settings),
        patch(
            "deep_agent.src.token_budget.service._mongo_repo",
            return_value=mock_repo,
        ),
    ):
        with pytest.raises(TokenUsageNotFoundError):
            await get_thread_token_usage("thread-1")
