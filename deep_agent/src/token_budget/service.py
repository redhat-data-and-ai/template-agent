"""Token usage tracking and extraction."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from deep_agent.src.agent.config import agent_config
from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


@dataclass(frozen=True)
class ThreadTokenUsage:
    """Thread token usage summary."""

    thread_id: str
    used: int
    input_tokens: int
    output_tokens: int


class TokenUsageUnavailableError(Exception):
    """Token usage storage is not configured or temporarily unreachable."""


class TokenUsageNotFoundError(Exception):
    """No token usage record exists for the requested thread."""

    def __init__(self, thread_id: str) -> None:
        self.thread_id = thread_id
        super().__init__(thread_id)

def _reasoning_tokens(usage: dict[str, Any]) -> int:
    """Return reasoning tokens from provider output_token_details (Gemini)."""
    details = usage.get("output_token_details")
    if not isinstance(details, dict):
        return 0
    value = details.get("reasoning")
    return int(value) if isinstance(value, int) else 0


def _usage_dict_to_counts(usage: dict[str, Any] | None) -> tuple[int, int]:
    """Map provider usage to billable input/output counts.

    Matches Langfuse **Total usage** (input + visible output + reasoning).
    Gemini often reports ``output_tokens`` as visible output only (e.g. 29) while
    ``output_token_details.reasoning`` holds the rest (e.g. 141). Prefer
    ``total_tokens - input_tokens`` when both are present.
    """
    if not usage:
        return 0, 0
    input_tokens = int(
        usage.get("input_tokens")
        or usage.get("input")
        or usage.get("prompt_tokens")
        or usage.get("prompt_token_count")
        or 0
    )
    total_tokens = int(usage.get("total_tokens") or usage.get("total") or 0)
    if total_tokens > 0 and total_tokens >= input_tokens:
        return input_tokens, total_tokens - input_tokens

    output_tokens = int(
        usage.get("output_tokens")
        or usage.get("output")
        or usage.get("completion_tokens")
        or usage.get("candidates_token_count")
        or 0
    )
    reasoning = _reasoning_tokens(usage)
    if reasoning:
        output_tokens += reasoning

    if input_tokens or output_tokens:
        return input_tokens, output_tokens
    if total_tokens:
        return 0, total_tokens
    return 0, 0


def _usage_from_generation(generation: Any) -> tuple[int, int]:
    """Extract billable tokens from a single LangChain generation."""
    message = getattr(generation, "message", None)
    if message is not None:
        in_t, out_t = extract_tokens_from_message(message)
        if in_t or out_t:
            return in_t, out_t

    gen_info = getattr(generation, "generation_info", None) or {}
    if isinstance(gen_info, dict):
        usage = gen_info.get("usage_metadata") or gen_info.get("token_usage") or {}
        if isinstance(usage, dict):
            return _usage_dict_to_counts(usage)

    return 0, 0


def extract_tokens_from_llm_result(response: Any) -> tuple[int, int]:
    """Extract billable token counts from a LangChain LLMResult."""
    input_tokens = 0
    output_tokens = 0
    generations = getattr(response, "generations", None) or []
    for generation_list in generations:
        for generation in generation_list:
            in_t, out_t = _usage_from_generation(generation)
            input_tokens += in_t
            output_tokens += out_t

    if input_tokens or output_tokens:
        return input_tokens, output_tokens

    llm_output = getattr(response, "llm_output", None) or {}
    if isinstance(llm_output, dict):
        token_usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        if isinstance(token_usage, dict):
            return _usage_dict_to_counts(token_usage)

    return 0, 0


def extract_tokens_from_chat_result(response: Any) -> tuple[int, int]:
    """Alias for chat model results — Langfuse routes these through on_llm_end."""
    return extract_tokens_from_llm_result(response)


def extract_tokens_from_message(message: Any) -> tuple[int, int]:
    """Extract billable token counts from a LangChain message object."""
    usage = getattr(message, "usage_metadata", None) or {}
    if isinstance(usage, dict) and usage:
        return _usage_dict_to_counts(usage)

    response_metadata = getattr(message, "response_metadata", None) or {}
    if isinstance(response_metadata, dict):
        nested_usage = (
            response_metadata.get("usage_metadata")
            or response_metadata.get("token_usage")
            or {}
        )
        if isinstance(nested_usage, dict):
            in_t, out_t = _usage_dict_to_counts(nested_usage)
            if in_t or out_t:
                return in_t, out_t

    return 0, 0


_mongo_repo_instance = None
_mongo_repo_lock = threading.Lock()


def _mongo_repo():
    """Return a process-wide Mongo repository (reuses the Motor client pool)."""
    global _mongo_repo_instance  # noqa: PLW0603

    if _mongo_repo_instance is None:
        with _mongo_repo_lock:
            if _mongo_repo_instance is None:
                from deep_agent.src.token_budget.mongo_repository import (
                    TokenUsageMongoRepository,
                )

                _mongo_repo_instance = TokenUsageMongoRepository(
                    settings.MONGODB_URI,
                    db_name=settings.MONGODB_DB,
                )
    return _mongo_repo_instance


_MAX_REASONABLE_TOKENS = 1_000_000


async def check_and_record(
    thread_id: str,
    input_tokens: int,
    output_tokens: int,
    *,
    user_id: str | None = None,
) -> None:
    """Increment thread usage, roll up daily user totals, and emit OTEL when enabled."""
    config = agent_config.get_token_budget_config()
    if not config.is_active:
        return
    if not thread_id or thread_id == "unknown":
        return
    if not settings.MONGODB_URI:
        logger.debug("token_budget_skipped_no_mongodb_uri")
        return
    if input_tokens <= 0 and output_tokens <= 0:
        return
    if input_tokens > _MAX_REASONABLE_TOKENS or output_tokens > _MAX_REASONABLE_TOKENS:
        logger.warning(
            "token_budget_suspicious_count",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return

    try:
        repo = _mongo_repo()
        agent_name = agent_config.get_name()
        row = await repo.increment_usage(
            thread_id,
            input_tokens,
            output_tokens,
            agent_name=agent_name,
        )
        total_delta = input_tokens + output_tokens
        daily_row = None
        if user_id and user_id != "unknown" and total_delta > 0:
            daily_row = await repo.increment_daily_usage(user_id, total_delta)
    except Exception:
        logger.warning(
            "token_budget_mongo_write_failed",
            exc_info=True,
        )
        return

    from deep_agent.src.token_budget.otel_emit import emit_daily_token_usage, emit_token_usage

    emit_token_usage(
        thread_id=thread_id,
        user_id=user_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cumulative_total=int(row["total_tokens"]),
        cumulative_input=int(row["input_tokens"]),
        cumulative_output=int(row["output_tokens"]),
        timestamp=row.get("updated_at"),
    )

    if daily_row is not None:
        emit_daily_token_usage(
            user_id=str(daily_row["user_id"]),
            total_tokens=int(daily_row["total_tokens"]),
            date=str(daily_row["date"]),
            timestamp=daily_row.get("updated_at"),
        )


async def get_thread_token_usage(thread_id: str) -> ThreadTokenUsage:
    """Return cumulative token usage for a thread."""
    config = agent_config.get_token_budget_config()
    if not config.is_active or not settings.MONGODB_URI:
        raise TokenUsageUnavailableError("token budget tracking is not configured")

    try:
        repo = _mongo_repo()
        row = await repo.get_thread_usage(thread_id)
    except Exception as exc:
        logger.warning(
            "token_budget_mongo_read_failed",
            exc_info=True,
        )
        raise TokenUsageUnavailableError("token usage storage unavailable") from exc

    if row is None:
        raise TokenUsageNotFoundError(thread_id)

    return ThreadTokenUsage(
        thread_id=thread_id,
        used=int(row["total_tokens"]),
        input_tokens=int(row["input_tokens"]),
        output_tokens=int(row["output_tokens"]),
    )
