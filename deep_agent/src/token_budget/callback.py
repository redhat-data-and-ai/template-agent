"""LangChain callback handler for per-thread token budget tracking."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import ChatGeneration, LLMResult

from deep_agent.src.token_budget.identity import resolve_thread_id, resolve_user_id
from deep_agent.src.token_budget.service import (
    check_and_record,
    extract_tokens_from_llm_result,
    extract_tokens_from_message,
)
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

THREAD_ID_METADATA_KEY = "token_budget_thread_id"
USER_ID_METADATA_KEY = "token_budget_user_id"


def _extract_from_metadata(
    metadata: dict[str, Any] | None,
    key: str,
    *,
    fallback_keys: tuple[str, ...] = (),
) -> str | None:
    """Return the first non-empty metadata value for key or fallback keys."""
    if not metadata:
        return None
    for metadata_key in (key, *fallback_keys):
        value = metadata.get(metadata_key)
        if value:
            return str(value)
    return None


def thread_id_from_metadata(metadata: dict[str, Any] | None) -> str | None:
    """Resolve thread_id from RunnableConfig metadata."""
    return _extract_from_metadata(
        metadata,
        THREAD_ID_METADATA_KEY,
        fallback_keys=("langfuse_session_id",),
    )


def user_id_from_metadata(metadata: dict[str, Any] | None) -> str | None:
    """Resolve the chatting user's id from RunnableConfig metadata."""
    return _extract_from_metadata(metadata, USER_ID_METADATA_KEY)


class TokenBudgetCallbackHandler(AsyncCallbackHandler):
    """Increment per-thread token usage after each LLM call."""

    async def _record_tokens(
        self,
        response: LLMResult,
        metadata: dict[str, Any] | None,
        extraction_fn: Any,
    ) -> None:
        """Shared logic for recording token usage from LLM responses."""
        thread_id = thread_id_from_metadata(metadata) or resolve_thread_id()
        if not thread_id:
            logger.debug("token_budget_callback_no_thread_id")
            return

        user_id = user_id_from_metadata(metadata) or resolve_user_id()

        input_tokens, output_tokens = extraction_fn(response)
        if input_tokens <= 0 and output_tokens <= 0:
            input_tokens, output_tokens = _tokens_from_generations(response)

        try:
            await check_and_record(
                thread_id,
                input_tokens,
                output_tokens,
                user_id=user_id,
            )
        except Exception:
            logger.warning(
                "token_budget_callback_failed",
                thread_id=thread_id,
                exc_info=True,
            )

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        await self._record_tokens(response, metadata, extract_tokens_from_llm_result)


def _tokens_from_generations(response: LLMResult) -> tuple[int, int]:
    input_tokens = 0
    output_tokens = 0
    for generation_list in response.generations:
        for generation in generation_list:
            if not isinstance(generation, ChatGeneration):
                continue
            message = generation.message
            in_t, out_t = extract_tokens_from_message(message)
            input_tokens += in_t
            output_tokens += out_t
    return input_tokens, output_tokens
