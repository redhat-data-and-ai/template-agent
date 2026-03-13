"""Utility functions for deep research pipeline."""

import logging
from typing import Any

from template_agent.src.core.deep_research.prompts import (
    INPUT_CLASSIFICATION_PROMPT,
)
from template_agent.src.core.utils import safe_json_parse

logger = logging.getLogger(__name__)

__all__ = ["sanitize_error_for_client", "get_setting"]


def get_setting(name: str, default: Any) -> Any:
    """Get setting with fallback to default."""
    try:
        from template_agent.src.settings import settings

        return getattr(settings, name, default)
    except Exception:
        return default


def sanitize_error_for_client(exc: BaseException) -> str:
    """Sanitize exception for safe display to client."""
    msg = str(exc)
    if len(msg) > 500:
        return msg[:497] + "..."
    return msg


GIBBERISH_RESPONSE = (
    "I noticed your message doesn't seem to contain a clear question. "
    "No worries — it happens!\n\n"
    "Here's what I can help you with in **Deep Research** mode:\n\n"
    "- **Data analysis** — Ask about trends, metrics, or patterns\n"
    "- **Comparative research** — Compare teams, products, regions, or time periods\n"
    "- **Exploratory queries** — Discover what data is available and explore it\n"
    "- **Aggregations & reports** — Get summaries, counts, and breakdowns\n\n"
    "Just type a data-related question and I'll run a thorough "
    "multi-step investigation for you!"
)


def get_raw_checkpointer(checkpointer: Any) -> Any:
    """Return the underlying checkpointer if wrapped, else the checkpointer itself."""
    if hasattr(checkpointer, "raw_checkpointer"):
        return getattr(checkpointer, "raw_checkpointer")
    return checkpointer


async def aput_checkpoint(
    checkpointer: Any,
    config: Any,
    checkpoint: dict,
    metadata: dict,
) -> None:
    """Persist checkpoint with metadata."""
    try:
        from langgraph.checkpoint.base import CheckpointTuple

        new_tuple = CheckpointTuple(
            config=config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=None,
        )
        if hasattr(checkpointer, "aput_tuple"):
            await checkpointer.aput_tuple(new_tuple)
    except Exception as exc:
        logger.warning("Checkpoint persistence failed: %s", exc)


async def classify_input_quality(message: str, model: Any) -> str:
    """Classify user input as research_query or gibberish."""
    prompt_text = INPUT_CLASSIFICATION_PROMPT.format(message=message)
    try:
        response = await model.ainvoke(prompt_text)
        content = str(response.content or "").strip()
        parsed = safe_json_parse(content, pattern=r"\{[\s\S]*\}")
        if parsed and isinstance(parsed, dict):
            classification = parsed.get("classification", "research_query")
            if classification in ("research_query", "gibberish"):
                return classification
    except Exception as exc:
        logger.warning("Input classification failed: %s", exc)
    return "research_query"
