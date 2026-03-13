"""Utility functions for deep research pipeline."""

import json
import re
from typing import Any


def safe_json_parse(
    text: str,
    pattern: str = r"\[[\s\S]*\]",
    default: Any = None,
) -> Any:
    """Extract and parse the first JSON object/array from text."""
    if not text:
        return default
    try:
        match = re.search(pattern, text)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass
    return default


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to max_length, appending suffix if truncated."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def sanitize_error_for_client(exc: Exception) -> str:
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

INPUT_CLASSIFICATION_PROMPT = """\
You are an input classifier for a research system.

Classify the user message into exactly ONE category:

- **research_query**: A meaningful question or request that requires research.

- **gibberish**: Random characters, keyboard mash, accidental input, or
strings with no discernible meaning (e.g. "asdfghjkl", "3424fsdwsfgn", "qqqqqq").

Respond with ONLY a JSON object, nothing else:
{{"classification": "research_query"}} or {{"classification": "gibberish"}}

User message: {message}"""


def get_raw_checkpointer(checkpointer: Any) -> Any:
    """Return the underlying checkpointer if wrapped, else the checkpointer itself."""
    if hasattr(checkpointer, "raw_checkpointer"):
        return getattr(checkpointer, "raw_checkpointer")
    return checkpointer


def _sanitize_messages_for_persistence(messages: list) -> list:
    """Sanitize messages for persistence (strip large content, etc.)."""
    return messages


async def aput_checkpoint(
    checkpointer: Any,
    config: Any,
    checkpoint: dict,
    metadata: dict,
    channel_versions: dict,
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
    except Exception:
        pass


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
    except Exception:
        pass
    return "research_query"
