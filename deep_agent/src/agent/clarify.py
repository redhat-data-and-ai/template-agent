"""Ask the user one clarifying question and wait for the answer.

The pause is a LangGraph interrupt. The payload is stored on the thread
checkpoint, so any pod can show the question and accept the answer.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from langgraph.types import interrupt
from pydantic import BaseModel, Field

_EMPTY_QUESTION = "No question was asked. Call clarify again with a non-empty question."
_NO_ANSWER = "The user did not provide an answer."
_NON_INTERACTIVE = "No preference. Continue with the most reasonable choice."

_DESCRIPTION = """Asks the user one question and waits until they answer.

Use this when you need a fact from the user that you cannot infer.
Put a string in options only when it is a complete answer.
If the user must supply the value, omit options.
The user may tick any number of suggested answers and may also type their own words.
The reply is one string.
If the answer still does not contain the fact, call clarify again.
Do not ask that question in your reply.
Ask one question at a time. Do not call any other tool in the same step."""


class ClarifyInput(BaseModel):
    """Arguments for the clarify tool."""

    question: str = Field(description="The one question to ask the user.")
    options: list[str] | None = Field(
        default=None,
        description="Suggested answers. The user may tick any number and may also type their own words.",
    )


def _clean_options(options: Any) -> list[str]:
    """Keep non-empty strings, in order. Drop anything else."""
    if not isinstance(options, list):
        return []
    return [item.strip() for item in options if isinstance(item, str) and item.strip()]


def _is_interactive() -> bool:
    """True only when the chat UI marked this run as interactive."""
    try:
        from langgraph.config import get_config
    except ImportError:
        return False
    cfg = get_config() or {}
    configurable = cfg.get("configurable") or {}
    return bool(configurable.get("interactive"))


def _ask(question: str, options: list[str] | None = None) -> str:
    """Pause for one answer. A blank question does not pause the run."""
    text = question.strip() if isinstance(question, str) else ""
    if not text:
        return _EMPTY_QUESTION
    # Headless / webhook / eval never set interactive — do not block them.
    if not _is_interactive():
        return _NON_INTERACTIVE
    answer = interrupt(
        {
            "type": "user_question",
            "question": text,
            "options": _clean_options(options),
        }
    )
    if not isinstance(answer, str) or not answer.strip():
        return _NO_ANSWER
    return answer.strip()


clarify = StructuredTool(
    name="clarify",
    description=_DESCRIPTION,
    func=_ask,
    args_schema=ClarifyInput,
)


def attach_clarify(tools: list[Any]) -> list[Any]:
    """Return tools plus clarify, unless that tool is already present."""
    if any(getattr(tool, "name", None) == "clarify" for tool in tools):
        return list(tools)
    return [*tools, clarify]
