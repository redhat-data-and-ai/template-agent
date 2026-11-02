"""Clarification tool - pauses the agent to ask the user a follow-up question.

Uses LangGraph's ``interrupt()`` to surface a question to the UI as a popup
modal. The user's response is returned as the tool result, enriching context
before the agent continues with searches.
"""

from __future__ import annotations

from langchain_core.tools import tool
from langgraph.types import interrupt

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


@tool
def clarify(question: str, options: list[str] | None = None) -> str:
    """Ask the user a clarifying question to enrich context before searching.

    Use this tool when the user's query is ambiguous, underspecified, or could
    benefit from additional context. Examples:
    - "Which fiscal year are you asking about?"
    - "Are you interested in revenue by segment or total company revenue?"
    - "Do you want a comparison across quarters or a single quarter deep dive?"

    The user will see a popup with your question. If you provide options,
    they will be shown as clickable buttons the user can pick from. The user
    can also type a custom answer.

    IMPORTANT: When you already have data from a prior tool call (e.g.,
    execute_sql returned multiple matching rows), pass the candidates as
    options so the user can pick instead of typing. For example, if a
    people search returned 5 matches, pass their names as options.

    Args:
        question: The clarifying question to ask the user.
        options: Optional list of choices to show as clickable buttons.
            Pass candidates from prior tool results when available.

    Returns:
        The user's response text (either a picked option or typed answer).
    """
    logger.info("Clarify tool invoked: %s (options=%d)", question, len(options or []))
    response = interrupt({
        "type": "clarify",
        "question": question,
        "options": options or [],
    })
    user_answer = str(response) if response else ""
    logger.info("Clarify response received: %s", user_answer[:100])
    return user_answer
