"""Parse ``/name`` tokens for subagents.

A token counts only when it matches a configured subagent name. The same
text the user typed is what the orchestrator sees, so recognition lives
here rather than in the chat box.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

# A slash command is a whole token: start of text or whitespace, then
# ``/name``. The name is every non-space character after the slash, with
# sentence punctuation such as ``.`` or ``,`` ignored. A name with a space
# does not match. ``https://example.com/analyst`` does not match. Matching
# is case-insensitive; the configured spelling is what gets returned.
_TOKEN = re.compile(r"(?:^|(?<=\s))/(\S+)")
_TRAILING_PUNCTUATION = ".,!?;:"


def message_text(content: Any) -> str:
    """Plain text from a message whose content is a string or text blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def find_requested_subagents(content: Any, catalog: Mapping[str, Any]) -> list[str]:
    """Return catalog keys named with ``/name`` in *content*, first occurrence order.

    Matching is case-insensitive. The token after ``/`` must equal a catalog
    name once trailing sentence punctuation is removed. Unknown names and
    slashes inside words or URLs are ignored. Duplicates appear once.
    """
    if not catalog:
        return []
    text = message_text(content)
    if not text:
        return []

    by_lower = {name.lower(): name for name in catalog}
    found: list[str] = []
    seen: set[str] = set()
    for match in _TOKEN.finditer(text):
        typed = match.group(1).rstrip(_TRAILING_PUNCTUATION).lower()
        canonical = by_lower.get(typed)
        if canonical is None or canonical in seen:
            continue
        seen.add(canonical)
        found.append(canonical)
    return found


def invocation_block(content: Any, catalog: Mapping[str, Any]) -> str:
    """System-prompt addendum for subagents named in *content*.

    A ``default`` or ``compiled`` subagent uses the ``task`` tool. An
    ``async`` subagent uses ``start_async_task``. Returns ``""`` when
    nothing matches. Only catalog keys are inserted.
    """
    names = find_requested_subagents(content, catalog)
    if not names:
        return ""

    task_names: list[str] = []
    async_names: list[str] = []
    ordered: list[str] = []
    for name in names:
        cfg = catalog.get(name)
        if isinstance(cfg, dict) and cfg.get("type") == "async":
            async_names.append(name)
            ordered.append(f"{name} with the start_async_task tool")
        else:
            task_names.append(name)
            ordered.append(f"{name} with the task tool")

    body = (
        "The user invoked these subagents for the latest message, in this order: "
        + ", ".join(ordered)
        + ". Delegate each one once, setting subagent_type to that name "
        "and description to the relevant part of the user's request. "
        "Decide from their descriptions and the user's question whether to "
        "run them one after another or at the same time. "
        "You may also call other subagents."
    )
    if task_names and async_names:
        body += (
            " Then answer from the task results, and report the async status"
            " without waiting for the remote run to finish."
        )
    elif task_names:
        body += " Then answer the user using their results."
    else:
        body += " Then tell the user the status the tool returned."
    return f"<invoked-subagents>\n{body}\n</invoked-subagents>"


def _one_line(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def mention_catalog(configs: Mapping[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Name and one-line description for every configured subagent.

    Prompt bodies, tools, and skills are omitted. The menu only needs
    enough to show a row.
    """
    items: list[dict[str, str]] = []
    for name in sorted(configs):
        cfg = configs[name]
        description = ""
        if isinstance(cfg, dict):
            description = _one_line(cfg.get("description", ""))
        items.append({"name": name, "description": description})
    return {"subagents": items}
