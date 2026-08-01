"""Inject user personalization context into the agent system prompt.

The injector appends two optional blocks to the base system prompt:

1. **User Memories** — facts the agent should recall across sessions
2. **User Rules** — custom instructions that shape agent behaviour

Both blocks are omitted when the corresponding list is empty, keeping
the prompt clean for users who haven't configured personalization.
"""

from __future__ import annotations


def inject_personalization(
    system_prompt: str,
    memories: list[str],
    rules: list[str],
) -> str:
    """Return *system_prompt* enriched with personalization blocks.

    Args:
        system_prompt: The base system prompt from config.
        memories: Plain-text user memories (newest first).
        rules: Plain-text user rules / custom instructions.

    Returns:
        The enriched system prompt. Unchanged if both lists are empty.
    """
    sections: list[str] = []

    if memories:
        lines = "\n".join(f"- {m}" for m in memories)
        sections.append(
            f"## User Memories\n\n"
            f"The following facts were saved by the user across prior sessions. "
            f"Treat them as persistent context — reference them when relevant "
            f"but do not repeat them verbatim unless asked.\n\n{lines}"
        )

    if rules:
        lines = "\n".join(f"- {r}" for r in rules)
        sections.append(
            f"## User Custom Instructions\n\n"
            f"The user has defined the following rules. Follow them for every "
            f"response unless they conflict with safety guidelines.\n\n{lines}"
        )

    if not sections:
        return system_prompt

    personalization_block = "\n\n---\n\n".join(sections)
    return f"{system_prompt}\n\n---\n\n{personalization_block}"
