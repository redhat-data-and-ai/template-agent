"""Platform memory file path and save/recall instructions.

Factory agents mount their own ``config/agent/`` (PROMPT.md, agent.yaml).
The memory file path and the instructions that tell the model how to write
it live here in the runtime package so those agents pick them up from the
image without a config-volume change.
"""

from __future__ import annotations

from pathlib import Path

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

USER_MEMORY_FILE = "/memories/user_profile.md"
USER_MEMORY_STORE_KEY = "user_profile.md"
STOCK_MEMORY_NAMESPACE = "memories"


def is_memory_store_key(key: str) -> bool:
    """Return True if *key* is the platform user-memory file.

    CompositeBackend strips the ``/memories/`` prefix, so the Store key is
    ``user_profile.md`` or ``/user_profile.md``. Accept the full path as well
    so callers can pass either form.
    """
    if not key:
        return False
    normalized = key.strip().lstrip("/")
    return normalized == USER_MEMORY_STORE_KEY or normalized.endswith(
        "/" + USER_MEMORY_STORE_KEY
    )


def memory_file_text(value: dict | None) -> str:
    """Return the memory file body from a LangGraph Store item value.

    deepagents 0.7.x persists ``FileData.content`` as a string. Older rows
    used a list of lines. Joining a string would split it into characters.
    """
    if not isinstance(value, dict):
        return ""
    raw = value.get("content", "")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        return "\n".join(str(line) for line in raw)
    return ""


_INSTRUCTIONS_PATH = Path(__file__).resolve().parent / "memory_instructions.j2"
_instructions_cache: str | None = None
_instructions_mtime: float | None = None


def load_memory_instructions() -> str:
    """Load the memory save/recall instructions (re-read when the template changes)."""
    global _instructions_cache, _instructions_mtime  # noqa: PLW0603
    try:
        mtime = _INSTRUCTIONS_PATH.stat().st_mtime
    except FileNotFoundError:
        logger.warning(
            "Memory instructions template not found at %s",
            _INSTRUCTIONS_PATH,
        )
        _instructions_cache = ""
        _instructions_mtime = None
        return _instructions_cache
    if _instructions_cache is None or _instructions_mtime != mtime:
        _instructions_cache = _INSTRUCTIONS_PATH.read_text()
        _instructions_mtime = mtime
    return _instructions_cache


def append_memory_instructions(system_prompt: str) -> str:
    """Append memory instructions to the system prompt when memory is enabled."""
    instructions = load_memory_instructions()
    if not instructions:
        return system_prompt
    return system_prompt.rstrip() + "\n\n---\n\n" + instructions


def resolve_memory_namespaces(configured: list[str] | None) -> list[str]:
    """Map stock YAML ``memories`` to the platform memory file.

    ``agent.yaml`` keeps ``namespaces: ["memories"]`` (template default).
    That token is not a Store file path — the runtime substitutes
    ``/memories/user_profile.md``. Custom YAML paths pass through unchanged.
    """
    if not configured:
        return [USER_MEMORY_FILE]
    if len(configured) == 1 and configured[0].strip("/") == STOCK_MEMORY_NAMESPACE:
        return [USER_MEMORY_FILE]
    return list(configured)
