"""Resolve thread and user identity from LangGraph RunnableConfig."""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime


def resolve_thread_id(runtime: Runtime[Any] | None = None) -> str | None:
    """Resolve thread_id from LangGraph runtime or active RunnableConfig."""
    if runtime is not None:
        execution_info = getattr(runtime, "execution_info", None)
        if execution_info is not None:
            thread_id = getattr(execution_info, "thread_id", None)
            if thread_id:
                return str(thread_id)

    try:
        from langgraph.config import get_config

        config = get_config()
        configurable = config.get("configurable") or {}
        thread_id = configurable.get("thread_id")
        if thread_id:
            return str(thread_id)
    except Exception:
        pass
    return None


def resolve_user_id(runtime: Runtime[Any] | None = None) -> str | None:
    """Resolve chatting user_id from LangGraph runtime or active RunnableConfig."""
    if runtime is not None:
        execution_info = getattr(runtime, "execution_info", None)
        if execution_info is not None:
            user_id = getattr(execution_info, "user_id", None)
            if user_id:
                return str(user_id)

    try:
        from langgraph.config import get_config

        config = get_config()
        configurable = config.get("configurable") or {}
        user_id = configurable.get("user_id")
        if user_id:
            return str(user_id)
    except Exception:
        pass
    return None
