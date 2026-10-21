"""Tell the model what the user has attached.

Without this, an attached file is invisible: it sits in the thread's
filesystem and the model has no reason to look. The user has to name
the path — "read /uploads/intake.txt" — which is not something anyone
should have to know, let alone type.

So the attachment list is injected into the conversation before the
model runs. The model then sees the files the way the user thinks of
them: as things they just handed over.
"""

from __future__ import annotations

from typing import Any

from deep_agent.src.settings import settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

# Keep the note short. It is prepended to every model call on a thread
# with attachments, so verbosity here is paid for repeatedly.
_MAX_LISTED = settings.ATTACHMENT_MAX_LISTED


def build_notice(paths: list[str]) -> str:
    """The note describing what is attached."""
    listed = paths[:_MAX_LISTED]
    lines = "\n".join(f"- {p}" for p in listed)
    more = (
        f"\n- …and {len(paths) - _MAX_LISTED} more" if len(paths) > _MAX_LISTED else ""
    )
    return (
        "The user has attached the following files to this conversation. "
        "Read them with `read_file` when they are relevant — the user "
        "refers to them by name, not by path, and expects you to already "
        "know they are here.\n"
        f"{lines}{more}"
    )


def attachment_paths(state: Any, mount: str = "/uploads/") -> list[str]:
    """Attached file paths from graph state, in a stable order."""
    files = state.get("files") if isinstance(state, dict) else None
    if not isinstance(files, dict):
        return []
    return sorted(p for p in files if isinstance(p, str) and p.startswith(mount))


def build_attachment_middleware(mount_path: str = "/uploads/") -> Any:
    """Middleware that announces attachments on each model call.

    The notice is added to the *request*, not to state. Returning it
    from `before_model` seemed simpler and was wrong: the `messages`
    channel reduces with `add_messages`, which appends, so a "replace
    the old notice" pass never replaced anything and the conversation
    accumulated one copy per turn — stored in the thread forever.

    Overriding the request leaves no trace in the transcript, and the
    list is rebuilt from state every call so it cannot go stale.
    """
    from langchain.agents.middleware.types import AgentMiddleware
    from langchain_core.messages import SystemMessage

    class AttachmentNoticeMiddleware(AgentMiddleware):
        """Tell the model which files the user has attached."""

        def wrap_model_call(self, request: Any, handler: Any) -> Any:
            return handler(self._with_notice(request))

        async def awrap_model_call(self, request: Any, handler: Any) -> Any:
            return await handler(self._with_notice(request))

        def _with_notice(self, request: Any) -> Any:
            paths = attachment_paths(getattr(request, "state", None), mount_path)
            if not paths:
                return request

            notice = build_notice(paths)
            existing = getattr(request, "system_message", None)

            # `system_message` is a SystemMessage, not a string —
            # passing a str here fails later with "'str' object has no
            # attribute 'content_blocks'", which names neither this
            # middleware nor the field.
            try:
                if existing is None:
                    updated = SystemMessage(content=notice)
                elif isinstance(getattr(existing, "content", None), str):
                    updated = existing.model_copy(
                        update={"content": f"{existing.content}\n\n{notice}".strip()}
                    )
                else:
                    # Structured content: append a block rather than
                    # stringifying what is already there.
                    blocks = list(existing.content or [])
                    blocks.append({"type": "text", "text": notice})
                    updated = existing.model_copy(update={"content": blocks})
                return request.override(system_message=updated)
            except Exception:
                logger.debug("Could not attach notice to request", exc_info=True)
                return request

    return AttachmentNoticeMiddleware()
