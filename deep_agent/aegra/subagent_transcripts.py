"""Saved subagent transcripts for a chat reload.

Subagent messages live in checkpoint namespaces, not in the parent thread
the page loads. This route returns each subagent conversation so the card
can show it after the live run is gone.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

router = APIRouter(tags=["threads"])

_THINKING_TYPES = frozenset({"thinking", "reasoning", "thought"})


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif (
                isinstance(block, dict)
                and block.get("type") == "text"
                and block.get("thought") is not True
            ):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def _thinking_from_content(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") == "redacted_thinking":
            continue
        is_thought = (
            block.get("thought") is True or block.get("type") in _THINKING_TYPES
        )
        if not is_thought:
            continue
        for key in ("thinking", "reasoning", "summary", "text"):
            value = block.get(key)
            if isinstance(value, str) and value:
                parts.append(value)
                break
    return "".join(parts)


def transcript_from_messages(messages: list[Any]) -> dict[str, Any] | None:
    """Turn one subagent message list into a delegation plus card entries."""
    delegation = ""
    entries: list[dict[str, Any]] = []
    tools_by_id: dict[str, dict[str, Any]] = {}

    for raw in messages:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("type") or "").lower()
        if kind in {"human", "humanmessage"}:
            if not delegation:
                delegation = _text_from_content(raw.get("content")).strip()
            continue
        if kind in {"tool", "toolmessage"}:
            tool_id = raw.get("tool_call_id")
            match = tools_by_id.get(tool_id) if isinstance(tool_id, str) else None
            if match is not None:
                match["content"] = _text_from_content(raw.get("content"))
            continue
        if kind not in {"ai", "aimessage"}:
            continue

        tool_calls: list[dict[str, Any]] = []
        for call in raw.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            name = call.get("name") if isinstance(call.get("name"), str) else ""
            raw_args = call.get("args")
            args = raw_args if isinstance(raw_args, dict) else {}
            if name == "task" and isinstance(args.get("subagent_type"), str):
                name = args["subagent_type"]
            tool = {"id": call.get("id") or "", "name": name, "args": args}
            tool_calls.append(tool)
            if isinstance(tool["id"], str) and tool["id"]:
                tools_by_id[tool["id"]] = tool

        thinking = _thinking_from_content(raw.get("content"))
        answer = _text_from_content(raw.get("content")).strip()
        if not thinking and not answer and not tool_calls:
            continue
        entry: dict[str, Any] = {"id": raw.get("id") or f"m-{len(entries)}"}
        if thinking:
            entry["thinking"] = thinking
        if answer:
            entry["content"] = answer
        if tool_calls:
            entry["tool_calls"] = tool_calls
        entries.append(entry)

    if not delegation and not entries:
        return None
    return {"delegation": delegation, "entries": entries}


def select_transcripts(namespaces: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Keep one conversation per subagent launch.

    Nested checkpoints share the first namespace segment. The list that
    starts with the delegation is the subagent's own conversation.
    """
    groups: dict[str, list[tuple[str, list[Any]]]] = {}
    for namespace, messages in namespaces.items():
        root = namespace.split("|", 1)[0]
        groups.setdefault(root, []).append((namespace, messages))

    chosen: list[dict[str, Any]] = []
    for root in sorted(groups):

        def _score(item: tuple[str, list[Any]]) -> tuple[bool, int]:
            messages = item[1]
            has_human = any(
                isinstance(message, dict)
                and str(message.get("type") or "").lower().startswith("human")
                for message in messages
            )
            return (has_human, len(messages))

        _namespace, messages = max(groups[root], key=_score)
        built = transcript_from_messages(messages)
        if built is not None:
            chosen.append(built)
    return chosen


async def _load_namespace_messages(thread_id: str) -> dict[str, list[Any]]:
    import msgpack
    import psycopg

    from deep_agent.src.settings import settings

    if not settings.database_uri:
        return {}

    def _ext_hook(code: int, data: bytes) -> Any:
        try:
            return msgpack.unpackb(data, raw=False, ext_hook=_ext_hook)
        except Exception:
            return data

    namespaces: dict[str, list[Any]] = {}
    async with await psycopg.AsyncConnection.connect(settings.database_uri) as conn:
        rows = await conn.execute(
            """
            SELECT DISTINCT ON (checkpoint_ns) checkpoint_ns, blob
            FROM checkpoint_blobs
            WHERE thread_id = %s
              AND checkpoint_ns != ''
              AND channel = 'messages'
            ORDER BY checkpoint_ns, version DESC
            """,
            (thread_id,),
        )
        async for namespace, blob in rows:
            if blob is None or not isinstance(namespace, str):
                continue
            try:
                items = msgpack.unpackb(bytes(blob), raw=False, ext_hook=_ext_hook)
            except Exception as exc:
                logger.debug("Skipping subagent blob for %s: %s", thread_id, exc)
                continue
            if not isinstance(items, (list, tuple)):
                continue
            messages: list[Any] = []
            for item in items:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) >= 3
                    and isinstance(item[2], dict)
                ):
                    messages.append(item[2])
            if messages:
                namespaces[namespace] = messages
    return namespaces


@router.get("/threads/{thread_id}/subagent-transcripts")
async def get_subagent_transcripts(thread_id: str, request: Request) -> dict[str, Any]:
    """Return saved subagent conversations for one thread."""
    from deep_agent.aegra.auth import ENABLE_AUTH
    from deep_agent.src.settings import settings

    user_id: str | None = None
    if ENABLE_AUTH:
        from deep_agent.aegra.auth_helpers import authenticated_user_id

        user_id = await authenticated_user_id(request, reject_anonymous=True)

    if not settings.database_uri:
        return {"transcripts": []}

    import psycopg

    try:
        async with await psycopg.AsyncConnection.connect(settings.database_uri) as conn:
            if user_id is not None:
                cursor = await conn.execute(
                    "SELECT 1 FROM thread WHERE thread_id = %s AND user_id = %s",
                    (thread_id, user_id),
                )
            else:
                cursor = await conn.execute(
                    "SELECT 1 FROM thread WHERE thread_id = %s",
                    (thread_id,),
                )
            if await cursor.fetchone() is None:
                raise HTTPException(
                    status_code=404, detail=f"Thread '{thread_id}' not found"
                )
        namespaces = await _load_namespace_messages(thread_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Subagent transcript read failed: %s", exc)
        return {"transcripts": []}

    return {"transcripts": select_transcripts(namespaces)}
