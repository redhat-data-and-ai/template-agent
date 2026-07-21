"""Outermost graph wrapper for PII token-map sharing and SSE stream restoration.

Controlled by PII_MIDDLEWARE_ENABLED.  When active it:

  1. Creates a per-request shared mutable dict (token_map_container) and
     registers it with the PIIScrubber via set_shared_container() so that
     PIIMiddleware can push the per-request token map back across LangGraph's
     ContextVar isolation boundary after scrubbing model inputs.

  2. For astream_events: restores every wire surface that can carry tokenized
     PII placeholders ([EMAIL_1] etc.) before it reaches the SSE client:
       - on_chat_model_stream chunks — buffered per run_id (one LLM call),
         assembled, and restored (both `content` and `tool_calls`/
         `tool_call_chunks`) as soon as that run's on_chat_model_end fires.
       - on_tool_start — the tool's `args` are restored immediately so the
         "arguments" the UI renders for a tool call are never raw tokens,
         even though the tool itself already executes with real values
         (PIIMiddleware restores the AIMessage before it's added to graph
         state, which is what LangGraph hands to the tool).

This wrapper is intentionally independent of Guardian / SafetyAwareRunnable.
When both are enabled the wrapping order is:

    PIIAwareRunnable  (outermost — sees final event stream)
      └─ SafetyAwareRunnable  (safety checks, refusal injection)
           └─ compiled graph

When only PII is enabled (Guardian off):

    PIIAwareRunnable
      └─ compiled graph
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


def _preview(value: Any, max_len: int = 120) -> str:
    if isinstance(value, str):
        return value[:max_len]
    return repr(value)[:max_len]


def _restore_text(text: Any, container: dict[str, str]) -> Any:
    if not isinstance(text, str) or not container:
        return text
    for token, value in container.items():
        if token in text:
            text = text.replace(token, value)
    return text


def _restore_content(content: Any, container: dict[str, str]) -> Any:
    if isinstance(content, str):
        return _restore_text(content, container)
    if isinstance(content, list):
        result = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                result.append(
                    {**block, "text": _restore_text(block.get("text", ""), container)}
                )
            else:
                result.append(block)
        return result
    return content


def _restore_args(args: Any, container: dict[str, str]) -> Any:
    if isinstance(args, dict):
        return {k: _restore_args(v, container) for k, v in args.items()}
    if isinstance(args, list):
        return [_restore_args(v, container) for v in args]
    if isinstance(args, str):
        return _restore_text(args, container)
    return args


def _restore_message_stream_data(data: Any, container: dict[str, str]) -> Any:
    """Restore tokens in a LangGraph messages-mode stream item.

    Messages mode yields [BaseMessageChunk, metadata_dict] pairs.
    _restore_args handles plain dicts/lists/strings but not Pydantic message
    objects — this function handles the chunk explicitly.
    """
    if not container:
        return data

    # LangGraph messages mode: data is [chunk, metadata] or (chunk, metadata)
    if isinstance(data, (list, tuple)) and len(data) == 2:
        chunk, metadata = data
        if hasattr(chunk, "content"):
            restored_content = _restore_content(chunk.content, container)
            tool_calls = getattr(chunk, "tool_calls", None) or []
            if tool_calls:
                restored_tcs = [
                    {**tc, "args": _restore_args(tc.get("args", {}), container)}
                    for tc in tool_calls
                ]
                chunk = chunk.model_copy(
                    update={"content": restored_content, "tool_calls": restored_tcs}
                )
            elif restored_content != chunk.content:
                chunk = chunk.model_copy(update={"content": restored_content})
        result = [chunk, metadata]
        return tuple(result) if isinstance(data, tuple) else result

    # Fallback for unexpected formats
    return _restore_args(data, container)


class PIIAwareRunnable:
    """Outermost graph wrapper that manages the PII token map across requests."""

    def __init__(self, runnable: Any) -> None:
        """Wrap *runnable* with PII token-map management."""
        self._runnable = runnable

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped runnable."""
        return getattr(self._runnable, name)

    def copy(self, **kwargs: Any) -> "PIIAwareRunnable":
        """Return a re-wrapped copy of the inner runnable."""
        return PIIAwareRunnable(self._runnable.copy(**kwargs))

    def with_config(self, config: Any = None, **kwargs: Any) -> "PIIAwareRunnable":
        """Re-wrap after with_config so the PIIAwareRunnable is not stripped by __getattr__."""
        if config is not None:
            inner = self._runnable.with_config(config, **kwargs)
        else:
            inner = self._runnable.with_config(**kwargs)
        return PIIAwareRunnable(inner)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _setup_container(self) -> tuple[dict[str, str], bool]:
        """Return the shared token-map container, registering one if needed.

        Reuses an already-registered container instead of unconditionally
        creating a new one. This matters because subagents are also wrapped
        in PIIAwareRunnable and invoked as a nested `ainvoke()` in the *same*
        async context as the orchestrator (not a separate asyncio.Task) — so
        without this check, a subagent call would silently replace the
        orchestrator's container reference (both the ContextVar and the
        scrubber's singleton `_instance_container`) partway through the
        request, orphaning it and losing any tokens produced afterwards.
        The first (outermost) call creates the container; every nested call
        within the same request reuses that same object.

        Returns (container, owned) — `owned` is True only for the call that
        created the container, so only that call is responsible for clearing
        it afterward (see _clear_container).
        """
        try:
            from deep_agent.src.pii import get_scrubber

            s = get_scrubber()
            if not s:
                logger.debug(
                    "pii_aware_runnable scrubber_not_found — PII middleware inactive"
                )
                return {}, False
            existing = s._get_shared_container()
            if existing is not None:
                logger.debug("pii_aware_runnable container_reused (nested runnable)")
                return existing, False
            container: dict[str, str] = {}
            s.set_shared_container(container)
            logger.debug("pii_aware_runnable container_registered=True")
            return container, True
        except Exception as exc:
            logger.warning("pii_aware_runnable container_setup_failed: %s", exc)
            return {}, False

    def _clear_container(self, owned: bool) -> None:
        """Clear the instance-level container reference on the scrubber after a request.

        Only the wrapper that created the container (owned=True) may clear it.
        A nested/reusing call clearing it would wipe the singleton fallback
        attribute out from under the still-running outer wrapper.
        """
        if not owned:
            return
        try:
            from deep_agent.src.pii import get_scrubber

            s = get_scrubber()
            if s and getattr(s, "_instance_container", None) is not None:
                s._instance_container = None
        except Exception:
            pass

    def _assemble_chunk(self, events: list[dict]) -> Any:
        """Sum the AIMessageChunk objects from one run's buffered events into one."""
        assembled = None
        for e in events:
            chunk = e.get("data", {}).get("chunk")
            if chunk is None:
                continue
            assembled = chunk if assembled is None else assembled + chunk
        return assembled

    def _restore_chunk(self, chunk: Any, container: dict[str, str]) -> Any:
        """Return a new AIMessageChunk with content and tool_calls fully restored."""
        from langchain_core.messages import AIMessageChunk

        content = _restore_content(getattr(chunk, "content", None), container)
        tool_calls = getattr(chunk, "tool_calls", None) or []
        if not tool_calls:
            return AIMessageChunk(content=content)

        restored_tool_calls = [
            {**tc, "args": _restore_args(tc.get("args", {}), container)}
            for tc in tool_calls
        ]
        tool_call_chunks = [
            {
                "name": tc.get("name"),
                "args": json.dumps(tc.get("args", {})),
                "id": tc.get("id"),
                "index": i,
            }
            for i, tc in enumerate(restored_tool_calls)
        ]
        return AIMessageChunk(content=content, tool_call_chunks=tool_call_chunks)

    async def _flush_run(
        self, run_id: Any, events: list[dict], container: dict[str, str]
    ) -> AsyncIterator[Any]:
        """Assemble, restore, and yield one LLM run's buffered chunks as a single event."""
        if not events:
            return
        assembled = self._assemble_chunk(events)
        last = events[-1]
        if assembled is None or not container:
            for e in events:
                yield e
            return
        restored = self._restore_chunk(assembled, container)
        logger.debug(
            "pii_aware_runnable sse_restore run_id=%s chunks=%d tool_calls=%d",
            run_id,
            len(events),
            len(restored.tool_calls or []),
        )
        yield {**last, "data": {"chunk": restored}}

    def _restore_tool_event_args(self, event: dict, container: dict[str, str]) -> dict:
        """Restore args on an on_tool_start event before it reaches the client."""
        if not container:
            return event
        data = event.get("data", {})
        tool_input = data.get("input")
        if isinstance(tool_input, dict) and "args" in tool_input:
            restored_input = {
                **tool_input,
                "args": _restore_args(tool_input["args"], container),
            }
            return {**event, "data": {**data, "input": restored_input}}
        if isinstance(tool_input, dict):
            return {
                **event,
                "data": {**data, "input": _restore_args(tool_input, container)},
            }
        return event

    def _restore_tool_event_output(
        self, event: dict, container: dict[str, str]
    ) -> dict:
        """Restore tokens in an on_tool_end event's output before it reaches the client.

        The tool result content — including nested dicts and list-of-block content
        (e.g. [{"type": "text", "text": {...}}]) — is recursively de-anonymized so
        the UI never renders raw PII placeholders for tool outputs or subagent results.
        """
        if not container:
            return event
        data = event.get("data", {})
        output = data.get("output")
        if output is None:
            return event
        restored_output = _restore_args(output, container)
        return {**event, "data": {**data, "output": restored_output}}

    # ── Core async interface ─────────────────────────────────────────────

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Invoke the wrapped runnable with PII container management."""
        _container, owned = self._setup_container()
        try:
            return await self._runnable.ainvoke(input, config, **kwargs)
        finally:
            self._clear_container(owned)

    async def astream(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Stream chunks from the wrapped runnable, restoring PII tokens in message events."""
        container, owned = self._setup_container()
        thread_id: str | None = (config or {}).get("configurable", {}).get("thread_id")

        def _token_map() -> dict[str, str]:
            if thread_id:
                try:
                    from deep_agent.src.pii.scrubber import _thread_token_maps

                    merged = dict(container)
                    merged.update(_thread_token_maps.get(thread_id, {}))
                    return merged
                except Exception:
                    pass
            return container

        async for chunk in self._runnable.astream(input, config, **kwargs):
            if isinstance(chunk, tuple) and len(chunk) == 2:
                mode, data = chunk
                if mode == "messages":
                    tok_map = _token_map()
                    if tok_map:
                        data = _restore_message_stream_data(data, tok_map)
                    yield (mode, data)
                    continue
            yield chunk

        self._clear_container(owned)

    async def astream_events(
        self, input: Any, config: Any = None, **kwargs: Any
    ) -> Any:
        """Stream events from the wrapped runnable, restoring PII tokens before emission."""
        container, owned = self._setup_container()

        # Thread-store is the primary source of truth, not the shared container.
        # The container (a ContextVar + singleton instance attribute) gets
        # hijacked every time ANY nested PIIAwareRunnable — e.g. a subagent's
        # own wrapper — calls _setup_container(), which re-registers a brand
        # new dict and silently orphans this one. _thread_token_maps, by
        # contrast, is a plain dict keyed by thread_id that every model call
        # (main graph or subagent, any nesting depth) merges into via
        # save_thread_map() during input sanitization — *before* the LLM
        # runs. Since real PII never reaches the model, any token the LLM
        # could possibly echo back was necessarily assigned during that
        # scrubbing step, so this map is always complete by the time SSE
        # events need restoring. Fall back to the container only when there
        # is no thread_id to key off (e.g. stateless one-off invocations).
        thread_id: str | None = (config or {}).get("configurable", {}).get("thread_id")

        def _token_map() -> dict[str, str]:
            """Return the best available token map at yield time."""
            if thread_id:
                try:
                    from deep_agent.src.pii.scrubber import _thread_token_maps

                    merged = dict(container)
                    merged.update(_thread_token_maps.get(thread_id, {}))
                    return merged
                except Exception:
                    pass
            return container

        # Buffer on_chat_model_stream events per run_id (one LLM call), so each
        # call's output is restored and flushed independently — as soon as that
        # call's on_chat_model_end fires — instead of merging every LLM call in
        # the whole graph run into a single chunk at the very end.
        buffers: dict[Any, list[dict]] = {}

        async for event in self._runnable.astream_events(input, config, **kwargs):
            event_type = event.get("event", "")
            run_id = event.get("run_id")

            if event_type == "on_chat_model_stream":
                buffers.setdefault(run_id, []).append(event)
                continue

            if event_type in ("on_chat_model_end", "on_llm_end") and run_id in buffers:
                async for restored_event in self._flush_run(
                    run_id, buffers.pop(run_id), _token_map()
                ):
                    yield restored_event
                yield event
                continue

            if event_type == "on_tool_start":
                event = self._restore_tool_event_args(event, _token_map())

            elif event_type == "on_tool_end":
                event = self._restore_tool_event_output(event, _token_map())

            yield event

        # Defensive: flush any run whose end event never arrived.
        for run_id, events in buffers.items():
            async for restored_event in self._flush_run(run_id, events, _token_map()):
                yield restored_event

        self._clear_container(owned)
