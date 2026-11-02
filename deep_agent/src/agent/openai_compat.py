"""OpenAI-compatible chat client that preserves per-tool-call ``extra_content``.

Some OpenAI-compatible gateways attach vendor metadata to each tool call under
an ``extra_content`` key. Gemini 3.x carries its ``thought_signature`` there and
rejects any follow-up request whose function calls arrive without it:

    400 Function call is missing a thought_signature in functionCall parts.

``langchain_openai`` discards the field while parsing a response, so the
signature is gone before the agent loop can echo it back, and every multi-turn
tool call fails on its second turn. This module captures the field on the way
in, stashes it on the message keyed by tool-call id, and re-attaches it on the
way out.

The relay is provider-agnostic: it keys off the presence of ``extra_content``
and never inspects the contents, so endpoints that do not send the field are
unaffected.
"""

from __future__ import annotations

from typing import Any

import openai
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI

# Key used both for the raw field on a tool call and for the stash on
# ``AIMessage.additional_kwargs``, where it holds {tool_call_id: [extra_content]}.
EXTRA_CONTENT_KEY = "extra_content"


def _extra_content_by_tool_call_id(tool_calls: Any) -> dict[str, list[Any]]:
    """Map tool-call ``id`` to its ``extra_content``, for well-formed entries only.

    Anything malformed is skipped rather than raised on: this runs inside a
    response-parsing path, where a vendor quirk must never become a crash.

    Each value is a single-element *list*, and that is load-bearing. LangChain
    merges ``additional_kwargs`` across streaming chunks with ``merge_dicts``,
    which recurses into nested dicts and **concatenates** strings on key
    collision. A plain ``{id: blob}`` stash would therefore turn a signature
    repeated across two deltas into ``"abc" + "abc" == "abcabc"`` — silently
    corrupting the very bytes this module exists to relay intact. ``merge_dicts``
    hands list values to ``merge_lists``, which appends entries that carry no
    integer ``index`` key, so duplicates accumulate harmlessly and the replay
    takes the first. Do not "simplify" this back to a bare dict.

    Args:
        tool_calls: Raw ``tool_calls`` value from a response message or delta.

    Returns:
        Mapping of tool-call id to a one-element list holding its extra content.
        Empty when the input is not a list or no entry carries usable extra
        content.
    """
    if not isinstance(tool_calls, list):
        return {}

    found: dict[str, list[Any]] = {}
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        call_id = tool_call.get("id")
        extra = tool_call.get(EXTRA_CONTENT_KEY)
        if isinstance(call_id, str) and call_id and isinstance(extra, dict) and extra:
            found[call_id] = [extra]
    return found


class ExtraContentChatOpenAI(ChatOpenAI):
    """``ChatOpenAI`` that round-trips per-tool-call ``extra_content``.

    Overrides the three points that see the raw wire payload. Each is a no-op
    when no tool call carries ``extra_content``, so models served by endpoints
    that never send it behave exactly as they do under plain ``ChatOpenAI``.
    """

    def _create_chat_result(
        self,
        response: dict | openai.BaseModel,
        generation_info: dict | None = None,
    ) -> ChatResult:
        """Capture ``extra_content`` from a non-streaming response."""
        result = super()._create_chat_result(response, generation_info)

        response_dict = (
            response if isinstance(response, dict) else response.model_dump()
        )
        choices = response_dict.get("choices") or []

        # super() builds one generation per choice, in order.
        for generation, choice in zip(result.generations, choices, strict=False):
            if not isinstance(generation.message, AIMessage) or not isinstance(
                choice, dict
            ):
                continue
            raw_message = choice.get("message")
            if not isinstance(raw_message, dict):
                continue
            captured = _extra_content_by_tool_call_id(raw_message.get("tool_calls"))
            if captured:
                generation.message.additional_kwargs[EXTRA_CONTENT_KEY] = captured

        return result

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        """Capture ``extra_content`` from a streaming delta.

        Keyed by tool-call id, not by ``index``: the gateway sends ``index:
        null`` on these deltas. See :func:`_extra_content_by_tool_call_id` for
        why each value is a list — chunk merging concatenates strings, so the
        list is what keeps a repeated signature from being doubled.
        """
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation_chunk is None or not isinstance(
            generation_chunk.message, AIMessageChunk
        ):
            return generation_chunk

        # Mirror how the library resolves choices, including the beta stream shape.
        choices = chunk.get("choices", []) or chunk.get("chunk", {}).get("choices", [])
        if not choices or not isinstance(choices[0], dict):
            return generation_chunk

        delta = choices[0].get("delta")
        if not isinstance(delta, dict):
            return generation_chunk

        captured = _extra_content_by_tool_call_id(delta.get("tool_calls"))
        if captured:
            generation_chunk.message.additional_kwargs[EXTRA_CONTENT_KEY] = captured

        return generation_chunk

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Re-attach captured ``extra_content`` to the outbound tool calls.

        Matching is by tool-call id rather than position, so message order and
        any history trimming are irrelevant. Stashed ids with no matching tool
        call in this payload are ignored.
        """
        payload: dict = super()._get_request_payload(input_, stop=stop, **kwargs)

        # The Responses API payload has a different shape and no "messages".
        if self._use_responses_api(payload):
            return payload
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return payload

        stashed: dict[str, Any] = {}
        for message in self._convert_input(input_).to_messages():
            if isinstance(message, AIMessage):
                captured = message.additional_kwargs.get(EXTRA_CONTENT_KEY)
                if isinstance(captured, dict):
                    stashed.update(captured)
        if not stashed:
            return payload

        # Values are lists so chunk merging appends rather than concatenating
        # (see _extra_content_by_tool_call_id). Repeats of one tool call carry
        # the same blob, so the first entry is the one to send.
        stashed = {
            call_id: entries[0] if isinstance(entries, list) else entries
            for call_id, entries in stashed.items()
            if not isinstance(entries, list) or entries
        }

        for message_dict in messages:
            if (
                not isinstance(message_dict, dict)
                or message_dict.get("role") != "assistant"
            ):
                continue
            for tool_call in message_dict.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                call_id = tool_call.get("id")
                if not isinstance(call_id, str):
                    continue
                extra = stashed.get(call_id)
                if extra is not None:
                    tool_call[EXTRA_CONTENT_KEY] = extra

        return payload
