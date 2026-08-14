"""PIIMiddleware — AgentMiddleware that anonymizes LLM inputs and de-anonymizes outputs.

Intercepts every model call via awrap_model_call/wrap_model_call so:

  1. PII is scrubbed from all messages (and the system message) before they
     reach the LLM — Langfuse and any other callbacks attached to the model
     see only tokenized placeholders.
  2. The token map is snapshotted into the shared container registered by
     PIIAwareRunnable so that SSE stream events can be de-anonymized before
     they reach the SSE client.
  3. PII is restored in the model response before LangGraph adds it to state,
     so tools and downstream nodes receive real values.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Awaitable, Callable, cast

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    BaseMessage,
    SystemMessage,
)

from deep_agent.src.pii.scrubber import _ID_LIKE_KEYS
from deep_agent.utils.pylogger import get_python_logger

if TYPE_CHECKING:
    from langchain.agents.middleware.types import ModelRequest, ModelResponse

    from deep_agent.src.pii.scrubber import PIIScrubber

logger = get_python_logger()


def _extract_text(content: Any, max_len: int = 120) -> str:
    if isinstance(content, str):
        return content[:max_len]
    if isinstance(content, list):
        parts = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return " ".join(parts)[:max_len]
    return repr(content)[:max_len]


def _synced_additional_kwargs(
    msg: Any, tool_calls: list[dict]
) -> dict[str, Any] | None:
    """Mirror tool_calls[0].args into additional_kwargs.function_call.arguments."""
    additional_kwargs = getattr(msg, "additional_kwargs", None)
    if not isinstance(additional_kwargs, dict):
        return None
    function_call = additional_kwargs.get("function_call")
    if not isinstance(function_call, dict) or not tool_calls:
        return None
    args = tool_calls[0].get("args", {})
    original_args = function_call.get("arguments")
    if isinstance(original_args, str):
        try:
            new_args: Any = json.dumps(args)
        except Exception:
            new_args = original_args
    else:
        new_args = args
    return {
        **additional_kwargs,
        "function_call": {**function_call, "arguments": new_args},
    }


class PIIMiddleware(AgentMiddleware):
    """AgentMiddleware that scrubs PII from model inputs and restores it in outputs.

    Registered via build_middleware_list() when pii.enabled is true in agent.yaml.
    The global scrubber must be initialised (init_pii_middleware called) before
    this middleware is instantiated.
    """

    def __init__(self, scrubber: "PIIScrubber") -> None:
        """Initialise with a pre-built PIIScrubber."""
        self._scrubber = scrubber

    # ── Input blocking ────────────────────────────────────────────────────

    def _check_input_blocked(self, state: Any) -> Any:
        """Scan the last human message; return a Command if a block-strategy rule matches.

        Blocking is implicit — no separate flag needed. Having any rule with
        strategy: block is sufficient to activate input blocking.
        """
        # Use pre-built block-only detector — skips email/phone/hash rules entirely.
        if not self._scrubber._block_detector:
            return None

        messages = (
            state.get("messages", [])
            if isinstance(state, dict)
            else getattr(state, "messages", [])
        )
        human_msg = next(
            (
                m
                for m in reversed(messages)
                if (
                    getattr(m, "type", None)
                    or (m.get("role") if isinstance(m, dict) else None)
                )
                in ("human", "user")
            ),
            None,
        )
        if not human_msg:
            return None

        content = (
            human_msg.get("content")
            if isinstance(human_msg, dict)
            else getattr(human_msg, "content", "")
        )
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        if not isinstance(content, str) or not content:
            return None

        matches = self._scrubber._block_detector.find_all(content)
        blocked = matches  # all matches are block-strategy by construction
        if not blocked:
            return None

        labels = ", ".join(sorted({m.label for m in blocked}))
        reply = (
            f"I'm unable to process this request as it contains sensitive information "
            f"({labels}) that is restricted by the PII policy. "
            "Please remove the sensitive data and try again."
        )
        from langchain_core.messages import AIMessage
        from langgraph.constants import END
        from langgraph.types import Command

        return Command(
            update={"messages": [AIMessage(content=reply)]},
            goto=END,
        )

    async def abefore_agent(self, state: Any, runtime: Any) -> Any:
        """Block requests containing PII with strategy: block before the agent runs."""
        return self._check_input_blocked(state)

    def before_agent(self, state: Any, runtime: Any) -> Any:
        """Block requests containing PII with strategy: block before the agent runs."""
        return self._check_input_blocked(state)

    # ── Thread-aware setup ────────────────────────────────────────────────

    def _get_thread_id(self) -> str | None:
        try:
            from langgraph.config import get_config

            config = get_config()
            return cast(
                str | None, (config or {}).get("configurable", {}).get("thread_id")
            )
        except Exception:
            return None

    def _setup_scrub(self) -> str | None:
        thread_id = self._get_thread_id()
        if thread_id:
            self._scrubber.load_thread_map(thread_id)
        return thread_id

    def _teardown_scrub(self, thread_id: str | None) -> None:
        if thread_id:
            self._scrubber.save_thread_map(thread_id)
        self._scrubber.snapshot_to_container()
        logger.debug(
            "pii_middleware snapshot tokens=%d",
            len(self._scrubber.snapshot_token_map()),
        )

    # ── Scrubbing helpers ─────────────────────────────────────────────────

    def _scrub_content(self, content: Any) -> Any:
        if isinstance(content, str):
            return self._scrubber.scrub(content)
        if isinstance(content, list):
            return [self._scrub_content_block(block) for block in content]
        return content

    def _scrub_content_block(self, block: Any) -> Any:
        if not isinstance(block, dict):
            return block
        block_type = block.get("type")
        if block_type == "text":
            return {**block, "text": self._scrubber.scrub(block.get("text", ""))}
        if block_type == "image_url":
            image_url = block.get("image_url")
            if isinstance(image_url, dict) and "url" in image_url:
                return {
                    **block,
                    "image_url": {
                        **image_url,
                        "url": self._scrubber.scrub(image_url["url"]),
                    },
                }
        # For any other block type, scrub all top-level string values
        return {
            k: self._scrubber.scrub(v) if isinstance(v, str) else v
            for k, v in block.items()
        }

    def _scrub_tool_args(self, args: dict[str, Any]) -> dict[str, Any]:
        return {
            k: self._scrub_value(v) if k not in _ID_LIKE_KEYS else v
            for k, v in args.items()
        }

    def _scrub_value(self, v: Any) -> Any:
        if isinstance(v, str):
            return self._scrubber.scrub(v)
        if isinstance(v, dict):
            return {
                k: self._scrub_value(val) if k not in _ID_LIKE_KEYS else val
                for k, val in v.items()
            }
        if isinstance(v, list):
            return [self._scrub_value(item) for item in v]
        return v

    def _scrub_message(self, msg: BaseMessage) -> BaseMessage:
        content = self._scrub_content(msg.content)
        kwargs: dict[str, Any] = {"content": content}
        if isinstance(msg, (AIMessage, AIMessageChunk)) and getattr(
            msg, "tool_calls", None
        ):
            scrubbed_tool_calls = [
                {**tc, "args": self._scrub_tool_args(tc.get("args", {}))}
                for tc in msg.tool_calls
            ]
            kwargs["tool_calls"] = scrubbed_tool_calls
            synced = _synced_additional_kwargs(msg, scrubbed_tool_calls)
            if synced is not None:
                kwargs["additional_kwargs"] = synced
        return msg.model_copy(update=kwargs)

    def _scrub_system(self, msg: SystemMessage | None) -> SystemMessage | None:
        if msg is None:
            return None
        content = self._scrub_content(msg.content)
        return msg.model_copy(update={"content": content})

    # ── Restoration helpers (token_map passed explicitly — no ContextVar dependency) ──

    @staticmethod
    def _restore_str(text: str, token_map: dict[str, str]) -> str:
        for token, value in token_map.items():
            if token in text:
                text = text.replace(token, value)
        return text

    @classmethod
    def _restore_content_with_map(cls, content: Any, token_map: dict[str, str]) -> Any:
        if isinstance(content, str):
            return cls._restore_str(content, token_map)
        if isinstance(content, list):
            result = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    result.append(
                        {
                            **block,
                            "text": cls._restore_str(block.get("text", ""), token_map),
                        }
                    )
                else:
                    result.append(block)
            return result
        return content

    @classmethod
    def _restore_tool_args_with_map(
        cls, args: dict[str, Any], token_map: dict[str, str]
    ) -> dict[str, Any]:
        return {
            k: cls._restore_str(v, token_map) if isinstance(v, str) else v
            for k, v in args.items()
        }

    @classmethod
    def _restore_message_with_map(
        cls, msg: BaseMessage, token_map: dict[str, str]
    ) -> BaseMessage:
        if not isinstance(msg, (AIMessage, AIMessageChunk)):
            return msg
        if not token_map:
            return msg
        raw_content = msg.content
        content = cls._restore_content_with_map(raw_content, token_map)
        kwargs: dict[str, Any] = {"content": content}
        if getattr(msg, "tool_calls", None):
            restored_tool_calls = [
                {
                    **tc,
                    "args": cls._restore_tool_args_with_map(
                        tc.get("args", {}), token_map
                    ),
                }
                for tc in msg.tool_calls
            ]
            kwargs["tool_calls"] = restored_tool_calls
            synced = _synced_additional_kwargs(msg, restored_tool_calls)
            if synced is not None:
                kwargs["additional_kwargs"] = synced
        logger.debug(
            "pii_middleware restore content_changed=%s",
            content != raw_content,
        )
        return msg.model_copy(update=kwargs)

    # ── Core middleware hooks ──────────────────────────────────────────────

    async def awrap_model_call(
        self,
        request: "ModelRequest",
        handler: Callable[["ModelRequest"], Awaitable["ModelResponse"]],
    ) -> "ModelResponse":
        """Scrub PII from the model request and restore it in the response (async)."""
        from langchain.agents.middleware.types import ModelResponse

        thread_id = self._setup_scrub()

        scrubbed_messages: list[AnyMessage] = [
            self._scrub_message(m) for m in request.messages
        ]
        scrubbed_system = self._scrub_system(request.system_message)

        # Snapshot NOW — before the async handler crosses any context boundary.
        # self._scrubber.restore() reads a ContextVar which may be empty after
        # await; the snapshot dict is a plain local variable, always available.
        token_map = self._scrubber.snapshot_token_map()
        self._teardown_scrub(thread_id)

        overrides: dict[str, Any] = {"messages": scrubbed_messages}
        if scrubbed_system is not None:
            overrides["system_message"] = scrubbed_system

        scrubbed_request = request.override(**overrides)
        logger.debug(
            "pii_middleware awrap_model_call scrubbed=%d token_map=%d",
            len(scrubbed_messages),
            len(token_map),
        )

        response = await handler(scrubbed_request)

        restored_result = [
            self._restore_message_with_map(m, token_map) for m in response.result
        ]
        return ModelResponse(
            result=restored_result, structured_response=response.structured_response
        )

    def wrap_model_call(
        self,
        request: "ModelRequest",
        handler: Callable[["ModelRequest"], "ModelResponse"],
    ) -> "ModelResponse":
        """Scrub PII from the model request and restore it in the response (sync)."""
        from langchain.agents.middleware.types import ModelResponse

        thread_id = self._setup_scrub()

        scrubbed_messages: list[AnyMessage] = [
            self._scrub_message(m) for m in request.messages
        ]
        scrubbed_system = self._scrub_system(request.system_message)

        token_map = self._scrubber.snapshot_token_map()
        self._teardown_scrub(thread_id)

        overrides: dict[str, Any] = {"messages": scrubbed_messages}
        if scrubbed_system is not None:
            overrides["system_message"] = scrubbed_system

        scrubbed_request = request.override(**overrides)

        response = handler(scrubbed_request)

        restored_result = [
            self._restore_message_with_map(m, token_map) for m in response.result
        ]
        return ModelResponse(
            result=restored_result, structured_response=response.structured_response
        )


def build_pii_middleware() -> PIIMiddleware | None:
    """Build PIIMiddleware from the global scrubber, or None if PII is not active."""
    try:
        from deep_agent.src.pii import get_scrubber

        scrubber = get_scrubber()
        if scrubber is None:
            logger.debug("PIIMiddleware: scrubber not initialised — skipping")
            return None
        return PIIMiddleware(scrubber)
    except Exception as exc:
        logger.warning("PIIMiddleware: failed to build: %s", exc)
        return None
