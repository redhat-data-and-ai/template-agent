"""Unit tests for PIIMiddleware — scrub/restore lifecycle, input blocking, and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)

from deep_agent.src.pii.middleware import (
    PIIMiddleware,
    _extract_text,
    _synced_additional_kwargs,
    build_pii_middleware,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@dataclass
class PIIMatch:
    start: int
    end: int
    value: str
    rule_name: str
    label: str
    action: str


def _mock_scrubber(
    *,
    has_block_detector: bool = False,
    block_matches: list[PIIMatch] | None = None,
) -> MagicMock:
    """Return a mock PIIScrubber with sensible defaults."""
    s = MagicMock()
    s.scrub.side_effect = lambda text: text.replace("user@example.com", "[EMAIL_1]")
    s.snapshot_token_map.return_value = {"[EMAIL_1]": "user@example.com"}
    s.load_thread_map = MagicMock()
    s.save_thread_map = MagicMock()
    s.snapshot_to_container = MagicMock()
    if has_block_detector:
        detector = MagicMock()
        detector.find_all.return_value = block_matches or []
        s._block_detector = detector
    else:
        s._block_detector = None
    return s


def _model_request(
    messages: list[Any],
    system_message: SystemMessage | None = None,
) -> MagicMock:
    """Build a lightweight ModelRequest mock."""
    req = MagicMock()
    req.messages = messages
    req.system_message = system_message
    req.override.side_effect = lambda **kw: MagicMock(
        messages=kw.get("messages", messages),
        system_message=kw.get("system_message", system_message),
        override=req.override,
    )
    return req


def _model_response(
    result: list[Any],
    structured_response: Any = None,
) -> MagicMock:
    resp = MagicMock()
    resp.result = result
    resp.structured_response = structured_response
    return resp


# ═══════════════════════════════════════════════════════════════════════════
# _extract_text
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractText:
    def test_string_truncated(self):
        long_str = "a" * 200
        result = _extract_text(long_str, max_len=120)
        assert len(result) == 120
        assert result == "a" * 120

    def test_string_short_unchanged(self):
        assert _extract_text("hello", max_len=120) == "hello"

    def test_list_extracts_text_blocks(self):
        content = [
            {"type": "text", "text": "Hello"},
            {"type": "image_url", "url": "http://img"},
            {"type": "text", "text": "World"},
        ]
        assert _extract_text(content) == "Hello World"

    def test_list_truncated(self):
        content = [{"type": "text", "text": "x" * 200}]
        result = _extract_text(content, max_len=50)
        assert len(result) == 50

    def test_list_skips_non_text(self):
        content = [{"type": "image", "url": "http://img"}]
        assert _extract_text(content) == ""

    def test_non_string_non_list_returns_repr(self):
        result = _extract_text(42)
        assert result == "42"

    def test_non_string_non_list_repr_truncated(self):
        result = _extract_text({"key": "v" * 300}, max_len=20)
        assert len(result) == 20

    def test_custom_max_len(self):
        assert len(_extract_text("a" * 50, max_len=10)) == 10


# ═══════════════════════════════════════════════════════════════════════════
# _synced_additional_kwargs
# ═══════════════════════════════════════════════════════════════════════════


class TestSyncedAdditionalKwargs:
    def test_returns_updated_kwargs_with_function_call(self):
        msg = MagicMock()
        msg.additional_kwargs = {
            "function_call": {"name": "fn", "arguments": '{"a": 1}'},
            "extra": "keep",
        }
        tool_calls = [{"args": {"a": 2}}]
        result = _synced_additional_kwargs(msg, tool_calls)
        assert result is not None
        assert result["extra"] == "keep"
        assert result["function_call"]["name"] == "fn"
        assert result["function_call"]["arguments"] == '{"a": 2}'

    def test_returns_none_when_no_additional_kwargs(self):
        msg = MagicMock(spec=[])
        assert _synced_additional_kwargs(msg, [{"args": {}}]) is None

    def test_returns_none_when_additional_kwargs_not_dict(self):
        msg = MagicMock()
        msg.additional_kwargs = "not-a-dict"
        assert _synced_additional_kwargs(msg, [{"args": {}}]) is None

    def test_returns_none_when_no_function_call(self):
        msg = MagicMock()
        msg.additional_kwargs = {"other": "value"}
        assert _synced_additional_kwargs(msg, [{"args": {}}]) is None

    def test_returns_none_when_empty_tool_calls(self):
        msg = MagicMock()
        msg.additional_kwargs = {"function_call": {"name": "fn", "arguments": "{}"}}
        assert _synced_additional_kwargs(msg, []) is None

    def test_non_string_original_args(self):
        msg = MagicMock()
        msg.additional_kwargs = {
            "function_call": {"name": "fn", "arguments": {"a": 1}},
        }
        tool_calls = [{"args": {"a": 2}}]
        result = _synced_additional_kwargs(msg, tool_calls)
        assert result is not None
        assert result["function_call"]["arguments"] == {"a": 2}

    def test_function_call_not_dict(self):
        msg = MagicMock()
        msg.additional_kwargs = {"function_call": "not-a-dict"}
        assert _synced_additional_kwargs(msg, [{"args": {}}]) is None


# ═══════════════════════════════════════════════════════════════════════════
# _check_input_blocked
# ═══════════════════════════════════════════════════════════════════════════


class TestCheckInputBlocked:
    def test_no_block_detector_returns_none(self):
        mw = PIIMiddleware(_mock_scrubber(has_block_detector=False))
        assert mw._check_input_blocked({"messages": []}) is None

    def test_no_messages_returns_none(self):
        mw = PIIMiddleware(_mock_scrubber(has_block_detector=True))
        assert mw._check_input_blocked({"messages": []}) is None

    def test_no_human_message_returns_none(self):
        match = PIIMatch(
            start=0, end=5, value="12345", rule_name="ssn", label="SSN", action="block"
        )
        scrubber = _mock_scrubber(has_block_detector=True, block_matches=[match])
        mw = PIIMiddleware(scrubber)
        ai = AIMessage(content="hello")
        assert mw._check_input_blocked({"messages": [ai]}) is None
        scrubber._block_detector.find_all.assert_not_called()

    def test_human_message_without_pii_returns_none(self):
        mw = PIIMiddleware(_mock_scrubber(has_block_detector=True, block_matches=[]))
        human = HumanMessage(content="just a normal message")
        assert mw._check_input_blocked({"messages": [human]}) is None

    def test_human_message_with_block_pii_returns_command(self):
        match = PIIMatch(
            start=0, end=5, value="12345", rule_name="ssn", label="SSN", action="block"
        )
        mw = PIIMiddleware(
            _mock_scrubber(has_block_detector=True, block_matches=[match])
        )
        human = HumanMessage(content="my ssn is 12345")
        result = mw._check_input_blocked({"messages": [human]})
        assert result is not None
        assert result.goto is not None
        msgs = result.update["messages"]
        assert len(msgs) == 1
        assert "SSN" in msgs[0].content
        assert "sensitive information" in msgs[0].content

    def test_handles_dict_style_messages(self):
        match = PIIMatch(
            start=0,
            end=5,
            value="secret",
            rule_name="r",
            label="SECRET",
            action="block",
        )
        mw = PIIMiddleware(
            _mock_scrubber(has_block_detector=True, block_matches=[match])
        )
        state = {
            "messages": [
                {"role": "user", "content": "secret data"},
            ]
        }
        result = mw._check_input_blocked(state)
        assert result is not None
        assert "SECRET" in result.update["messages"][0].content

    def test_handles_list_content(self):
        match = PIIMatch(
            start=0,
            end=5,
            value="secret",
            rule_name="r",
            label="SECRET",
            action="block",
        )
        mw = PIIMiddleware(
            _mock_scrubber(has_block_detector=True, block_matches=[match])
        )
        human = HumanMessage(content=[{"type": "text", "text": "secret data"}])
        result = mw._check_input_blocked({"messages": [human]})
        assert result is not None

    def test_empty_string_content_returns_none(self):
        match = PIIMatch(
            start=0, end=5, value="12345", rule_name="ssn", label="SSN", action="block"
        )
        scrubber = _mock_scrubber(has_block_detector=True, block_matches=[match])
        mw = PIIMiddleware(scrubber)
        human = HumanMessage(content="")
        assert mw._check_input_blocked({"messages": [human]}) is None
        scrubber._block_detector.find_all.assert_not_called()

    def test_state_as_object_with_messages_attr(self):
        scrubber = _mock_scrubber(has_block_detector=True, block_matches=[])
        mw = PIIMiddleware(scrubber)
        state = MagicMock()
        state.messages = [HumanMessage(content="hi")]
        assert mw._check_input_blocked(state) is None

    def test_picks_last_human_message(self):
        match = PIIMatch(
            start=0, end=5, value="ssn", rule_name="r", label="SSN", action="block"
        )
        mw = PIIMiddleware(
            _mock_scrubber(has_block_detector=True, block_matches=[match])
        )
        h1 = HumanMessage(content="first")
        h2 = HumanMessage(content="my ssn is 12345")
        result = mw._check_input_blocked({"messages": [h1, h2]})
        assert result is not None

    @pytest.mark.asyncio
    async def test_abefore_agent_delegates(self):
        mw = PIIMiddleware(_mock_scrubber(has_block_detector=False))
        assert await mw.abefore_agent({"messages": []}, None) is None

    def test_before_agent_delegates(self):
        mw = PIIMiddleware(_mock_scrubber(has_block_detector=False))
        assert mw.before_agent({"messages": []}, None) is None


# ═══════════════════════════════════════════════════════════════════════════
# Thread-aware setup / teardown
# ═══════════════════════════════════════════════════════════════════════════


class TestThreadSetup:
    @patch(
        "deep_agent.src.pii.middleware.PIIMiddleware._get_thread_id", return_value="t1"
    )
    def test_setup_scrub_loads_thread_map(self, _get_tid):
        scrubber = _mock_scrubber()
        mw = PIIMiddleware(scrubber)
        tid = mw._setup_scrub()
        assert tid == "t1"
        scrubber.load_thread_map.assert_called_once_with("t1")

    @patch(
        "deep_agent.src.pii.middleware.PIIMiddleware._get_thread_id", return_value=None
    )
    def test_setup_scrub_without_thread_id(self, _get_tid):
        scrubber = _mock_scrubber()
        mw = PIIMiddleware(scrubber)
        tid = mw._setup_scrub()
        assert tid is None
        scrubber.load_thread_map.assert_not_called()

    def test_teardown_scrub_saves_when_thread_id(self):
        scrubber = _mock_scrubber()
        mw = PIIMiddleware(scrubber)
        mw._teardown_scrub("t1")
        scrubber.save_thread_map.assert_called_once_with("t1")
        scrubber.snapshot_to_container.assert_called_once()

    def test_teardown_scrub_skips_save_without_thread_id(self):
        scrubber = _mock_scrubber()
        mw = PIIMiddleware(scrubber)
        mw._teardown_scrub(None)
        scrubber.save_thread_map.assert_not_called()
        scrubber.snapshot_to_container.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# _scrub_content
# ═══════════════════════════════════════════════════════════════════════════


class TestScrubContent:
    def test_scrubs_string(self):
        mw = PIIMiddleware(_mock_scrubber())
        assert mw._scrub_content("user@example.com") == "[EMAIL_1]"

    def test_scrubs_list_text_blocks(self):
        mw = PIIMiddleware(_mock_scrubber())
        blocks = [{"type": "text", "text": "user@example.com"}]
        result = mw._scrub_content(blocks)
        assert result[0]["text"] == "[EMAIL_1]"

    def test_scrubs_list_image_url_block(self):
        scrubber = _mock_scrubber()
        scrubber.scrub.side_effect = lambda t: t.replace(
            "http://example.com/user@example.com", "[URL_1]"
        )
        mw = PIIMiddleware(scrubber)
        blocks = [
            {
                "type": "image_url",
                "image_url": {"url": "http://example.com/user@example.com"},
            }
        ]
        result = mw._scrub_content(blocks)
        assert result[0]["image_url"]["url"] == "[URL_1]"

    def test_scrubs_unknown_block_type_string_values(self):
        mw = PIIMiddleware(_mock_scrubber())
        blocks = [{"type": "custom", "data": "user@example.com", "num": 42}]
        result = mw._scrub_content(blocks)
        assert result[0]["data"] == "[EMAIL_1]"
        assert result[0]["num"] == 42

    def test_non_string_non_list_passthrough(self):
        mw = PIIMiddleware(_mock_scrubber())
        assert mw._scrub_content(42) == 42

    def test_non_dict_block_passthrough(self):
        mw = PIIMiddleware(_mock_scrubber())
        result = mw._scrub_content(["just a string"])
        assert result == ["just a string"]


# ═══════════════════════════════════════════════════════════════════════════
# _scrub_tool_args / _scrub_value
# ═══════════════════════════════════════════════════════════════════════════


class TestScrubToolArgs:
    def test_scrubs_string_values(self):
        mw = PIIMiddleware(_mock_scrubber())
        result = mw._scrub_tool_args({"query": "user@example.com", "count": 5})
        assert result["query"] == "[EMAIL_1]"
        assert result["count"] == 5

    def test_skips_id_like_keys(self):
        mw = PIIMiddleware(_mock_scrubber())
        result = mw._scrub_tool_args(
            {
                "id": "user@example.com",
                "run_id": "user@example.com",
                "tool_call_id": "user@example.com",
                "thread_id": "user@example.com",
                "query": "user@example.com",
            }
        )
        assert result["id"] == "user@example.com"
        assert result["run_id"] == "user@example.com"
        assert result["tool_call_id"] == "user@example.com"
        assert result["thread_id"] == "user@example.com"
        assert result["query"] == "[EMAIL_1]"

    def test_scrubs_nested_dict(self):
        mw = PIIMiddleware(_mock_scrubber())
        result = mw._scrub_value({"inner": "user@example.com"})
        assert result["inner"] == "[EMAIL_1]"

    def test_scrubs_nested_list(self):
        mw = PIIMiddleware(_mock_scrubber())
        result = mw._scrub_value(["user@example.com", 42])
        assert result == ["[EMAIL_1]", 42]

    def test_scrub_value_non_string_passthrough(self):
        mw = PIIMiddleware(_mock_scrubber())
        assert mw._scrub_value(123) == 123
        assert mw._scrub_value(None) is None

    def test_scrub_value_skips_id_keys_in_nested_dict(self):
        mw = PIIMiddleware(_mock_scrubber())
        result = mw._scrub_value({"id": "user@example.com", "name": "user@example.com"})
        assert result["id"] == "user@example.com"
        assert result["name"] == "[EMAIL_1]"


# ═══════════════════════════════════════════════════════════════════════════
# _scrub_message
# ═══════════════════════════════════════════════════════════════════════════


class TestScrubMessage:
    def test_scrubs_ai_message_content(self):
        mw = PIIMiddleware(_mock_scrubber())
        msg = AIMessage(content="user@example.com")
        result = mw._scrub_message(msg)
        assert result.content == "[EMAIL_1]"

    def test_scrubs_tool_calls_args(self):
        mw = PIIMiddleware(_mock_scrubber())
        msg = AIMessage(
            content="ok",
            tool_calls=[
                {"name": "search", "args": {"q": "user@example.com"}, "id": "tc1"}
            ],
        )
        result = mw._scrub_message(msg)
        assert result.tool_calls[0]["args"]["q"] == "[EMAIL_1]"

    def test_handles_message_without_tool_calls(self):
        mw = PIIMiddleware(_mock_scrubber())
        msg = HumanMessage(content="user@example.com")
        result = mw._scrub_message(msg)
        assert result.content == "[EMAIL_1]"

    def test_scrubs_ai_message_chunk(self):
        mw = PIIMiddleware(_mock_scrubber())
        msg = AIMessageChunk(content="user@example.com")
        result = mw._scrub_message(msg)
        assert result.content == "[EMAIL_1]"

    def test_scrubs_system_message(self):
        mw = PIIMiddleware(_mock_scrubber())
        msg = SystemMessage(content="user@example.com")
        result = mw._scrub_system(msg)
        assert result.content == "[EMAIL_1]"

    def test_scrub_system_none(self):
        mw = PIIMiddleware(_mock_scrubber())
        assert mw._scrub_system(None) is None

    def test_synced_additional_kwargs_in_scrub_message(self):
        mw = PIIMiddleware(_mock_scrubber())
        msg = AIMessage(
            content="",
            additional_kwargs={
                "function_call": {
                    "name": "fn",
                    "arguments": '{"q": "user@example.com"}',
                }
            },
            tool_calls=[{"name": "fn", "args": {"q": "user@example.com"}, "id": "tc1"}],
        )
        result = mw._scrub_message(msg)
        ak = result.additional_kwargs
        assert "[EMAIL_1]" in ak["function_call"]["arguments"]


# ═══════════════════════════════════════════════════════════════════════════
# _restore_str / _restore_content_with_map / _restore_tool_args_with_map
# ═══════════════════════════════════════════════════════════════════════════


class TestRestoreStr:
    def test_replaces_tokens(self):
        result = PIIMiddleware._restore_str(
            "Contact [EMAIL_1] please",
            {"[EMAIL_1]": "user@example.com"},
        )
        assert result == "Contact user@example.com please"

    def test_no_tokens_unchanged(self):
        text = "no tokens here"
        assert PIIMiddleware._restore_str(text, {"[X]": "y"}) == text

    def test_multiple_tokens(self):
        result = PIIMiddleware._restore_str(
            "[EMAIL_1] and [PHONE_1]",
            {"[EMAIL_1]": "a@b.com", "[PHONE_1]": "555-1234"},
        )
        assert result == "a@b.com and 555-1234"

    def test_empty_token_map(self):
        assert PIIMiddleware._restore_str("hello", {}) == "hello"


class TestRestoreContentWithMap:
    def test_restores_string(self):
        result = PIIMiddleware._restore_content_with_map(
            "[EMAIL_1]", {"[EMAIL_1]": "a@b.com"}
        )
        assert result == "a@b.com"

    def test_restores_list_text_blocks(self):
        content = [
            {"type": "text", "text": "hi [EMAIL_1]"},
            {"type": "image_url", "url": "http://x"},
        ]
        result = PIIMiddleware._restore_content_with_map(
            content, {"[EMAIL_1]": "a@b.com"}
        )
        assert result[0]["text"] == "hi a@b.com"
        assert result[1] == {"type": "image_url", "url": "http://x"}

    def test_non_string_non_list_passthrough(self):
        assert PIIMiddleware._restore_content_with_map(42, {"[X]": "y"}) == 42


class TestRestoreToolArgsWithMap:
    def test_restores_string_values(self):
        result = PIIMiddleware._restore_tool_args_with_map(
            {"q": "[EMAIL_1]", "count": 5},
            {"[EMAIL_1]": "a@b.com"},
        )
        assert result["q"] == "a@b.com"
        assert result["count"] == 5


# ═══════════════════════════════════════════════════════════════════════════
# _restore_message_with_map
# ═══════════════════════════════════════════════════════════════════════════


class TestRestoreMessageWithMap:
    def test_restores_ai_message_content(self):
        msg = AIMessage(content="[EMAIL_1]")
        result = PIIMiddleware._restore_message_with_map(msg, {"[EMAIL_1]": "a@b.com"})
        assert result.content == "a@b.com"

    def test_restores_tool_calls(self):
        msg = AIMessage(
            content="ok",
            tool_calls=[{"name": "fn", "args": {"q": "[EMAIL_1]"}, "id": "tc1"}],
        )
        result = PIIMiddleware._restore_message_with_map(msg, {"[EMAIL_1]": "a@b.com"})
        assert result.tool_calls[0]["args"]["q"] == "a@b.com"

    def test_non_ai_message_returned_as_is(self):
        msg = HumanMessage(content="[EMAIL_1]")
        result = PIIMiddleware._restore_message_with_map(msg, {"[EMAIL_1]": "a@b.com"})
        assert result is msg
        assert result.content == "[EMAIL_1]"

    def test_empty_token_map_returns_original(self):
        msg = AIMessage(content="[EMAIL_1]")
        result = PIIMiddleware._restore_message_with_map(msg, {})
        assert result is msg

    def test_restores_ai_message_chunk(self):
        msg = AIMessageChunk(content="[EMAIL_1]")
        result = PIIMiddleware._restore_message_with_map(msg, {"[EMAIL_1]": "a@b.com"})
        assert result.content == "a@b.com"

    def test_system_message_returned_as_is(self):
        msg = SystemMessage(content="[EMAIL_1]")
        result = PIIMiddleware._restore_message_with_map(msg, {"[EMAIL_1]": "a@b.com"})
        assert result is msg

    def test_restores_with_synced_additional_kwargs(self):
        msg = AIMessage(
            content="ok",
            additional_kwargs={
                "function_call": {"name": "fn", "arguments": '{"q": "[EMAIL_1]"}'}
            },
            tool_calls=[{"name": "fn", "args": {"q": "[EMAIL_1]"}, "id": "tc1"}],
        )
        result = PIIMiddleware._restore_message_with_map(msg, {"[EMAIL_1]": "a@b.com"})
        assert result.tool_calls[0]["args"]["q"] == "a@b.com"
        assert "a@b.com" in result.additional_kwargs["function_call"]["arguments"]


# ═══════════════════════════════════════════════════════════════════════════
# awrap_model_call (async)
# ═══════════════════════════════════════════════════════════════════════════


class TestAwrapModelCall:
    @pytest.mark.asyncio
    @patch(
        "deep_agent.src.pii.middleware.PIIMiddleware._get_thread_id", return_value="t1"
    )
    async def test_scrubs_request_calls_handler_restores_response(self, _get_tid):
        scrubber = _mock_scrubber()
        mw = PIIMiddleware(scrubber)

        ai_msg = AIMessage(content="[EMAIL_1]")
        handler_response = _model_response(result=[ai_msg])

        handler = AsyncMock(return_value=handler_response)
        request = _model_request(messages=[HumanMessage(content="user@example.com")])

        with patch(
            "langchain.agents.middleware.types.ModelResponse",
            side_effect=lambda result, structured_response: _model_response(
                result, structured_response
            ),
        ):
            response = await mw.awrap_model_call(request, handler)

        handler.assert_awaited_once()
        scrubbed_req = handler.call_args[0][0]
        assert scrubbed_req.messages[0].content == "[EMAIL_1]"

        assert response.result[0].content == "user@example.com"
        scrubber.load_thread_map.assert_called_once_with("t1")
        scrubber.save_thread_map.assert_called_once_with("t1")
        scrubber.snapshot_to_container.assert_called_once()

    @pytest.mark.asyncio
    @patch(
        "deep_agent.src.pii.middleware.PIIMiddleware._get_thread_id", return_value=None
    )
    async def test_handles_system_message(self, _get_tid):
        scrubber = _mock_scrubber()
        mw = PIIMiddleware(scrubber)

        ai_msg = AIMessage(content="done")
        handler_response = _model_response(result=[ai_msg])
        handler = AsyncMock(return_value=handler_response)
        request = _model_request(
            messages=[HumanMessage(content="hi")],
            system_message=SystemMessage(content="You are user@example.com"),
        )

        with patch(
            "langchain.agents.middleware.types.ModelResponse",
            side_effect=lambda result, structured_response: _model_response(
                result, structured_response
            ),
        ):
            await mw.awrap_model_call(request, handler)

        scrubbed_req = handler.call_args[0][0]
        assert scrubbed_req.system_message.content == "You are [EMAIL_1]"

    @pytest.mark.asyncio
    @patch(
        "deep_agent.src.pii.middleware.PIIMiddleware._get_thread_id", return_value=None
    )
    async def test_no_system_message(self, _get_tid):
        scrubber = _mock_scrubber()
        mw = PIIMiddleware(scrubber)

        handler_response = _model_response(result=[AIMessage(content="ok")])
        handler = AsyncMock(return_value=handler_response)
        request = _model_request(messages=[HumanMessage(content="hi")])

        with patch(
            "langchain.agents.middleware.types.ModelResponse",
            side_effect=lambda result, structured_response: _model_response(
                result, structured_response
            ),
        ):
            await mw.awrap_model_call(request, handler)

        call_kwargs = request.override.call_args[1]
        assert "system_message" not in call_kwargs

    @pytest.mark.asyncio
    @patch(
        "deep_agent.src.pii.middleware.PIIMiddleware._get_thread_id", return_value=None
    )
    async def test_structured_response_preserved(self, _get_tid):
        scrubber = _mock_scrubber()
        mw = PIIMiddleware(scrubber)

        handler_response = _model_response(
            result=[AIMessage(content="ok")],
            structured_response={"key": "value"},
        )
        handler = AsyncMock(return_value=handler_response)
        request = _model_request(messages=[HumanMessage(content="hi")])

        captured = {}

        def mock_model_response(result, structured_response):
            captured["structured_response"] = structured_response
            return _model_response(result, structured_response)

        with patch(
            "langchain.agents.middleware.types.ModelResponse",
            side_effect=mock_model_response,
        ):
            await mw.awrap_model_call(request, handler)

        assert captured["structured_response"] == {"key": "value"}


# ═══════════════════════════════════════════════════════════════════════════
# wrap_model_call (sync)
# ═══════════════════════════════════════════════════════════════════════════


class TestWrapModelCall:
    @patch(
        "deep_agent.src.pii.middleware.PIIMiddleware._get_thread_id", return_value="t1"
    )
    def test_sync_scrubs_and_restores(self, _get_tid):
        scrubber = _mock_scrubber()
        mw = PIIMiddleware(scrubber)

        ai_msg = AIMessage(content="[EMAIL_1]")
        handler_response = _model_response(result=[ai_msg])
        handler = MagicMock(return_value=handler_response)
        request = _model_request(messages=[HumanMessage(content="user@example.com")])

        with patch(
            "langchain.agents.middleware.types.ModelResponse",
            side_effect=lambda result, structured_response: _model_response(
                result, structured_response
            ),
        ):
            response = mw.wrap_model_call(request, handler)

        handler.assert_called_once()
        scrubbed_req = handler.call_args[0][0]
        assert scrubbed_req.messages[0].content == "[EMAIL_1]"
        assert response.result[0].content == "user@example.com"

    @patch(
        "deep_agent.src.pii.middleware.PIIMiddleware._get_thread_id", return_value=None
    )
    def test_sync_handles_system_message(self, _get_tid):
        scrubber = _mock_scrubber()
        mw = PIIMiddleware(scrubber)

        handler_response = _model_response(result=[AIMessage(content="done")])
        handler = MagicMock(return_value=handler_response)
        request = _model_request(
            messages=[HumanMessage(content="hi")],
            system_message=SystemMessage(content="You are user@example.com"),
        )

        with patch(
            "langchain.agents.middleware.types.ModelResponse",
            side_effect=lambda result, structured_response: _model_response(
                result, structured_response
            ),
        ):
            mw.wrap_model_call(request, handler)

        scrubbed_req = handler.call_args[0][0]
        assert scrubbed_req.system_message.content == "You are [EMAIL_1]"

    @patch(
        "deep_agent.src.pii.middleware.PIIMiddleware._get_thread_id", return_value="t1"
    )
    def test_sync_teardown_saves_thread_map(self, _get_tid):
        scrubber = _mock_scrubber()
        mw = PIIMiddleware(scrubber)

        handler_response = _model_response(result=[AIMessage(content="ok")])
        handler = MagicMock(return_value=handler_response)
        request = _model_request(messages=[HumanMessage(content="hello")])

        with patch(
            "langchain.agents.middleware.types.ModelResponse",
            side_effect=lambda result, structured_response: _model_response(
                result, structured_response
            ),
        ):
            mw.wrap_model_call(request, handler)

        scrubber.load_thread_map.assert_called_once_with("t1")
        scrubber.save_thread_map.assert_called_once_with("t1")
        scrubber.snapshot_to_container.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# build_pii_middleware
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildPiiMiddleware:
    @patch("deep_agent.src.pii.get_scrubber")
    def test_returns_middleware_when_scrubber_available(self, mock_get):
        mock_get.return_value = _mock_scrubber()
        result = build_pii_middleware()
        assert isinstance(result, PIIMiddleware)

    @patch("deep_agent.src.pii.get_scrubber")
    def test_returns_none_when_no_scrubber(self, mock_get):
        mock_get.return_value = None
        assert build_pii_middleware() is None

    @patch("deep_agent.src.pii.get_scrubber")
    def test_returns_none_on_import_error(self, mock_get):
        mock_get.side_effect = ImportError("no module")
        assert build_pii_middleware() is None

    @patch("deep_agent.src.pii.get_scrubber")
    def test_returns_none_on_runtime_error(self, mock_get):
        mock_get.side_effect = RuntimeError("boom")
        assert build_pii_middleware() is None
