"""Unit tests for OPA authorization middleware.

Covers _NoStreamModel proxy, text extraction helpers, and the three
middleware hooks: abefore_model, awrap_model_call, awrap_tool_call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from deep_agent.src.opa.middleware import (
    OPAMiddleware,
    _BLOCKED_MODEL_MESSAGE,
    _BLOCKED_TOOL_MESSAGE,
    _BLOCKED_TRAJECTORY_MESSAGE,
    _NoStreamModel,
)
from deep_agent.src.opa.service import OpaResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_response(messages: list | None = None):
    """Build a lightweight ModelResponse-like object."""
    resp = MagicMock()
    resp.result = messages or []
    return resp


def _model_request(model=None, messages=None):
    """Build a lightweight ModelRequest-like object that supports .override()."""
    req = MagicMock()
    req.model = model or MagicMock()
    req.messages = messages or []

    def _make_override(parent):
        def _override(**kwargs):
            new_req = MagicMock()
            new_req.model = kwargs.get("model", parent.model)
            new_req.messages = kwargs.get("messages", parent.messages)
            new_req.override = _make_override(new_req)
            return new_req

        return _override

    req.override = _make_override(req)
    return req


def _tool_call_request(tool_call=None):
    """Build a lightweight ToolCallRequest-like object."""
    req = MagicMock()
    req.tool_call = tool_call or {"name": "test_tool", "id": "call_123"}
    return req


# ===================================================================
# _NoStreamModel
# ===================================================================


class TestNoStreamModel:
    """_NoStreamModel delegates to the wrapped model and re-applies TAG_NOSTREAM."""

    def test_bind_tools_delegates_and_tags(self):
        inner = MagicMock()
        bound = MagicMock()
        tagged = MagicMock()
        inner.bind_tools.return_value = bound
        bound.with_config.return_value = tagged

        proxy = _NoStreamModel(inner)
        result = proxy.bind_tools("a", key="b")

        inner.bind_tools.assert_called_once_with("a", key="b")
        bound.with_config.assert_called_once()
        call_kwargs = bound.with_config.call_args
        assert "nostream" in call_kwargs.kwargs["tags"][0]
        assert result is tagged

    def test_bind_delegates_and_tags(self):
        inner = MagicMock()
        bound = MagicMock()
        tagged = MagicMock()
        inner.bind.return_value = bound
        bound.with_config.return_value = tagged

        proxy = _NoStreamModel(inner)
        result = proxy.bind("x", y=1)

        inner.bind.assert_called_once_with("x", y=1)
        bound.with_config.assert_called_once()
        call_kwargs = bound.with_config.call_args
        assert "nostream" in call_kwargs.kwargs["tags"][0]
        assert result is tagged

    def test_getattr_delegates_to_wrapped_model(self):
        inner = MagicMock()
        inner.temperature = 0.7
        inner.some_method.return_value = "delegated"

        proxy = _NoStreamModel(inner)

        assert proxy.temperature == 0.7
        assert proxy.some_method() == "delegated"

    def test_getattr_raises_attribute_error_for_missing(self):
        inner = MagicMock(spec=[])
        proxy = _NoStreamModel(inner)

        with pytest.raises(AttributeError):
            _ = proxy.nonexistent_attr


# ===================================================================
# _extract_text
# ===================================================================


class TestExtractText:
    """OPAMiddleware._extract_text extracts text from the last AIMessage."""

    def setup_method(self):
        self.mw = OPAMiddleware()

    def test_returns_text_from_last_ai_message(self):
        response = _model_response(
            [
                AIMessage(content="first"),
                AIMessage(content="second"),
            ]
        )
        assert self.mw._extract_text(response) == "second"

    def test_skips_non_ai_messages(self):
        response = _model_response(
            [
                HumanMessage(content="human"),
                ToolMessage(content="tool result", tool_call_id="tc1"),
                AIMessage(content="ai response"),
            ]
        )
        assert self.mw._extract_text(response) == "ai response"

    def test_returns_empty_string_when_no_ai_messages(self):
        response = _model_response(
            [
                HumanMessage(content="human"),
                ToolMessage(content="tool", tool_call_id="tc1"),
            ]
        )
        assert self.mw._extract_text(response) == ""

    def test_returns_empty_string_when_empty_result(self):
        response = _model_response([])
        assert self.mw._extract_text(response) == ""

    def test_returns_empty_string_when_all_ai_messages_empty(self):
        response = _model_response(
            [
                AIMessage(content=""),
                AIMessage(content=""),
            ]
        )
        assert self.mw._extract_text(response) == ""

    def test_handles_ai_message_with_content_blocks(self):
        msg = AIMessage(content=[{"type": "text", "text": "block content"}])
        response = _model_response([msg])
        assert self.mw._extract_text(response) == "block content"

    def test_skips_empty_ai_and_returns_earlier_nonempty(self):
        response = _model_response(
            [
                AIMessage(content="good"),
                AIMessage(content=""),
            ]
        )
        assert self.mw._extract_text(response) == "good"


# ===================================================================
# abefore_model
# ===================================================================


class TestAbforeModel:
    """OPAMiddleware.abefore_model evaluates trajectory before model calls."""

    def setup_method(self):
        self.mw = OPAMiddleware()

    @pytest.mark.asyncio
    async def test_empty_messages_returns_none(self):
        result = await self.mw.abefore_model({"messages": []}, MagicMock())
        assert result is None

    @pytest.mark.asyncio
    async def test_missing_messages_key_returns_none(self):
        result = await self.mw.abefore_model({}, MagicMock())
        assert result is None

    @pytest.mark.asyncio
    @patch("deep_agent.src.opa.middleware.evaluate_trajectory", new_callable=AsyncMock)
    async def test_opa_allows_returns_none(self, mock_eval):
        mock_eval.return_value = OpaResult(allowed=True)
        state = {"messages": [HumanMessage(content="hello")]}

        result = await self.mw.abefore_model(state, MagicMock())

        assert result is None
        mock_eval.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("deep_agent.src.opa.middleware.evaluate_trajectory", new_callable=AsyncMock)
    async def test_opa_denies_returns_jump_to_end(self, mock_eval):
        mock_eval.return_value = OpaResult(
            allowed=False, denial_reasons=["harmful content"]
        )
        state = {"messages": [HumanMessage(content="bad input")]}

        result = await self.mw.abefore_model(state, MagicMock())

        assert result is not None
        assert result["jump_to"] == "end"
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == _BLOCKED_TRAJECTORY_MESSAGE

    @pytest.mark.asyncio
    @patch("deep_agent.src.opa.middleware.evaluate_trajectory", new_callable=AsyncMock)
    async def test_filters_opa_retry_messages(self, mock_eval):
        mock_eval.return_value = OpaResult(allowed=True)
        retry_msg = HumanMessage(
            content="retry feedback",
            additional_kwargs={"opa_retry": True},
        )
        normal_msg = HumanMessage(content="normal")
        state = {"messages": [normal_msg, retry_msg]}

        await self.mw.abefore_model(state, MagicMock())

        trajectory = mock_eval.call_args[0][0]
        assert len(trajectory) == 1
        assert trajectory[0].content == "normal"

    @pytest.mark.asyncio
    @patch("deep_agent.src.opa.middleware.evaluate_trajectory", new_callable=AsyncMock)
    async def test_non_base_messages_filtered_from_trajectory(self, mock_eval):
        mock_eval.return_value = OpaResult(allowed=True)
        state = {"messages": [HumanMessage(content="ok"), "not_a_message"]}

        await self.mw.abefore_model(state, MagicMock())

        trajectory = mock_eval.call_args[0][0]
        assert len(trajectory) == 1

    @pytest.mark.asyncio
    @patch("deep_agent.src.opa.middleware.evaluate_trajectory", new_callable=AsyncMock)
    async def test_denies_with_empty_reasons(self, mock_eval):
        mock_eval.return_value = OpaResult(allowed=False, denial_reasons=[])
        state = {"messages": [HumanMessage(content="input")]}

        result = await self.mw.abefore_model(state, MagicMock())

        assert result is not None
        assert result["jump_to"] == "end"


# ===================================================================
# awrap_model_call
# ===================================================================


class TestAwrapModelCall:
    """OPAMiddleware.awrap_model_call generates silently and checks OPA."""

    def setup_method(self):
        self.mw = OPAMiddleware()

    @pytest.mark.asyncio
    @patch("deep_agent.src.opa.middleware.get_opa_max_retries", return_value=2)
    @patch("deep_agent.src.opa.middleware.evaluate_message", new_callable=AsyncMock)
    async def test_opa_allows_returns_original(self, mock_eval, _mock_retries):
        mock_eval.return_value = OpaResult(allowed=True)
        ai_msg = AIMessage(content="safe answer", id="msg1", name="assistant")
        response = _model_response([ai_msg])
        handler = AsyncMock(return_value=response)
        request = _model_request(messages=[HumanMessage(content="hi")])

        result = await self.mw.awrap_model_call(request, handler)

        assert result is response
        mock_eval.assert_awaited_once_with("llm_response", agent_message="safe answer")

    @pytest.mark.asyncio
    @patch("deep_agent.src.opa.middleware.get_opa_max_retries", return_value=1)
    @patch("deep_agent.src.opa.middleware.evaluate_message", new_callable=AsyncMock)
    async def test_opa_denies_after_max_retries_returns_blocked(
        self, mock_eval, _mock_retries
    ):
        mock_eval.return_value = OpaResult(allowed=False, denial_reasons=["toxic"])
        ai_msg = AIMessage(content="bad output", id="orig_id", name="orig_name")
        response = _model_response([ai_msg])
        handler = AsyncMock(return_value=response)
        request = _model_request(messages=[HumanMessage(content="prompt")])

        result = await self.mw.awrap_model_call(request, handler)

        assert len(result.result) == 1
        blocked_msg = result.result[0]
        assert blocked_msg.content == _BLOCKED_MODEL_MESSAGE
        assert blocked_msg.id == "orig_id"
        assert blocked_msg.name == "orig_name"
        assert mock_eval.await_count == 2  # initial + 1 retry

    @pytest.mark.asyncio
    @patch("deep_agent.src.opa.middleware.get_opa_max_retries", return_value=2)
    @patch("deep_agent.src.opa.middleware.evaluate_message", new_callable=AsyncMock)
    async def test_empty_text_returns_without_opa_check(self, mock_eval, _mock_retries):
        ai_msg = AIMessage(content="")
        response = _model_response([ai_msg])
        handler = AsyncMock(return_value=response)
        request = _model_request()

        result = await self.mw.awrap_model_call(request, handler)

        assert result is response
        mock_eval.assert_not_awaited()

    @pytest.mark.asyncio
    @patch("deep_agent.src.opa.middleware.get_opa_max_retries", return_value=2)
    @patch("deep_agent.src.opa.middleware.evaluate_message", new_callable=AsyncMock)
    async def test_whitespace_only_text_returns_without_opa_check(
        self, mock_eval, _mock_retries
    ):
        ai_msg = AIMessage(content="   \n  ")
        response = _model_response([ai_msg])
        handler = AsyncMock(return_value=response)
        request = _model_request()

        result = await self.mw.awrap_model_call(request, handler)

        assert result is response
        mock_eval.assert_not_awaited()

    @pytest.mark.asyncio
    @patch("deep_agent.src.opa.middleware.get_opa_max_retries", return_value=2)
    @patch("deep_agent.src.opa.middleware.evaluate_message", new_callable=AsyncMock)
    async def test_retries_with_feedback_when_blocked(self, mock_eval, _mock_retries):
        mock_eval.side_effect = [
            OpaResult(allowed=False, denial_reasons=["profanity"]),
            OpaResult(allowed=True),
        ]
        ai_msg = AIMessage(content="response text", id="m1", name="bot")
        response = _model_response([ai_msg])
        handler = AsyncMock(return_value=response)
        request = _model_request(messages=[HumanMessage(content="prompt")])

        result = await self.mw.awrap_model_call(request, handler)

        assert result is response
        assert handler.await_count == 2
        second_call_req = handler.call_args_list[1][0][0]
        last_msg = second_call_req.messages[-1]
        assert isinstance(last_msg, HumanMessage)
        assert last_msg.additional_kwargs.get("opa_retry") is True
        assert "profanity" in last_msg.content

    @pytest.mark.asyncio
    @patch("deep_agent.src.opa.middleware.get_opa_max_retries", return_value=1)
    @patch("deep_agent.src.opa.middleware.evaluate_message", new_callable=AsyncMock)
    async def test_denial_with_no_reasons_uses_policy_violation(
        self, mock_eval, _mock_retries
    ):
        mock_eval.return_value = OpaResult(allowed=False, denial_reasons=[])
        ai_msg = AIMessage(content="some output", id="m1", name="bot")
        response = _model_response([ai_msg])
        handler = AsyncMock(return_value=response)
        request = _model_request(messages=[HumanMessage(content="prompt")])

        result = await self.mw.awrap_model_call(request, handler)

        second_call_req = handler.call_args_list[1][0][0]
        last_msg = second_call_req.messages[-1]
        assert "policy violation" in last_msg.content

    @pytest.mark.asyncio
    @patch("deep_agent.src.opa.middleware.get_opa_max_retries", return_value=0)
    @patch("deep_agent.src.opa.middleware.evaluate_message", new_callable=AsyncMock)
    async def test_zero_retries_blocks_immediately(self, mock_eval, _mock_retries):
        mock_eval.return_value = OpaResult(allowed=False, denial_reasons=["disallowed"])
        ai_msg = AIMessage(content="output", id="m1", name="bot")
        response = _model_response([ai_msg])
        handler = AsyncMock(return_value=response)
        request = _model_request(messages=[])

        result = await self.mw.awrap_model_call(request, handler)

        assert result.result[0].content == _BLOCKED_MODEL_MESSAGE
        assert handler.await_count == 1

    @pytest.mark.asyncio
    @patch("deep_agent.src.opa.middleware.get_opa_max_retries", return_value=2)
    @patch("deep_agent.src.opa.middleware.evaluate_message", new_callable=AsyncMock)
    async def test_handler_receives_no_stream_model(self, mock_eval, _mock_retries):
        mock_eval.return_value = OpaResult(allowed=True)
        ai_msg = AIMessage(content="fine", id="m1", name="bot")
        response = _model_response([ai_msg])
        handler = AsyncMock(return_value=response)
        request = _model_request(messages=[])

        await self.mw.awrap_model_call(request, handler)

        called_req = handler.call_args_list[0][0][0]
        assert isinstance(called_req.model, _NoStreamModel)

    @pytest.mark.asyncio
    @patch("deep_agent.src.opa.middleware.get_opa_max_retries", return_value=3)
    @patch("deep_agent.src.opa.middleware.evaluate_message", new_callable=AsyncMock)
    async def test_allowed_on_second_retry_stops_early(self, mock_eval, _mock_retries):
        mock_eval.side_effect = [
            OpaResult(allowed=False, denial_reasons=["r1"]),
            OpaResult(allowed=False, denial_reasons=["r2"]),
            OpaResult(allowed=True),
        ]
        ai_msg = AIMessage(content="text", id="m1", name="bot")
        response = _model_response([ai_msg])
        handler = AsyncMock(return_value=response)
        request = _model_request(messages=[HumanMessage(content="hi")])

        result = await self.mw.awrap_model_call(request, handler)

        assert result is response
        assert handler.await_count == 3
        assert mock_eval.await_count == 3


# ===================================================================
# _extract_tool_content
# ===================================================================


class TestExtractToolContent:
    """OPAMiddleware._extract_tool_content extracts text from tool results."""

    def setup_method(self):
        self.mw = OPAMiddleware()

    def test_extracts_string_content_from_tool_message(self):
        msg = ToolMessage(content="tool output", tool_call_id="tc1")
        assert self.mw._extract_tool_content(msg) == "tool output"

    def test_extracts_non_string_content_from_tool_message(self):
        msg = ToolMessage(content={"key": "value"}, tool_call_id="tc1")
        result = self.mw._extract_tool_content(msg)
        assert "key" in result
        assert "value" in result

    def test_returns_empty_for_empty_tool_message_content(self):
        msg = ToolMessage(content="", tool_call_id="tc1")
        assert self.mw._extract_tool_content(msg) == ""

    def test_extracts_from_command_with_tool_messages(self):
        tool_msg = ToolMessage(content="cmd output", tool_call_id="tc1")
        cmd = Command(update={"messages": [tool_msg]})
        assert self.mw._extract_tool_content(cmd) == "cmd output"

    def test_returns_empty_for_command_without_messages(self):
        cmd = Command(update={"other": "data"})
        assert self.mw._extract_tool_content(cmd) == ""

    def test_returns_empty_for_command_with_empty_messages(self):
        cmd = Command(update={"messages": []})
        assert self.mw._extract_tool_content(cmd) == ""

    def test_returns_empty_for_command_with_none_update(self):
        cmd = Command(update=None)
        assert self.mw._extract_tool_content(cmd) == ""

    def test_command_extracts_from_last_tool_message(self):
        first = ToolMessage(content="first", tool_call_id="tc1")
        second = ToolMessage(content="second", tool_call_id="tc2")
        cmd = Command(update={"messages": [first, second]})
        assert self.mw._extract_tool_content(cmd) == "second"

    def test_command_skips_non_tool_messages(self):
        human = HumanMessage(content="human")
        tool = ToolMessage(content="tool data", tool_call_id="tc1")
        cmd = Command(update={"messages": [tool, human]})
        assert self.mw._extract_tool_content(cmd) == "tool data"

    def test_command_extracts_non_string_tool_content(self):
        tool_msg = ToolMessage(content=["item1", "item2"], tool_call_id="tc1")
        cmd = Command(update={"messages": [tool_msg]})
        result = self.mw._extract_tool_content(cmd)
        assert "item1" in result

    def test_tool_message_with_none_content(self):
        msg = MagicMock(spec=ToolMessage)
        msg.content = None
        assert self.mw._extract_tool_content(msg) == ""

    def test_returns_empty_for_non_tool_non_command(self):
        result = self.mw._extract_tool_content(HumanMessage(content="nope"))
        assert result == ""


# ===================================================================
# awrap_tool_call
# ===================================================================


class TestAwrapToolCall:
    """OPAMiddleware.awrap_tool_call checks OPA on tool outputs."""

    def setup_method(self):
        self.mw = OPAMiddleware()

    @pytest.mark.asyncio
    @patch("deep_agent.src.opa.middleware.evaluate_message", new_callable=AsyncMock)
    async def test_opa_allows_returns_original(self, mock_eval):
        mock_eval.return_value = OpaResult(allowed=True)
        tool_msg = ToolMessage(content="safe result", tool_call_id="tc1")
        handler = AsyncMock(return_value=tool_msg)
        request = _tool_call_request({"name": "my_tool", "id": "call_1"})

        result = await self.mw.awrap_tool_call(request, handler)

        assert result is tool_msg
        mock_eval.assert_awaited_once_with("tool_response", result="safe result")

    @pytest.mark.asyncio
    @patch("deep_agent.src.opa.middleware.evaluate_message", new_callable=AsyncMock)
    async def test_opa_denies_returns_command_with_error(self, mock_eval):
        mock_eval.return_value = OpaResult(
            allowed=False, denial_reasons=["sensitive data"]
        )
        tool_msg = ToolMessage(content="secret data", tool_call_id="tc1")
        handler = AsyncMock(return_value=tool_msg)
        request = _tool_call_request({"name": "my_tool", "id": "call_1"})

        result = await self.mw.awrap_tool_call(request, handler)

        assert isinstance(result, Command)
        error_msg = result.update["messages"][0]
        assert isinstance(error_msg, ToolMessage)
        assert _BLOCKED_TOOL_MESSAGE in error_msg.content
        assert "sensitive data" in error_msg.content
        assert error_msg.tool_call_id == "call_1"
        assert error_msg.name == "my_tool"
        assert error_msg.status == "error"
        assert error_msg.additional_kwargs.get("opa_retry") is True

    @pytest.mark.asyncio
    @patch("deep_agent.src.opa.middleware.evaluate_message", new_callable=AsyncMock)
    async def test_empty_tool_content_returns_without_opa(self, mock_eval):
        tool_msg = ToolMessage(content="", tool_call_id="tc1")
        handler = AsyncMock(return_value=tool_msg)
        request = _tool_call_request({"name": "t", "id": "c1"})

        result = await self.mw.awrap_tool_call(request, handler)

        assert result is tool_msg
        mock_eval.assert_not_awaited()

    @pytest.mark.asyncio
    @patch("deep_agent.src.opa.middleware.evaluate_message", new_callable=AsyncMock)
    async def test_whitespace_only_content_returns_without_opa(self, mock_eval):
        tool_msg = ToolMessage(content="   \n  ", tool_call_id="tc1")
        handler = AsyncMock(return_value=tool_msg)
        request = _tool_call_request({"name": "t", "id": "c1"})

        result = await self.mw.awrap_tool_call(request, handler)

        assert result is tool_msg
        mock_eval.assert_not_awaited()

    @pytest.mark.asyncio
    @patch("deep_agent.src.opa.middleware.evaluate_message", new_callable=AsyncMock)
    async def test_handles_dict_style_tool_call(self, mock_eval):
        mock_eval.return_value = OpaResult(allowed=False, denial_reasons=["blocked"])
        tool_msg = ToolMessage(content="output", tool_call_id="tc1")
        handler = AsyncMock(return_value=tool_msg)
        request = _tool_call_request({"name": "dict_tool", "id": "dict_id"})

        result = await self.mw.awrap_tool_call(request, handler)

        assert isinstance(result, Command)
        error_msg = result.update["messages"][0]
        assert error_msg.tool_call_id == "dict_id"
        assert error_msg.name == "dict_tool"

    @pytest.mark.asyncio
    @patch("deep_agent.src.opa.middleware.evaluate_message", new_callable=AsyncMock)
    async def test_handles_object_style_tool_call(self, mock_eval):
        mock_eval.return_value = OpaResult(
            allowed=False, denial_reasons=["not allowed"]
        )
        tool_msg = ToolMessage(content="output", tool_call_id="tc1")
        handler = AsyncMock(return_value=tool_msg)

        tool_call_obj = MagicMock()
        tool_call_obj.name = "obj_tool"
        tool_call_obj.id = "obj_id"
        request = _tool_call_request(tool_call_obj)

        result = await self.mw.awrap_tool_call(request, handler)

        assert isinstance(result, Command)
        error_msg = result.update["messages"][0]
        assert error_msg.tool_call_id == "obj_id"
        assert error_msg.name == "obj_tool"

    @pytest.mark.asyncio
    @patch("deep_agent.src.opa.middleware.evaluate_message", new_callable=AsyncMock)
    async def test_denial_no_reasons_uses_policy_violation(self, mock_eval):
        mock_eval.return_value = OpaResult(allowed=False, denial_reasons=[])
        tool_msg = ToolMessage(content="output", tool_call_id="tc1")
        handler = AsyncMock(return_value=tool_msg)
        request = _tool_call_request({"name": "t", "id": "c1"})

        result = await self.mw.awrap_tool_call(request, handler)

        error_msg = result.update["messages"][0]
        assert "policy violation" in error_msg.content

    @pytest.mark.asyncio
    @patch("deep_agent.src.opa.middleware.evaluate_message", new_callable=AsyncMock)
    async def test_denial_multiple_reasons_joined(self, mock_eval):
        mock_eval.return_value = OpaResult(
            allowed=False, denial_reasons=["reason1", "reason2"]
        )
        tool_msg = ToolMessage(content="output", tool_call_id="tc1")
        handler = AsyncMock(return_value=tool_msg)
        request = _tool_call_request({"name": "t", "id": "c1"})

        result = await self.mw.awrap_tool_call(request, handler)

        error_msg = result.update["messages"][0]
        assert "reason1; reason2" in error_msg.content

    @pytest.mark.asyncio
    @patch("deep_agent.src.opa.middleware.evaluate_message", new_callable=AsyncMock)
    async def test_command_result_opa_allows(self, mock_eval):
        mock_eval.return_value = OpaResult(allowed=True)
        inner_tool = ToolMessage(content="cmd data", tool_call_id="tc1")
        cmd = Command(update={"messages": [inner_tool]})
        handler = AsyncMock(return_value=cmd)
        request = _tool_call_request({"name": "t", "id": "c1"})

        result = await self.mw.awrap_tool_call(request, handler)

        assert result is cmd
        mock_eval.assert_awaited_once_with("tool_response", result="cmd data")

    @pytest.mark.asyncio
    @patch("deep_agent.src.opa.middleware.evaluate_message", new_callable=AsyncMock)
    async def test_command_result_opa_denies(self, mock_eval):
        mock_eval.return_value = OpaResult(allowed=False, denial_reasons=["bad tool"])
        inner_tool = ToolMessage(content="cmd data", tool_call_id="tc1")
        cmd = Command(update={"messages": [inner_tool]})
        handler = AsyncMock(return_value=cmd)
        request = _tool_call_request({"name": "cmd_tool", "id": "cmd_id"})

        result = await self.mw.awrap_tool_call(request, handler)

        assert isinstance(result, Command)
        error_msg = result.update["messages"][0]
        assert _BLOCKED_TOOL_MESSAGE in error_msg.content
        assert error_msg.tool_call_id == "cmd_id"
        assert error_msg.name == "cmd_tool"
