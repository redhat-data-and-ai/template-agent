"""Unit tests for deep_agent.src.guardrails.callback."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from deep_agent.src.guardrails import InputContentSafetyError, ToolContentSafetyError
from deep_agent.src.guardrails.callback import (
    GraniteGuardianCallbackHandler,
    _extract_content,
    _extract_messages_to_scan,
    _extract_output_text,
)


class TestExtractContent:
    def test_string_content(self):
        msg = HumanMessage(content="hello world")
        assert _extract_content(msg) == "hello world"

    def test_list_content_with_dicts(self):
        msg = HumanMessage(content=[{"text": "foo"}, {"text": "bar"}])
        result = _extract_content(msg)
        assert "foo" in result
        assert "bar" in result

    def test_list_content_with_non_dict(self):
        msg = HumanMessage(content=["plain string"])
        assert "plain string" in _extract_content(msg)

    def test_list_content_dict_missing_text_key(self):
        msg = HumanMessage(content=[{"type": "image"}])
        assert _extract_content(msg) == ""


class TestExtractMessagesToScan:
    def test_empty_messages_returns_empty(self):
        assert _extract_messages_to_scan([]) == []

    def test_scans_last_human_message(self):
        human = HumanMessage(content="scan me")
        result = _extract_messages_to_scan([[human]])
        assert result == [("scan me", "input")]

    def test_ignores_ai_messages(self):
        ai = AIMessage(content="I am AI")
        result = _extract_messages_to_scan([[ai]])
        assert result == []

    def test_skips_empty_human_content(self):
        human = HumanMessage(content="")
        result = _extract_messages_to_scan([[human]])
        assert result == []

    def test_only_last_batch_is_scanned(self):
        old_human = HumanMessage(content="old message")
        new_human = HumanMessage(content="new message")
        result = _extract_messages_to_scan([[old_human], [new_human]])
        assert result == [("new message", "input")]

    def test_stops_at_first_human_in_batch(self):
        h1 = HumanMessage(content="first human")
        h2 = HumanMessage(content="second human")
        result = _extract_messages_to_scan([[h1, h2]])
        assert len(result) == 1


class TestExtractOutputText:
    def test_extracts_message_content_string(self):
        msg = AIMessage(content="hello")
        gen = ChatGeneration(message=msg, text="")
        result_obj = LLMResult(generations=[[gen]])
        assert _extract_output_text(result_obj) == "hello"

    def test_extracts_message_content_list(self):
        msg = AIMessage(content=[{"text": "chunk"}])
        gen = ChatGeneration(message=msg, text="")
        result_obj = LLMResult(generations=[[gen]])
        assert "chunk" in _extract_output_text(result_obj)

    def test_falls_back_to_text_field(self):
        gen = MagicMock()
        gen.message = None
        gen.text = "fallback text"
        response = MagicMock()
        response.generations = [[gen]]
        assert _extract_output_text(response) == "fallback text"

    def test_returns_empty_string_when_no_content(self):
        gen = MagicMock()
        gen.message = None
        gen.text = ""
        response = MagicMock()
        response.generations = [[gen]]
        assert _extract_output_text(response) == ""


class TestGraniteGuardianCallbackHandler:
    def test_init_creates_empty_scanned_set(self):
        handler = GraniteGuardianCallbackHandler()
        assert handler._scanned == set()

    def test_already_scanned_false_first_time(self):
        handler = GraniteGuardianCallbackHandler()
        assert handler._already_scanned("hello") is False

    def test_already_scanned_true_second_time(self):
        handler = GraniteGuardianCallbackHandler()
        handler._already_scanned("hello")
        assert handler._already_scanned("hello") is True

    def test_already_scanned_different_content(self):
        handler = GraniteGuardianCallbackHandler()
        handler._already_scanned("hello")
        assert handler._already_scanned("world") is False

    @pytest.mark.asyncio
    async def test_on_tool_start_logs_and_returns(self):
        handler = GraniteGuardianCallbackHandler()
        await handler.on_tool_start(
            serialized={"name": "my_tool"},
            input_str="input",
            run_id=uuid4(),
        )

    @pytest.mark.asyncio
    async def test_on_tool_end_logs_and_returns(self):
        handler = GraniteGuardianCallbackHandler()
        await handler.on_tool_end(output="result", run_id=uuid4())

    @pytest.mark.asyncio
    async def test_on_tool_error_logs_warning(self):
        handler = GraniteGuardianCallbackHandler()
        await handler.on_tool_error(error=RuntimeError("boom"), run_id=uuid4())

    @pytest.mark.asyncio
    async def test_on_tool_start_never_raises_preserving_parallel_batch(self):
        """Tool-start callback must never raise — parallel tool isolation requires it.

        If one tool in a batch were blocked by raising here, all other in-flight
        tools would be cancelled. Instead, unsafe tool args are handled by
        GuardianToolProxy which returns a ToolMessage so the batch continues.
        """
        handler = GraniteGuardianCallbackHandler()
        run1, run2 = uuid4(), uuid4()
        # Both calls must complete without raising, even for "harmful" input
        await handler.on_tool_start(
            serialized={"name": "tool_a"}, input_str="harmful content", run_id=run1
        )
        await handler.on_tool_start(
            serialized={"name": "tool_b"}, input_str="safe content", run_id=run2
        )

    @pytest.mark.asyncio
    async def test_on_chat_model_start_skips_when_runtime_disabled(self):
        """enabled=false / runtime-disabled: callback must return immediately, no guardian call."""
        handler = GraniteGuardianCallbackHandler()
        human = HumanMessage(content="some input")

        with (
            patch("deep_agent.src.guardrails.get_guardrails_config", return_value=None),
            patch(
                "deep_agent.src.guardrails.client.check_safety", new=AsyncMock()
            ) as mock_safety,
        ):
            await handler.on_chat_model_start(
                serialized={}, messages=[[human]], run_id=uuid4()
            )
        mock_safety.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_llm_end_skips_when_runtime_disabled(self):
        """enabled=false / runtime-disabled: callback must return immediately, no guardian call."""
        handler = GraniteGuardianCallbackHandler()
        msg = AIMessage(content="some response")
        gen = ChatGeneration(message=msg, text="")
        response = LLMResult(generations=[[gen]])

        with (
            patch("deep_agent.src.guardrails.get_guardrails_config", return_value=None),
            patch(
                "deep_agent.src.guardrails.client.check_safety", new=AsyncMock()
            ) as mock_safety,
        ):
            await handler.on_llm_end(response=response, run_id=uuid4())
        mock_safety.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_chat_model_start_safe_content_passes(self):
        handler = GraniteGuardianCallbackHandler()
        human = HumanMessage(content="safe input")

        with (
            patch(
                "deep_agent.src.guardrails.get_guardrails_config",
                return_value=MagicMock(enabled=True),
            ),
            patch(
                "deep_agent.src.guardrails.client.check_safety",
                new=AsyncMock(return_value=(True, "No")),
            ),
            patch(
                "deep_agent.src.guardrails.client.check_injection",
                new=AsyncMock(return_value=(True, "No")),
            ),
        ):
            await handler.on_chat_model_start(
                serialized={},
                messages=[[human]],
                run_id=uuid4(),
            )

    @pytest.mark.asyncio
    async def test_on_chat_model_start_unsafe_input_raises(self):
        handler = GraniteGuardianCallbackHandler()
        human = HumanMessage(content="unsafe content")

        with (
            patch(
                "deep_agent.src.guardrails.get_guardrails_config",
                return_value=MagicMock(enabled=True),
            ),
            patch(
                "deep_agent.src.guardrails.client.check_safety",
                new=AsyncMock(return_value=(False, "Yes")),
            ),
        ):
            with pytest.raises(InputContentSafetyError):
                await handler.on_chat_model_start(
                    serialized={},
                    messages=[[human]],
                    run_id=uuid4(),
                )

    @pytest.mark.asyncio
    async def test_on_chat_model_start_unsafe_injection_raises(self):
        handler = GraniteGuardianCallbackHandler()
        human = HumanMessage(content="inject me")

        with (
            patch(
                "deep_agent.src.guardrails.get_guardrails_config",
                return_value=MagicMock(enabled=True),
            ),
            patch(
                "deep_agent.src.guardrails.client.check_safety",
                new=AsyncMock(return_value=(True, "No")),
            ),
            patch(
                "deep_agent.src.guardrails.client.check_injection",
                new=AsyncMock(return_value=(False, "Yes")),
            ),
        ):
            with pytest.raises(InputContentSafetyError):
                await handler.on_chat_model_start(
                    serialized={},
                    messages=[[human]],
                    run_id=uuid4(),
                )

    @pytest.mark.asyncio
    async def test_on_chat_model_start_skips_already_scanned(self):
        handler = GraniteGuardianCallbackHandler()
        human = HumanMessage(content="already seen")

        with (
            patch(
                "deep_agent.src.guardrails.get_guardrails_config",
                return_value=MagicMock(enabled=True),
            ),
            patch(
                "deep_agent.src.guardrails.client.check_safety",
                new=AsyncMock(return_value=(True, "No")),
            ) as mock_safety,
            patch(
                "deep_agent.src.guardrails.client.check_injection",
                new=AsyncMock(return_value=(True, "No")),
            ),
        ):
            await handler.on_chat_model_start(
                serialized={}, messages=[[human]], run_id=uuid4()
            )
            await handler.on_chat_model_start(
                serialized={}, messages=[[human]], run_id=uuid4()
            )
            assert mock_safety.call_count == 1

    @pytest.mark.asyncio
    async def test_on_llm_end_safe_output_passes(self):
        handler = GraniteGuardianCallbackHandler()
        msg = AIMessage(content="safe response")
        gen = ChatGeneration(message=msg, text="")
        response = LLMResult(generations=[[gen]])

        with (
            patch(
                "deep_agent.src.guardrails.get_guardrails_config",
                return_value=MagicMock(enabled=True),
            ),
            patch(
                "deep_agent.src.guardrails.client.check_safety",
                new=AsyncMock(return_value=(True, "No")),
            ),
        ):
            await handler.on_llm_end(response=response, run_id=uuid4())

    @pytest.mark.asyncio
    async def test_on_llm_end_unsafe_output_raises(self):
        handler = GraniteGuardianCallbackHandler()
        msg = AIMessage(content="unsafe response")
        gen = ChatGeneration(message=msg, text="")
        response = LLMResult(generations=[[gen]])

        with (
            patch(
                "deep_agent.src.guardrails.get_guardrails_config",
                return_value=MagicMock(enabled=True),
            ),
            patch(
                "deep_agent.src.guardrails.client.check_safety",
                new=AsyncMock(return_value=(False, "Yes")),
            ),
        ):
            with pytest.raises(ToolContentSafetyError):
                await handler.on_llm_end(response=response, run_id=uuid4())

    @pytest.mark.asyncio
    async def test_on_llm_end_empty_content_skips_check(self):
        handler = GraniteGuardianCallbackHandler()
        gen = MagicMock()
        gen.message = None
        gen.text = ""
        response = MagicMock()
        response.generations = [[gen]]

        with (
            patch(
                "deep_agent.src.guardrails.get_guardrails_config",
                return_value=MagicMock(enabled=True),
            ),
            patch(
                "deep_agent.src.guardrails.client.check_safety",
                new=AsyncMock(return_value=(True, "No")),
            ) as mock_safety,
        ):
            await handler.on_llm_end(response=response, run_id=uuid4())
            mock_safety.assert_not_called()
