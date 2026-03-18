"""Comprehensive pytest tests for the deep research manager module.

Tests AgentManager routing logic, follow-up classification, standard agent
fallback, and error handling in the deep research path.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from template_agent.src.core.manager import (
    AgentManager,
    _content_to_str,
)
from template_agent.src.schema import StreamRequest


async def _empty_async_gen(*args, **kwargs):
    """Empty async generator for mocking."""
    return  # pragma: no cover - yield below makes this an async generator
    yield


# ---------------------------------------------------------------------------
# TestContentToStr
# ---------------------------------------------------------------------------


class TestContentToStr:
    """Test cases for _content_to_str content normalization helper."""

    def test_content_to_str_returns_empty_for_none(self):
        """None content returns empty string."""
        result = _content_to_str(None)
        assert result == ""

    def test_content_to_str_returns_string_unchanged(self):
        """String content is returned as-is."""
        content = "hello world"
        result = _content_to_str(content)
        assert result == "hello world"

    def test_content_to_str_converts_list_via_agent_utils(self):
        """List content is converted via convert_message_content_to_string."""
        content = [{"type": "text", "text": "foo"}, {"type": "text", "text": "bar"}]
        with patch(
            "template_agent.src.core.manager.convert_message_content_to_string",
            return_value="foobar",
        ) as mock_convert:
            result = _content_to_str(content)
        mock_convert.assert_called_once_with(content)
        assert result == "foobar"


# ---------------------------------------------------------------------------
# TestShouldUseDeepResearch
# ---------------------------------------------------------------------------


class TestShouldUseDeepResearch:
    """Test cases for _should_use_deep_research routing logic."""

    def test_returns_false_when_deep_research_disabled_in_settings(self):
        """When DEEP_RESEARCH_ENABLED is False, always returns False."""
        request = StreamRequest(
            message="test",
            deep_research_enabled=True,
        )
        manager = AgentManager()

        with patch(
            "template_agent.src.core.manager.settings",
            MagicMock(DEEP_RESEARCH_ENABLED=False),
        ):
            result = manager._should_use_deep_research(request)

        assert result is False

    def test_returns_true_when_deep_research_enabled_on_request(self):
        """When DEEP_RESEARCH_ENABLED and request.deep_research_enabled, returns True."""
        request = StreamRequest(
            message="test",
            deep_research_enabled=True,
        )
        manager = AgentManager()

        with patch(
            "template_agent.src.core.manager.settings",
            MagicMock(DEEP_RESEARCH_ENABLED=True),
        ):
            result = manager._should_use_deep_research(request)

        assert result is True

    def test_returns_true_when_deep_research_resume_on_request(self):
        """When DEEP_RESEARCH_ENABLED and request.deep_research_resume, returns True."""
        request = StreamRequest(
            message="test",
            deep_research_enabled=False,
            deep_research_resume=True,
        )
        manager = AgentManager()

        with patch(
            "template_agent.src.core.manager.settings",
            MagicMock(DEEP_RESEARCH_ENABLED=True),
        ):
            result = manager._should_use_deep_research(request)

        assert result is True

    def test_returns_false_when_neither_enabled_nor_resume(self):
        """When neither deep_research_enabled nor deep_research_resume, returns False."""
        request = StreamRequest(
            message="test",
            deep_research_enabled=False,
            deep_research_resume=False,
        )
        manager = AgentManager()

        with patch(
            "template_agent.src.core.manager.settings",
            MagicMock(DEEP_RESEARCH_ENABLED=True),
        ):
            result = manager._should_use_deep_research(request)

        assert result is False


# ---------------------------------------------------------------------------
# TestClassifyFollowUp
# ---------------------------------------------------------------------------


class TestClassifyFollowUp:
    """Test cases for _classify_follow_up classification."""

    @pytest.mark.asyncio
    async def test_classify_returns_answer_directly_when_llm_says_so(self):
        """LLM returning 'answer_directly' is passed through."""
        manager = AgentManager()
        mock_response = MagicMock()
        mock_response.content = "answer_directly"

        with patch(
            "langchain_google_genai.ChatGoogleGenerativeAI",
        ) as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm_cls.return_value = mock_llm

            result = await manager._classify_follow_up(
                query="test",
                findings_text="findings",
                conversation_text="conv",
            )

        assert result == "answer_directly"

    @pytest.mark.asyncio
    async def test_classify_returns_needs_research_when_llm_says_so(self):
        """LLM returning 'needs_research' is passed through."""
        manager = AgentManager()
        mock_response = MagicMock()
        mock_response.content = "needs_research"

        with patch(
            "langchain_google_genai.ChatGoogleGenerativeAI",
        ) as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm_cls.return_value = mock_llm

            result = await manager._classify_follow_up(
                query="test",
                findings_text="findings",
                conversation_text="conv",
            )

        assert result == "needs_research"

    @pytest.mark.asyncio
    async def test_classify_defaults_to_needs_research_on_unexpected_value(self):
        """Unexpected classifier output defaults to needs_research."""
        manager = AgentManager()
        mock_response = MagicMock()
        mock_response.content = "invalid_decision"

        with patch(
            "langchain_google_genai.ChatGoogleGenerativeAI",
        ) as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm_cls.return_value = mock_llm

            result = await manager._classify_follow_up(
                query="test",
                findings_text="findings",
                conversation_text="conv",
            )

        assert result == "needs_research"

    @pytest.mark.asyncio
    async def test_classify_defaults_to_needs_research_on_llm_exception(self):
        """LLM exception defaults to needs_research."""
        manager = AgentManager()

        with patch(
            "langchain_google_genai.ChatGoogleGenerativeAI",
        ) as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("API down"))
            mock_llm_cls.return_value = mock_llm

            result = await manager._classify_follow_up(
                query="test",
                findings_text="findings",
                conversation_text="conv",
            )

        assert result == "needs_research"


# ---------------------------------------------------------------------------
# TestStreamFollowUpAnswer
# ---------------------------------------------------------------------------


class TestStreamFollowUpAnswer:
    """Test cases for _stream_follow_up_answer."""

    @pytest.mark.asyncio
    async def test_stream_follow_up_yields_tokens_and_message(self):
        """Follow-up answer streams tokens then final message."""
        request = StreamRequest(
            message="What was X?",
            thread_id="t1",
            session_id="s1",
        )
        manager = AgentManager()

        async def mock_chunk_gen(*args, **kwargs):
            chunk = MagicMock()
            chunk.content = "Hello "
            yield chunk
            chunk2 = MagicMock()
            chunk2.content = "world"
            yield chunk2

        with patch(
            "langchain_google_genai.ChatGoogleGenerativeAI",
        ) as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.astream = mock_chunk_gen
            mock_llm_cls.return_value = mock_llm

            with patch(
                "template_agent.src.core.manager.save_conversation_turn",
            ) as mock_save:
                events = []
                async for event in manager._stream_follow_up_answer(
                    request,
                    findings_text="findings",
                    conversation_text="conv",
                ):
                    events.append(event)

        token_events = [e for e in events if e.get("type") == "token"]
        message_events = [e for e in events if e.get("type") == "message"]

        assert len(token_events) == 2
        assert token_events[0]["content"] == "Hello "
        assert token_events[1]["content"] == "world"
        assert len(message_events) == 1
        assert message_events[0]["content"]["content"] == "Hello world"
        assert message_events[0]["content"]["content"] == "Hello world"
        mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_follow_up_yields_error_on_llm_failure(self):
        """Follow-up answer yields error event when LLM fails."""
        request = StreamRequest(
            message="What was X?",
            thread_id="t1",
        )
        manager = AgentManager()

        with patch(
            "langchain_google_genai.ChatGoogleGenerativeAI",
        ) as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.astream = MagicMock(side_effect=RuntimeError("LLM failed"))
            mock_llm_cls.return_value = mock_llm

            events = []
            async for event in manager._stream_follow_up_answer(
                request,
                findings_text="findings",
                conversation_text="conv",
            ):
                events.append(event)

        error_events = [e for e in events if e.get("type") == "error"]
        assert len(error_events) == 1
        assert "LLM failed" in error_events[0]["content"]["message"]
        assert error_events[0]["content"]["error_type"] == "followup_error"
        assert error_events[0]["content"]["recoverable"] is True


# ---------------------------------------------------------------------------
# TestStreamResponseRouting
# ---------------------------------------------------------------------------


class TestStreamResponseRouting:
    """Test stream_response routing between deep research and standard agent."""

    @pytest.mark.asyncio
    async def test_stream_response_routes_to_deep_research_when_enabled(self):
        """When deep research enabled, routes to _stream_deep_research."""
        request = StreamRequest(
            message="test",
            deep_research_enabled=True,
        )
        manager = AgentManager()

        with patch(
            "template_agent.src.core.manager.settings",
            MagicMock(DEEP_RESEARCH_ENABLED=True),
        ):
            with patch.object(
                manager,
                "_stream_deep_research",
                side_effect=_empty_async_gen,
            ) as mock_dr:
                events = []
                async for event in manager.stream_response(request):
                    events.append(event)

        mock_dr.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_stream_response_routes_to_standard_agent_when_deep_research_disabled(
        self,
    ):
        """When deep research disabled, routes to standard agent."""
        request = StreamRequest(
            message="test",
            deep_research_enabled=False,
        )
        manager = AgentManager()

        with patch(
            "template_agent.src.core.manager.settings",
            MagicMock(DEEP_RESEARCH_ENABLED=False),
        ):
            with patch(
                "template_agent.src.core.manager.get_template_agent",
            ) as mock_get_agent:
                mock_agent = MagicMock()
                mock_agent.astream = _empty_async_gen
                mock_agent.aget_state = AsyncMock(return_value=MagicMock(tasks=[]))
                mock_agent.__aenter__ = AsyncMock(return_value=mock_agent)
                mock_agent.__aexit__ = AsyncMock(return_value=None)
                mock_get_agent.return_value = mock_agent

                events = []
                async for event in manager.stream_response(request):
                    events.append(event)

        mock_get_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_response_yields_error_on_standard_agent_exception(self):
        """Standard agent exception yields error event."""
        request = StreamRequest(
            message="test",
            deep_research_enabled=False,
        )
        manager = AgentManager()

        with patch(
            "template_agent.src.core.manager.settings",
            MagicMock(DEEP_RESEARCH_ENABLED=False),
        ):
            with patch(
                "template_agent.src.core.manager.get_template_agent",
            ) as mock_get_agent:
                mock_agent = MagicMock()
                mock_agent.__aenter__ = AsyncMock(return_value=mock_agent)
                mock_agent.__aexit__ = AsyncMock(return_value=None)

                with patch.object(
                    manager,
                    "_handle_input",
                    side_effect=ValueError("Agent failed"),
                ):
                    mock_get_agent.return_value = mock_agent

                    events = []
                    async for event in manager.stream_response(request):
                        events.append(event)

        error_events = [e for e in events if e.get("type") == "error"]
        assert len(error_events) == 1
        assert error_events[0]["content"]["error_type"] == "agent_error"
        assert error_events[0]["content"]["recoverable"] is False


# ---------------------------------------------------------------------------
# TestStreamDeepResearchErrorHandling
# ---------------------------------------------------------------------------


class TestStreamDeepResearchErrorHandling:
    """Test error handling in _stream_deep_research path."""

    @pytest.mark.asyncio
    async def test_stream_deep_research_yields_error_on_runtime_error(self):
        """RuntimeError when entering deep research yields error event and fallback."""
        request = StreamRequest(
            message="test",
            thread_id="t1",
        )
        manager = AgentManager()

        with patch(
            "template_agent.src.core.manager.settings",
            MagicMock(
                DEEP_RESEARCH_ENABLED=True,
                USE_INMEMORY_SAVER=False,
                DEEP_RESEARCH_DEFAULT_MODEL="gemini-2.5-flash",
            ),
        ):
            with patch(
                "template_agent.src.core.deep_research.streaming.get_deep_research_agent",
                side_effect=RuntimeError("Deep research failed"),
            ):
                events = []
                async for event in manager._stream_deep_research(request):
                    events.append(event)

        message_events = [e for e in events if e.get("type") == "message"]

        assert len(message_events) == 1
        assert "Please try again" in message_events[0]["content"]["content"]

    @pytest.mark.asyncio
    async def test_stream_deep_research_follow_up_answer_directly_path(self):
        """When cached findings exist and classifier says answer_directly, streams follow-up."""
        request = StreamRequest(
            message="What was the result?",
            thread_id="t1",
        )
        manager = AgentManager()

        mock_ctx = MagicMock()
        mock_ctx.base_model = MagicMock()
        mock_dr_agent = MagicMock()
        mock_dr_agent.ctx = mock_ctx

        class MockAsyncCM:
            def __init__(self, result):
                self.result = result

            async def __aenter__(self):
                return self.result

            async def __aexit__(self, *args):
                return None

        def mock_get_agent(*args, **kwargs):
            return MockAsyncCM(mock_dr_agent)

        with patch(
            "template_agent.src.core.manager.settings",
            MagicMock(
                DEEP_RESEARCH_ENABLED=True,
                USE_INMEMORY_SAVER=False,
                DEEP_RESEARCH_DEFAULT_MODEL="gemini-2.5-flash",
                DEEP_RESEARCH_REQUIRE_PLAN_APPROVAL=False,
            ),
        ):
            with patch(
                "template_agent.src.core.deep_research.streaming.get_deep_research_agent",
                side_effect=mock_get_agent,
            ):
                with patch(
                    "template_agent.src.core.deep_research.nodes._cache.load_findings_in_memory",
                    return_value={"q1": [{"content": "a1"}]},
                ):
                    with patch(
                        "template_agent.src.core.deep_research.nodes._cache.format_cached_findings_for_prompt",
                        return_value="findings text",
                    ):
                        with patch(
                            "template_agent.src.core.deep_research.streaming.select_relevant_findings",
                            new_callable=AsyncMock,
                            return_value="triage text",
                        ):
                            with patch(
                                "template_agent.src.core.manager.load_conversation_history",
                                return_value=[],
                            ):
                                with patch(
                                    "template_agent.src.core.manager.format_conversation_for_prompt",
                                    return_value="conv text",
                                ):
                                    with patch.object(
                                        manager,
                                        "_classify_follow_up",
                                        new_callable=AsyncMock,
                                        return_value="answer_directly",
                                    ):
                                        call_log = []

                                        async def mock_follow_up_impl(
                                            req, findings_text, conversation_text
                                        ):
                                            call_log.append(
                                                (
                                                    req,
                                                    findings_text,
                                                    conversation_text,
                                                )
                                            )
                                            return  # pragma: no cover
                                            yield

                                        with patch.object(
                                            manager,
                                            "_stream_follow_up_answer",
                                            side_effect=mock_follow_up_impl,
                                        ):
                                            events = []
                                            async for (
                                                event
                                            ) in manager._stream_deep_research(request):
                                                events.append(event)

        assert len(call_log) == 1
        assert call_log[0][1] == "triage text"
        assert call_log[0][2] == "conv text"
