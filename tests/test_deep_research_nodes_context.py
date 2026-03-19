"""Comprehensive pytest tests for deep research nodes _context module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from template_agent.src.core.deep_research.nodes import _context as context_module
from template_agent.src.core.deep_research.state import (
    DeepResearchState,
    Finding,
    ResearchContext,
)


def _make_ctx(**overrides) -> ResearchContext:
    """Create minimal ResearchContext for testing."""
    ctx = ResearchContext(tools=[MagicMock()], base_model=MagicMock())
    for k, v in overrides.items():
        setattr(ctx, k, v)
    return ctx


def _make_finding(
    subquery: str = "q1", answer: str = "a1", **overrides: object
) -> Finding:
    """Create a minimal Finding for testing."""
    base: dict[str, object] = {"subquery": subquery, "answer": answer}
    base.update(overrides)
    return base  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# _process_finding_hierarchical
# ---------------------------------------------------------------------------


class TestProcessFindingHierarchical:
    """Tests for _process_finding_hierarchical."""

    @pytest.mark.asyncio
    async def test_returns_none_empty_none_when_hierarchical_context_disabled(self):
        """When ENABLE_HIERARCHICAL_CONTEXT is False, returns (None, [], None)."""
        ctx = _make_ctx()
        finding = _make_finding()
        state: DeepResearchState = {
            "query": "test",
            "thread_id": "t1",
            "current_phase": "supervisor",
        }

        with patch.object(
            context_module, "_get_setting", return_value=False
        ) as mock_get:
            result = await context_module._process_finding_hierarchical(
                ctx, finding, state
            )

        assert result == (None, [], None)
        mock_get.assert_called()

    @pytest.mark.asyncio
    async def test_calls_manager_process_new_finding_when_enabled(self):
        """When enabled, delegates to context manager process_new_finding."""
        ctx = _make_ctx()
        finding = _make_finding(subquery="sq1", answer="ans1")
        immediate = {
            "recent_findings": [],
            "recent_subqueries": [],
            "window_size": 8,
            "slide_step": 4,
        }
        state: DeepResearchState = {
            "query": "test",
            "thread_id": "t1",
            "current_phase": "supervisor",
            "immediate_context": immediate,
            "finding_cards": [],
            "research_memory": None,
        }

        expected_immediate = {
            "recent_findings": [finding],
            "recent_subqueries": ["sq1"],
        }
        expected_cards = [{"subquery": "sq1", "summary": "s1"}]
        expected_memory = {"key_insights": ["i1"]}

        mock_mgr = MagicMock()
        mock_mgr.process_new_finding = AsyncMock(
            return_value=(expected_immediate, expected_cards, expected_memory)
        )

        with patch.object(
            context_module,
            "_get_setting",
            side_effect=lambda name, default: (
                True
                if name == "ENABLE_HIERARCHICAL_CONTEXT"
                else {"CONTEXT_WINDOW_SIZE": 8, "CONTEXT_SLIDE_STEP": 4}.get(
                    name, default
                )
            ),
        ):
            with patch.object(
                context_module, "create_context_manager", return_value=mock_mgr
            ):
                result = await context_module._process_finding_hierarchical(
                    ctx, finding, state
                )

        assert result == (expected_immediate, expected_cards, expected_memory)
        mock_mgr.process_new_finding.assert_awaited_once_with(
            finding, immediate, [], None
        )

    @pytest.mark.asyncio
    async def test_builds_default_immediate_context_when_state_empty(self):
        """When state has no immediate_context, builds default from settings."""
        ctx = _make_ctx()
        finding = _make_finding()
        state: dict = {
            "query": "test",
            "thread_id": "t1",
            "current_phase": "supervisor",
        }

        mock_mgr = MagicMock()
        mock_mgr.process_new_finding = AsyncMock(
            return_value=({"recent_findings": [finding]}, [], None)
        )

        with patch.object(
            context_module,
            "_get_setting",
            side_effect=lambda name, default: (
                True
                if name == "ENABLE_HIERARCHICAL_CONTEXT"
                else {"CONTEXT_WINDOW_SIZE": 8, "CONTEXT_SLIDE_STEP": 4}.get(
                    name, default
                )
            ),
        ):
            with patch.object(
                context_module, "create_context_manager", return_value=mock_mgr
            ):
                await context_module._process_finding_hierarchical(ctx, finding, state)

        call_args = mock_mgr.process_new_finding.call_args
        passed_immediate = call_args[0][1]
        assert passed_immediate["recent_findings"] == []
        assert passed_immediate["window_size"] == 8
        assert passed_immediate["slide_step"] == 4

    @pytest.mark.asyncio
    async def test_returns_unchanged_on_manager_exception(self):
        """On exception in process_new_finding, returns existing context unchanged."""
        ctx = _make_ctx()
        finding = _make_finding()
        immediate = {
            "recent_findings": [],
            "recent_subqueries": [],
            "window_size": 8,
            "slide_step": 4,
        }
        cards = [{"subquery": "c1", "summary": "s1"}]
        memory = {"key_insights": []}
        state: DeepResearchState = {
            "query": "test",
            "thread_id": "t1",
            "current_phase": "supervisor",
            "immediate_context": immediate,
            "finding_cards": cards,
            "research_memory": memory,
        }

        mock_mgr = MagicMock()
        mock_mgr.process_new_finding = AsyncMock(side_effect=RuntimeError("LLM failed"))

        with patch.object(
            context_module,
            "_get_setting",
            side_effect=lambda name, default: (
                True
                if name == "ENABLE_HIERARCHICAL_CONTEXT"
                else {"CONTEXT_WINDOW_SIZE": 8, "CONTEXT_SLIDE_STEP": 4}.get(
                    name, default
                )
            ),
        ):
            with patch.object(
                context_module, "create_context_manager", return_value=mock_mgr
            ):
                with patch.object(context_module, "logger") as mock_logger:
                    result = await context_module._process_finding_hierarchical(
                        ctx, finding, state
                    )

        assert result == (immediate, cards, memory)
        mock_logger.warning.assert_called_once()


# ---------------------------------------------------------------------------
# _emit_context_usage
# ---------------------------------------------------------------------------


class TestEmitContextUsage:
    """Tests for _emit_context_usage."""

    def test_returns_early_when_hierarchical_context_disabled(self):
        """When ENABLE_HIERARCHICAL_CONTEXT is False, does not emit."""
        ctx = _make_ctx()
        ctx.emit = MagicMock()
        state: DeepResearchState = {
            "query": "q",
            "thread_id": "t",
            "current_phase": "supervisor",
        }

        with patch.object(context_module, "_get_setting", return_value=False):
            context_module._emit_context_usage(ctx, state, "supervisor")

        ctx.emit.assert_not_called()

    def test_emits_with_normal_status_when_usage_under_70_percent(self):
        """Emits event with status 'normal' when usage < 70%."""
        ctx = _make_ctx()
        ctx.emit = MagicMock()
        ctx.model_name = "claude-3"
        state: DeepResearchState = {
            "query": "q",
            "thread_id": "t",
            "current_phase": "supervisor",
            "immediate_context": {"recent_findings": []},
            "finding_cards": [],
            "research_memory": None,
        }

        with patch.object(context_module, "_get_setting", return_value=True):
            with patch.object(
                context_module, "estimate_state_tokens", return_value=50_000
            ):
                with patch.object(
                    context_module, "get_max_context_tokens", return_value=200_000
                ):
                    with patch.object(
                        context_module, "emit_context_usage_update"
                    ) as mock_emit_fn:
                        mock_emit_fn.return_value = {"type": "context_usage"}
                        context_module._emit_context_usage(ctx, state, "supervisor")

        mock_emit_fn.assert_called_once()
        call_kwargs = mock_emit_fn.call_args[1]
        assert call_kwargs["current_tokens"] == 50_000
        assert call_kwargs["max_tokens"] == 200_000
        assert call_kwargs["usage_percent"] == pytest.approx(25.0)
        assert call_kwargs["status"] == "normal"
        assert call_kwargs["stage"] == "supervisor"
        ctx.emit.assert_called_once()

    def test_emits_with_warning_status_when_usage_over_70_percent(self):
        """Emits event with status 'warning' when usage > 70%."""
        ctx = _make_ctx()
        ctx.emit = MagicMock()
        ctx.model_name = "claude-3"
        state: DeepResearchState = {
            "query": "q",
            "thread_id": "t",
            "current_phase": "supervisor",
            "immediate_context": {},
            "finding_cards": [],
            "research_memory": None,
        }

        with patch.object(context_module, "_get_setting", return_value=True):
            with patch.object(
                context_module, "estimate_state_tokens", return_value=150_000
            ):
                with patch.object(
                    context_module, "get_max_context_tokens", return_value=200_000
                ):
                    with patch.object(
                        context_module, "emit_context_usage_update"
                    ) as mock_emit_fn:
                        mock_emit_fn.return_value = {}
                        context_module._emit_context_usage(ctx, state, "synthesis")

        call_kwargs = mock_emit_fn.call_args[1]
        assert call_kwargs["status"] == "warning"
        assert call_kwargs["usage_percent"] == pytest.approx(75.0)

    def test_emits_with_critical_status_when_usage_over_90_percent(self):
        """Emits event with status 'critical' when usage > 90%."""
        ctx = _make_ctx()
        ctx.emit = MagicMock()
        ctx.model_name = "claude-3"
        state: DeepResearchState = {
            "query": "q",
            "thread_id": "t",
            "current_phase": "supervisor",
            "immediate_context": {},
            "finding_cards": [],
            "research_memory": None,
        }

        with patch.object(context_module, "_get_setting", return_value=True):
            with patch.object(
                context_module, "estimate_state_tokens", return_value=190_000
            ):
                with patch.object(
                    context_module, "get_max_context_tokens", return_value=200_000
                ):
                    with patch.object(
                        context_module, "emit_context_usage_update"
                    ) as mock_emit_fn:
                        mock_emit_fn.return_value = {}
                        context_module._emit_context_usage(ctx, state, "supervisor")

        call_kwargs = mock_emit_fn.call_args[1]
        assert call_kwargs["status"] == "critical"
        assert call_kwargs["usage_percent"] == pytest.approx(95.0)

    def test_uses_findings_board_when_present(self):
        """Uses findings_from_board when findings_board is in state."""
        ctx = _make_ctx()
        ctx.emit = MagicMock()
        ctx.model_name = "claude-3"
        board = {"sq1": {"finding": _make_finding("sq1", "a1")}}
        state: DeepResearchState = {
            "query": "q",
            "thread_id": "t",
            "current_phase": "supervisor",
            "findings_board": board,
            "immediate_context": None,
            "finding_cards": [],
            "research_memory": None,
        }

        with patch.object(context_module, "_get_setting", return_value=True):
            with patch.object(
                context_module,
                "findings_from_board",
                return_value={"sq1": _make_finding()},
            ) as mock_fb:
                with patch.object(
                    context_module, "estimate_state_tokens", return_value=10_000
                ):
                    with patch.object(
                        context_module, "get_max_context_tokens", return_value=100_000
                    ):
                        with patch.object(
                            context_module, "emit_context_usage_update"
                        ) as mock_emit_fn:
                            mock_emit_fn.return_value = {}
                            context_module._emit_context_usage(ctx, state, "supervisor")

        mock_fb.assert_called_once_with(board)

    def test_handles_zero_max_tokens_gracefully(self):
        """When max_tokens is 0, usage_percent is 0 and no division error."""
        ctx = _make_ctx()
        ctx.emit = MagicMock()
        ctx.model_name = "unknown"
        state: DeepResearchState = {
            "query": "q",
            "thread_id": "t",
            "current_phase": "supervisor",
            "immediate_context": {},
            "finding_cards": [],
            "research_memory": None,
        }

        with patch.object(context_module, "_get_setting", return_value=True):
            with patch.object(
                context_module, "estimate_state_tokens", return_value=50_000
            ):
                with patch.object(
                    context_module, "get_max_context_tokens", return_value=0
                ):
                    with patch.object(
                        context_module, "emit_context_usage_update"
                    ) as mock_emit_fn:
                        mock_emit_fn.return_value = {}
                        context_module._emit_context_usage(ctx, state, "supervisor")

        call_kwargs = mock_emit_fn.call_args[1]
        assert call_kwargs["usage_percent"] == pytest.approx(0.0)

    def test_logs_warning_and_does_not_raise_on_exception(self):
        """On exception, logs warning and does not propagate."""
        ctx = _make_ctx()
        ctx.emit = MagicMock(side_effect=RuntimeError("emit failed"))
        ctx.model_name = "claude-3"
        state: DeepResearchState = {
            "query": "q",
            "thread_id": "t",
            "current_phase": "supervisor",
            "immediate_context": {},
            "finding_cards": [],
            "research_memory": None,
        }

        with patch.object(context_module, "_get_setting", return_value=True):
            with patch.object(
                context_module, "estimate_state_tokens", return_value=10_000
            ):
                with patch.object(
                    context_module, "get_max_context_tokens", return_value=100_000
                ):
                    with patch.object(context_module, "logger") as mock_logger:
                        context_module._emit_context_usage(ctx, state, "supervisor")

        mock_logger.warning.assert_called_once()


# ---------------------------------------------------------------------------
# _format_hierarchical_context_for_synthesis
# ---------------------------------------------------------------------------


class TestFormatHierarchicalContextForSynthesis:
    """Tests for _format_hierarchical_context_for_synthesis."""

    def test_returns_empty_string_when_hierarchical_context_disabled(self):
        """When ENABLE_HIERARCHICAL_CONTEXT is False, returns ''."""
        state: DeepResearchState = {
            "query": "q",
            "thread_id": "t",
            "current_phase": "synthesis",
        }
        ctx = _make_ctx()

        with patch.object(context_module, "_get_setting", return_value=False):
            result = context_module._format_hierarchical_context_for_synthesis(
                state, ctx
            )

        assert result == ""

    def test_returns_empty_string_when_immediate_context_missing(self):
        """When immediate_context is None or absent, returns ''."""
        state: DeepResearchState = {
            "query": "q",
            "thread_id": "t",
            "current_phase": "synthesis",
            "immediate_context": None,
        }
        ctx = _make_ctx()

        with patch.object(context_module, "_get_setting", return_value=True):
            result = context_module._format_hierarchical_context_for_synthesis(
                state, ctx
            )

        assert result == ""

    def test_returns_formatted_context_from_manager_when_enabled(self):
        """When enabled and immediate_context present, returns manager output."""
        immediate = {"recent_findings": [_make_finding()], "recent_subqueries": ["q1"]}
        state: DeepResearchState = {
            "query": "What is X?",
            "thread_id": "t",
            "current_phase": "synthesis",
            "immediate_context": immediate,
            "finding_cards": [{"subquery": "q1", "answer": "s1"}],
            "research_memory": {"key_insights": ["i1"]},
        }
        ctx = _make_ctx()

        expected = (
            "## RESEARCH CONTEXT\nOriginal Query: What is X?\n\n## FINDING SUMMARIES"
        )

        mock_mgr = MagicMock()
        mock_mgr.format_for_synthesis = MagicMock(return_value=expected)

        with patch.object(context_module, "_get_setting", return_value=True):
            with patch.object(
                context_module, "create_context_manager", return_value=mock_mgr
            ):
                result = context_module._format_hierarchical_context_for_synthesis(
                    state, ctx
                )

        assert result == expected
        mock_mgr.format_for_synthesis.assert_called_once_with(
            immediate,
            [{"subquery": "q1", "answer": "s1"}],
            {"key_insights": ["i1"]},
            "What is X?",
        )

    def test_returns_empty_string_on_manager_exception(self):
        """On exception in format_for_synthesis, returns '' and logs warning."""
        immediate = {"recent_findings": [], "recent_subqueries": []}
        state: DeepResearchState = {
            "query": "q",
            "thread_id": "t",
            "current_phase": "synthesis",
            "immediate_context": immediate,
            "finding_cards": [],
            "research_memory": None,
        }
        ctx = _make_ctx()

        mock_mgr = MagicMock()
        mock_mgr.format_for_synthesis = MagicMock(
            side_effect=ValueError("format failed")
        )

        with patch.object(context_module, "_get_setting", return_value=True):
            with patch.object(
                context_module, "create_context_manager", return_value=mock_mgr
            ):
                with patch.object(context_module, "logger") as mock_logger:
                    result = context_module._format_hierarchical_context_for_synthesis(
                        state, ctx
                    )

        assert result == ""
        mock_logger.warning.assert_called_once()
