"""Comprehensive pytest tests for the deep research context_manager module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from template_agent.src.core.deep_research.context_manager import (
    FindingCard,
    HierarchicalContextManager,
    ImmediateContext,
    ResearchMemory,
    create_context_manager,
    create_initial_hierarchical_context,
    estimate_finding_tokens,
    estimate_state_tokens,
    estimate_tokens,
)
from template_agent.src.core.deep_research.state import Finding, ResearchContext


def _make_ctx(**overrides):
    """Create minimal ResearchContext for testing with mocked LLM."""
    base_model = AsyncMock()
    ctx = ResearchContext(tools=[MagicMock(name="tool1")], base_model=base_model)
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


def _make_immediate_context(
    findings: list[Finding] | None = None,
    window_size: int = 8,
    slide_step: int = 4,
) -> ImmediateContext:
    """Create ImmediateContext for testing."""
    findings = findings or []
    return ImmediateContext(
        recent_findings=findings,
        recent_subqueries=[f.get("subquery", "") for f in findings],
        window_size=window_size,
        slide_step=slide_step,
    )


def _make_finding_card(
    subquery: str = "c1", answer: str = "a1", **overrides: object
) -> FindingCard:
    """Create a minimal FindingCard for testing."""
    base: dict[str, object] = {
        "subquery": subquery,
        "answer": answer,
    }
    base.update(overrides)
    return base  # type: ignore[return-value]


class TestHierarchicalContextManagerInit:
    """Test cases for HierarchicalContextManager initialization."""

    def test_init_uses_default_window_and_slide_from_settings(self):
        """Manager uses CONTEXT_WINDOW_SIZE and CONTEXT_SLIDE_STEP from settings."""
        ctx = _make_ctx()
        with patch(
            "template_agent.src.core.deep_research.context_manager._get_setting"
        ) as mock_get:
            mock_get.side_effect = lambda name, default: {
                "CONTEXT_WINDOW_SIZE": 8,
                "CONTEXT_SLIDE_STEP": 4,
            }.get(name, default)
            mgr = HierarchicalContextManager(ctx)
        assert mgr.ctx is ctx
        assert mgr.window_size == 8
        assert mgr.slide_step == 4

    def test_init_accepts_window_size_override(self):
        """Manager accepts explicit window_size override."""
        ctx = _make_ctx()
        mgr = HierarchicalContextManager(ctx, window_size=6)
        assert mgr.window_size == 6

    def test_init_clamps_slide_step_when_exceeds_window_size(self):
        """Slide step is clamped to window_size when larger."""
        ctx = _make_ctx()
        with patch(
            "template_agent.src.core.deep_research.context_manager.logger"
        ) as mock_logger:
            mgr = HierarchicalContextManager(ctx, window_size=4, slide_step=8)
        assert mgr.slide_step == 4
        mock_logger.warning.assert_called_once()


class TestProcessNewFinding:
    """Test cases for process_new_finding."""

    @pytest.mark.asyncio
    async def test_process_new_finding_adds_to_immediate_context_without_sliding(self):
        """Adding a finding when under window size appends to immediate context."""
        ctx = _make_ctx()
        ctx.base_model.ainvoke = AsyncMock()
        mgr = HierarchicalContextManager(ctx, window_size=8, slide_step=4)

        immediate = _make_immediate_context(findings=[], window_size=8, slide_step=4)
        finding = _make_finding(subquery="q1", answer="a1")

        updated_immediate, cards, memory = await mgr.process_new_finding(
            finding, immediate, [], None
        )

        assert len(updated_immediate["recent_findings"]) == 1
        assert updated_immediate["recent_findings"][0]["subquery"] == "q1"
        assert updated_immediate["recent_subqueries"] == ["q1"]
        assert cards == []
        assert memory is None
        ctx.base_model.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_new_finding_triggers_slide_when_window_exceeded(self):
        """When findings exceed window size, oldest are compressed to cards."""
        ctx = _make_ctx()
        ctx.base_model.ainvoke = AsyncMock(
            return_value=MagicMock(
                content='{"summary":"s","key_facts":[],"data_highlights":{},'
                '"source_citations":[],"quality_score":0.8,"has_visualization":false}'
            )
        )
        mgr = HierarchicalContextManager(ctx, window_size=4, slide_step=2)

        findings = [_make_finding(subquery=f"q{i}", answer=f"a{i}") for i in range(4)]
        immediate = _make_immediate_context(
            findings=findings, window_size=4, slide_step=2
        )

        new_finding = _make_finding(subquery="q5", answer="a5")
        updated_immediate, cards, _ = await mgr.process_new_finding(
            new_finding, immediate, [], None
        )

        assert len(updated_immediate["recent_findings"]) == 3
        assert updated_immediate["recent_findings"][0]["subquery"] == "q2"
        assert len(cards) == 2
        assert ctx.base_model.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_process_new_finding_triggers_consolidation_when_threshold_met(self):
        """When cards since consolidation meet threshold, research memory is consolidated."""
        ctx = _make_ctx()
        ctx.base_model.ainvoke = AsyncMock(
            return_value=MagicMock(
                content='{"summary":"s","key_facts":[],"data_highlights":{},'
                '"source_citations":[],"quality_score":0.8,"has_visualization":false}'
            )
        )
        mgr = HierarchicalContextManager(ctx, window_size=2, slide_step=1)

        with patch(
            "template_agent.src.core.deep_research.context_manager._get_setting"
        ) as mock_get:
            mock_get.side_effect = lambda name, default: {
                "RESEARCH_MEMORY_CONSOLIDATION_THRESHOLD": 2,
            }.get(name, default)

            findings = [_make_finding(subquery="q1", answer="a1")]
            immediate = _make_immediate_context(
                findings=findings, window_size=2, slide_step=1
            )
            cards = []

            for i in range(2, 5):
                new_finding = _make_finding(subquery=f"q{i}", answer=f"a{i}")
                immediate, cards, memory = await mgr.process_new_finding(
                    new_finding, immediate, cards, None
                )

            assert memory is not None
            assert "completed_count" in memory or "key_insights" in memory


class TestCompressToFindingCard:
    """Test cases for compress_to_finding_card."""

    @pytest.mark.asyncio
    async def test_compress_to_finding_card_returns_error_card_when_error_present(self):
        """Finding with error returns card without LLM call."""
        ctx = _make_ctx()
        ctx.base_model.ainvoke = AsyncMock()
        mgr = HierarchicalContextManager(ctx)

        finding = _make_finding(subquery="q1", answer="", error="Something failed")

        card = await mgr.compress_to_finding_card(finding)

        assert card["subquery"] == "q1"
        assert card["summary"] == "Something failed"
        assert card["quality_score"] == pytest.approx(0.0)
        ctx.base_model.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_compress_to_finding_card_returns_error_card_when_no_answer(self):
        """Finding with no answer returns fallback card without LLM call."""
        ctx = _make_ctx()
        ctx.base_model.ainvoke = AsyncMock()
        mgr = HierarchicalContextManager(ctx)

        finding = _make_finding(subquery="q1", answer="")

        card = await mgr.compress_to_finding_card(finding)

        assert card["subquery"] == "q1"
        assert card["summary"] == "No answer available"
        ctx.base_model.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_compress_to_finding_card_uses_llm_response_when_valid_json(self):
        """Valid LLM JSON response produces structured FindingCard."""
        ctx = _make_ctx()
        ctx.base_model.ainvoke = AsyncMock(
            return_value=MagicMock(
                content='{"summary":"Compressed summary","key_facts":["f1","f2"],'
                '"data_highlights":{"k":"v"},"source_citations":["c1"],'
                '"quality_score":0.9,"has_visualization":true}'
            )
        )
        mgr = HierarchicalContextManager(ctx)

        finding = _make_finding(subquery="q1", answer="Full answer")

        card = await mgr.compress_to_finding_card(finding)

        assert card["subquery"] == "q1"
        assert card["summary"] == "Compressed summary"
        assert card["key_facts"] == ["f1", "f2"]
        assert card["quality_score"] == pytest.approx(0.9)
        assert card["has_visualization"] is True
        ctx.base_model.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_compress_to_finding_card_returns_fallback_on_llm_exception(self):
        """LLM exception yields fallback card with truncated answer."""
        ctx = _make_ctx()
        ctx.base_model.ainvoke = AsyncMock(side_effect=RuntimeError("LLM failed"))
        mgr = HierarchicalContextManager(ctx)

        finding = _make_finding(subquery="q1", answer="x" * 100)

        card = await mgr.compress_to_finding_card(finding)

        assert card["subquery"] == "q1"
        assert "x" in card["summary"]
        assert card["quality_score"] == pytest.approx(0.5)


class TestConsolidateResearchMemory:
    """Test cases for consolidate_research_memory."""

    @pytest.mark.asyncio
    async def test_consolidate_research_memory_returns_empty_when_no_cards(self):
        """Empty cards list returns empty memory when no current memory."""
        ctx = _make_ctx()
        ctx.base_model.ainvoke = AsyncMock()
        mgr = HierarchicalContextManager(ctx)

        memory = await mgr.consolidate_research_memory([], None)

        assert memory["completed_count"] == 0
        assert memory["key_insights"] == []
        ctx.base_model.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_consolidate_research_memory_merges_insights_from_llm_response(self):
        """Valid LLM response merges insights into research memory."""
        ctx = _make_ctx()
        ctx.base_model.ainvoke = AsyncMock(
            return_value=MagicMock(
                content='{"plan_summary":"plan","key_insights":["i1","i2"],'
                '"data_summary":"data","failed_subqueries":[],'
                '"access_denied_subqueries":[],"themes":["t1"]}'
            )
        )
        mgr = HierarchicalContextManager(ctx)

        cards = [_make_finding_card(subquery="c1", summary="s1")]
        memory = await mgr.consolidate_research_memory(cards, None)

        assert memory["key_insights"] == ["i1", "i2"]
        assert memory["themes"] == ["t1"]
        assert memory["completed_count"] == 1


class TestFormatForSynthesis:
    """Test cases for format_for_synthesis."""

    def test_format_for_synthesis_includes_query_header(self):
        """Formatted output includes original query in header."""
        ctx = _make_ctx()
        mgr = HierarchicalContextManager(ctx)

        result = mgr.format_for_synthesis(
            _make_immediate_context(),
            [],
            None,
            query="What is the revenue trend?",
        )

        assert "RESEARCH CONTEXT" in result
        assert "revenue trend" in result

    def test_format_for_synthesis_includes_research_overview_when_memory_present(self):
        """Research memory produces overview section."""
        ctx = _make_ctx()
        mgr = HierarchicalContextManager(ctx)

        memory: ResearchMemory = {
            "plan_summary": "plan",
            "completed_count": 5,
            "total_count": 10,
            "key_insights": ["insight1"],
            "data_summary": "",
            "failed_subqueries": [],
            "access_denied_subqueries": [],
            "themes": [],
            "consolidated_at": "2025-01-01T00:00:00",
        }

        result = mgr.format_for_synthesis(
            _make_immediate_context(),
            [],
            memory,
            query="q",
        )

        assert "RESEARCH OVERVIEW" in result
        assert "5/10" in result
        assert "insight1" in result

    def test_format_for_synthesis_includes_finding_summaries_from_cards(self):
        """Finding cards produce summaries section."""
        ctx = _make_ctx()
        mgr = HierarchicalContextManager(ctx)

        cards = [
            _make_finding_card(subquery="q1", summary="s1", key_facts=["f1"]),
        ]

        result = mgr.format_for_synthesis(
            _make_immediate_context(),
            cards,
            None,
            query="q",
        )

        assert "FINDING SUMMARIES" in result
        assert "q1" in result
        assert "s1" in result
        assert "f1" in result

    def test_format_for_synthesis_includes_detailed_findings_from_immediate(self):
        """Immediate context findings produce detailed section."""
        ctx = _make_ctx()
        mgr = HierarchicalContextManager(ctx)

        findings = [_make_finding(subquery="q1", answer="a1")]
        immediate = _make_immediate_context(findings=findings)

        result = mgr.format_for_synthesis(immediate, [], None, query="q")

        assert "DETAILED RECENT FINDINGS" in result
        assert "q1" in result
        assert "a1" in result

    def test_format_for_synthesis_returns_no_findings_when_all_empty(self):
        """Empty context returns 'No findings available.'."""
        ctx = _make_ctx()
        mgr = HierarchicalContextManager(ctx)

        result = mgr.format_for_synthesis(
            _make_immediate_context(),
            [],
            None,
            query="",
        )

        assert result == "No findings available."


class TestFormatForSubquery:
    """Test cases for format_for_subquery."""

    def test_format_for_subquery_includes_insights_from_memory(self):
        """Research memory insights appear in subquery context."""
        ctx = _make_ctx()
        mgr = HierarchicalContextManager(ctx)

        memory: ResearchMemory = {
            "key_insights": ["insight1", "insight2"],
        }

        result = mgr.format_for_subquery("new q", [], memory)

        assert "Key insights" in result
        assert "insight1" in result

    def test_format_for_subquery_returns_empty_when_no_context(self):
        """Empty memory and cards yield empty string."""
        ctx = _make_ctx()
        mgr = HierarchicalContextManager(ctx)

        result = mgr.format_for_subquery("q", [], None)

        assert result == ""


class TestEstimateTokens:
    """Test cases for estimate_tokens module function."""

    def test_estimate_tokens_returns_zero_for_empty_string(self):
        """Empty string returns 0 tokens."""
        assert estimate_tokens("") == 0

    def test_estimate_tokens_returns_heuristic_for_gemini_model(self):
        """Gemini model uses len/4 heuristic."""
        text = "hello world"
        result = estimate_tokens(text, model_name="gemini-1.5-pro")
        assert result == max(1, len(text) // 4)

    def test_estimate_tokens_returns_positive_for_non_empty_text(self):
        """Non-empty text returns positive token count."""
        result = estimate_tokens("This is a sample sentence.")
        assert result >= 1


class TestEstimateFindingTokens:
    """Test cases for estimate_finding_tokens."""

    def test_estimate_finding_tokens_sums_subquery_answer_and_tool_results(self):
        """Token count includes subquery, answer, and tool results."""
        finding = _make_finding(
            subquery="q1",
            answer="a1",
            tool_results=["tr1", "tr2"],
        )
        result = estimate_finding_tokens(finding)
        assert result >= 1


class TestEstimateStateTokens:
    """Test cases for estimate_state_tokens module function."""

    def test_estimate_state_tokens_includes_base_overhead(self):
        """Total includes base_overhead."""
        result = estimate_state_tokens(
            findings={},
            immediate_context=None,
            finding_cards=[],
            research_memory=None,
            base_overhead=15000,
        )
        assert result == 15000

    def test_estimate_state_tokens_adds_immediate_context_findings(self):
        """Immediate context findings contribute to token count."""
        immediate = _make_immediate_context(
            findings=[
                _make_finding(subquery="q1", answer="a1"),
            ]
        )
        result = estimate_state_tokens(
            findings={},
            immediate_context=immediate,
            finding_cards=[],
            research_memory=None,
            base_overhead=100,
        )
        assert result > 100

    def test_estimate_state_tokens_adds_finding_cards(self):
        """Finding cards contribute to token count."""
        cards = [
            _make_finding_card(subquery="c1", summary="s1", key_facts=["f1"]),
        ]
        result = estimate_state_tokens(
            findings={},
            immediate_context=None,
            finding_cards=cards,
            research_memory=None,
            base_overhead=100,
        )
        assert result > 100

    def test_estimate_state_tokens_adds_research_memory(self):
        """Research memory contributes to token count."""
        memory: ResearchMemory = {
            "plan_summary": "plan",
            "data_summary": "data",
            "key_insights": ["i1"],
            "themes": ["t1"],
        }
        result = estimate_state_tokens(
            findings={},
            immediate_context=None,
            finding_cards=[],
            research_memory=memory,
            base_overhead=100,
        )
        assert result > 100


class TestTokenizeAndCosineSimilarity:
    """Test cases for _tokenize and _cosine_similarity via _find_related_cards."""

    def test_find_related_cards_returns_empty_for_empty_subquery(self):
        """Empty subquery returns no related cards."""
        ctx = _make_ctx()
        mgr = HierarchicalContextManager(ctx)
        cards = [_make_finding_card(subquery="c1", summary="s1")]

        result = mgr._find_related_cards("", cards)

        assert result == []

    def test_find_related_cards_returns_semantically_similar_cards(self):
        """Cards with overlapping terms are returned as related."""
        ctx = _make_ctx()
        mgr = HierarchicalContextManager(ctx)
        cards = [
            _make_finding_card(
                subquery="revenue growth",
                summary="revenue increased significantly",
                key_facts=["growth", "revenue"],
            ),
        ]

        result = mgr._find_related_cards(
            "revenue growth analysis", cards, threshold=0.1
        )

        assert len(result) >= 1
        assert result[0]["subquery"] == "revenue growth"


class TestCreateContextManager:
    """Test cases for create_context_manager factory."""

    def test_create_context_manager_returns_hierarchical_manager(self):
        """Factory returns HierarchicalContextManager instance."""
        ctx = _make_ctx()
        mgr = create_context_manager(ctx)
        assert isinstance(mgr, HierarchicalContextManager)
        assert mgr.ctx is ctx


class TestCreateInitialHierarchicalContext:
    """Test cases for create_initial_hierarchical_context."""

    def test_create_initial_hierarchical_context_returns_three_tuple(self):
        """Factory returns (immediate, cards, memory) tuple."""
        immediate, cards, memory = create_initial_hierarchical_context(["q1", "q2"])

        assert isinstance(immediate, dict)
        assert immediate["recent_findings"] == []
        assert immediate["recent_subqueries"] == []
        assert cards == []
        assert memory["total_count"] == 2
        assert "subqueries" in memory["plan_summary"]
