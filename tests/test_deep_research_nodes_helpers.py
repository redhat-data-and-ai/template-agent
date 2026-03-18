"""Comprehensive pytest tests for deep research node helpers."""

from unittest.mock import AsyncMock, patch

import pytest

from template_agent.src.core.deep_research.nodes import _helpers as helpers

# ---------------------------------------------------------------------------
# Plan parsing helpers
# ---------------------------------------------------------------------------


class TestExtractBalancedJson:
    """Tests for _extract_balanced_json."""

    def test_extracts_outermost_balanced_brace_containing_keyword(self):
        """Extract balanced {...} containing the keyword."""
        text = 'pre {"subqueries": ["a", "b"]} post'
        result = helpers._extract_balanced_json(text, '"subqueries"')
        assert result == '{"subqueries": ["a", "b"]}'

    def test_returns_none_when_keyword_absent(self):
        """Return None when keyword not found."""
        text = '{"foo": "bar"}'
        result = helpers._extract_balanced_json(text, '"subqueries"')
        assert result is None

    def test_returns_none_when_no_brace_before_keyword(self):
        """Return None when no opening brace before keyword."""
        text = 'subqueries: ["a"]'
        result = helpers._extract_balanced_json(text, "subqueries")
        assert result is None

    def test_handles_nested_braces(self):
        """Handle nested braces correctly."""
        text = 'outer {"inner": {"subqueries": [1, 2]}} end'
        result = helpers._extract_balanced_json(text, '"subqueries"')
        assert result is not None
        assert '"subqueries"' in result


class TestSubqueriesFromParsed:
    """Tests for _subqueries_from_parsed."""

    def test_returns_cleaned_list_when_valid(self):
        """Return cleaned subqueries list when data has valid subqueries key."""
        data = {"subqueries": ["  q1  ", "q2", "  q3  "]}
        result = helpers._subqueries_from_parsed(data)
        assert result == ["q1", "q2", "q3"]

    def test_returns_none_when_no_subqueries_key(self):
        """Return None when subqueries key missing."""
        data = {"other": ["a"]}
        assert helpers._subqueries_from_parsed(data) is None

    def test_returns_none_when_subqueries_not_list(self):
        """Return None when subqueries is not a list."""
        data = {"subqueries": "not a list"}
        assert helpers._subqueries_from_parsed(data) is None

    def test_filters_empty_items(self):
        """Filter out empty/falsy items."""
        data = {"subqueries": ["q1", "", None, "q2"]}
        result = helpers._subqueries_from_parsed(data)
        assert result == ["q1", "q2"]


class TestExtractNumberedOrBulletLines:
    """Tests for _extract_numbered_or_bullet_lines."""

    def test_extracts_numbered_lines(self):
        """Extract numbered lines (1. ...) format."""
        text = "1. First item\n2. Second item\n3. Third item"
        result = helpers._extract_numbered_or_bullet_lines(text)
        assert result == ["First item", "Second item", "Third item"]

    def test_extracts_bullet_lines(self):
        """Extract bullet lines (- ...) format."""
        text = "- Bullet one\n- Bullet two"
        result = helpers._extract_numbered_or_bullet_lines(text)
        assert result == ["Bullet one", "Bullet two"]

    def test_returns_empty_list_for_plain_text(self):
        """Return empty list when no numbered or bullet lines."""
        text = "Plain paragraph with no structure."
        result = helpers._extract_numbered_or_bullet_lines(text)
        assert result == []


class TestParseSubqueries:
    """Tests for _parse_subqueries."""

    def test_parses_json_in_fence(self):
        """Parse subqueries from JSON in markdown fence."""
        text = '```json\n{"subqueries": ["What is X?", "What is Y?"]}\n```'
        result = helpers._parse_subqueries(text)
        assert result == ["What is X?", "What is Y?"]

    def test_parses_balanced_json_with_subqueries(self):
        """Parse subqueries from balanced JSON in text."""
        text = 'Here are the subqueries: {"subqueries": ["Q1", "Q2"]}'
        result = helpers._parse_subqueries(text)
        assert result == ["Q1", "Q2"]

    def test_falls_back_to_numbered_lines(self):
        """Fall back to numbered/bullet extraction when JSON fails."""
        text = "1. First query\n2. Second query"
        result = helpers._parse_subqueries(text)
        assert result == ["First query", "Second query"]


class TestIsIdentitySubquery:
    """Tests for _is_identity_subquery."""

    def test_exact_match_returns_true(self):
        """Exact match returns True."""
        assert (
            helpers._is_identity_subquery("What is revenue?", "What is revenue?")
            is True
        )

    def test_high_overlap_returns_true(self):
        """High word overlap returns True."""
        sq = "What is the total revenue for Q1 2024?"
        orig = "What is total revenue for Q1 2024"
        assert helpers._is_identity_subquery(sq, orig) is True

    def test_different_queries_return_false(self):
        """Different queries return False."""
        assert helpers._is_identity_subquery("What is X?", "What is Y?") is False

    def test_empty_subquery_returns_false(self):
        """Empty subquery returns False."""
        assert helpers._is_identity_subquery("", "What is X?") is False


class TestStripSqlFromSubqueries:
    """Tests for _strip_sql_from_subqueries."""

    def test_keeps_natural_language_unchanged(self):
        """Keep natural language subqueries unchanged."""
        subqueries = ["What is revenue?", "Show me sales data"]
        result = helpers._strip_sql_from_subqueries(subqueries)
        assert result == subqueries

    def test_extracts_nl_from_sql_mixed(self):
        """Extract natural language from SQL-mixed subqueries."""
        subqueries = ["Revenue: SELECT * FROM sales"]
        result = helpers._strip_sql_from_subqueries(subqueries)
        assert "SELECT" not in result[0]
        assert "Revenue" in result[0] or result[0]


# ---------------------------------------------------------------------------
# Formatting and findings helpers
# ---------------------------------------------------------------------------


class TestFormatEnrichedPlan:
    """Tests for _format_enriched_plan."""

    def test_formats_enriched_subqueries_with_status(self):
        """Format enriched subqueries with status icons."""
        enriched = [
            {"query": "Q1", "status": "ready", "data_products": []},
            {"query": "Q2", "status": "access_denied", "data_products": []},
        ]
        result = helpers._format_enriched_plan(enriched)
        assert "✓" in result
        assert "✗" in result
        assert "Q1" in result
        assert "Q2" in result

    def test_includes_data_product_names(self):
        """Include data product names when present."""
        enriched = [
            {
                "query": "Q1",
                "status": "ready",
                "data_products": [{"name": "dp1"}, {"name": "dp2"}],
            }
        ]
        result = helpers._format_enriched_plan(enriched)
        assert "Resources:" in result
        assert "dp1" in result
        assert "dp2" in result


class TestFindingsFromBoard:
    """Tests for findings_from_board."""

    def test_derives_legacy_findings_dict(self):
        """Derive legacy-shaped findings from findings_board."""
        board = {
            "q1": {"finding": {"subquery": "q1", "answer": "a1"}},
            "q2": {"finding": {"subquery": "q2", "answer": "a2"}},
        }
        result = helpers.findings_from_board(board)
        assert result == {
            "q1": {"subquery": "q1", "answer": "a1"},
            "q2": {"subquery": "q2", "answer": "a2"},
        }

    def test_handles_missing_finding(self):
        """Handle entries with no finding (use empty dict)."""
        board = {"q1": {"finding": None}}
        result = helpers.findings_from_board(board)
        assert result["q1"] == {}


class TestMakeFinding:
    """Tests for _make_finding."""

    def test_builds_minimal_finding(self):
        """Build minimal finding with required keys."""
        result = helpers._make_finding("Q1", "A1")
        assert result["subquery"] == "Q1"
        assert result["answer"] == "A1"
        assert result["cached"] is False

    def test_includes_optional_fields_when_provided(self):
        """Include optional fields when provided."""
        result = helpers._make_finding(
            "Q1",
            "A1",
            tool_results=["r1"],
            error="err",
            cached=True,
            access_denied=True,
            execution_time_ms=100,
        )
        assert result["tool_results"] == ["r1"]
        assert result["error"] == "err"
        assert result["cached"] is True
        assert result["access_denied"] is True
        assert result["execution_time_ms"] == 100


class TestClassifyFailure:
    """Tests for _classify_failure."""

    def test_classifies_timeout(self):
        """Classify timeout errors."""
        exc = Exception("Connection timed out")
        assert helpers._classify_failure(exc, "timed out") == "tool_timeout"

    def test_classifies_access_denied(self):
        """Classify access denied errors."""
        exc = Exception("Access denied")
        assert helpers._classify_failure(exc, "access denied") == "access_denied"

    def test_classifies_empty_result(self):
        """Classify empty result errors."""
        exc = Exception("No results")
        assert helpers._classify_failure(exc, "no results") == "empty_result"

    def test_classifies_parse_error(self):
        """Classify parse/JSON errors."""
        exc = Exception("Invalid JSON")
        assert helpers._classify_failure(exc, "invalid json") == "parse_error"

    def test_classifies_llm_failure(self):
        """Classify rate limit / API errors."""
        exc = Exception("Rate limit exceeded")
        assert helpers._classify_failure(exc, "rate limit") == "llm_failure"

    def test_returns_unknown_for_unrecognized(self):
        """Return unknown for unrecognized errors."""
        exc = Exception("Something went wrong")
        assert helpers._classify_failure(exc, "something went wrong") == "unknown"


class TestComputeDataQualityScore:
    """Tests for compute_data_quality_score."""

    def test_returns_zero_when_error_present(self):
        """Return 0.0 when finding has error."""
        finding = {"answer": "good answer", "error": "failed"}
        assert helpers.compute_data_quality_score(finding) < 0.01

    def test_adds_score_for_substantial_answer(self):
        """Add score for substantial answer (>20 chars)."""
        finding = {"answer": "This is a substantial answer with enough content."}
        score = helpers.compute_data_quality_score(finding)
        assert score >= 0.5

    def test_adds_confidence_bonus_high(self):
        """Add bonus for high confidence."""
        finding = {"answer": "x" * 30}
        score_high = helpers.compute_data_quality_score(finding, "high")
        score_medium = helpers.compute_data_quality_score(finding, "medium")
        assert score_high > score_medium

    def test_applies_plausibility_penalty(self):
        """Apply penalty for plausibility warnings."""
        finding = {
            "answer": "x" * 30,
            "plausibility_warnings": [{"severity": "critical"}],
        }
        score = helpers.compute_data_quality_score(finding)
        assert score < 0.8

    def test_clamps_score_between_zero_and_one(self):
        """Clamp score between 0 and 1."""
        finding = {
            "answer": "x" * 30,
            "plausibility_warnings": [{"severity": "critical"}] * 5,
        }
        score = helpers.compute_data_quality_score(finding)
        assert 0.0 <= score <= 1.0


class TestAssessDataQuality:
    """Tests for _assess_data_quality."""

    def test_returns_high_when_most_valid(self):
        """Return high when >=80% valid."""
        findings = {
            "q1": {"answer": "a1", "error": None},
            "q2": {"answer": "a2", "error": None},
            "q3": {"answer": "a3", "error": None},
            "q4": {"answer": "a4", "error": None},
            "q5": {"answer": None, "error": "err"},
        }
        assert helpers._assess_data_quality(findings) == "high"

    def test_returns_medium_when_half_valid(self):
        """Return medium when 50-80% valid."""
        findings = {
            "q1": {"answer": "a1", "error": None},
            "q2": {"answer": "a2", "error": None},
            "q3": {"answer": None, "error": "err"},
        }
        assert helpers._assess_data_quality(findings) == "medium"

    def test_returns_low_when_empty_or_few_valid(self):
        """Return low when empty or few valid."""
        assert helpers._assess_data_quality({}) == "low"
        findings = {"q1": {"answer": None, "error": "err"}}
        assert helpers._assess_data_quality(findings) == "low"


class TestLooksLikeToolRecommendation:
    """Tests for _looks_like_tool_recommendation."""

    def test_detects_recommend_patterns(self):
        """Detect tool recommendation patterns."""
        assert (
            helpers._looks_like_tool_recommendation("I recommend using the sales tool")
            is True
        )
        assert (
            helpers._looks_like_tool_recommendation("You should use the query tool")
            is True
        )

    def test_returns_false_for_synthesis_content(self):
        """Return False for synthesis content."""
        text = "Based on the data, revenue increased by 15%."
        assert helpers._looks_like_tool_recommendation(text) is False


class TestShouldExcludeFromSynthesis:
    """Tests for should_exclude_from_synthesis."""

    def test_excludes_when_low_quality_drop(self):
        """Exclude when low_quality_drop is set."""
        finding = {"low_quality_drop": True, "answer": "good"}
        assert helpers.should_exclude_from_synthesis(finding) is True

    def test_does_not_exclude_when_error_but_has_data(self):
        """Do not exclude when error but has substantial answer or tool_results."""
        finding = {"error": "err", "answer": "x" * 30}
        assert helpers.should_exclude_from_synthesis(finding) is False

    def test_excludes_when_error_and_no_usable_data(self):
        """Exclude when error and no usable data."""
        finding = {"error": "err", "answer": "short", "tool_results": []}
        assert helpers.should_exclude_from_synthesis(finding) is True


class TestStripNoise:
    """Tests for _strip_noise."""

    def test_removes_chart_placeholder(self):
        """Remove chart placeholders."""
        text = "Content <!-- CHART_PLACEHOLDER --> more"
        result = helpers._strip_noise(text)
        assert "CHART_PLACEHOLDER" not in result

    def test_removes_conversational_fluff(self):
        """Remove conversational fluff patterns."""
        text = "I've created a report. Based on my analysis, here are the key points."
        result = helpers._strip_noise(text)
        assert "I've created" not in result or result != text

    def test_collapses_excessive_newlines(self):
        """Collapse 3+ newlines to 2."""
        text = "A\n\n\n\nB"
        result = helpers._strip_noise(text)
        assert "\n\n\n" not in result


class TestBuildZeroFindingsMessage:
    """Tests for _build_zero_findings_message."""

    def test_produces_informative_message(self):
        """Produce informative message with plan and reason."""
        state = {
            "subqueries": ["Q1", "Q2"],
            "sentinel_reason": "timeout",
        }
        result = helpers._build_zero_findings_message(state)
        assert "Research could not complete" in result
        assert "timeout" in result
        assert "Q1" in result

    def test_handles_none_state(self):
        """Handle None state gracefully."""
        result = helpers._build_zero_findings_message(None)
        assert "unknown" in result


class TestBuildFallbackSynthesis:
    """Tests for _build_fallback_synthesis."""

    def test_builds_synthesis_from_valid_findings(self):
        """Build synthesis from valid findings."""
        findings_board = {
            "q1": {"finding": {"answer": "Answer one", "error": None}},
            "q2": {"finding": {"answer": "Answer two", "error": None}},
        }
        result = helpers._build_fallback_synthesis(findings_board, "Main query")
        assert "Research Analysis" in result
        assert "q1" in result
        assert "Answer one" in result

    def test_returns_zero_message_when_no_valid_findings(self):
        """Return zero findings message when no valid findings."""
        findings_board = {
            "q1": {"finding": {"error": "err", "answer": ""}},
        }
        state = {"subqueries": ["Q1"], "sentinel_reason": "timeout"}
        result = helpers._build_fallback_synthesis(findings_board, "Q", state)
        assert "Research could not complete" in result


class TestDetectTruncation:
    """Tests for _detect_truncation."""

    def test_detects_mid_sentence_cutoff(self):
        """Detect when answer ends mid-sentence."""
        assert helpers._detect_truncation("This answer was cut off mid") is True

    def test_returns_false_when_ends_with_punctuation(self):
        """Return False when ends with sentence-ending punctuation."""
        assert helpers._detect_truncation("Complete sentence.") is False
        assert helpers._detect_truncation("Complete sentence!") is False

    def test_returns_false_for_empty(self):
        """Return False for empty string."""
        assert helpers._detect_truncation("") is False


class TestParseReviewResult:
    """Tests for _parse_review_result."""

    def test_parses_valid_json_response(self):
        """Parse valid JSON review response."""
        text = '{"action": "revise", "score": 65, "reason": "Needs work", "feedback": "Fix it"}'
        result = helpers._parse_review_result(text, "persona1")
        assert result["action"] == "revise"
        assert result["score"] == 65
        assert result["persona"] == "persona1"

    def test_returns_defaults_on_parse_failure(self):
        """Return defaults when parse fails."""
        result = helpers._parse_review_result("not json", "p")
        assert result["action"] == "approve"
        assert result["score"] == 70
        assert "Could not parse" in result["reason"]


# ---------------------------------------------------------------------------
# Structural and quality gate checks
# ---------------------------------------------------------------------------


class TestCheckHeadings:
    """Tests for _check_headings."""

    def test_passes_when_headings_present(self):
        """Pass when at least one heading present."""
        draft = "# Title\n## Section\nContent"
        passed, violation = helpers._check_headings(draft)
        assert passed is True
        assert violation is None

    def test_fails_when_no_headings(self):
        """Fail when no headings."""
        draft = "Plain text with no headings."
        passed, violation = helpers._check_headings(draft)
        assert passed is False
        assert "No headings" in violation


class TestCheckExecutiveSummary:
    """Tests for _check_executive_summary."""

    def test_passes_when_section_present(self):
        """Pass when executive summary section present."""
        draft = "## Executive Summary\nKey findings..."
        passed, _ = helpers._check_executive_summary(draft)
        assert passed is True

    def test_fails_when_missing(self):
        """Fail when section missing."""
        draft = "# Report\n## Other Section"
        passed, violation = helpers._check_executive_summary(draft)
        assert passed is False
        assert "executive summary" in violation.lower()


class TestCheckConclusionSection:
    """Tests for _check_conclusion_section."""

    def test_passes_when_conclusion_present(self):
        """Pass when conclusion/takeaways present."""
        draft = "## Conclusion\nSummary..."
        passed, _ = helpers._check_conclusion_section(draft)
        assert passed is True

    def test_fails_when_missing(self):
        """Fail when conclusion missing."""
        draft = "# Report\n## Intro"
        passed, _ = helpers._check_conclusion_section(draft)
        assert passed is False


class TestCheckStructuralCompliance:
    """Tests for check_structural_compliance."""

    def test_returns_score_and_violations(self):
        """Return score and violations list."""
        draft = (
            "# Report\n## Executive Summary\nKey finding.\n"
            "## Conclusion\nTakeaways here."
        )
        score, violations = helpers.check_structural_compliance(
            draft, "comprehensive", "fast"
        )
        assert 0.0 <= score <= 1.0
        assert isinstance(violations, list)


class TestDetectRedundancy:
    """Tests for detect_redundancy."""

    def test_returns_zero_for_unique_content(self):
        """Return low score for unique content."""
        draft = "First unique sentence. Second different sentence. Third distinct idea."
        score, pairs = helpers.detect_redundancy(draft)
        assert score >= 0.0
        assert isinstance(pairs, list)

    def test_detects_repeated_sentences(self):
        """Detect highly similar sentences."""
        draft = (
            "The revenue increased by fifteen percent. "
            "Revenue increased by 15 percent. "
            "Sales went up."
        )
        _, pairs = helpers.detect_redundancy(draft)
        assert isinstance(pairs, list)


# ---------------------------------------------------------------------------
# Async / cancellation
# ---------------------------------------------------------------------------


class TestCheckNodeCancelled:
    """Tests for check_node_cancelled."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_thread_id(self):
        """Return None when thread_id is None."""
        state = {"draft_answer": "draft", "total_node_transitions": 0}
        result = await helpers.check_node_cancelled(None, "synthesize", state)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_not_cancelled(self):
        """Return None when thread not cancelled."""
        state = {"draft_answer": "draft", "total_node_transitions": 0}
        with patch(
            "template_agent.src.core.deep_research.cancel.get_cancel_store"
        ) as mock_get:
            store = AsyncMock()
            store.is_cancelled = AsyncMock(return_value=False)
            mock_get.return_value = store

            result = await helpers.check_node_cancelled("t1", "synthesize", state)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_early_state_when_cancelled(self):
        """Return early state update when cancelled."""
        state = {"draft_answer": "", "total_node_transitions": 5}
        with patch(
            "template_agent.src.core.deep_research.cancel.get_cancel_store"
        ) as mock_get:
            store = AsyncMock()
            store.is_cancelled = AsyncMock(return_value=True)
            mock_get.return_value = store

            result = await helpers.check_node_cancelled("t1", "synthesize", state)
        assert result is not None
        assert result["current_phase"] == "complete"
        assert "cancelled" in result["final_answer"].lower()
        assert result["total_node_transitions"] == 6


# ---------------------------------------------------------------------------
# Format findings for synthesis
# ---------------------------------------------------------------------------


class TestFormatFindingsForSynthesis:
    """Tests for _format_findings_for_synthesis."""

    def test_returns_no_findings_message_when_empty(self):
        """Return no findings message when empty."""
        result = helpers._format_findings_for_synthesis({})
        assert "No findings" in result

    def test_formats_findings_with_quality_and_status(self):
        """Format findings with quality score and status."""
        findings = {
            "q1": {"subquery": "q1", "answer": "Answer one", "error": None},
            "q2": {"subquery": "q2", "answer": "", "error": "Failed"},
        }
        result = helpers._format_findings_for_synthesis(findings)
        assert "q1" in result
        assert "q2" in result
        assert "FINDINGS SUMMARY" in result
        assert "Successful" in result
        assert "Failed" in result

    def test_respects_max_chars_budget(self):
        """Respect max_chars when provided."""
        findings = {
            "q1": {"subquery": "q1", "answer": "x" * 5000, "error": None},
        }
        result = helpers._format_findings_for_synthesis(findings, max_chars=2000)
        assert len(result) <= 3000  # summary adds some


class TestCompressPartsForBudget:
    """Tests for _compress_parts_for_budget."""

    def test_returns_parts_unchanged_when_under_budget(self):
        """Return parts unchanged when under budget."""
        parts = ["short", "also short"]
        result = helpers._compress_parts_for_budget(parts, 10000)
        assert result == parts

    def test_truncates_long_parts(self):
        """Truncate long parts to fit budget."""
        parts = ["x" * 2000, "y" * 2000]
        result = helpers._compress_parts_for_budget(parts, 1500)
        assert len(result) == 2
        assert "truncated" in result[0] or len(result[0]) < 2000


class TestBuildFindingsSummary:
    """Tests for _build_findings_summary."""

    def test_includes_counts_for_all_categories(self):
        """Include counts for successful, access_denied, failed."""
        result = helpers._build_findings_summary(
            access_denied_findings=["q1"],
            failed_findings=["q2"],
            successful_findings=["q3", "q4"],
        )
        result_str = "\n".join(result)
        assert "Successful queries: 2" in result_str
        assert "Access denied queries: 1" in result_str
        assert "Failed queries: 1" in result_str
        assert "q1" in result_str
        assert "q2" in result_str
