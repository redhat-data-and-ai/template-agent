"""Comprehensive pytest tests for the deep research SUPERVISOR node."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from template_agent.src.core.deep_research.nodes._helpers import SubAgentResult
from template_agent.src.core.deep_research.nodes.supervisor import (
    MassFailureInput,
    SupervisorContinueInput,
    SupervisorStateBase,
    WorkerResultContext,
    _apply_conflict_metadata,
    _apply_plausibility_penalties,
    _build_context_prefix,
    _build_supervisor_completeness_return,
    _build_supervisor_state,
    _check_mass_failure,
    _create_error_finding,
    _create_timeout_finding,
    _detect_and_resolve_conflicts,
    _detect_delegation_cycle,
    _extract_answers_for_conflict_detection,
    _finalize_best_finding,
    _generate_alternative_approach,
    _get_cached_subagent_result,
    _get_status_error_result,
    _is_non_retryable_failure,
    _process_and_deposit_worker_result,
    _retry_loop_handle_exception,
    _supervisor_check_all_completed,
    _supervisor_check_empty_plan,
    _supervisor_check_sentinel,
    _supervisor_run_debate,
    _try_continue_research,
)
from template_agent.src.core.deep_research.state import (
    PHASE_COMPLETENESS,
    PHASE_SUPERVISOR,
    PHASE_SYNTHESIZE,
    FindingEntry,
    ResearchContext,
    SupervisorRound,
)


def _make_ctx(**overrides):
    """Create minimal ResearchContext for testing."""
    base_model = AsyncMock()
    ctx = ResearchContext(tools=[MagicMock(name="tool1")], base_model=base_model)
    for k, v in overrides.items():
        setattr(ctx, k, v)
    return ctx


def _make_state(**overrides):
    """Create minimal DeepResearchState dict with supervisor context."""
    base = {
        "query": "test query",
        "thread_id": "t1",
        "current_phase": "supervisor",
        "subqueries": ["sq1", "sq2"],
        "findings_board": {},
        "supervisor_rounds": [],
        "current_round": 0,
        "max_rounds": 3,
        "agent_messages": [],
        "pending_subqueries": ["sq1", "sq2"],
        "completed_subqueries": [],
    }
    base.update(overrides)
    return base


class TestExtractAnswersForConflictDetection:
    """Tests for _extract_answers_for_conflict_detection."""

    def test_extract_answers_returns_empty_when_no_findings(self):
        """Empty recent_findings yields empty answers and quality_scores."""
        findings_board = {}
        recent_findings = []

        answers, quality_scores = _extract_answers_for_conflict_detection(
            findings_board, recent_findings
        )

        assert answers == {}
        assert quality_scores == {}

    def test_extract_answers_skips_entries_with_error(self):
        """Findings with error are excluded from answers."""
        findings_board = {
            "sq1": FindingEntry(
                finding={"subquery": "sq1", "answer": "ans1", "error": "failed"},
                quality_score=0.5,
            ),
        }
        recent_findings = ["sq1"]

        answers, quality_scores = _extract_answers_for_conflict_detection(
            findings_board, recent_findings
        )

        assert answers == {}
        assert quality_scores == {}

    def test_extract_answers_includes_valid_findings(self):
        """Valid findings with answers are extracted."""
        findings_board = {
            "sq1": FindingEntry(
                finding={"subquery": "sq1", "answer": "Answer one"},
                quality_score=0.8,
                data_quality_score=0.85,
            ),
            "sq2": FindingEntry(
                finding={"subquery": "sq2", "answer": "Answer two"},
                quality_score=0.7,
            ),
        }
        recent_findings = ["sq1", "sq2"]

        answers, quality_scores = _extract_answers_for_conflict_detection(
            findings_board, recent_findings
        )

        assert answers["sq1"] == "Answer one"
        assert answers["sq2"] == "Answer two"
        assert "sq1" in quality_scores
        assert "sq2" in quality_scores

    def test_extract_answers_skips_missing_board_entries(self):
        """Subqueries not in findings_board are skipped."""
        findings_board = {"sq1": FindingEntry(finding={"answer": "a1"})}
        recent_findings = ["sq1", "sq_nonexistent"]

        answers, _ = _extract_answers_for_conflict_detection(
            findings_board, recent_findings
        )

        assert "sq1" in answers
        assert "sq_nonexistent" not in answers


class TestApplyConflictMetadata:
    """Tests for _apply_conflict_metadata."""

    def test_apply_conflict_metadata_adds_has_conflict_tag(self):
        """Conflicting findings get has_conflict tag."""
        findings_board = {
            "sq1": FindingEntry(finding={"answer": "a1"}, tags=[]),
            "sq2": FindingEntry(finding={"answer": "a2"}, tags=[]),
        }
        answers = {"sq1": "a1", "sq2": "a2"}
        conflicts = [{"finding_indices": [1, 2]}]

        _apply_conflict_metadata(findings_board, answers, conflicts)

        assert "has_conflict" in (findings_board["sq1"].get("tags") or [])
        assert "has_conflict" in (findings_board["sq2"].get("tags") or [])

    def test_apply_conflict_metadata_does_not_duplicate_tag(self):
        """Already tagged entries are not duplicated."""
        findings_board = {
            "sq1": FindingEntry(finding={"answer": "a1"}, tags=["has_conflict"]),
        }
        answers = {"sq1": "a1"}
        conflicts = [{"finding_indices": [1]}]

        _apply_conflict_metadata(findings_board, answers, conflicts)

        tags = findings_board["sq1"].get("tags") or []
        assert tags.count("has_conflict") == 1

    def test_apply_conflict_metadata_ignores_invalid_indices(self):
        """Indices out of range are ignored."""
        findings_board = {"sq1": FindingEntry(finding={"answer": "a1"}, tags=[])}
        answers = {"sq1": "a1"}
        conflicts = [{"finding_indices": [99, 0]}]

        _apply_conflict_metadata(findings_board, answers, conflicts)

        assert "has_conflict" not in (findings_board["sq1"].get("tags") or [])


class TestApplyPlausibilityPenalties:
    """Tests for _apply_plausibility_penalties."""

    def test_plausible_returns_unchanged(self):
        """Plausible findings keep original score and confidence."""
        score, conf = _apply_plausibility_penalties(0.8, "high", {"plausible": True})
        assert score == pytest.approx(0.8)
        assert conf == "high"

    def test_implausible_applies_penalty(self):
        """Implausible findings get penalty and confidence downgrade."""
        plausibility = {
            "plausible": False,
            "warnings": [{"severity": "major"}],
        }
        score, conf = _apply_plausibility_penalties(0.8, "high", plausibility)
        assert score < 0.8
        assert conf == "medium"

    def test_critical_severity_applies_larger_penalty(self):
        """Critical severity applies max penalty."""
        plausibility = {
            "plausible": False,
            "warnings": [{"severity": "critical"}],
        }
        score, _ = _apply_plausibility_penalties(0.9, "high", plausibility)
        assert score < 0.7


class TestIsNonRetryableFailure:
    """Tests for _is_non_retryable_failure."""

    def test_access_denied_is_non_retryable(self):
        """access_denied failures are not retried."""
        assert _is_non_retryable_failure("access_denied", 1, 3) is True

    def test_tool_timeout_is_non_retryable(self):
        """tool_timeout failures are not retried."""
        assert _is_non_retryable_failure("tool_timeout", 1, 3) is True

    def test_last_attempt_is_non_retryable(self):
        """Last attempt is non-retryable regardless of failure class."""
        assert _is_non_retryable_failure("parse_error", 2, 2) is True

    def test_retryable_failure_returns_false(self):
        """Retryable failures return False."""
        assert _is_non_retryable_failure("parse_error", 1, 3) is False


class TestCreateErrorFinding:
    """Tests for _create_error_finding."""

    def test_create_error_finding_sets_error_and_failure_class(self):
        """Error finding has error message and failure_class."""
        finding = _create_error_finding("sq1", "Access denied", "access_denied")
        assert finding["error"] == "Access denied"
        assert finding["failure_class"] == "access_denied"
        assert finding["subquery"] == "sq1"


class TestCreateTimeoutFinding:
    """Tests for _create_timeout_finding."""

    def test_create_timeout_finding_sets_tool_timeout(self):
        """Timeout finding has tool_timeout failure class."""
        finding = _create_timeout_finding("sq1")
        assert finding["error"] == "Execution timed out"
        assert finding["failure_class"] == "tool_timeout"


class TestBuildContextPrefix:
    """Tests for _build_context_prefix."""

    def test_first_attempt_returns_prefix(self):
        """First attempt returns non-empty prefix."""
        prefix = _build_context_prefix("cross context", 1)
        assert len(prefix) > 0

    def test_subsequent_attempts_return_empty(self):
        """Attempt > 1 returns empty string."""
        assert _build_context_prefix("cross context", 2) == ""


class TestRetryLoopHandleException:
    """Tests for _retry_loop_handle_exception."""

    def test_non_retryable_returns_early_tuple(self):
        """Non-retryable failure returns early return tuple."""
        result = _retry_loop_handle_exception(
            Exception("access denied"),
            "sq1",
            attempt=1,
            effective_retries=3,
            index=0,
            best_finding=None,
            best_score=0.0,
            prior_failure_class=None,
        )
        early_ret, _, _ = result
        assert early_ret is not None
        assert early_ret[0]["error"] is not None
        assert early_ret[2] == "low"

    def test_retryable_returns_none_and_new_best(self):
        """Retryable failure returns None for early_ret and new best."""
        result = _retry_loop_handle_exception(
            Exception("parse error"),
            "sq1",
            attempt=1,
            effective_retries=3,
            index=0,
            best_finding=None,
            best_score=0.0,
            prior_failure_class=None,
        )
        early_ret, new_best, new_prior = result
        assert early_ret is None
        assert new_best is not None
        assert new_prior == "parse_error"


class TestFinalizeBestFinding:
    """Tests for _finalize_best_finding."""

    def test_plausibility_warnings_set_data_quality_alert(self):
        """Plausibility warnings set data_quality_alert."""
        finding = {"answer": "ans", "plausibility_warnings": [{"severity": "minor"}]}
        _finalize_best_finding(finding, None, 0.8, "sq1")
        assert finding.get("data_quality_alert") is True

    def test_low_score_sets_low_quality_drop(self):
        """Score below threshold sets low_quality_drop."""
        finding = {"answer": "ans"}
        _finalize_best_finding(finding, None, 0.2, "sq1")
        assert finding.get("low_quality_drop") is True

    def test_prior_failure_class_propagated(self):
        """Prior failure class is set when no error in finding."""
        finding = {"answer": "ans"}
        _finalize_best_finding(finding, "parse_error", 0.5, "sq1")
        assert finding.get("failure_class") == "parse_error"


class TestGetStatusErrorResult:
    """Tests for _get_status_error_result."""

    def test_access_denied_returns_subagent_result(self):
        """access_denied status returns SubAgentResult."""
        result = _get_status_error_result("sq1", 1, 2, "access_denied")
        assert result is not None
        assert result.finding["error"] == "Access denied"
        assert result.quality_score == pytest.approx(0.0)

    def test_no_data_products_returns_subagent_result(self):
        """no_data_products status returns SubAgentResult."""
        result = _get_status_error_result("sq1", 1, 2, "no_data_products")
        assert result is not None
        assert "No data sources" in result.finding["error"]

    def test_ready_returns_none(self):
        """ready status returns None."""
        assert _get_status_error_result("sq1", 1, 2, "ready") is None


class TestDetectDelegationCycle:
    """Tests for _detect_delegation_cycle."""

    def test_empty_rounds_returns_proposed_unchanged(self):
        """Empty supervisor_rounds returns proposed follow-ups as-is."""
        proposed = ["new query one", "new query two"]
        result = _detect_delegation_cycle([], proposed)
        assert result == proposed

    def test_repeated_follow_up_filtered(self):
        """Follow-up that repeats prior delegated query is filtered."""
        rounds = [
            SupervisorRound(delegated_subqueries=["What is revenue in Q1?"]),
        ]
        proposed = ["What is revenue in Q1?"]
        result = _detect_delegation_cycle(rounds, proposed)
        assert len(result) < len(proposed)

    def test_novel_follow_up_retained(self):
        """Novel follow-up is retained."""
        rounds = [
            SupervisorRound(delegated_subqueries=["What is revenue?"]),
        ]
        proposed = ["What is profit margin?"]
        result = _detect_delegation_cycle(rounds, proposed)
        assert "What is profit margin?" in result


class TestBuildSupervisorState:
    """Tests for _build_supervisor_state."""

    def test_build_supervisor_state_includes_phase(self):
        """State includes current_phase and base fields."""
        base = SupervisorStateBase(
            execution_start_update={},
            findings_board={},
            agent_messages=[],
            supervisor_rounds=[],
            completed=[],
            round_num=1,
            transitions=2,
            immediate_context=None,
            finding_cards=[],
            research_memory=None,
        )
        state = _build_supervisor_state(base, current_phase=PHASE_SUPERVISOR)
        assert state["current_phase"] == PHASE_SUPERVISOR
        assert state["current_round"] == 1
        assert state["total_node_transitions"] == 2

    def test_build_supervisor_state_includes_extra_kwargs(self):
        """Extra kwargs are merged into state."""
        base = SupervisorStateBase(
            execution_start_update={},
            findings_board={},
            agent_messages=[],
            supervisor_rounds=[],
            completed=[],
            round_num=1,
            transitions=1,
            immediate_context=None,
            finding_cards=[],
            research_memory=None,
        )
        state = _build_supervisor_state(
            base,
            current_phase=PHASE_COMPLETENESS,
            total_subqueries_executed=5,
            research_abort_reason="test",
        )
        assert state["total_subqueries_executed"] == 5
        assert state["research_abort_reason"] == "test"


class TestTryContinueResearch:
    """Tests for _try_continue_research."""

    def test_continue_research_with_follow_ups_returns_state(self):
        """continue_research decision with follow-ups returns state update."""
        ctx = _make_ctx()
        events = []
        inp = SupervisorContinueInput(
            decision="continue_research",
            round_num=1,
            max_rounds=3,
            follow_ups=["new sq 1"],
            subqueries=["sq1", "sq2"],
            to_execute=["sq1"],
            completed=[],
            supervisor_rounds=[],
            total_sq_executed=1,
            state=_make_state(),
            execution_start_update={},
            findings_board={},
            agent_messages=[],
            transitions=1,
            immediate_context=None,
            finding_cards=[],
            research_memory=None,
            findings_count_history=[],
            fallback_count=0,
            ctx=ctx,
            events=events,
        )
        result = _try_continue_research(inp)
        assert result is not None
        state_update, _ = result
        assert state_update["current_phase"] == PHASE_SUPERVISOR
        assert "pending_subqueries" in state_update

    def test_proceed_decision_returns_none(self):
        """proceed_to_completeness decision returns None."""
        inp = SupervisorContinueInput(
            decision="proceed_to_completeness",
            round_num=1,
            max_rounds=3,
            follow_ups=[],
            subqueries=[],
            to_execute=[],
            completed=[],
            supervisor_rounds=[],
            total_sq_executed=0,
            state=_make_state(),
            execution_start_update={},
            findings_board={},
            agent_messages=[],
            transitions=1,
            immediate_context=None,
            finding_cards=[],
            research_memory=None,
            findings_count_history=[],
            fallback_count=0,
            ctx=_make_ctx(),
            events=[],
        )
        assert _try_continue_research(inp) is None

    def test_max_rounds_reached_returns_none(self):
        """When round_num >= max_rounds, returns None."""
        inp = SupervisorContinueInput(
            decision="continue_research",
            round_num=3,
            max_rounds=3,
            follow_ups=["sq"],
            subqueries=[],
            to_execute=[],
            completed=[],
            supervisor_rounds=[],
            total_sq_executed=0,
            state=_make_state(),
            execution_start_update={},
            findings_board={},
            agent_messages=[],
            transitions=1,
            immediate_context=None,
            finding_cards=[],
            research_memory=None,
            findings_count_history=[],
            fallback_count=0,
            ctx=_make_ctx(),
            events=[],
        )
        assert _try_continue_research(inp) is None


class TestCheckMassFailure:
    """Tests for _check_mass_failure."""

    def test_high_success_rate_returns_none(self):
        """Success rate >= 20% returns None."""
        findings_board = {
            "sq1": FindingEntry(finding={"answer": "a1"}),
            "sq2": FindingEntry(finding={"answer": "a2"}),
        }
        inp = MassFailureInput(
            to_execute=["sq1", "sq2"],
            findings_received=["sq1", "sq2"],
            findings_board=findings_board,
            round_num=2,
            execution_start_update={},
            agent_messages=[],
            supervisor_rounds=[],
            completed=[],
            transitions=1,
            immediate_context=None,
            finding_cards=[],
            research_memory=None,
            total_sq_executed=2,
            findings_count_history=[],
            ctx=_make_ctx(),
            events=[],
        )
        assert _check_mass_failure(inp) is None

    def test_mass_failure_round_0_returns_none(self):
        """Round 0 (round_num < 1) never triggers mass failure."""
        findings_board = {
            "sq1": FindingEntry(finding={"error": "failed"}),
            "sq2": FindingEntry(finding={"error": "failed"}),
        }
        inp = MassFailureInput(
            to_execute=["sq1", "sq2"],
            findings_received=["sq1", "sq2"],
            findings_board=findings_board,
            round_num=0,
            execution_start_update={},
            agent_messages=[],
            supervisor_rounds=[],
            completed=[],
            transitions=1,
            immediate_context=None,
            finding_cards=[],
            research_memory=None,
            total_sq_executed=2,
            findings_count_history=[],
            ctx=_make_ctx(),
            events=[],
        )
        assert _check_mass_failure(inp) is None

    def test_mass_failure_returns_synthesize_phase(self):
        """Mass failure routes to PHASE_SYNTHESIZE when success_rate < 20%."""
        findings_board = {
            "sq1": FindingEntry(finding={"error": "failed"}),
            "sq2": FindingEntry(finding={"error": "failed"}),
            "sq3": FindingEntry(finding={"error": "failed"}),
            "sq4": FindingEntry(finding={"error": "failed"}),
            "sq5": FindingEntry(finding={"error": "failed"}),
        }
        inp = MassFailureInput(
            to_execute=["sq1", "sq2", "sq3", "sq4", "sq5"],
            findings_received=["sq1", "sq2", "sq3", "sq4", "sq5"],
            findings_board=findings_board,
            round_num=2,
            execution_start_update={},
            agent_messages=[],
            supervisor_rounds=[],
            completed=[],
            transitions=1,
            immediate_context=None,
            finding_cards=[],
            research_memory=None,
            total_sq_executed=5,
            findings_count_history=[],
            ctx=_make_ctx(),
            events=[],
        )
        result = _check_mass_failure(inp)
        assert result is not None
        state_update, _ = result
        assert state_update["current_phase"] == PHASE_SYNTHESIZE
        assert "research_abort_reason" in state_update


class TestBuildSupervisorCompletenessReturn:
    """Tests for _build_supervisor_completeness_return."""

    def test_returns_completeness_phase(self):
        """Return includes PHASE_COMPLETENESS."""
        state, events = _build_supervisor_completeness_return(
            execution_start_update={},
            findings_board={},
            agent_messages=[],
            supervisor_rounds=[],
            completed=[],
            round_num=1,
            transitions=1,
            immediate_context=None,
            finding_cards=[],
            research_memory=None,
            events=[],
        )
        assert state["current_phase"] == PHASE_COMPLETENESS
        assert isinstance(events, list)


class TestResearchSupervisorNode:
    """Tests for research_supervisor_node (main entry point)."""

    @pytest.mark.asyncio
    async def test_supervisor_empty_plan_transitions_to_completeness(self):
        """Empty plan (no pending, no completed, round 1) goes to completeness."""
        state = _make_state(
            pending_subqueries=[],
            completed_subqueries=[],
            current_round=0,
        )
        ctx = _make_ctx()

        with patch(
            "template_agent.src.core.deep_research.nodes.supervisor.get_cancel_store"
        ) as mock_get_store:
            mock_store = MagicMock()
            mock_store.is_cancelled = AsyncMock(return_value=False)
            mock_get_store.return_value = mock_store
            from template_agent.src.core.deep_research.nodes.supervisor import (
                research_supervisor_node,
            )

            updates, _ = await research_supervisor_node(state, ctx)

        assert updates["current_phase"] == PHASE_COMPLETENESS

    @pytest.mark.asyncio
    async def test_supervisor_cancellation_returns_early(self):
        """Cancelled thread returns early with PHASE_COMPLETE."""
        state = _make_state(thread_id="t1", pending_subqueries=["sq1"])
        ctx = _make_ctx()

        with patch(
            "template_agent.src.core.deep_research.nodes.supervisor.get_cancel_store"
        ) as mock_get_store:
            mock_store = MagicMock()
            mock_store.is_cancelled = AsyncMock(return_value=True)
            mock_get_store.return_value = mock_store
            from template_agent.src.core.deep_research.nodes.supervisor import (
                research_supervisor_node,
            )

            updates, _ = await research_supervisor_node(state, ctx)

        assert updates.get("current_phase") == "complete"
        assert updates.get("should_stop") is True


# ---------------------------------------------------------------------------
# New test classes for improved coverage (target: 65%+)
# ---------------------------------------------------------------------------


class TestGenerateAlternativeApproach:
    """Tests for _generate_alternative_approach."""

    @pytest.mark.asyncio
    async def test_generate_alternative_returns_none_when_llm_fails(self):
        """LLM exception returns None."""
        ctx = _make_ctx()
        warnings = [
            {
                "metric": "revenue",
                "value": "1e99",
                "severity": "major",
                "possible_cause": "unit error",
            }
        ]

        with patch(
            "template_agent.src.core.deep_research.nodes.supervisor.tracked_invoke",
            new_callable=AsyncMock,
            side_effect=Exception("LLM error"),
        ):
            result = await _generate_alternative_approach(
                ctx, "What is revenue?", warnings
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_alternative_returns_none_when_response_too_short(self):
        """Empty or very short LLM response returns None."""
        ctx = _make_ctx()
        mock_response = MagicMock()
        mock_response.content = "ok"
        warnings = [
            {"metric": "x", "value": "1", "severity": "minor", "possible_cause": "?"}
        ]

        with patch(
            "template_agent.src.core.deep_research.nodes.supervisor.tracked_invoke",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await _generate_alternative_approach(ctx, "query", warnings)

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_alternative_returns_alternative_when_valid(self):
        """Valid LLM response returns alternative query."""
        ctx = _make_ctx()
        mock_response = MagicMock()
        mock_response.content = "What is the total revenue broken down by region?"
        warnings = [
            {
                "metric": "revenue",
                "value": "1e99",
                "severity": "major",
                "possible_cause": "unit",
            }
        ]

        with patch(
            "template_agent.src.core.deep_research.nodes.supervisor.tracked_invoke",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await _generate_alternative_approach(
                ctx, "What is revenue?", warnings
            )

        assert result == "What is the total revenue broken down by region?"


class TestProcessAndDepositWorkerResult:
    """Tests for _process_and_deposit_worker_result."""

    @pytest.mark.asyncio
    async def test_deposit_adds_finding_to_board(self):
        """Worker result is deposited on findings_board."""
        ctx = _make_ctx()
        findings_board = {}
        completed = []
        findings_received = []
        agent_messages = []
        events = []

        result = SubAgentResult(
            finding={"subquery": "sq1", "answer": "Answer one", "error": None},
            quality_score=0.85,
            confidence="high",
            summary="Summary",
            events=[],
        )
        worker_ctx = WorkerResultContext(
            sq="sq1",
            result=result,
            round_num=1,
            to_execute=["sq1"],
            total=1,
            findings_board=findings_board,
            completed=completed,
            findings_received=findings_received,
            agent_messages=agent_messages,
            immediate_context=None,
            finding_cards=[],
            research_memory=None,
            events=events,
        )

        _, _, _ = await _process_and_deposit_worker_result(ctx, worker_ctx)

        assert "sq1" in findings_board
        assert findings_board["sq1"]["finding"]["answer"] == "Answer one"
        assert "sq1" in completed
        assert "sq1" in findings_received
        assert len(agent_messages) == 1

    @pytest.mark.asyncio
    async def test_deposit_error_finding_sets_exclude_from_synthesis(self):
        """Error finding with no usable data gets exclude_from_synthesis."""
        ctx = _make_ctx()
        findings_board = {}
        completed = []
        findings_received = []
        agent_messages = []
        events = []

        result = SubAgentResult(
            finding={"subquery": "sq1", "answer": "", "error": "Access denied"},
            quality_score=0.0,
            confidence="low",
            summary="Access denied",
            events=[],
        )
        worker_ctx = WorkerResultContext(
            sq="sq1",
            result=result,
            round_num=1,
            to_execute=["sq1"],
            total=1,
            findings_board=findings_board,
            completed=completed,
            findings_received=findings_received,
            agent_messages=agent_messages,
            immediate_context=None,
            finding_cards=[],
            research_memory=None,
            events=events,
        )

        await _process_and_deposit_worker_result(ctx, worker_ctx)

        assert findings_board["sq1"].get("exclude_from_synthesis") is True


class TestGetCachedSubagentResult:
    """Tests for _get_cached_subagent_result."""

    def test_returns_none_when_no_cached_finding(self):
        """No cached finding returns None."""
        findings_board = {}
        enriched = {"query": "sq1", "status": "ready"}
        result = _get_cached_subagent_result("sq1", enriched, findings_board, 1, 2)
        assert result is None

    def test_returns_subagent_result_when_source_cached_and_key_matches(self):
        """Cached source with matching key returns SubAgentResult."""
        findings_board = {
            "cached_sq": FindingEntry(
                finding={"subquery": "cached_sq", "answer": "Cached answer"},
                quality_score=0.8,
            ),
        }
        enriched = {
            "query": "sq1",
            "status": "ready",
            "source": "cached",
            "cached_finding_key": "cached_sq",
        }
        cached = _get_cached_subagent_result("sq1", enriched, findings_board, 1, 2)
        assert cached is not None
        assert cached.finding.get("answer") == "Cached answer"
        assert cached.quality_score == pytest.approx(0.8)


class TestSupervisorCheckSentinel:
    """Tests for _supervisor_check_sentinel."""

    def test_returns_none_when_sentinel_not_triggered(self):
        """Sentinel not triggered returns None."""
        state = _make_state()
        ctx = _make_ctx()
        events = []

        with patch(
            "template_agent.src.core.deep_research.nodes.supervisor.check_loop_sentinel",
            return_value=(False, None, None),
        ):
            result = _supervisor_check_sentinel(state, ctx, 1, events)

        assert result is None

    def test_returns_state_update_when_sentinel_triggered(self):
        """Sentinel triggered returns state update with forced phase."""
        state = _make_state()
        ctx = _make_ctx()
        events = []

        with patch(
            "template_agent.src.core.deep_research.nodes.supervisor.check_loop_sentinel",
            return_value=(True, "Max subqueries exceeded", PHASE_SYNTHESIZE),
        ):
            result = _supervisor_check_sentinel(state, ctx, 1, events)

        assert result is not None
        state_update, evts = result
        assert state_update["current_phase"] == PHASE_SYNTHESIZE
        assert state_update.get("sentinel_triggered") is True
        assert len(evts) > 0


class TestSupervisorCheckEmptyPlan:
    """Tests for _supervisor_check_empty_plan."""

    def test_returns_none_when_pending_exists(self):
        """Has pending subqueries returns None."""
        result = _supervisor_check_empty_plan(
            pending=["sq1"],
            completed=[],
            round_num=1,
            execution_start_update={},
            transitions=1,
            ctx=_make_ctx(),
            events=[],
        )
        assert result is None

    def test_returns_none_when_round_not_one(self):
        """Round 2 with empty pending returns None (not empty plan case)."""
        result = _supervisor_check_empty_plan(
            pending=[],
            completed=[],
            round_num=2,
            execution_start_update={},
            transitions=1,
            ctx=_make_ctx(),
            events=[],
        )
        assert result is None

    def test_returns_state_update_when_empty_plan(self):
        """Empty plan (no pending, no completed, round 1) returns completeness."""
        events = []
        result = _supervisor_check_empty_plan(
            pending=[],
            completed=[],
            round_num=1,
            execution_start_update={},
            transitions=1,
            ctx=_make_ctx(),
            events=events,
        )
        assert result is not None
        state_update, _ = result
        assert state_update["current_phase"] == PHASE_COMPLETENESS


class TestSupervisorCheckAllCompleted:
    """Tests for _supervisor_check_all_completed."""

    def test_returns_none_when_to_execute_non_empty(self):
        """Still has work to execute returns None."""
        result = _supervisor_check_all_completed(
            to_execute=["sq1"],
            round_num=2,
            execution_start_update={},
            findings_board={},
            agent_messages=[],
            supervisor_rounds=[],
            completed=[],
            transitions=1,
            immediate_context=None,
            finding_cards=[],
            research_memory=None,
            ctx=_make_ctx(),
            events=[],
        )
        assert result is None

    def test_returns_none_when_round_one(self):
        """Round 1 never triggers all-completed early return."""
        result = _supervisor_check_all_completed(
            to_execute=[],
            round_num=1,
            execution_start_update={},
            findings_board={},
            agent_messages=[],
            supervisor_rounds=[],
            completed=[],
            transitions=1,
            immediate_context=None,
            finding_cards=[],
            research_memory=None,
            ctx=_make_ctx(),
            events=[],
        )
        assert result is None

    def test_returns_completeness_when_all_done_round_gt_one(self):
        """All done and round > 1 returns completeness phase."""
        events = []
        result = _supervisor_check_all_completed(
            to_execute=[],
            round_num=2,
            execution_start_update={},
            findings_board={"sq1": FindingEntry(finding={"answer": "a1"})},
            agent_messages=[],
            supervisor_rounds=[],
            completed=["sq1"],
            transitions=1,
            immediate_context=None,
            finding_cards=[],
            research_memory=None,
            ctx=_make_ctx(),
            events=events,
        )
        assert result is not None
        state_update, _ = result
        assert state_update["current_phase"] == PHASE_COMPLETENESS


class TestSupervisorRunDebate:
    """Tests for _supervisor_run_debate."""

    @pytest.mark.asyncio
    async def test_returns_board_unchanged_when_fewer_than_two_findings(self):
        """Fewer than 2 findings returns board unchanged."""
        ctx = _make_ctx()
        findings_board = {"sq1": FindingEntry(finding={"answer": "a1"})}
        findings_received = ["sq1"]
        events = []

        result = await _supervisor_run_debate(
            ctx, findings_board, findings_received, "query", events
        )

        assert result == findings_board
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_runs_conflict_detection_when_two_or_more_findings(self):
        """Two or more findings triggers conflict detection."""
        ctx = _make_ctx()
        findings_board = {
            "sq1": FindingEntry(finding={"answer": "a1"}),
            "sq2": FindingEntry(finding={"answer": "a2"}),
        }
        findings_received = ["sq1", "sq2"]
        events = []

        with patch(
            "template_agent.src.core.deep_research.nodes.supervisor._detect_and_resolve_conflicts",
            new_callable=AsyncMock,
            return_value=(findings_board, []),
        ):
            await _supervisor_run_debate(
                ctx, findings_board, findings_received, "query", events
            )

        assert len(events) > 0


class TestDetectAndResolveConflicts:
    """Tests for _detect_and_resolve_conflicts."""

    @pytest.mark.asyncio
    async def test_returns_early_when_fewer_than_two_recent_findings(self):
        """Fewer than 2 recent findings returns early."""
        ctx = _make_ctx()
        findings_board = {"sq1": FindingEntry(finding={"answer": "a1"})}
        recent_findings = ["sq1"]

        board, events = await _detect_and_resolve_conflicts(
            ctx, findings_board, recent_findings, "query"
        )

        assert board == findings_board
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_returns_early_when_fewer_than_two_answers(self):
        """Fewer than 2 valid answers (e.g. one has error) returns early."""
        ctx = _make_ctx()
        findings_board = {
            "sq1": FindingEntry(finding={"answer": "a1"}),
            "sq2": FindingEntry(finding={"answer": "", "error": "failed"}),
        }
        recent_findings = ["sq1", "sq2"]

        board, events = await _detect_and_resolve_conflicts(
            ctx, findings_board, recent_findings, "query"
        )

        assert board == findings_board
        assert len(events) == 0


class TestResearchSupervisorNodeAdditionalScenarios:
    """Additional tests for research_supervisor_node."""

    @pytest.mark.asyncio
    async def test_supervisor_sentinel_triggered_returns_early(self):
        """Sentinel triggered at start returns early."""
        state = _make_state(pending_subqueries=["sq1"])
        ctx = _make_ctx()

        with patch(
            "template_agent.src.core.deep_research.nodes.supervisor.get_cancel_store"
        ) as mock_get_store:
            mock_store = MagicMock()
            mock_store.is_cancelled = AsyncMock(return_value=False)
            mock_get_store.return_value = mock_store
            with patch(
                "template_agent.src.core.deep_research.nodes.supervisor._supervisor_check_sentinel",
                return_value=(
                    {"current_phase": PHASE_SYNTHESIZE, "sentinel_triggered": True},
                    [],
                ),
            ):
                from template_agent.src.core.deep_research.nodes.supervisor import (
                    research_supervisor_node,
                )

                updates, _ = await research_supervisor_node(state, ctx)

        assert updates.get("current_phase") == PHASE_SYNTHESIZE
        assert updates.get("sentinel_triggered") is True

    @pytest.mark.asyncio
    async def test_supervisor_empty_plan_round_one_transitions_to_completeness(self):
        """Empty plan at round 1 transitions to completeness."""
        state = _make_state(
            pending_subqueries=[],
            completed_subqueries=[],
            current_round=0,
        )
        ctx = _make_ctx()

        with patch(
            "template_agent.src.core.deep_research.nodes.supervisor.get_cancel_store"
        ) as mock_get_store:
            mock_store = MagicMock()
            mock_store.is_cancelled = AsyncMock(return_value=False)
            mock_get_store.return_value = mock_store
            from template_agent.src.core.deep_research.nodes.supervisor import (
                research_supervisor_node,
            )

            updates, _ = await research_supervisor_node(state, ctx)

        assert updates["current_phase"] == PHASE_COMPLETENESS

    @pytest.mark.asyncio
    async def test_supervisor_cached_worker_returns_cached_result(self):
        """Worker with cached result uses cached finding."""
        state = _make_state(
            pending_subqueries=["sq1"],
            completed_subqueries=[],
            current_round=0,
            enriched_subqueries=[
                {
                    "query": "sq1",
                    "status": "ready",
                    "source": "cached",
                    "cached_finding_key": "cached_sq",
                }
            ],
            findings_board={
                "cached_sq": FindingEntry(
                    finding={"subquery": "cached_sq", "answer": "Cached answer"},
                    quality_score=0.8,
                ),
            },
        )
        ctx = _make_ctx()

        with patch(
            "template_agent.src.core.deep_research.nodes.supervisor.get_cancel_store"
        ) as mock_get_store:
            mock_store = MagicMock()
            mock_store.is_cancelled = AsyncMock(return_value=False)
            mock_get_store.return_value = mock_store
            with patch(
                "template_agent.src.core.deep_research.nodes.supervisor._execute_worker_uncached",
                new_callable=AsyncMock,
            ) as mock_uncached:
                from template_agent.src.core.deep_research.nodes.supervisor import (
                    research_supervisor_node,
                )

                updates, _ = await research_supervisor_node(state, ctx)

                mock_uncached.assert_not_called()

        assert "sq1" in updates.get("completed_subqueries", [])
        assert "sq1" in updates.get("findings_board", {})

    @pytest.mark.asyncio
    async def test_supervisor_status_access_denied_skips_worker_execution(self):
        """Subquery with access_denied status skips worker execution."""
        state = _make_state(
            pending_subqueries=["sq1"],
            completed_subqueries=[],
            current_round=0,
            enriched_subqueries=[
                {"query": "sq1", "status": "access_denied"},
            ],
        )
        ctx = _make_ctx()

        with patch(
            "template_agent.src.core.deep_research.nodes.supervisor.get_cancel_store"
        ) as mock_get_store:
            mock_store = MagicMock()
            mock_store.is_cancelled = AsyncMock(return_value=False)
            mock_get_store.return_value = mock_store
            with patch(
                "template_agent.src.core.deep_research.nodes.supervisor._execute_worker_uncached",
                new_callable=AsyncMock,
            ) as mock_uncached:
                with patch(
                    "template_agent.src.core.deep_research.nodes.supervisor._supervisor_reflect",
                    new_callable=AsyncMock,
                    return_value={
                        "decision": "proceed_to_completeness",
                        "coverage_pct": 50,
                        "gaps": [],
                        "follow_up_subqueries": [],
                    },
                ):
                    from template_agent.src.core.deep_research.nodes.supervisor import (
                        research_supervisor_node,
                    )

                    updates, _ = await research_supervisor_node(state, ctx)

                    mock_uncached.assert_not_called()

        assert "sq1" in updates.get("completed_subqueries", [])
