"""Deep research state definitions.

This module defines the typed state for the hierarchical multi-agent
deep research pipeline.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from template_agent.src.core.deep_research.token_tracker import TokenUsageTracker

__all__ = [
    "PHASE_PLAN",
    "PHASE_SUBQUERY",
    "PHASE_SUPERVISOR",
    "PHASE_SYNTHESIS",
    "PHASE_DONE",
    "DEFAULT_MAX_ITERATIONS",
    "ReviewAction",
    "FailureClass",
    "Finding",
    "ReviewResult",
    "AgentMessage",
    "FindingEntry",
    "SupervisorRound",
    "FindingCard",
    "ResearchMemory",
    "ImmediateContext",
    "QualityDimensionScore",
    "QualityMatrix",
    "DeepResearchStateRequired",
    "DeepResearchState",
    "ResearchContext",
]

# Phase constants
PHASE_PLAN = "plan"
PHASE_SUBQUERY = "subquery"
PHASE_SUPERVISOR = "supervisor"
PHASE_SYNTHESIS = "synthesis"
PHASE_DONE = "done"

# Deep research graph phases (for streaming pipeline)
PHASE_TRIAGE = "triage"
PHASE_PROBE = "probe"
PHASE_AWAIT_APPROVAL = "await_approval"
PHASE_COMPLETENESS = "completeness"
PHASE_SYNTHESIZE = "synthesize"
PHASE_VISUALIZE = "visualize"
PHASE_REVIEW = "review"
PHASE_COMPLETE = "complete"

DEFAULT_MAX_ITERATIONS = 10


class ReviewAction(str, Enum):
    """Action taken by the supervisor when reviewing a finding."""

    ACCEPT = "accept"
    REJECT = "reject"
    REVISE = "revise"
    DEFER = "defer"


class FailureClass(str, Enum):
    """Classification of why a finding failed or was rejected."""

    TOOL_ERROR = "tool_error"
    TIMEOUT = "timeout"
    INVALID_RESPONSE = "invalid_response"
    LOW_QUALITY = "low_quality"
    PLAUSIBILITY_CONCERN = "plausibility_concern"
    DATA_QUALITY = "data_quality"
    OTHER = "other"


class Finding(TypedDict, total=False):
    """A single research finding from a subquery execution."""

    subquery: str
    answer: str
    tool_results: list[Any]
    error: str | None
    failure_class: str | None
    cached: bool
    execution_time_ms: float | None
    plausibility_concern: bool
    plausibility_warnings: list[str]
    data_quality_alert: bool
    low_quality_drop: bool
    data_quality_score: float | None
    access_denied: bool
    resources_used: list[str]


class ReviewResult(TypedDict, total=False):
    """Result of supervisor review of a finding."""

    action: str
    feedback: str | None
    revised_answer: str | None


class AgentMessage(TypedDict, total=False):
    """Message exchanged between agents in the pipeline."""

    from_agent: str
    to_agent: str
    message_type: str
    content: str
    metadata: dict[str, Any]
    timestamp: str


class FindingEntry(TypedDict, total=False):
    """Entry in the findings log with metadata."""

    finding: Finding
    review: ReviewResult | None
    index: int


class SupervisorRound(TypedDict, total=False):
    """One round of supervisor-managed research."""

    round_number: int
    delegated_subqueries: list[str]
    findings_received: int
    gaps_identified: list[str]
    follow_ups_spawned: list[str]
    coverage_assessment: str


class FindingCard(TypedDict, total=False):
    """Card representation of a finding for display or aggregation."""

    subquery: str
    answer: str
    status: str
    error: str | None
    review_action: str | None
    plausibility_concern: bool
    data_quality_alert: bool
    low_quality_drop: bool
    execution_time_ms: float | None
    index: int


class ResearchMemory(TypedDict, total=False):
    """Persistent memory across research iterations."""

    accepted_findings: list[FindingCard]
    rejected_subqueries: list[str]
    synthesis_notes: list[str]
    iteration_count: int


class ImmediateContext(TypedDict, total=False):
    """Context passed to the current node for immediate use."""

    current_subquery: str | None
    current_finding: Finding | None
    pending_review: list[FindingEntry]


class QualityDimensionScore(TypedDict, total=False):
    """Score for a single quality dimension."""

    name: str
    score: float
    weight: float
    rationale: str | None


class QualityMatrix(TypedDict, total=False):
    """Aggregated quality scores across dimensions."""

    dimensions: list[QualityDimensionScore]
    overall: float
    passed: bool


class DeepResearchStateRequired(TypedDict):
    """Required fields for deep research state."""

    query: str
    thread_id: str | None
    current_phase: str


class DeepResearchState(DeepResearchStateRequired, total=False):
    """Full state for the deep research pipeline.

    Every field used by create_initial_state or returned by a node must be
    declared here so LangGraph creates a channel for it.
    """

    # Plan / subqueries
    context: str
    subqueries: list[str]
    enriched_subqueries: list[dict[str, Any]]
    discovered_data_products: list[dict[str, Any]]
    plan_approved: bool
    plan_modified: bool
    pending_subqueries: list[str]
    completed_subqueries: list[str]
    understanding: str

    # Tool discovery
    tool_inventory: str
    tool_names: list[str]
    probe_result: str

    # Research
    findings: dict[str, Any]
    findings_board: dict[str, Any]
    agent_messages: list[dict[str, Any]]
    supervisor_rounds: list[SupervisorRound]
    current_round: int
    max_rounds: int

    # Synthesis / review
    coverage_complete: bool
    draft_answer: str
    synthesis_iteration: int
    visualizations: list[dict[str, Any]]
    visualization_attempted: bool
    review_results: list[dict[str, Any]]
    current_review: dict[str, Any] | None
    final_answer: str
    query_type: str
    query_type_confidence: float

    # Iteration / sentinel
    iteration: int
    max_iterations: int
    error: str | None
    should_stop: bool
    node_transitions: int
    total_node_transitions: int
    sentinel_triggered: bool
    sentinel_reason: str | None
    research_abort_reason: str | None

    # Complexity assessment
    triage_decision: str | None
    cached_findings_text: str
    assessed_max_subqueries: int
    assessed_max_supervisor_rounds: int
    assessed_max_review_iterations: int
    query_complexity_class: str | None
    complexity_reasoning: str | None

    # Timing / budget
    research_start_time: float
    execution_start_time: float
    pre_plan_elapsed_seconds: float
    human_wait_seconds: float
    total_subqueries_executed: int
    max_total_subqueries: int
    max_node_transitions: int
    max_session_seconds: float
    findings_count_history: list[int]
    coverage_history: list[float]

    # Mode / fallback tracking
    _mode_config: Any | None
    fallback_count: int

    # Internal overrides
    _user_max_rounds_override: int | None
    _user_max_iterations_override: int | None

    # Events (consumed by stream_mode="updates")
    _pending_events: list[dict[str, Any]]


@dataclass
class ResearchContext:
    """Shared context for deep research operations."""

    tools: list[Any]
    base_model: Any
    research_agent: Any = None
    checkpointer: Any = None
    user_id: str | None = None
    event_queue: asyncio.Queue | None = field(default=None, repr=False)
    token_tracker: TokenUsageTracker | None = field(default=None, repr=False)
    llm_semaphore: asyncio.Semaphore | None = field(default=None, repr=False)
    max_subqueries_override: int | None = None
    _max_mode: bool = False
    model_name: str | None = None
    root_tracer: Any = None
    mode_config: Any = None
    _event_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    @property
    def max_mode(self) -> bool:
        """Whether max mode (aggressive iteration) is enabled."""
        return self._max_mode

    def emit(self, event: dict[str, Any]) -> None:
        """Emit an event to the queue if configured."""
        if self.event_queue is not None:
            try:
                self.event_queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def emit_or_append(
        self,
        event: dict[str, Any] | None = None,
        fallback_list: list | None = None,
        *,
        key: str | None = None,
        value: Any = None,
    ) -> None:
        """Emit live if streaming, otherwise append to batch list.

        Usage: ctx.emit_or_append(emit_xxx(...), events)
        """
        if event is not None and fallback_list is not None:
            fallback_list.append(event)
            self.emit(event)
        elif key is not None:
            self.emit({key: value})

    async def async_emit_or_append(
        self,
        event: dict[str, Any] | None = None,
        fallback_list: list | None = None,
        *,
        key: str | None = None,
        value: Any = None,
    ) -> None:
        """Async emit or append, acquiring lock if needed."""
        async with self._event_lock:
            if event is not None and fallback_list is not None:
                fallback_list.append(event)
                self.emit(event)
            elif key is not None:
                self.emit({key: value})

    def llm_call_kwargs(self) -> dict[str, Any]:
        """Default kwargs for LLM calls (timeout 120s, root_tracer if available)."""
        kwargs: dict[str, Any] = {"timeout": 120}
        if self.root_tracer is not None:
            kwargs["root_tracer"] = self.root_tracer
        return kwargs

    def get_llm_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """Get LLM kwargs with optional overrides."""
        base = self.llm_call_kwargs()
        base.update(overrides)
        return base

    def get_tool_names(self) -> list[str]:
        """Return names of available tools."""
        return [getattr(t, "name", str(t)) for t in self.tools]

    def format_tool_inventory(self, max_tools: int = 50) -> str:
        """Format tool names for display, truncated to max_tools."""
        names = self.get_tool_names()
        if len(names) <= max_tools:
            return ", ".join(names)
        return ", ".join(names[:max_tools]) + f" ... (+{len(names) - max_tools} more)"
