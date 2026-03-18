"""Research mode configuration — single source of truth.

All per-mode settings live here: subquery bounds, execution budgets,
per-stage token limits, context budget ratios, quality gates, time
reservations, and prompt instructions.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class QualityGate:
    """Per-dimension quality thresholds and structural constraints."""

    coverage_min: float
    factual_grounding_min: float
    data_utilization_min: float
    synthesis_quality_min: float
    structural_compliance_min: float
    communication_quality_min: float
    actionability_min: float
    target_word_count_range: tuple[int, int]
    min_tables: int
    min_sections: int
    max_sections: int
    cross_refs_required: bool
    confidence_table_required: bool
    methodology_required: bool


@dataclass(frozen=True)
class ResearchModeConfig:
    """Complete per-mode configuration for deep research."""

    name: str
    min_subqueries: int
    max_subqueries: int
    recommended_subqueries: int
    max_supervisor_rounds: int
    max_review_iterations: int
    max_node_transitions: int
    session_timeout_floor: int
    per_subquery_allowance: float
    synthesis_reserved_seconds: float
    review_reserved_seconds: float
    default_reviewer_count: int
    min_reviewer_count: int
    worker_max_output_tokens: int | None
    planning_max_output_tokens: int | None
    synthesis_max_output_tokens: int
    context_budget_ratio: float
    quality_gate: QualityGate
    completeness_threshold: int
    planning_instruction: str
    worker_instruction: str
    synthesis_instruction: str
    review_instruction: str
    complexity_hint: str

    @property
    def max_output_tokens(self) -> int:
        """Return the synthesis max output tokens limit."""
        return self.synthesis_max_output_tokens

    @property
    def session_timeout_seconds(self) -> int:
        """Return the session timeout floor in seconds."""
        return self.session_timeout_floor


# ── Structural templates injected as {mode_instruction} ──────────────

_FAST_SYNTHESIS_INSTRUCTION = (
    "Produce a 400-1000 word executive briefing.\n"
    "STRUCTURE (follow exactly):\n"
    "  # [Topic]\n"
    "  ## Key Finding — 2-3 sentences with the primary answer and top-line number\n"
    "  ## Data Summary — ONE consolidated table of key metrics\n"
    "  ## Key Takeaways — 3-5 bullet insights with specific numbers\n"
    "RULES:\n"
    "- Maximum 3 sections total. One table max.\n"
    "- No methodology, no per-subquery sections.\n"
    "- No 'Limitations' unless data retrieval actually failed.\n"
    "- No separate 'Recommendations' section — fold actionable insights into takeaways.\n"
    "- Lead with the single most important number."
)

_EXTENDED_SYNTHESIS_INSTRUCTION = (
    "Produce a 1000-3000 word structured analysis.\n"
    "STRUCTURE (follow exactly):\n"
    "  # [Topic] - Analysis\n"
    "  ## Executive Summary — 2-3 sentences with key numbers\n"
    "  ## [Theme 1 Name] — analysis with embedded data table\n"
    "  ## [Theme 2 Name] — analysis with table\n"
    "  ## [Theme 3-5 Name] — as needed\n"
    "  ## Cross-Cutting Insights — patterns across themes, correlations\n"
    "  ## Recommendations — 3-5 specific, data-backed recommendations\n"
    "RULES:\n"
    "- 4-6 thematic sections grouped by THEME, NOT per-subquery.\n"
    "- Each theme must have at least one data table.\n"
    "- Cross-Cutting Insights section is mandatory.\n"
    "- Recommendations must cite specific data, NOT defer analysis.\n"
    "- No methodology section."
)

_FAST_MAX_SYNTHESIS_INSTRUCTION = (
    "Produce a 2000-4000 word comprehensive executive report.\n"
    "STRUCTURE (follow exactly):\n"
    "  # [Topic] - Research Report\n"
    "  ## Executive Summary — 3-5 sentences covering all key findings\n"
    "  ## [Dimension 1]\n"
    "    ### Key Metrics — data table\n"
    "    ### Analysis — detailed prose with numbers and comparisons\n"
    "    ### Implications — what this means for the business/user\n"
    "  ## [Dimension 2-5] — same 3-part structure\n"
    "  ## Cross-Finding Analysis — correlations, contradictions, patterns\n"
    "  ## Confidence Assessment — table: Claim | Confidence | Basis\n"
    "  ## Recommendations — prioritized, data-backed, actionable\n"
    "  ## Conclusion — summary with forward-looking implications\n"
    "RULES:\n"
    "- 5-8 thematic sections, each with metrics/analysis/implications sub-structure.\n"
    "- Cross-Finding Analysis section is mandatory.\n"
    "- Confidence Assessment table is mandatory.\n"
    "- Every claim must cite which finding it came from.\n"
    "- Writing must be executive-quality: crisp, no filler, no verbose prose."
)

_EXTENDED_MAX_SYNTHESIS_INSTRUCTION = (
    "Produce a 4000-8000 word comprehensive research report.\n"
    "STRUCTURE (follow exactly):\n"
    "  # [Topic] - Comprehensive Research Report\n"
    "  ## Executive Summary — 4-6 sentences: scope, methodology, top findings, key recommendation\n"
    "  ## Research Scope & Methodology — subqueries executed, data sources, coverage, quality notes\n"
    "  ## [Dimension 1]\n"
    "    ### Overview — context and why this dimension matters\n"
    "    ### Detailed Findings — multiple tables, category breakdowns\n"
    "    ### Anomalies & Edge Cases — outliers, unexpected values, quality flags\n"
    "    ### Cross-References — links to findings in other dimensions\n"
    "  ## [Dimension 2-N] — same 4-part depth for each\n"
    "  ## Cross-Dimensional Analysis — correlations, conflict resolution, emergent patterns\n"
    "  ## Data Quality & Confidence — per-dimension table: Dimension | Quality | Confidence | Notes\n"
    "  ## Risk Factors & Limitations — known gaps, quality concerns, scope limitations\n"
    "  ## Recommendations\n"
    "    ### Immediate Actions — data-backed, specific\n"
    "    ### Medium-Term Improvements — strategic recommendations\n"
    "  ## Conclusion — comprehensive synthesis with strategic implications\n"
    "RULES:\n"
    "- 6-12 thematic sections, each with overview/findings/anomalies/cross-refs sub-structure.\n"
    "- Methodology section is mandatory (this is the ONLY mode that includes it).\n"
    "- Anomaly analysis per dimension is mandatory.\n"
    "- Data Quality & Confidence table is mandatory.\n"
    "- Tiered recommendations (immediate + medium-term).\n"
    "- Maximum depth and breadth. Leave no stone unturned."
)


MODES: dict[str, ResearchModeConfig] = {
    "fast": ResearchModeConfig(
        name="fast",
        min_subqueries=3,
        max_subqueries=7,
        recommended_subqueries=5,
        max_supervisor_rounds=1,
        max_review_iterations=1,
        max_node_transitions=20,
        session_timeout_floor=180,
        per_subquery_allowance=60.0,
        synthesis_reserved_seconds=60.0,
        review_reserved_seconds=30.0,
        default_reviewer_count=4,
        min_reviewer_count=2,
        worker_max_output_tokens=None,
        planning_max_output_tokens=None,
        synthesis_max_output_tokens=16384,
        context_budget_ratio=0.75,
        completeness_threshold=70,
        quality_gate=QualityGate(
            coverage_min=0.50,
            factual_grounding_min=0.60,
            data_utilization_min=0.45,
            synthesis_quality_min=0.40,
            structural_compliance_min=0.50,
            communication_quality_min=0.50,
            actionability_min=0.35,
            target_word_count_range=(400, 1000),
            min_tables=0,
            min_sections=2,
            max_sections=4,
            cross_refs_required=False,
            confidence_table_required=False,
            methodology_required=False,
        ),
        planning_instruction=(
            "Generate focused subqueries that directly answer the question. "
            "Each subquery maps to one data retrieval. No tangential exploration."
        ),
        worker_instruction=(
            "Concise, data-driven answer. Report key metrics. "
            "Avoid tangential analysis."
        ),
        synthesis_instruction=_FAST_SYNTHESIS_INSTRUCTION,
        review_instruction=(
            "Verify factual accuracy. Accept if the core question is answered "
            "with correct data. Do not penalize for brevity or missing dimensions."
        ),
        complexity_hint="User wants a quick, focused answer.",
    ),
    "extended": ResearchModeConfig(
        name="extended",
        min_subqueries=5,
        max_subqueries=12,
        recommended_subqueries=7,
        max_supervisor_rounds=2,
        max_review_iterations=2,
        max_node_transitions=30,
        session_timeout_floor=300,
        per_subquery_allowance=90.0,
        synthesis_reserved_seconds=90.0,
        review_reserved_seconds=60.0,
        default_reviewer_count=6,
        min_reviewer_count=3,
        worker_max_output_tokens=None,
        planning_max_output_tokens=None,
        synthesis_max_output_tokens=32768,
        context_budget_ratio=0.75,
        completeness_threshold=75,
        quality_gate=QualityGate(
            coverage_min=0.60,
            factual_grounding_min=0.65,
            data_utilization_min=0.55,
            synthesis_quality_min=0.55,
            structural_compliance_min=0.55,
            communication_quality_min=0.55,
            actionability_min=0.45,
            target_word_count_range=(1000, 3000),
            min_tables=2,
            min_sections=4,
            max_sections=6,
            cross_refs_required=False,
            confidence_table_required=False,
            methodology_required=False,
        ),
        planning_instruction=(
            "Cover all dimensions of the question. Include subqueries for "
            "context, comparisons, and breakdowns by relevant categories "
            "(time, priority, type, team)."
        ),
        worker_instruction=(
            "Thorough answer with context. Note patterns, anomalies, "
            "connections. Include breakdowns. Highlight surprises."
        ),
        synthesis_instruction=_EXTENDED_SYNTHESIS_INSTRUCTION,
        review_instruction=(
            "Verify accuracy and check for missing dimensions. Ensure thematic "
            "grouping (not per-subquery dumping). Flag missing breakdowns or "
            "comparisons."
        ),
        complexity_hint="User wants thorough analysis with context and breakdowns.",
    ),
    "fast_max": ResearchModeConfig(
        name="fast_max",
        min_subqueries=7,
        max_subqueries=15,
        recommended_subqueries=10,
        max_supervisor_rounds=3,
        max_review_iterations=2,
        max_node_transitions=40,
        session_timeout_floor=360,
        per_subquery_allowance=75.0,
        synthesis_reserved_seconds=120.0,
        review_reserved_seconds=60.0,
        default_reviewer_count=6,
        min_reviewer_count=3,
        worker_max_output_tokens=None,
        planning_max_output_tokens=None,
        synthesis_max_output_tokens=32768,
        context_budget_ratio=0.85,
        completeness_threshold=80,
        quality_gate=QualityGate(
            coverage_min=0.70,
            factual_grounding_min=0.70,
            data_utilization_min=0.60,
            synthesis_quality_min=0.65,
            structural_compliance_min=0.60,
            communication_quality_min=0.60,
            actionability_min=0.55,
            target_word_count_range=(2000, 4000),
            min_tables=3,
            min_sections=5,
            max_sections=8,
            cross_refs_required=True,
            confidence_table_required=True,
            methodology_required=False,
        ),
        planning_instruction=(
            "Cover the question exhaustively. Generate subqueries for all "
            "relevant dimensions, comparisons, and trend analysis. Prioritize "
            "breadth-first: cover all angles before going deep. Aim for "
            "thorough coverage achievable in 3 supervisor rounds."
        ),
        worker_instruction=(
            "Provide thorough analysis with all relevant breakdowns. "
            "Prioritize data completeness over prose. Include all numbers, "
            "row counts, and computed metrics. Flag any anomalies."
        ),
        synthesis_instruction=_FAST_MAX_SYNTHESIS_INSTRUCTION,
        review_instruction=(
            "Verify accuracy, completeness, cross-referencing, and structural "
            "compliance. Ensure the 3-part section structure (metrics/analysis/"
            "implications) is followed. Reject if confidence assessment is missing."
        ),
        complexity_hint="Thorough, structured research. Prioritize quality over quantity.",
    ),
    "extended_max": ResearchModeConfig(
        name="extended_max",
        min_subqueries=10,
        max_subqueries=20,
        recommended_subqueries=15,
        max_supervisor_rounds=5,
        max_review_iterations=3,
        max_node_transitions=50,
        session_timeout_floor=600,
        per_subquery_allowance=90.0,
        synthesis_reserved_seconds=180.0,
        review_reserved_seconds=90.0,
        default_reviewer_count=8,
        min_reviewer_count=4,
        worker_max_output_tokens=None,
        planning_max_output_tokens=None,
        synthesis_max_output_tokens=65536,
        context_budget_ratio=0.90,
        completeness_threshold=85,
        quality_gate=QualityGate(
            coverage_min=0.80,
            factual_grounding_min=0.75,
            data_utilization_min=0.65,
            synthesis_quality_min=0.70,
            structural_compliance_min=0.65,
            communication_quality_min=0.65,
            actionability_min=0.60,
            target_word_count_range=(4000, 8000),
            min_tables=4,
            min_sections=6,
            max_sections=12,
            cross_refs_required=True,
            confidence_table_required=True,
            methodology_required=True,
        ),
        planning_instruction=(
            "Exhaustively explore the question AND connected topics. Generate "
            "subqueries for edge cases, correlations, historical trends, "
            "anomaly detection, and cross-dimensional analysis. Cast the "
            "widest possible net."
        ),
        worker_instruction=(
            "Comprehensive analysis. Explore edge cases, anomalies, "
            "historical context, correlations. If data reveals something "
            "unexpected, investigate it. Include all breakdowns and "
            "cross-tabulations."
        ),
        synthesis_instruction=_EXTENDED_MAX_SYNTHESIS_INSTRUCTION,
        review_instruction=(
            "Demand excellence. Check accuracy, completeness, cross-dimensional "
            "analysis, anomaly identification, methodology transparency, and "
            "confidence assessments. The answer should satisfy a domain expert. "
            "Flag gaps in analysis, missing cross-references, unsupported claims."
        ),
        complexity_hint=(
            "Publication-quality exhaustive research. Explore broadly and "
            "analyze deeply."
        ),
    ),
}

# Legacy alias so old code referencing MODES["max"] still works.
# New code should use "fast_max" or "extended_max" via resolve_mode().
MODES["max"] = MODES["extended_max"]


@dataclass(frozen=True)
class ModelSpec:
    """Provider-enforced limits for a supported model family."""

    context_window: int
    max_output_tokens: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "claude-sonnet-4": ModelSpec(context_window=200_000, max_output_tokens=64_000),
    "gemini-2.5-pro": ModelSpec(context_window=1_000_000, max_output_tokens=65_536),
    "gemini-2.5-flash": ModelSpec(context_window=1_000_000, max_output_tokens=65_536),
}

_DEFAULT_SPEC = ModelSpec(context_window=200_000, max_output_tokens=64_000)


def resolve_model_spec(model_name: str | None) -> ModelSpec:
    """Look up a model's spec using substring matching.

    Handles versioned names like ``claude-sonnet-4@20250514``.
    Falls back to ``_DEFAULT_SPEC`` when the model is unknown.
    """
    if not model_name:
        return _DEFAULT_SPEC
    lower = model_name.lower()
    for family, spec in MODEL_SPECS.items():
        if family in lower:
            return spec
    return _DEFAULT_SPEC


def resolve_mode(
    model_name: str | None,
    max_mode: bool,
    depth: str = "auto",
) -> ResearchModeConfig:
    """Determine the research mode from depth preference and max_mode flag.

    The ``depth`` parameter is the primary selector:
      - ``"fast"``     -> fast / fast_max
      - ``"extended"`` -> extended / extended_max
      - ``"auto"``     -> infer from model name (gemini -> extended, else fast)

    ``max_mode`` upgrades the selected depth to its _max variant.

    The model name only influences mode selection when ``depth="auto"``
    so that the structural template is no longer tightly coupled to the
    LLM provider.

    When ``max_mode`` is selected the synthesis token budget is clamped
    to the model's hard output-token limit.
    """
    is_extended = depth == "extended" or (
        depth == "auto" and model_name and "gemini" in model_name.lower()
    )

    if max_mode:
        mode = MODES["extended_max"] if is_extended else MODES["fast_max"]
        spec = resolve_model_spec(model_name)
        if mode.synthesis_max_output_tokens > spec.max_output_tokens:
            mode = dataclasses.replace(
                mode,
                synthesis_max_output_tokens=spec.max_output_tokens,
            )
        return mode

    return MODES["extended"] if is_extended else MODES["fast"]
