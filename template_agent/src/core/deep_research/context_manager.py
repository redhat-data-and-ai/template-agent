"""Hierarchical Context Manager for Deep Research.

This module implements DeepMiner-style sliding window context management
for the deep research pipeline, enabling 50+ subquery sessions within
token limits by compressing older findings while preserving recent detail.

Architecture:
    Level 1 (Immediate Context): Full-detail recent findings with complete
        tool results. Sliding window of size CONTEXT_WINDOW_SIZE.

    Level 2 (Working Memory): Compressed FindingCards with summaries, key
        facts, and data highlights. Created when findings slide out of Level 1.

    Level 3 (Research Memory): High-level research state with cross-cutting
        insights, themes, and progress tracking. Periodically consolidated.

References:
    - DeepMiner: "Beyond Turn Limits" (arXiv:2510.08276)
    - AgentFold: "Long-Horizon Web Agents" (arXiv:2510.24699)
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, TypedDict

from template_agent.src.core.deep_research.state import Finding, ResearchContext
from template_agent.src.core.utils import safe_json_parse, truncate_text
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

# Defaults when settings are not available
_CONTEXT_WINDOW_SIZE = 8
_CONTEXT_SLIDE_STEP = 4
_RESEARCH_MEMORY_CONSOLIDATION_THRESHOLD = 10
_FINDING_CARD_MAX_SUMMARY_WORDS = 80
_FINDING_CARD_MAX_KEY_FACTS = 5


def _get_setting(name: str, default: Any) -> Any:
    """Get setting with fallback to default."""
    try:
        from template_agent.src.settings import settings

        return getattr(settings, name, default)
    except Exception:
        return default


class FindingCard(TypedDict, total=False):
    """Compressed finding card for Level 2 context."""

    subquery: str
    summary: str
    key_facts: list[str]
    data_highlights: dict[str, Any]
    source_citations: list[str]
    quality_score: float
    has_visualization: bool
    has_tool_data: bool
    resources_used: list[str]
    compressed_at: str


class ImmediateContext(TypedDict, total=False):
    """Level 1 immediate context with sliding window."""

    recent_findings: list[Finding]
    recent_subqueries: list[str]
    window_size: int
    slide_step: int


class ResearchMemory(TypedDict, total=False):
    """Level 3 research memory with consolidated insights."""

    plan_summary: str
    completed_count: int
    total_count: int
    key_insights: list[str]
    data_summary: str
    failed_subqueries: list[str]
    access_denied_subqueries: list[str]
    themes: list[str]
    consolidated_at: str


class HierarchicalContextManager:
    """Manages three-level context hierarchy for deep research.

    Implements the sliding window mechanism from DeepMiner paper:
    - Only keeps recent N findings in full detail (Level 1)
    - Compresses older findings to FindingCards (Level 2)
    - Periodically consolidates to ResearchMemory (Level 3)

    This enables sustained research sessions of 50+ subqueries within
    standard 32k-200k context windows by managing token budget across levels.
    """

    def __init__(
        self,
        ctx: ResearchContext,
        window_size: int | None = None,
        slide_step: int | None = None,
    ):
        """Initialize the context manager.

        Args:
            ctx: Research context with base_model for compression LLM calls.
            window_size: Override for CONTEXT_WINDOW_SIZE setting.
            slide_step: Override for CONTEXT_SLIDE_STEP setting.
        """
        self.ctx = ctx
        self.window_size = window_size or _get_setting(
            "CONTEXT_WINDOW_SIZE", _CONTEXT_WINDOW_SIZE
        )
        self.slide_step = slide_step or _get_setting(
            "CONTEXT_SLIDE_STEP", _CONTEXT_SLIDE_STEP
        )

        if self.slide_step > self.window_size:
            logger.warning(
                "CONTEXT_SLIDE_STEP (%d) > CONTEXT_WINDOW_SIZE (%d), clamping to window_size",
                self.slide_step,
                self.window_size,
            )
            self.slide_step = self.window_size

    async def process_new_finding(
        self,
        finding: Finding,
        immediate_context: ImmediateContext,
        finding_cards: list[FindingCard],
        research_memory: ResearchMemory | None,
    ) -> tuple[ImmediateContext, list[FindingCard], ResearchMemory | None]:
        """Process a new finding through the hierarchical context.

        1. Add finding to immediate context (Level 1)
        2. If window full, slide: compress oldest findings to cards (Level 2)
        3. If enough new cards, consolidate research memory (Level 3)

        Args:
            finding: The new finding to process.
            immediate_context: Current Level 1 context.
            finding_cards: Current Level 2 cards.
            research_memory: Current Level 3 memory.

        Returns:
            Tuple of (updated_immediate, updated_cards, updated_memory)
        """
        recent_findings = list(immediate_context.get("recent_findings", []))
        recent_subqueries = list(immediate_context.get("recent_subqueries", []))

        subquery = finding.get("subquery", "")
        recent_findings.append(finding)
        recent_subqueries.append(subquery)

        if len(recent_findings) > self.window_size:
            logger.info(
                "Sliding context window: %d findings > window_size %d",
                len(recent_findings),
                self.window_size,
            )

            num_to_compress = self.slide_step

            findings_to_compress = recent_findings[:num_to_compress]
            new_cards = []
            for f in findings_to_compress:
                card = await self.compress_to_finding_card(f)
                new_cards.append(card)

            recent_findings = recent_findings[num_to_compress:]
            recent_subqueries = recent_subqueries[num_to_compress:]
            finding_cards = finding_cards + new_cards

            logger.info(
                "Compressed %d findings to cards. Immediate: %d, Cards: %d",
                num_to_compress,
                len(recent_findings),
                len(finding_cards),
            )

        updated_immediate: ImmediateContext = {
            "recent_findings": recent_findings,
            "recent_subqueries": recent_subqueries,
            "window_size": self.window_size,
            "slide_step": self.slide_step,
        }

        consolidation_threshold = _get_setting(
            "RESEARCH_MEMORY_CONSOLIDATION_THRESHOLD",
            _RESEARCH_MEMORY_CONSOLIDATION_THRESHOLD,
        )
        cards_since_consolidation = self._count_cards_since_consolidation(
            finding_cards, research_memory
        )

        if cards_since_consolidation >= consolidation_threshold:
            logger.info(
                "Consolidating research memory: %d new cards",
                cards_since_consolidation,
            )
            research_memory = await self.consolidate_research_memory(
                finding_cards, research_memory
            )

        return updated_immediate, finding_cards, research_memory

    async def compress_to_finding_card(self, finding: Finding) -> FindingCard:
        """Compress a full Finding to a FindingCard using LLM.

        Preserves: key facts, data highlights, citations.
        Discards: raw tool_results.

        Args:
            finding: The finding to compress.

        Returns:
            Compressed FindingCard.
        """
        subquery = finding.get("subquery", "")
        answer = finding.get("answer", "")
        tool_results = finding.get("tool_results", [])
        resources_used = finding.get("resources_used", [])
        error = finding.get("error")

        if error or not answer:
            return FindingCard(
                subquery=subquery,
                summary=error or "No answer available",
                key_facts=[],
                data_highlights={},
                source_citations=[],
                quality_score=0.0,
                has_visualization=False,
                has_tool_data=bool(tool_results),
                resources_used=resources_used,
                compressed_at=datetime.now(timezone.utc).isoformat(),
            )

        prompt = self._build_compression_prompt(
            subquery=subquery,
            answer=answer,
            tool_results=tool_results,
        )

        try:
            response = await self.ctx.base_model.ainvoke(prompt)
            content = str(response.content or "").strip()

            parsed = safe_json_parse(content)
            if parsed:
                raw_facts = parsed.get("key_facts", [])
                key_facts = raw_facts if isinstance(raw_facts, list) else []
                max_facts = _get_setting(
                    "FINDING_CARD_MAX_KEY_FACTS", _FINDING_CARD_MAX_KEY_FACTS
                )
                return FindingCard(
                    subquery=subquery,
                    summary=str(parsed.get("summary", answer)),
                    key_facts=key_facts[:max_facts],
                    data_highlights=parsed.get("data_highlights", {}),
                    source_citations=parsed.get("source_citations", []),
                    quality_score=float(parsed.get("quality_score", 0.7)),
                    has_visualization=bool(parsed.get("has_visualization", False)),
                    has_tool_data=bool(tool_results),
                    resources_used=resources_used,
                    compressed_at=datetime.now(timezone.utc).isoformat(),
                )
        except Exception as e:
            logger.warning("FindingCard compression failed: %s", e)

        return self._create_fallback_card(finding)

    async def consolidate_research_memory(
        self,
        cards: list[FindingCard],
        current_memory: ResearchMemory | None,
    ) -> ResearchMemory:
        """Consolidate FindingCards into ResearchMemory.

        Extracts cross-cutting insights, updates themes, and creates
        a high-level view of the research progress.

        Args:
            cards: All FindingCards accumulated so far.
            current_memory: Existing research memory (if any).

        Returns:
            Updated ResearchMemory.
        """
        if not cards:
            return current_memory or self._create_empty_memory()

        prompt = self._build_consolidation_prompt(cards, current_memory)

        try:
            response = await self.ctx.base_model.ainvoke(prompt)
            content = str(response.content or "").strip()

            parsed = safe_json_parse(content)
            if parsed:
                existing_insights = (
                    current_memory.get("key_insights", []) if current_memory else []
                )
                new_insights = parsed.get("key_insights", [])
                merged_insights = self._deduplicate_insights(
                    existing_insights + new_insights
                )

                existing_themes = (
                    current_memory.get("themes", []) if current_memory else []
                )
                new_themes = parsed.get("themes", [])
                all_themes = [
                    t if isinstance(t, str) else str(t.get("name", t))
                    for t in existing_themes + new_themes
                    if t
                ]
                merged_themes = list(dict.fromkeys(all_themes))[:10]

                return ResearchMemory(
                    plan_summary=parsed.get(
                        "plan_summary",
                        current_memory.get("plan_summary", "")
                        if current_memory
                        else "",
                    ),
                    completed_count=len(cards),
                    total_count=(
                        current_memory.get("total_count", 0) if current_memory else 0
                    ),
                    key_insights=merged_insights[:15],
                    data_summary=parsed.get("data_summary", ""),
                    failed_subqueries=parsed.get("failed_subqueries", []),
                    access_denied_subqueries=parsed.get("access_denied_subqueries", []),
                    themes=merged_themes,
                    consolidated_at=datetime.now(timezone.utc).isoformat(),
                )
        except Exception as e:
            logger.warning("Research memory consolidation failed: %s", e)

        return self._create_fallback_memory(cards, current_memory)

    def format_for_synthesis(
        self,
        immediate_context: ImmediateContext,
        finding_cards: list[FindingCard],
        research_memory: ResearchMemory | None,
        query: str,
    ) -> str:
        """Format hierarchical context for synthesis prompt.

        Structure:
        1. Research Overview (from Level 3 memory)
        2. Finding Summaries (from Level 2 cards)
        3. Detailed Recent Findings (from Level 1 immediate)

        Args:
            immediate_context: Level 1 context.
            finding_cards: Level 2 cards.
            research_memory: Level 3 memory.
            query: Original user query (used in header).

        Returns:
            Formatted context string for synthesis prompt.
        """
        sections = []

        if query:
            sections.append(
                f"## RESEARCH CONTEXT\nOriginal Query: {truncate_text(query, 200)}"
            )

        if research_memory:
            overview = self._format_research_overview(research_memory)
            if overview:
                sections.append(overview)

        if finding_cards:
            summaries = self._format_finding_summaries(finding_cards)
            if summaries:
                sections.append(summaries)

        recent_findings = immediate_context.get("recent_findings", [])
        if recent_findings:
            detailed = self._format_detailed_findings(recent_findings)
            if detailed:
                sections.append(detailed)

        if not sections:
            return "No findings available."

        return "\n\n".join(sections)

    def format_for_subquery(
        self,
        subquery: str,
        finding_cards: list[FindingCard],
        research_memory: ResearchMemory | None,
    ) -> str:
        """Format cross-context for a new subquery execution.

        Provides relevant context from completed research without
        overwhelming the subquery agent with all findings.

        Args:
            subquery: The subquery about to be executed.
            finding_cards: Level 2 cards for context.
            research_memory: Level 3 memory for overview.

        Returns:
            Relevant cross-context string.
        """
        parts = []

        if research_memory:
            insights = research_memory.get("key_insights", [])
            if insights:
                parts.append("Key insights from prior research:")
                for insight in insights[:5]:
                    parts.append(f"  - {insight}")

        related = self._find_related_cards(subquery, finding_cards)
        if related:
            parts.append("\nRelated findings:")
            for card in related[:3]:
                card_subquery = card.get("subquery", "")
                summary = truncate_text(card.get("summary", ""), 500)
                parts.append(f"  [{card_subquery}]: {summary}")

        return "\n".join(parts) if parts else ""

    def _build_compression_prompt(
        self,
        subquery: str,
        answer: str,
        tool_results: list[str],
    ) -> list[dict[str, str]]:
        """Build the LLM prompt for compressing a finding to a card."""
        tool_text = "\n".join(
            f"- {truncate_text(tr, 3000)}" for tr in tool_results[:10]
        )

        max_words = _get_setting(
            "FINDING_CARD_MAX_SUMMARY_WORDS", _FINDING_CARD_MAX_SUMMARY_WORDS
        )
        max_facts = _get_setting(
            "FINDING_CARD_MAX_KEY_FACTS", _FINDING_CARD_MAX_KEY_FACTS
        )

        return [
            {
                "role": "system",
                "content": f"""You compress research findings into compact summary cards.
Extract the essential information while discarding verbose tool output.

Output JSON with these fields:
- summary: {max_words}-word summary capturing the key answer
- key_facts: Up to {max_facts} bullet points of important facts/numbers
- data_highlights: Key statistics as JSON (e.g., {{"total": 1500, "growth": "15%"}})
- source_citations: List of tools/data sources used
- quality_score: Confidence 0.0-1.0
- has_visualization: true if charts were generated""",
            },
            {
                "role": "user",
                "content": f"""Compress this finding:

SUBQUERY: {subquery}

ANSWER:
{answer}

TOOL RESULTS:
{tool_text}

Respond with JSON only.""",
            },
        ]

    def _build_consolidation_prompt(
        self,
        cards: list[FindingCard],
        current_memory: ResearchMemory | None,
    ) -> list[dict[str, str]]:
        """Build prompt for consolidating cards into research memory."""
        card_summaries = []
        for card in cards:
            sq = card.get("subquery", "")
            summary = truncate_text(card.get("summary", ""), 500)
            facts = ", ".join(card.get("key_facts", [])[:5])
            card_summaries.append(f"[{sq}]: {summary}\n  Facts: {facts}")

        existing_context = ""
        if current_memory:
            existing_insights = current_memory.get("key_insights", [])
            if existing_insights:
                existing_context = "Previous insights:\n" + "\n".join(
                    f"- {i}" for i in existing_insights[:10]
                )

        return [
            {
                "role": "system",
                "content": """You consolidate research findings into a high-level memory.
Identify cross-cutting insights, emergent themes, and research gaps.

Output JSON with:
- plan_summary: Brief description of research scope
- key_insights: Cross-subquery insights (not repetition of individual findings)
- data_summary: Aggregated key numbers/statistics
- themes: Emergent categories/patterns
- failed_subqueries: List of failed/empty subqueries
- access_denied_subqueries: List of access-denied subqueries""",
            },
            {
                "role": "user",
                "content": "Consolidate these research findings:\n\n"
                + "\n".join(card_summaries)
                + f"\n\n{existing_context}"
                + "\n\nExtract cross-cutting insights and themes. Respond with JSON only.",
            },
        ]

    def _create_fallback_card(self, finding: Finding) -> FindingCard:
        """Create a FindingCard without LLM when compression fails."""
        subquery = finding.get("subquery", "")
        answer = finding.get("answer", "")
        tool_results = finding.get("tool_results", [])
        resources_used = finding.get("resources_used", [])

        return FindingCard(
            subquery=subquery,
            summary=truncate_text(answer, 2000),
            key_facts=[],
            data_highlights={},
            source_citations=[],
            quality_score=0.5,
            has_visualization=False,
            has_tool_data=bool(tool_results),
            resources_used=resources_used,
            compressed_at=datetime.now(timezone.utc).isoformat(),
        )

    def _create_empty_memory(self) -> ResearchMemory:
        """Create an empty research memory."""
        return ResearchMemory(
            plan_summary="",
            completed_count=0,
            total_count=0,
            key_insights=[],
            data_summary="",
            failed_subqueries=[],
            access_denied_subqueries=[],
            themes=[],
            consolidated_at=datetime.now(timezone.utc).isoformat(),
        )

    def _create_fallback_memory(
        self,
        cards: list[FindingCard],
        current_memory: ResearchMemory | None,
    ) -> ResearchMemory:
        """Create research memory without LLM when consolidation fails."""
        failed = []
        access_denied = []

        for card in cards:
            if card.get("quality_score", 0) == 0:
                subquery = card.get("subquery", "")
                summary = card.get("summary", "").lower()
                if "access denied" in summary:
                    access_denied.append(subquery)
                else:
                    failed.append(subquery)

        return ResearchMemory(
            plan_summary=(
                current_memory.get("plan_summary", "") if current_memory else ""
            ),
            completed_count=len(cards),
            total_count=current_memory.get("total_count", 0) if current_memory else 0,
            key_insights=(
                current_memory.get("key_insights", []) if current_memory else []
            ),
            data_summary="",
            failed_subqueries=failed,
            access_denied_subqueries=access_denied,
            themes=current_memory.get("themes", []) if current_memory else [],
            consolidated_at=datetime.now(timezone.utc).isoformat(),
        )

    def _count_cards_since_consolidation(
        self,
        cards: list[FindingCard],
        memory: ResearchMemory | None,
    ) -> int:
        """Count FindingCards created since last consolidation."""
        if not memory:
            return len(cards)

        consolidated_at = memory.get("consolidated_at")
        if not consolidated_at:
            return len(cards)

        try:
            threshold = datetime.fromisoformat(consolidated_at)
        except (ValueError, TypeError):
            return len(cards)

        count = 0
        for card in cards:
            raw_ts = card.get("compressed_at", "")
            if not raw_ts:
                continue
            try:
                if datetime.fromisoformat(raw_ts) > threshold:
                    count += 1
            except (ValueError, TypeError):
                continue

        return count

    def _deduplicate_insights(self, insights: list[str]) -> list[str]:
        """Remove duplicate or very similar insights."""
        seen: set[str] = set()
        unique = []

        for insight in insights:
            normalized = insight.lower().strip()[:100]
            if normalized not in seen:
                seen.add(normalized)
                unique.append(insight)

        return unique

    _STOPWORDS: set[str] = {
        "the",
        "a",
        "an",
        "of",
        "in",
        "for",
        "and",
        "or",
        "is",
        "are",
        "what",
        "how",
        "which",
        "when",
        "where",
        "who",
        "why",
        "to",
        "by",
        "on",
        "at",
        "from",
        "with",
        "that",
        "this",
        "it",
        "be",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "not",
        "no",
        "all",
        "each",
        "every",
        "any",
        "can",
        "will",
        "was",
        "were",
        "been",
    }

    @staticmethod
    def _tokenize(text: str) -> Counter:
        """Tokenize text into word frequency vector, removing stopwords."""
        words = re.findall(r"[a-z0-9]+", text.lower())
        return Counter(
            w
            for w in words
            if w not in HierarchicalContextManager._STOPWORDS and len(w) > 1
        )

    @staticmethod
    def _cosine_similarity(vec_a: Counter, vec_b: Counter) -> float:
        """Compute cosine similarity between two word frequency vectors."""
        if not vec_a or not vec_b:
            return 0.0
        common_keys = vec_a.keys() & vec_b.keys()
        dot = sum(vec_a[k] * vec_b[k] for k in common_keys)
        mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
        mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
        if mag_a <= 0.0 or mag_b <= 0.0:
            return 0.0
        return dot / (mag_a * mag_b)

    def _find_related_cards(
        self,
        subquery: str,
        cards: list[FindingCard],
        threshold: float = 0.25,
    ) -> list[FindingCard]:
        """Find cards semantically related to a subquery via TF cosine similarity."""
        sq_vec = self._tokenize(subquery)
        if not sq_vec:
            return []

        scored: list[tuple[float, FindingCard]] = []
        for card in cards:
            card_text = " ".join(
                filter(
                    None,
                    [
                        card.get("subquery", ""),
                        card.get("summary", ""),
                        " ".join(card.get("key_facts", [])),
                    ],
                )
            )
            card_vec = self._tokenize(card_text)
            sim = self._cosine_similarity(sq_vec, card_vec)
            if sim >= threshold:
                scored.append((sim, card))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [card for _, card in scored]

    def _format_research_overview(self, memory: ResearchMemory) -> str:
        """Format Level 3 research memory for synthesis."""
        parts = ["## RESEARCH OVERVIEW"]

        completed = memory.get("completed_count", 0)
        total = memory.get("total_count", 0) or completed
        parts.append(f"Progress: {completed}/{total} subqueries completed")

        insights = memory.get("key_insights", [])
        if insights:
            parts.append("\nKey Insights Discovered:")
            for insight in insights[:8]:
                parts.append(f"  - {insight}")

        themes = memory.get("themes", [])
        if themes:
            parts.append(f"\nThemes: {', '.join(themes[:5])}")

        data_summary = memory.get("data_summary", "")
        if data_summary:
            parts.append(f"\nData Summary: {data_summary}")

        failed = memory.get("failed_subqueries", [])
        access_denied = memory.get("access_denied_subqueries", [])
        if failed or access_denied:
            parts.append("\nLimitations:")
            if failed:
                parts.append(f"  Failed queries: {', '.join(failed[:3])}")
            if access_denied:
                parts.append(f"  Access denied: {', '.join(access_denied[:3])}")

        return "\n".join(parts)

    def _format_finding_summaries(self, cards: list[FindingCard]) -> str:
        """Format Level 2 FindingCards for synthesis."""
        parts = ["## FINDING SUMMARIES (Compressed)"]

        for card in cards:
            subquery = card.get("subquery", "")
            summary = card.get("summary", "")
            key_facts = card.get("key_facts", [])
            data_highlights = card.get("data_highlights", {})

            entry = [f"\n### {subquery}"]
            entry.append(summary)

            if key_facts:
                facts_text = "; ".join(key_facts)
                entry.append(f"Key Facts: {facts_text}")

            if data_highlights:
                try:
                    highlights_text = json.dumps(data_highlights, default=str)
                    entry.append(f"Data: {highlights_text}")
                except Exception:
                    pass

            parts.append("\n".join(entry))

        return "\n".join(parts)

    def _format_single_finding(self, finding: Finding) -> str:
        """Format a single finding for detailed output."""
        subquery = finding.get("subquery", "")
        answer = finding.get("answer", "")
        tool_results = finding.get("tool_results", [])
        error = finding.get("error")

        entry = [f"\n### {subquery}"]

        if error:
            entry.append(f"**Error:** {error}")
            return "\n".join(entry)

        if answer:
            entry.append(f"**Answer:** {answer}")
            if tool_results:
                entry.append("\n**Tool Results:**")
                for tr in tool_results:
                    entry.append(f"  - {tr}")

        return "\n".join(entry)

    def _format_detailed_findings(self, findings: list[Finding]) -> str:
        """Format Level 1 immediate findings with full detail."""
        parts = ["## DETAILED RECENT FINDINGS (Full Detail)"]
        for finding in findings:
            parts.append(self._format_single_finding(finding))
        return "\n".join(parts)


def create_context_manager(ctx: ResearchContext) -> HierarchicalContextManager:
    """Create a HierarchicalContextManager with default settings."""
    return HierarchicalContextManager(ctx)


def create_initial_hierarchical_context(
    subqueries: list[str],
) -> tuple[ImmediateContext, list[FindingCard], ResearchMemory]:
    """Create initial hierarchical context for a new research session."""
    window_size = _get_setting("CONTEXT_WINDOW_SIZE", _CONTEXT_WINDOW_SIZE)
    slide_step = _get_setting("CONTEXT_SLIDE_STEP", _CONTEXT_SLIDE_STEP)

    immediate_context: ImmediateContext = {
        "recent_findings": [],
        "recent_subqueries": [],
        "window_size": window_size,
        "slide_step": slide_step,
    }

    finding_cards: list[FindingCard] = []

    research_memory: ResearchMemory = {
        "plan_summary": f"Research plan with {len(subqueries)} subqueries",
        "completed_count": 0,
        "total_count": len(subqueries),
        "key_insights": [],
        "data_summary": "",
        "failed_subqueries": [],
        "access_denied_subqueries": [],
        "themes": [],
        "consolidated_at": datetime.now(timezone.utc).isoformat(),
    }

    return immediate_context, finding_cards, research_memory


def _get_tiktoken_encoding():
    """Lazily load and cache the tiktoken encoding."""
    enc = getattr(_get_tiktoken_encoding, "_cached", None)
    if enc is not None:
        return enc
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        _get_tiktoken_encoding._cached = enc
        return enc
    except Exception:
        _get_tiktoken_encoding._cached = None
        return None


def estimate_tokens(text: str, model_name: str | None = None) -> int:
    """Estimate token count using tiktoken (cl100k_base) with heuristic fallback."""
    if not text:
        return 0
    if not isinstance(text, str):
        text = str(text)
    if model_name and "gemini" in model_name.lower():
        return max(1, len(text) // 4)
    enc = _get_tiktoken_encoding()
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    return max(1, len(text) // 4)


def estimate_finding_tokens(finding: Finding) -> int:
    """Estimate tokens for a complete finding."""
    total = 0

    total += estimate_tokens(finding.get("subquery", ""))
    total += estimate_tokens(finding.get("answer", ""))

    tool_results = finding.get("tool_results", [])
    for result in tool_results:
        total += estimate_tokens(str(result))

    return total


def estimate_state_tokens(
    findings: dict[str, Finding],
    immediate_context: ImmediateContext | None,
    finding_cards: list[FindingCard],
    research_memory: ResearchMemory | None,
    base_overhead: int = 15000,
) -> int:
    """Estimate total context tokens from the research state."""
    total = base_overhead

    if immediate_context:
        for f in immediate_context.get("recent_findings", []):
            total += estimate_finding_tokens(f)
    elif findings:
        for finding in findings.values():
            total += estimate_finding_tokens(finding)

    for card in finding_cards:
        total += estimate_tokens(card.get("subquery", ""))
        total += estimate_tokens(card.get("summary", ""))
        for fact in card.get("key_facts", []):
            total += estimate_tokens(fact)

    if research_memory:
        total += estimate_tokens(research_memory.get("plan_summary", ""))
        total += estimate_tokens(research_memory.get("data_summary", ""))
        for insight in research_memory.get("key_insights", []):
            total += estimate_tokens(insight)
        for theme in research_memory.get("themes", []):
            total += estimate_tokens(theme)

    return total


def get_max_context_tokens(model_name: str | None = None) -> int:
    """Get the maximum context token limit for a model."""
    from template_agent.src.core.deep_research.mode_config import resolve_model_spec

    return resolve_model_spec(model_name).context_window
