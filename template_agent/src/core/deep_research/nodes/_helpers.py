"""Shared helper functions and utilities for deep research nodes."""

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from template_agent.src.core.deep_research.state import (
    DeepResearchState,
    Finding,
    FindingEntry,
)
from template_agent.src.core.deep_research.utils import get_setting as _get_setting
from template_agent.src.core.utils import (
    safe_json_parse,
    simplify_error_for_display,
    truncate_text,
)
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(_get_setting("PYTHON_LOG_LEVEL", "INFO"))


# ---------------------------------------------------------------------------
# Plan node helpers (_parse_subqueries, _is_identity_subquery, etc.)
# ---------------------------------------------------------------------------


def _extract_balanced_json(text: str, keyword: str) -> str | None:
    """Find the outermost balanced {...} containing keyword."""
    key_pos = text.find(keyword)
    if key_pos == -1:
        return None
    brace = text.rfind("{", 0, key_pos)
    if brace == -1:
        return None
    depth, end = 0, brace
    for i in range(brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        if depth == 0:
            end = i + 1
            break
    return text[brace:end]


def _subqueries_from_parsed(data: Any) -> List[str] | None:
    """Return cleaned subqueries list if data contains valid subqueries key."""
    if data and "subqueries" in data and isinstance(data["subqueries"], list):
        return [str(sq).strip() for sq in data["subqueries"] if sq]
    return None


def _extract_numbered_or_bullet_lines(text: str) -> List[str]:
    """Extract numbered (1. ...) or bulleted (- ...) lines."""
    lines: List[str] = []
    for raw_line in text.split("\n"):
        stripped = raw_line.strip()
        match = re.match(r"^\d+[.)]\s+(\S.*)$", stripped)
        if match:
            lines.append(match.group(1).strip())
        elif stripped.startswith("- "):
            lines.append(stripped[2:].strip())
    return lines


def _parse_subqueries(response_text: str) -> List[str]:
    """Parse subqueries from LLM response."""
    text = response_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{[^\}]*\})\s*```", text)
    if fence_match:
        text = fence_match.group(1)
    result = _subqueries_from_parsed(safe_json_parse(text))
    if result is not None:
        return result
    if not fence_match:
        fragment = _extract_balanced_json(text, '"subqueries"')
        if fragment:
            result = _subqueries_from_parsed(safe_json_parse(fragment))
            if result is not None:
                return result
    return _extract_numbered_or_bullet_lines(response_text)


def _is_identity_subquery(subquery: str, original_query: str) -> bool:
    """Check if a subquery is just the original query restated."""
    _STRIP_RE = re.compile(r"[^\w\s]")
    norm_sq = _STRIP_RE.sub("", subquery.lower()).split()
    norm_orig = _STRIP_RE.sub("", original_query.lower()).split()
    if not norm_sq or not norm_orig:
        return False
    if norm_sq == norm_orig:
        return True
    sq_set = set(norm_sq)
    orig_set = set(norm_orig)
    if not sq_set or not orig_set:
        return False
    overlap = len(sq_set & orig_set)
    union = len(sq_set | orig_set)
    return (overlap / union) > 0.85


_SQL_KEYWORDS = re.compile(
    r"\b(SELECT|INSERT|UPDATE|DELETE|FROM|WHERE|GROUP\s+BY|ORDER\s+BY|JOIN)\b",
    re.IGNORECASE,
)


def _strip_sql_from_subqueries(subqueries: List[str]) -> List[str]:
    """Remove raw SQL from subqueries, keeping natural language portions."""
    cleaned = []
    for sq in subqueries:
        if not _SQL_KEYWORDS.search(sq):
            cleaned.append(sq)
            continue
        parts = re.split(r"SELECT\b[^;]*?FROM", sq, flags=re.IGNORECASE)
        nl_parts = [
            p.strip().rstrip(":;.,")
            for p in parts
            if p and p.strip() and not _SQL_KEYWORDS.search(p)
        ]
        if nl_parts:
            cleaned.append(nl_parts[0])
        else:
            cleaned.append(sq)
    return cleaned


_STATUS_ICONS: dict[str, str] = {"ready": "✓", "access_denied": "✗"}


def _format_enriched_plan(enriched_subqueries: List[Dict[str, Any]]) -> str:
    """Format enriched subqueries for display."""
    lines = []
    for i, eq in enumerate(enriched_subqueries, 1):
        query = eq.get("query", "")
        status = eq.get("status", "unknown")
        data_products = eq.get("data_products", [])
        dp_names = (
            [dp.get("name", "?") for dp in data_products]
            if isinstance(data_products, list)
            else []
        )
        dp_str = ", ".join(dp_names) if dp_names else "none"
        status_icon = _STATUS_ICONS.get(status, "?")
        lines.append(f"{i}. {status_icon} {query}")
        if dp_str != "none":
            lines.append(f"   Resources: [{dp_str}]")
        lines.append(f"   Status: {status}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node cancellation and findings
# ---------------------------------------------------------------------------


async def check_node_cancelled(
    thread_id: str | None,
    node_name: str,
    state: DeepResearchState,
) -> Optional[Dict[str, Any]]:
    """Return early state update if the thread has been cancelled, else None.

    Intended for use at the top of synthesize, visualize, and review nodes
    so cancellation takes effect without waiting for the supervisor loop.
    """
    if not thread_id:
        return None
    try:
        from template_agent.src.core.deep_research.cancel import get_cancel_store

        store = get_cancel_store()
        if not await store.is_cancelled(thread_id):
            return None
    except Exception as exc:
        logger.warning("check_node_cancelled failed for thread %s: %s", thread_id, exc)
        return None

    from template_agent.src.core.deep_research.state import PHASE_COMPLETE

    logger.info(
        "Node %s: thread %s is cancelled, returning early", node_name, thread_id
    )
    return {
        "current_phase": PHASE_COMPLETE,
        "final_answer": state.get("draft_answer", "")
        or "Research was cancelled by the user.",
        "total_node_transitions": state.get("total_node_transitions", 0) + 1,
    }


def findings_from_board(board: dict) -> dict:
    """Derive a legacy-shaped findings dict from the findings_board."""
    return {sq: entry.get("finding") or {} for sq, entry in board.items()}


def _make_finding(
    subquery: str,
    answer: str = "",
    *,
    tool_results: Optional[List[str]] = None,
    error: Optional[str] = None,
    cached: bool = False,
    access_denied: bool = False,
    execution_time_ms: Optional[int] = None,
) -> Finding:
    """Build a Finding dict with consistent keys."""
    finding: Finding = {
        "subquery": subquery,
        "answer": answer,
        "cached": cached,
    }
    if tool_results is not None:
        finding["tool_results"] = tool_results
    if error is not None:
        finding["error"] = error
    if access_denied:
        finding["access_denied"] = True
    if execution_time_ms is not None:
        finding["execution_time_ms"] = execution_time_ms
    return finding


def _classify_failure(exc: Exception, error_msg: str) -> str:
    """Classify an exception into a failure class string for targeted recovery."""
    error_lower = error_msg.lower()
    if isinstance(exc, asyncio.TimeoutError) or "timed out" in error_lower:
        return "tool_timeout"
    if (
        "access denied" in error_lower
        or "permission" in error_lower
        or "forbidden" in error_lower
    ):
        return "access_denied"
    if (
        "no results" in error_lower
        or "empty" in error_lower
        or "no data" in error_lower
    ):
        return "empty_result"
    if "parse" in error_lower or "json" in error_lower or "invalid" in error_lower:
        return "parse_error"
    if "rate limit" in error_lower or "429" in error_lower or "api" in error_lower:
        return "llm_failure"
    return "unknown"


def compute_data_quality_score(
    finding: Finding,
    self_eval_confidence: str = "medium",
) -> float:
    """Compute objective data quality score (0.0-1.0) for a finding.

    Combines structural quality (has answer, no errors) with semantic
    quality (plausibility of values). Findings flagged with
    plausibility warnings receive a penalty proportional to the
    worst severity level.
    """
    if finding.get("error"):
        return 0.0

    score = 0.0
    answer = finding.get("answer", "")
    if answer and len(answer.strip()) > 20:
        score += 0.5
    if not finding.get("error"):
        score += 0.3
    if self_eval_confidence == "high":
        score += 0.2
    elif self_eval_confidence == "medium":
        score += 0.1

    plausibility_warnings = finding.get("plausibility_warnings", [])
    if plausibility_warnings:
        severity_penalties = {"minor": 0.1, "major": 0.2, "critical": 0.3}
        penalties = []
        for w in plausibility_warnings:
            sev = w.get("severity", "minor") if isinstance(w, dict) else "minor"
            penalties.append(severity_penalties.get(sev, 0.1))
        if penalties:
            score -= max(penalties)

    return max(0.0, min(1.0, score))


def _assess_data_quality(findings: Dict[str, Finding]) -> str:
    """Assess overall data quality of findings. Returns 'high', 'medium', or 'low'."""
    if not findings:
        return "low"
    valid = sum(1 for f in findings.values() if not f.get("error") and f.get("answer"))
    total = len(findings)
    if total == 0:
        return "low"
    ratio = valid / total
    if ratio >= 0.8:
        return "high"
    if ratio >= 0.5:
        return "medium"
    return "low"


def _looks_like_tool_recommendation(text: str) -> bool:
    """True if text appears to recommend tools rather than synthesize findings."""
    lower = text.lower()
    patterns = [
        "recommend using",
        "suggest using",
        "you should use",
        "which data product",
        "which tool to",
        "execute the following",
        "run the query",
    ]
    return any(p in lower for p in patterns)


def should_exclude_from_synthesis(finding: Finding) -> bool:
    """True when a finding is unusable for synthesis."""
    if finding.get("low_quality_drop"):
        return True
    if not finding.get("error"):
        return False
    answer = finding.get("answer", "").strip()
    tool_results = finding.get("tool_results", [])
    has_data = bool(tool_results or (answer and len(answer) >= 20))
    return (not answer or len(answer) < 20) and not has_data


def _summarize_findings_board(findings_board: Dict[str, FindingEntry]) -> str:
    """Summarize findings board for cross-context."""
    if not findings_board:
        return "No findings collected yet."
    parts = []
    for sq, entry in list(findings_board.items())[:20]:
        finding = entry.get("finding") or {}
        if finding.get("error"):
            continue
        ans = finding.get("answer", "")
        if ans:
            parts.append(f"- {truncate_text(sq, 80)}: {truncate_text(ans, 150)}")
    return "\n".join(parts) if parts else "No findings collected yet."


def _find_related_findings(
    subquery: str, findings_board: Dict[str, FindingEntry]
) -> str:
    """Find related findings for cross-context (simplified)."""
    return _summarize_findings_board(findings_board)


def _parse_review_result(response_text: str, persona: str) -> Dict[str, Any]:
    """Parse review result from LLM response."""
    data = safe_json_parse(response_text)
    if data:
        try:
            return {
                "action": data.get("action") or "approve",
                "score": int(data.get("score") or 70),
                "reason": data.get("reason") or "",
                "feedback": data.get("feedback") or "",
                "data_issues": data.get("data_issues") or [],
                "deferred_insights": data.get("deferred_insights") or [],
                "follow_up_subqueries": data.get("follow_up_subqueries") or [],
                "persona": persona,
                "dimensions": data.get("dimensions")
                if isinstance(data.get("dimensions"), dict)
                else {},
            }
        except (TypeError, ValueError):
            pass
    return {
        "action": "approve",
        "score": 70,
        "reason": "Could not parse review response",
        "feedback": "",
        "data_issues": [],
        "deferred_insights": [],
        "follow_up_subqueries": [],
        "persona": persona,
        "dimensions": {},
    }


_EMOJI_RE = re.compile(
    r"[\U0001F300-\U0001F9FF\U00002600-\U000027BF\U0001FA00-\U0001FA6F"
    r"\U0001FA70-\U0001FAFF\U00002702-\U000027B0]+",
)

_CONVERSATIONAL_RE = re.compile(
    r"(?i)^(?:I've created|Based on my analysis|Here are the key|"
    r"Let me |I'll |I can |I would |I'd |In summary,? |"
    r"As you can see|As shown |Looking at |Overall,? |"
    r"Here's |Certainly|Of course|Great question|"
    r"I've analyzed|I've examined|I've reviewed).*$",
    re.MULTILINE,
)


def _strip_noise(text: str) -> str:
    """Remove chart placeholders, conversational fluff, and emojis."""
    cleaned = re.sub(r"<!--\s*CHART_PLACEHOLDER[^>]*-->", "", text)
    cleaned = _EMOJI_RE.sub("", cleaned)
    cleaned = _CONVERSATIONAL_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _build_zero_findings_message(
    state: DeepResearchState | Dict[str, Any] | None,
) -> str:
    """Produce an informative message when research produced zero valid findings."""
    plan = state.get("subqueries", []) if state else []
    reason = state.get("sentinel_reason", "unknown") if state else "unknown"
    plan_text = (
        "\n".join(f"- {sq}" for sq in plan[:10]) if plan else "No plan available"
    )
    return (
        "## Research Summary\n\n"
        "Research could not complete within the time budget.\n\n"
        f"**Reason:** {reason}\n\n"
        "**Research plan prepared:**\n"
        f"{plan_text}\n\n"
        "**What to try:** Run this query again. The system will reuse the "
        "cached plan and skip directly to execution."
    )


def _build_fallback_synthesis(
    findings_board: Dict[str, Any],
    query: str,
    state: DeepResearchState | Dict[str, Any] | None = None,
    mode_name: str = "fast",
) -> str:
    """Build a best-effort synthesis when LLM synthesis fails.

    This is a last-resort fallback used only when the LLM API call itself
    errors (not for normal sentinel-driven flow, which now always runs
    LLM synthesis on the first pass).

    Produces a structured report with section headers and a conclusion.
    Individual findings are progressively shortened to stay within a
    reasonable output size.
    """
    valid: list[tuple[str, str]] = []
    for sq, entry in findings_board.items():
        finding = (entry.get("finding") or {}) if isinstance(entry, dict) else {}
        if finding.get("error"):
            continue
        answer = finding.get("answer", "")
        if answer:
            valid.append((sq, _strip_noise(answer)))

    if not valid:
        return _build_zero_findings_message(state)

    budget_by_mode = {
        "fast": 40_000,
        "extended": 60_000,
        "fast_max": 60_000,
        "extended_max": 80_000,
        "max": 80_000,
    }
    total_budget = budget_by_mode.get(mode_name, 60_000)
    header_budget = 600
    conclusion_budget = 600
    per_finding_budget = max(
        500,
        (total_budget - header_budget - conclusion_budget) // len(valid),
    )

    topics = [sq for sq, _ in valid[:5]]
    topics_str = ", ".join(topics)
    if len(valid) > 5:
        topics_str += f", and {len(valid) - 5} more"

    parts = [
        f"# Research Analysis\n\n"
        f"*Analysis of **{query}** based on {len(valid)} research findings "
        f"covering: {topics_str}.*\n"
    ]

    for sq, answer in valid:
        if not answer.strip():
            continue
        if len(answer) <= per_finding_budget:
            parts.append(f"---\n\n### {sq}\n\n{answer}\n")
        else:
            summarized = answer[: per_finding_budget - 100].rsplit(". ", 1)[0] + "."
            parts.append(f"---\n\n### {sq}\n\n{summarized}\n")

    parts.append(
        "\n---\n\n## Conclusion\n\n"
        f"This analysis synthesized {len(valid)} research findings addressing "
        f"*{query}*. The sections above cover the key dimensions discovered "
        f"during the research session."
    )

    return "\n".join(parts)


@dataclass
class SubAgentResult:
    """Result from a self-reflecting research sub-agent."""

    finding: Finding
    quality_score: float
    confidence: str
    summary: str
    events: List[Dict[str, Any]]


def _detect_truncation(answer: str) -> bool:
    """True when a worker answer appears to have been cut off mid-sentence."""
    stripped = answer.rstrip()
    if not stripped:
        return False
    if stripped[-1] in '.!?)"]':
        return False
    if stripped.endswith("```") or stripped.endswith("---"):
        return False
    return True


def _get_data_quality_score(
    subquery: str,
    finding: Finding,
    findings_board: Dict[str, FindingEntry] | None,
) -> float:
    """Get data quality score from findings_board or finding."""
    if findings_board and subquery in findings_board:
        entry = findings_board[subquery]
        val = entry.get("data_quality_score") or entry.get("quality_score") or 0.5
        return float(val)
    if finding.get("data_quality_score") is not None:
        return float(finding.get("data_quality_score") or 0.5)
    return 0.5


def _append_status_and_classify(
    finding: Finding,
    subquery: str,
    entry_parts: list[str],
) -> tuple[bool, list[str], list[str], list[str]]:
    """Append status to entry_parts and return (has_usable_data, access_denied, failed, successful)."""
    access_denied_list: list[str] = []
    failed_list: list[str] = []
    successful_list: list[str] = []
    access_denied = finding.get("access_denied", False)
    error = finding.get("error")
    answer = finding.get("answer", "")
    cached = finding.get("cached", False)

    if access_denied:
        entry_parts.append(
            "Status: ACCESS DENIED - User does not have access to required data"
        )
        access_denied_list.append(subquery)
        return False, access_denied_list, failed_list, successful_list

    if error:
        simplified_error = simplify_error_for_display(error)
        entry_parts.append(f"Status: QUERY FAILED - {simplified_error}")
        failed_list.append(subquery)
        return False, access_denied_list, failed_list, successful_list

    if answer:
        cleaned_answer = _strip_noise(answer)
        if _detect_truncation(cleaned_answer):
            cleaned_answer += "\n[TRUNCATED - use available data only]"
        entry_parts.append(f"Answer: {cleaned_answer}")
        successful_list.append(subquery)
        if cached:
            entry_parts.append("(cached from previous research)")
        return True, access_denied_list, failed_list, successful_list

    return False, access_denied_list, failed_list, successful_list


def _format_single_finding_for_synthesis(
    subquery: str,
    finding: Finding,
    findings_board: Dict[str, FindingEntry] | None,
) -> tuple[str, list[str], list[str], list[str]]:
    """Format one finding. Returns (formatted_text, access_denied, failed, successful)."""
    dq_score = _get_data_quality_score(subquery, finding, findings_board)
    entry_parts = [f"[Quality: {dq_score:.2f}] Subquery: {subquery}"]

    has_usable_data, access_denied_list, failed_list, successful_list = (
        _append_status_and_classify(finding, subquery, entry_parts)
    )

    if not has_usable_data:
        entry_parts.append(
            "WARNING: This finding has NO usable data. "
            "Do NOT write a report section about it. "
            "Do NOT fabricate numbers. Mention in Limitations only."
        )

    tool_results = finding.get("tool_results", [])
    if tool_results:
        entry_parts.append("Tool Results:")
        for tr in tool_results[:15]:
            entry_parts.append(f"  - {truncate_text(str(tr), 2000)}")

    return "\n".join(entry_parts), access_denied_list, failed_list, successful_list


def _compress_parts_for_budget(parts: list[str], max_chars: int) -> list[str]:
    """Compress parts when total budget is set. Returns compressed parts."""
    if not parts or max_chars <= 0:
        return parts
    summary_reserve = 500
    available = max(1000, max_chars - summary_reserve)
    per_finding_budget = available // max(1, len(parts))
    compressed: list[str] = []
    for part in parts:
        if len(part) <= per_finding_budget:
            compressed.append(part)
        else:
            trimmed = part[:per_finding_budget].rsplit(". ", 1)[0] + "."
            compressed.append(trimmed + "\n[...truncated for context budget]")
    return compressed


def _build_findings_summary(
    access_denied_findings: list[str],
    failed_findings: list[str],
    successful_findings: list[str],
) -> list[str]:
    """Build summary parts for findings."""
    summary_parts = ["--- FINDINGS SUMMARY ---"]
    summary_parts.append(f"Successful queries: {len(successful_findings)}")
    summary_parts.append(f"Access denied queries: {len(access_denied_findings)}")
    summary_parts.append(f"Failed queries: {len(failed_findings)}")
    if access_denied_findings:
        summary_parts.append("\nQueries with access denied (include in final answer):")
        for sq in access_denied_findings:
            summary_parts.append(f"  - {sq}")
    if failed_findings:
        summary_parts.append("\nQueries that failed (include in final answer):")
        for sq in failed_findings:
            summary_parts.append(f"  - {sq}")
    return summary_parts


def _format_findings_for_synthesis(
    findings: Dict[str, Finding],
    findings_board: Dict[str, FindingEntry] | None = None,
    max_chars: int | None = None,
) -> str:
    """Format findings dict into text for synthesis prompt.

    Preserves full tool results so the synthesizer has access to actual data.
    Includes data_quality_score per finding when findings_board is provided.

    When *max_chars* is set, each finding is allocated a proportional
    budget and individually trimmed rather than naively slicing the end.
    """
    if not findings:
        return "No findings available."

    parts: list[str] = []
    access_denied_findings: list[str] = []
    failed_findings: list[str] = []
    successful_findings: list[str] = []

    for subquery, finding in findings.items():
        part, acc_den, failed, succ = _format_single_finding_for_synthesis(
            subquery, finding, findings_board
        )
        parts.append(part)
        access_denied_findings.extend(acc_den)
        failed_findings.extend(failed)
        successful_findings.extend(succ)

    if max_chars is not None:
        parts = _compress_parts_for_budget(parts, max_chars)

    summary_parts = _build_findings_summary(
        access_denied_findings, failed_findings, successful_findings
    )
    return "\n\n".join(parts) + "\n\n" + "\n".join(summary_parts)


def _check_headings(draft: str) -> tuple[bool, str | None]:
    """Check for at least one heading. Returns (passed, violation)."""
    headings = re.findall(r"^#{1,3}\s+.+", draft, re.MULTILINE)
    return (bool(headings), None if headings else "No headings found in draft")


def _check_executive_summary(draft: str) -> tuple[bool, str | None]:
    """Check for executive summary section. Returns (passed, violation)."""
    if re.search(
        r"(?i)##?\s+(executive\s+summary|key\s+finding|summary|overview)",
        draft,
    ):
        return True, None
    return False, "Missing executive summary / key finding section"


def _check_conclusion_section(draft: str) -> tuple[bool, str | None]:
    """Check for conclusion/takeaways section. Returns (passed, violation)."""
    if re.search(
        r"(?i)##?\s+(conclusion|key\s+takeaways|takeaways|recommendations)",
        draft,
    ):
        return True, None
    return False, "Missing conclusion / takeaways section"


def _check_no_meta_commentary(draft: str) -> tuple[bool, str | None]:
    """Check no meta-commentary in output. Returns (passed, violation)."""
    meta_patterns = [
        "I've created",
        "Based on my analysis",
        "I'll ",
        "I can ",
        "I would ",
        "I'd ",
        "Let me ",
    ]
    has_meta = any(p.lower() in draft.lower() for p in meta_patterns)
    return (
        not has_meta,
        None if not has_meta else "Meta-commentary detected in output",
    )


def _check_word_count(word_count: int, gate: Any) -> tuple[bool, str | None]:
    """Check word count within mode range. Returns (passed, violation)."""
    lo, hi = gate.target_word_count_range
    if lo <= word_count <= hi * 1.2:
        return True, None
    return False, f"Word count {word_count} outside target range {lo}-{hi}"


def _check_section_count(section_count: int, gate: Any) -> tuple[bool, str | None]:
    """Check section count within range. Returns (passed, violation)."""
    if gate.min_sections <= section_count <= gate.max_sections + 2:
        return True, None
    return False, (
        f"Section count {section_count} outside range "
        f"{gate.min_sections}-{gate.max_sections}"
    )


def _check_table_count(table_count: int, gate: Any) -> tuple[bool, str | None]:
    """Check minimum tables. Returns (passed, violation)."""
    if table_count >= gate.min_tables:
        return True, None
    return False, f"Only {table_count} tables, minimum is {gate.min_tables}"


def _run_gate_checks(
    _draft: str,
    gate: Any | None,
    word_count: int,
    section_count: int,
    table_count: int,
    checks_total: int,
    checks_passed: int,
    violations: list[str],
) -> tuple[int, int]:
    """Run quality gate checks. Returns (checks_total, checks_passed)."""
    if not gate:
        return checks_total, checks_passed
    passed, violation = _check_word_count(word_count, gate)
    checks_total += 1
    if passed:
        checks_passed += 1
    elif violation:
        violations.append(violation)
    passed, violation = _check_section_count(section_count, gate)
    checks_total += 1
    if passed:
        checks_passed += 1
    elif violation:
        violations.append(violation)
    passed, violation = _check_table_count(table_count, gate)
    checks_total += 1
    if passed:
        checks_passed += 1
    elif violation:
        violations.append(violation)
    return checks_total, checks_passed


def _run_structural_checks(draft: str) -> tuple[int, int, list[str]]:
    """Run base structural checks. Returns (checks_total, checks_passed, violations)."""
    violations: list[str] = []
    checks_total = 0
    checks_passed = 0
    for check_fn in [
        _check_headings,
        _check_executive_summary,
        _check_conclusion_section,
        _check_no_meta_commentary,
    ]:
        passed, violation = check_fn(draft)
        checks_total += 1
        if passed:
            checks_passed += 1
        elif violation:
            violations.append(violation)
    return checks_total, checks_passed, violations


def _apply_gate_confidence_checks(
    draft: str, gate: Any, checks_total: int, checks_passed: int, violations: list[str]
) -> tuple[int, int]:
    """Apply gate-specific confidence and methodology checks when gate requires them."""
    if not gate or not gate.confidence_table_required:
        return checks_total, checks_passed

    checks_total += 1
    if re.search(r"(?i)confidence", draft):
        checks_passed += 1
    else:
        violations.append("Confidence assessment table required but missing")

    if gate.methodology_required:
        checks_total += 1
        if re.search(r"(?i)^##?\s+\S*methodolog", draft, re.MULTILINE):
            checks_passed += 1
        else:
            violations.append("Methodology section required but missing")
    return checks_total, checks_passed


def check_structural_compliance(
    draft: str,
    _query_type: str,
    mode_name: str,
) -> tuple[float, list[str]]:
    """Fast algorithmic check that the draft follows structural expectations.

    Returns (score 0.0-1.0, list of violation descriptions).
    """
    from template_agent.src.core.deep_research.mode_config import MODES

    violations: list[str] = []
    mode_cfg = MODES.get(mode_name, MODES.get("fast"))
    gate = mode_cfg.quality_gate if mode_cfg else None

    word_count = len(draft.split())
    headings = re.findall(r"^#{1,3}\s+.+", draft, re.MULTILINE)
    section_count = len(headings)
    tables = re.findall(r"^\|.+\|$", draft, re.MULTILINE)
    table_count = len(tables) // 2  # header + separator = 1 table

    checks_total, checks_passed, violations = _run_structural_checks(draft)

    checks_total, checks_passed = _run_gate_checks(
        draft,
        gate,
        word_count,
        section_count,
        table_count,
        checks_total,
        checks_passed,
        violations,
    )

    if gate:
        checks_total, checks_passed = _apply_gate_confidence_checks(
            draft, gate, checks_total, checks_passed, violations
        )

    score = checks_passed / checks_total if checks_total > 0 else 0.0
    return score, violations


def _find_redundant_sentence_pairs(sentences: list[str]) -> list[str]:
    """Find sentence pairs with high Jaccard overlap (redundancy)."""
    redundant_pairs: list[str] = []
    for i in range(len(sentences)):
        words_i = set(sentences[i].lower().split())
        for j in range(i + 1, min(i + 20, len(sentences))):
            words_j = set(sentences[j].lower().split())
            union = words_i | words_j
            if not union:
                continue
            jaccard = len(words_i & words_j) / len(union)
            if jaccard > 0.70:
                redundant_pairs.append(
                    f"High overlap ({jaccard:.0%}): "
                    f"'{truncate_text(sentences[i], 60)}' vs "
                    f"'{truncate_text(sentences[j], 60)}'"
                )
                if len(redundant_pairs) >= 10:
                    return redundant_pairs
        if len(redundant_pairs) >= 10:
            break
    return redundant_pairs


def _find_repeated_table_headers(draft: str) -> list[str]:
    """Find repeated table headers in the draft."""
    table_headers = re.findall(r"^\|(.+)\|$", draft, re.MULTILINE)
    seen_headers: set[str] = set()
    repeated: list[str] = []
    for h in table_headers:
        normalized = re.sub(r"\s+", " ", h.strip().lower())
        if normalized in seen_headers:
            repeated.append(f"Repeated table header: {h.strip()[:60]}")
        seen_headers.add(normalized)
    return repeated


def detect_redundancy(draft: str) -> tuple[float, list[str]]:
    """Detect redundant content in the draft using word-overlap heuristics.

    Returns (redundancy_score 0.0-1.0 where 0 is no redundancy, list of
    descriptions of redundant passages).
    """
    sentences = re.split(r"(?<=[.!?])\s+", draft)
    sentences = [s.strip() for s in sentences if len(s.split()) >= 6]

    if len(sentences) < 2:
        return 0.0, []

    redundant_pairs = _find_redundant_sentence_pairs(sentences)
    redundant_pairs.extend(_find_repeated_table_headers(draft))

    total_sentences = max(1, len(sentences))
    score = min(1.0, len(redundant_pairs) / (total_sentences * 0.1))
    return score, redundant_pairs
