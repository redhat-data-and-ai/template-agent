"""Eval case management — tag metrics, keyword normalization, and YAML file loading."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

CASES_FILE = "eval_cases.yaml"

# ── Default metrics per tag ───────────────────────────────────────────────────
# Loaded from tag_metrics.yaml alongside this file.
# Falls back to the hardcoded dict below if the file is missing.

_TAG_METRICS_FILE = Path(__file__).parent / "tag_metrics.yaml"
_DEFAULT_FALLBACK: dict[str, Any] = {"turn_metrics": ["custom:answer_correctness"]}
_tag_metrics_cache: dict[str, dict[str, Any]] | None = None
_tool_turn_metrics_cache: list[str] | None = None
_metric_thresholds_cache: dict[str, float | None] | None = None


def get_tool_turn_metrics() -> list[str]:
    """Return metrics only added to turns where tool calls are expected (from tag_metrics.yaml)."""
    global _tool_turn_metrics_cache
    if _tool_turn_metrics_cache is not None:
        return _tool_turn_metrics_cache
    data = yaml.safe_load(_TAG_METRICS_FILE.read_text(encoding="utf-8")) or {}
    _tool_turn_metrics_cache = [
        m for m in data.get("tool_turn_metrics", []) if isinstance(m, str)
    ]
    return _tool_turn_metrics_cache


def get_metric_thresholds() -> dict[str, float | None]:
    """Return metric thresholds from tag_metrics.yaml — single source of truth."""
    global _metric_thresholds_cache
    if _metric_thresholds_cache is not None:
        return _metric_thresholds_cache
    data = yaml.safe_load(_TAG_METRICS_FILE.read_text(encoding="utf-8")) or {}
    _metric_thresholds_cache = {
        k: v for k, v in data.get("metric_thresholds", {}).items()
    }
    return _metric_thresholds_cache


def _load_tag_metrics() -> dict[str, dict[str, Any]]:
    """Load tag_metrics.yaml once and cache it. Raises if the file is missing or invalid."""
    global _tag_metrics_cache
    if _tag_metrics_cache is not None:
        return _tag_metrics_cache
    data = yaml.safe_load(_TAG_METRICS_FILE.read_text(encoding="utf-8")) or {}
    _tag_metrics_cache = {k: v for k, v in data.items() if isinstance(v, dict)}
    log.info(
        "Loaded tag metrics from %s (%d tags)",
        _TAG_METRICS_FILE,
        len(_tag_metrics_cache),
    )
    return _tag_metrics_cache


def get_defaults_for_tag(tag: str) -> dict[str, Any]:
    """Return a copy of the default metric config for the given tag."""
    return copy.deepcopy(_load_tag_metrics().get(tag, _DEFAULT_FALLBACK))


# ── Keyword normalization ─────────────────────────────────────────────────────


def _normalize_keywords(keywords: list | str) -> list[list[str]]:
    """Normalize keywords to list[list[str]] (the format lightspeed-eval expects)."""
    if isinstance(keywords, str):
        return [[kw.strip()] for kw in keywords.split(",") if kw.strip()]
    if isinstance(keywords, list) and keywords:
        if isinstance(keywords[0], list):
            return keywords
        return [[str(kw)] for kw in keywords]
    return keywords


# ── File loading ──────────────────────────────────────────────────────────────


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    """Load cases from eval_cases.yaml, normalizing keyword fields at read time.

    Accepts a file path or a directory (in which case eval_cases.yaml is appended).
    """
    p = Path(path)
    if p.is_dir():
        p = p / CASES_FILE
    if not p.exists():
        log.warning("eval_cases file not found: %s", p)
        return []
    cases = yaml.safe_load(p.read_text()) or []
    for conv in cases:
        for turn in conv.get("turns", []):
            if "expected_keywords" in turn:
                turn["expected_keywords"] = _normalize_keywords(
                    turn["expected_keywords"]
                )
    return cases


def filter_cases_by_tag(path: str | Path, tag: str) -> list[dict[str, Any]]:
    """Return all conversations matching the given tag."""
    return [c for c in load_cases(path) if c.get("tag") == tag]
