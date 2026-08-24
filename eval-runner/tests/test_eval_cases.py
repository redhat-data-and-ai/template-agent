"""Tests for eval_cases.py — tag metrics, normalization, file loading."""

from __future__ import annotations

from pathlib import Path

import eval_cases
import pytest
import yaml

# ── _normalize_keywords ────────────────────────────────────────────────────────


def test_normalize_keywords_comma_string() -> None:
    result = eval_cases._normalize_keywords("22.9, Normal")
    assert result == [["22.9"], ["Normal"]]


def test_normalize_keywords_flat_list() -> None:
    result = eval_cases._normalize_keywords(["22.9", "Normal"])
    assert result == [["22.9"], ["Normal"]]


def test_normalize_keywords_already_nested() -> None:
    nested = [["22.9", "22.8"], ["Normal"]]
    result = eval_cases._normalize_keywords(nested)
    assert result == nested


def test_normalize_keywords_empty_list() -> None:
    assert eval_cases._normalize_keywords([]) == []


def test_normalize_keywords_string_strips_whitespace() -> None:
    result = eval_cases._normalize_keywords("  a ,  b  ")
    assert result == [["a"], ["b"]]


# ── get_defaults_for_tag ───────────────────────────────────────────────────────


def test_get_defaults_tool_use() -> None:
    d = eval_cases.get_defaults_for_tag("tool_use")
    assert "custom:answer_correctness" in d["turn_metrics"]


def test_get_defaults_hitl() -> None:
    d = eval_cases.get_defaults_for_tag("hitl")
    assert d.get("hitl") is True
    assert "custom:intent_eval" in d["turn_metrics"]


def test_get_defaults_non_hitl() -> None:
    d = eval_cases.get_defaults_for_tag("non_hitl")
    assert "custom:answer_correctness" in d["turn_metrics"]
    assert "geval:tone_safety" in d["turn_metrics"]


def test_get_defaults_multi_turn() -> None:
    d = eval_cases.get_defaults_for_tag("multi_turn")
    assert "custom:answer_correctness" in d["turn_metrics"]
    assert "deepeval:knowledge_retention" in d.get("conversation_metrics", [])


def test_get_defaults_structured_output_falls_back_to_default() -> None:
    # structured_output removed from tags — falls back to _DEFAULT_FALLBACK
    d = eval_cases.get_defaults_for_tag("structured_output")
    assert d == {"turn_metrics": ["custom:answer_correctness"]}


def test_get_defaults_unknown_tag() -> None:
    d = eval_cases.get_defaults_for_tag("unknown_tag")
    assert d == {"turn_metrics": ["custom:answer_correctness"]}


def test_get_defaults_returns_deep_copy() -> None:
    """Mutations to the returned dict must not affect subsequent calls."""
    d1 = eval_cases.get_defaults_for_tag("tool_use")
    d1["turn_metrics"].append("injected")
    d2 = eval_cases.get_defaults_for_tag("tool_use")
    assert "injected" not in d2["turn_metrics"]


# ── get_tool_turn_metrics ──────────────────────────────────────────────────────


def test_get_tool_turn_metrics_returns_list() -> None:
    metrics = eval_cases.get_tool_turn_metrics()
    assert isinstance(metrics, list)


def test_get_tool_turn_metrics_contains_faithfulness() -> None:
    metrics = eval_cases.get_tool_turn_metrics()
    assert "ragas:faithfulness" in metrics


def test_get_tool_turn_metrics_is_cached() -> None:
    m1 = eval_cases.get_tool_turn_metrics()
    m2 = eval_cases.get_tool_turn_metrics()
    assert m1 is m2  # same object — cached


# ── get_metric_thresholds ─────────────────────────────────────────────────────


def test_get_metric_thresholds_returns_dict() -> None:
    thresholds = eval_cases.get_metric_thresholds()
    assert isinstance(thresholds, dict)


def test_get_metric_thresholds_faithfulness_threshold() -> None:
    thresholds = eval_cases.get_metric_thresholds()
    assert thresholds.get("ragas:faithfulness") == 0.7


def test_get_metric_thresholds_answer_correctness() -> None:
    thresholds = eval_cases.get_metric_thresholds()
    assert thresholds.get("custom:answer_correctness") == 0.75


def test_get_metric_thresholds_is_cached() -> None:
    t1 = eval_cases.get_metric_thresholds()
    t2 = eval_cases.get_metric_thresholds()
    assert t1 is t2  # same object — cached


# ── load_cases ────────────────────────────────────────────────────────────────


def test_load_cases_returns_empty_for_missing_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    missing = tmp_path / "eval_cases.yaml"
    cases = eval_cases.load_cases(missing)
    assert cases == []
    assert "not found" in caplog.text


def test_load_cases_from_file(tmp_path: Path) -> None:
    data = [
        {
            "conversation_group_id": "c1",
            "tag": "non_hitl",
            "turns": [
                {
                    "turn_id": "t1",
                    "query": "Q",
                    "expected_response": "A",
                    "turn_metrics": ["custom:answer_correctness"],
                }
            ],
        }
    ]
    f = tmp_path / "eval_cases.yaml"
    f.write_text(yaml.dump(data))
    cases = eval_cases.load_cases(f)
    assert len(cases) == 1
    assert cases[0]["conversation_group_id"] == "c1"


def test_load_cases_normalizes_expected_keywords(tmp_path: Path) -> None:
    data = [
        {
            "conversation_group_id": "c1",
            "tag": "non_hitl",
            "turns": [
                {
                    "turn_id": "t1",
                    "query": "Q",
                    "expected_response": "A",
                    "turn_metrics": [],
                    "expected_keywords": "22.9, Normal",
                }
            ],
        }
    ]
    f = tmp_path / "eval_cases.yaml"
    f.write_text(yaml.dump(data))
    cases = eval_cases.load_cases(f)
    assert cases[0]["turns"][0]["expected_keywords"] == [["22.9"], ["Normal"]]


def test_load_cases_accepts_directory(tmp_path: Path) -> None:
    data = [{"conversation_group_id": "c1", "tag": "tool_use", "turns": []}]
    (tmp_path / "eval_cases.yaml").write_text(yaml.dump(data))
    cases = eval_cases.load_cases(tmp_path)  # pass directory
    assert len(cases) == 1


# ── filter_cases_by_tag ───────────────────────────────────────────────────────


def test_filter_cases_by_tag_match(tmp_path: Path) -> None:
    data = [
        {"conversation_group_id": "c1", "tag": "tool_use", "turns": []},
        {"conversation_group_id": "c2", "tag": "hitl", "turns": []},
    ]
    f = tmp_path / "eval_cases.yaml"
    f.write_text(yaml.dump(data))
    results = eval_cases.filter_cases_by_tag(f, "tool_use")
    assert len(results) == 1
    assert results[0]["tag"] == "tool_use"


def test_filter_cases_by_tag_no_match(tmp_path: Path) -> None:
    data = [{"conversation_group_id": "c1", "tag": "tool_use", "turns": []}]
    f = tmp_path / "eval_cases.yaml"
    f.write_text(yaml.dump(data))
    assert eval_cases.filter_cases_by_tag(f, "hitl") == []


def test_filter_cases_by_tag_empty(tmp_path: Path) -> None:
    f = tmp_path / "eval_cases.yaml"
    f.write_text(yaml.dump([]))
    assert eval_cases.filter_cases_by_tag(f, "tool_use") == []
