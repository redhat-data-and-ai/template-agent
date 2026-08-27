"""Tests for eval_postgres.py — DB access layer."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import eval_postgres
import pytest

# ── _compute_config_hash ──────────────────────────────────────────────────────


def test_compute_config_hash_returns_16_chars(tmp_path: Path) -> None:
    h = eval_postgres._compute_config_hash(str(tmp_path))
    assert len(h) == 16


def test_compute_config_hash_changes_with_content(tmp_path: Path) -> None:
    (tmp_path / "system.yaml").write_text("model: gpt4")
    h1 = eval_postgres._compute_config_hash(str(tmp_path))
    (tmp_path / "system.yaml").write_text("model: gpt3")
    h2 = eval_postgres._compute_config_hash(str(tmp_path))
    assert h1 != h2


def test_compute_config_hash_excludes_evals_dir(tmp_path: Path) -> None:
    """Changes inside evals/ must not affect the hash."""
    (tmp_path / "system.yaml").write_text("model: gpt4")
    h1 = eval_postgres._compute_config_hash(str(tmp_path))
    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "eval_cases.yaml").write_text("- query: hi")
    h2 = eval_postgres._compute_config_hash(str(tmp_path))
    assert h1 == h2


def test_compute_config_hash_excludes_deployment_dir(tmp_path: Path) -> None:
    """Changes inside deployment/ must not affect the hash."""
    (tmp_path / "system.yaml").write_text("model: gpt4")
    h1 = eval_postgres._compute_config_hash(str(tmp_path))
    deployment = tmp_path / "deployment"
    deployment.mkdir()
    (deployment / "config.yaml").write_text("replicas: 3")
    h2 = eval_postgres._compute_config_hash(str(tmp_path))
    assert h1 == h2


def test_compute_config_hash_missing_dir() -> None:
    h = eval_postgres._compute_config_hash("/nonexistent/path/xyz")
    assert len(h) == 16  # empty hash — consistent return type


def test_compute_config_hash_ignores_non_config_extensions(tmp_path: Path) -> None:
    (tmp_path / "system.yaml").write_text("a: 1")
    h1 = eval_postgres._compute_config_hash(str(tmp_path))
    (tmp_path / "notes.txt").write_text("irrelevant")
    h2 = eval_postgres._compute_config_hash(str(tmp_path))
    assert h1 == h2


def test_compute_config_hash_includes_json_and_md(tmp_path: Path) -> None:
    (tmp_path / "system.yaml").write_text("a: 1")
    h1 = eval_postgres._compute_config_hash(str(tmp_path))
    (tmp_path / "readme.md").write_text("# docs")
    (tmp_path / "tools.json").write_text("{}")
    h2 = eval_postgres._compute_config_hash(str(tmp_path))
    assert h1 != h2


def test_get_config_hash_prefers_env_var() -> None:
    with patch.object(eval_postgres, "_env_hash", "fixed-hash-from-env"):
        assert eval_postgres._get_config_hash() == "fixed-hash-from-env"


# ── ensure_table ──────────────────────────────────────────────────────────────


def test_ensure_table_executes_create(
    mock_psycopg2_conn: tuple[MagicMock, MagicMock],
) -> None:
    conn, cursor = mock_psycopg2_conn
    with patch("eval_postgres._get_conn", return_value=conn):
        eval_postgres.ensure_table()
    assert eval_postgres._table_ensured is True
    cursor.execute.assert_called()
    conn.close.assert_called_once()


def test_ensure_table_idempotent(
    mock_psycopg2_conn: tuple[MagicMock, MagicMock],
) -> None:
    """Second call must not hit the DB."""
    conn, cursor = mock_psycopg2_conn
    eval_postgres._table_ensured = True
    with patch("eval_postgres._get_conn", return_value=conn):
        eval_postgres.ensure_table()
    conn.cursor.assert_not_called()


def test_ensure_table_handles_db_error(caplog: pytest.LogCaptureFixture) -> None:
    with patch("eval_postgres._get_conn", side_effect=Exception("connection refused")):
        eval_postgres.ensure_table()
    assert eval_postgres._table_ensured is False
    assert "eval_postgres_ensure_table_failed" in caplog.text


# ── write_eval_result ─────────────────────────────────────────────────────────


def test_write_eval_result_success(
    mock_psycopg2_conn: tuple[MagicMock, MagicMock], caplog: pytest.LogCaptureFixture
) -> None:
    conn, cursor = mock_psycopg2_conn
    cursor.fetchone.return_value = (42,)
    with caplog.at_level(logging.INFO):
        with patch("eval_postgres._get_conn", return_value=conn):
            eval_postgres.write_eval_result(
                passed=8,
                failed=2,
                errors=0,
                eval_score=0.8,
                ls_run_ids=["run1"],
            )
    cursor.execute.assert_called_once()
    args = cursor.execute.call_args[0][1]
    assert args[0] == ["run1"]  # ls_run_ids
    assert args[1] == 0.8  # eval_score
    assert "eval_result_written_to_postgres" in caplog.text
    conn.close.assert_called_once()


def test_write_eval_result_logs_warning_when_no_match(
    mock_psycopg2_conn: tuple[MagicMock, MagicMock], caplog: pytest.LogCaptureFixture
) -> None:
    conn, cursor = mock_psycopg2_conn
    cursor.fetchone.return_value = None
    with patch("eval_postgres._get_conn", return_value=conn):
        eval_postgres.write_eval_result(
            passed=0,
            failed=1,
            errors=0,
            eval_score=0.0,
        )
    assert "eval_postgres_no_matching_record" in caplog.text


def test_write_eval_result_handles_db_error(caplog: pytest.LogCaptureFixture) -> None:
    with patch("eval_postgres._get_conn", side_effect=Exception("timeout")):
        eval_postgres.write_eval_result(
            passed=0,
            failed=0,
            errors=1,
            eval_score=0.0,
        )
    assert "eval_postgres_write_failed" in caplog.text


def test_write_eval_result_uses_default_config_hash(
    mock_psycopg2_conn: tuple[MagicMock, MagicMock],
) -> None:
    """config_hash should fall back to _get_config_hash() when not provided."""
    conn, cursor = mock_psycopg2_conn
    cursor.fetchone.return_value = (1,)
    with patch("eval_postgres._get_conn", return_value=conn):
        eval_postgres.write_eval_result(passed=1, failed=0, errors=0, eval_score=1.0)
    cursor.execute.assert_called_once()
    args = cursor.execute.call_args[0][1]
    assert args[9] == eval_postgres._get_config_hash()


def test_write_eval_result_serialises_results_detail(
    mock_psycopg2_conn: tuple[MagicMock, MagicMock],
) -> None:
    conn, cursor = mock_psycopg2_conn
    cursor.fetchone.return_value = (1,)
    detail: dict[str, Any] = {"turns": [], "summary": {}}
    with patch("eval_postgres._get_conn", return_value=conn):
        eval_postgres.write_eval_result(
            passed=1,
            failed=0,
            errors=0,
            eval_score=1.0,
            results_detail=detail,
        )
    args = cursor.execute.call_args[0][1]
    assert args[6] == json.dumps(detail)


def test_write_eval_result_passes_explicit_config_hash(
    mock_psycopg2_conn: tuple[MagicMock, MagicMock],
) -> None:
    conn, cursor = mock_psycopg2_conn
    cursor.fetchone.return_value = (1,)
    with patch("eval_postgres._get_conn", return_value=conn):
        eval_postgres.write_eval_result(
            passed=1,
            failed=0,
            errors=0,
            eval_score=1.0,
            config_hash="abc123",
        )
    args = cursor.execute.call_args[0][1]
    assert args[9] == "abc123"


# ── load_results_since ────────────────────────────────────────────────────────


def test_load_results_since_empty_returns_empty(
    mock_psycopg2_conn: tuple[MagicMock, MagicMock], caplog: pytest.LogCaptureFixture
) -> None:
    conn, cursor = mock_psycopg2_conn
    cursor.fetchall.return_value = []
    with caplog.at_level(logging.WARNING):
        with patch("eval_postgres._get_conn", return_value=conn):
            result = eval_postgres.load_results_since(datetime.now(UTC))
    assert result == {}
    assert "No evaluation_results rows found since" in caplog.text


def test_load_results_since_aggregates_correctly(
    mock_psycopg2_conn: tuple[MagicMock, MagicMock],
) -> None:
    conn, cursor = mock_psycopg2_conn
    now = datetime.now(UTC)
    cursor.fetchall.return_value = [
        {
            "run_id": "run1",
            "result": "PASS",
            "metric_identifier": "custom:tool_eval",
            "conversation_group_id": "conv1",
            "score": 1.0,
            "timestamp": now,
        },
        {
            "run_id": "run1",
            "result": "FAIL",
            "metric_identifier": "custom:tool_eval",
            "conversation_group_id": "conv1",
            "score": 0.0,
            "timestamp": now,
        },
    ]
    with patch("eval_postgres._get_conn", return_value=conn):
        result = eval_postgres.load_results_since(now)

    overall = result["summary"]["summary_stats"]["overall"]
    assert overall["PASS"] == 1
    assert overall["FAIL"] == 1
    assert overall["pass_rate"] == 0.5
    assert len(result["turns"]) == 2
    assert result["ls_run_ids"] == ["run1"]


def test_load_results_since_serialises_datetime_in_turns(
    mock_psycopg2_conn: tuple[MagicMock, MagicMock],
) -> None:
    conn, cursor = mock_psycopg2_conn
    now = datetime.now(UTC)
    cursor.fetchall.return_value = [
        {
            "run_id": "r1",
            "result": "PASS",
            "metric_identifier": "m",
            "conversation_group_id": "c",
            "score": 1.0,
            "timestamp": now,
        },
    ]
    with patch("eval_postgres._get_conn", return_value=conn):
        result = eval_postgres.load_results_since(now)
    assert isinstance(result["turns"][0]["timestamp"], str)


def test_load_results_since_serialises_float_scores(
    mock_psycopg2_conn: tuple[MagicMock, MagicMock],
) -> None:
    conn, cursor = mock_psycopg2_conn
    now = datetime.now(UTC)
    cursor.fetchall.return_value = [
        {
            "run_id": "r1",
            "result": "PASS",
            "metric_identifier": "m",
            "conversation_group_id": "c",
            "score": 0.95,
            "timestamp": now,
        },
    ]
    with patch("eval_postgres._get_conn", return_value=conn):
        result = eval_postgres.load_results_since(now)
    assert result["turns"][0]["score"] == "0.95"


def test_load_results_since_handles_db_error(caplog: pytest.LogCaptureFixture) -> None:
    with patch("eval_postgres._get_conn", side_effect=Exception("db down")):
        result = eval_postgres.load_results_since(datetime.now(UTC))
    assert result == {}
    assert "load_results_since_failed" in caplog.text


def test_load_results_since_error_result_counted_as_error(
    mock_psycopg2_conn: tuple[MagicMock, MagicMock],
) -> None:
    conn, cursor = mock_psycopg2_conn
    now = datetime.now(UTC)
    cursor.fetchall.return_value = [
        {
            "run_id": "r1",
            "result": "UNKNOWN",
            "metric_identifier": "m",
            "conversation_group_id": "c",
            "score": 0.0,
            "timestamp": now,
        },
    ]
    with patch("eval_postgres._get_conn", return_value=conn):
        result = eval_postgres.load_results_since(now)
    assert result["summary"]["summary_stats"]["overall"]["ERROR"] == 1


def test_load_results_since_by_metric_and_conversation(
    mock_psycopg2_conn: tuple[MagicMock, MagicMock],
) -> None:
    conn, cursor = mock_psycopg2_conn
    now = datetime.now(UTC)
    cursor.fetchall.return_value = [
        {
            "run_id": "r1",
            "result": "PASS",
            "metric_identifier": "custom:tool_eval",
            "conversation_group_id": "conv1",
            "score": 1.0,
            "timestamp": now,
        },
        {
            "run_id": "r1",
            "result": "FAIL",
            "metric_identifier": "custom:intent_eval",
            "conversation_group_id": "conv2",
            "score": 0.0,
            "timestamp": now,
        },
    ]
    with patch("eval_postgres._get_conn", return_value=conn):
        result = eval_postgres.load_results_since(now)

    by_metric = result["summary"]["summary_stats"]["by_metric"]
    assert by_metric["custom:tool_eval"]["pass"] == 1
    assert by_metric["custom:intent_eval"]["fail"] == 1

    by_conv = result["summary"]["summary_stats"]["by_conversation"]
    assert by_conv["conv1"]["pass"] == 1
    assert by_conv["conv2"]["fail"] == 1


# ── get_results_by_run_id ─────────────────────────────────────────────────────


def test_get_results_by_run_id_returns_rows(
    mock_psycopg2_conn: tuple[MagicMock, MagicMock],
) -> None:
    conn, cursor = mock_psycopg2_conn
    cursor.description = [
        ("conversation_group_id",),
        ("turn_id",),
        ("metric_identifier",),
        ("result",),
        ("score",),
        ("reason",),
    ]
    cursor.fetchall.return_value = [
        ("conv1", "turn_1", "custom:tool_eval", "PASS", 1.0, "ok")
    ]
    with patch("eval_postgres._get_conn", return_value=conn):
        rows = eval_postgres.get_results_by_run_id("run1")
    assert len(rows) == 1
    assert rows[0]["result"] == "PASS"
    assert rows[0]["score"] == 1.0
    conn.close.assert_called_once()


def test_get_results_by_run_id_empty(
    mock_psycopg2_conn: tuple[MagicMock, MagicMock],
) -> None:
    conn, cursor = mock_psycopg2_conn
    cursor.description = [("conversation_group_id",), ("result",)]
    cursor.fetchall.return_value = []
    with patch("eval_postgres._get_conn", return_value=conn):
        rows = eval_postgres.get_results_by_run_id("nonexistent")
    assert rows == []


# ── _dataset_to_eval_cases ────────────────────────────────────────────────────


def _make_dataset(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {"cases": cases}


def _simple_turn(
    user_msg: str = "What is my BMI?", expected: str = "Your BMI is 22.9"
) -> dict[str, Any]:
    return {
        "id": "t1",
        "userMessage": user_msg,
        "expectedResponse": expected,
        "expectedIntent": "",
        "expectedKeywords": [""],
        "toolCallEnabled": False,
        "toolCallOrdered": False,
        "expectedToolCalls": [],
    }


def test_dataset_to_eval_cases_basic() -> None:
    tc = {
        "id": "abc",
        "name": "calc_bmi",
        "description": "BMI test",
        "tag": "non_hitl",
        "turns": [_simple_turn()],
        "createdAt": "2026-01-01T00:00:00Z",
    }
    cases = eval_postgres._dataset_to_eval_cases(_make_dataset([tc]))
    assert len(cases) == 1
    c = cases[0]
    assert c["conversation_group_id"] == "calc_bmi"
    assert c["tag"] == "non_hitl"
    assert len(c["turns"]) == 1
    assert c["turns"][0]["query"] == "What is my BMI?"


def test_dataset_to_eval_cases_auto_adds_metrics() -> None:
    tc = {
        "id": "abc",
        "name": "calc_bmi",
        "description": "",
        "tag": "non_hitl",
        "turns": [_simple_turn()],
        "createdAt": "2026-01-01",
    }
    cases = eval_postgres._dataset_to_eval_cases(_make_dataset([tc]))
    metrics = cases[0]["turns"][0]["turn_metrics"]
    assert "custom:answer_correctness" in metrics
    assert "geval:tone_safety" in metrics


def test_dataset_to_eval_cases_tool_call_enabled() -> None:
    turn = {
        **_simple_turn(),
        "toolCallEnabled": True,
        "toolCallOrdered": False,
        "expectedToolCalls": [{"toolName": "calculate_bmi", "arguments": []}],
    }
    tc = {
        "id": "abc",
        "name": "bmi",
        "description": "",
        "tag": "non_hitl",
        "turns": [turn],
        "createdAt": "2026-01-01",
    }
    cases = eval_postgres._dataset_to_eval_cases(_make_dataset([tc]))
    c = cases[0]
    assert c["tag"] == "tool_use"  # promoted when tool calls present
    t = c["turns"][0]
    assert "custom:tool_eval" in t["turn_metrics"]
    assert "ragas:faithfulness" in t["turn_metrics"]
    assert t.get("expected_tool_calls") is not None


def test_dataset_to_eval_cases_tool_call_with_args() -> None:
    turn = {
        **_simple_turn(),
        "toolCallEnabled": True,
        "toolCallOrdered": False,
        "expectedToolCalls": [
            {
                "toolName": "calculate_bmi",
                "arguments": [{"key": "height_cm", "value": ""}],
            }
        ],
    }
    tc = {
        "id": "abc",
        "name": "bmi",
        "description": "",
        "tag": "non_hitl",
        "turns": [turn],
        "createdAt": "2026-01-01",
    }
    cases = eval_postgres._dataset_to_eval_cases(_make_dataset([tc]))
    t = cases[0]["turns"][0]
    # Empty value should become ".*"
    assert t["expected_tool_calls"][0][0]["arguments"]["height_cm"] == ".*"


def test_dataset_to_eval_cases_intent_adds_metric() -> None:
    turn = {**_simple_turn(), "expectedIntent": "calculate BMI"}
    tc = {
        "id": "abc",
        "name": "bmi",
        "description": "",
        "tag": "non_hitl",
        "turns": [turn],
        "createdAt": "2026-01-01",
    }
    cases = eval_postgres._dataset_to_eval_cases(_make_dataset([tc]))
    t = cases[0]["turns"][0]
    assert t["expected_intent"] == "calculate BMI"
    assert "custom:intent_eval" in t["turn_metrics"]


def test_dataset_to_eval_cases_keywords_adds_metric() -> None:
    turn = {**_simple_turn(), "expectedKeywords": ["Normal, 22.9"]}
    tc = {
        "id": "abc",
        "name": "bmi",
        "description": "",
        "tag": "non_hitl",
        "turns": [turn],
        "createdAt": "2026-01-01",
    }
    cases = eval_postgres._dataset_to_eval_cases(_make_dataset([tc]))
    t = cases[0]["turns"][0]
    assert "custom:keywords_eval" in t["turn_metrics"]
    assert t["expected_keywords"] == [["Normal", "22.9"]]


def test_dataset_to_eval_cases_hitl_last_turn_only() -> None:
    turns = [_simple_turn("turn 1"), _simple_turn("turn 2")]
    tc = {
        "id": "abc",
        "name": "hitl",
        "description": "",
        "tag": "hitl",
        "turns": turns,
        "createdAt": "2026-01-01",
    }
    cases = eval_postgres._dataset_to_eval_cases(_make_dataset([tc]))
    t_last = cases[0]["turns"][-1]
    t_first = cases[0]["turns"][0]
    assert t_last.get("hitl") is True
    assert t_last.get("expected_intent") == "request approval before taking action"
    assert t_first.get("hitl") is None  # NOT applied to first turn


def test_dataset_to_eval_cases_multi_turn_conversation_metrics() -> None:
    turns = [_simple_turn("t1"), _simple_turn("t2")]
    tc = {
        "id": "abc",
        "name": "multi",
        "description": "",
        "tag": "multi_turn",
        "turns": turns,
        "createdAt": "2026-01-01",
    }
    cases = eval_postgres._dataset_to_eval_cases(_make_dataset([tc]))
    assert "deepeval:knowledge_retention" in cases[0].get("conversation_metrics", [])


def test_dataset_to_eval_cases_empty_dataset() -> None:
    cases = eval_postgres._dataset_to_eval_cases({"cases": []})
    assert cases == []
