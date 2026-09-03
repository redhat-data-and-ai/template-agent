"""Unit tests for deep_agent/aegra/eval_routes.py."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import deep_agent.aegra.eval_routes as er

# ── helpers ───────────────────────────────────────────────────────────────────


class _AsyncCursor:
    """Async cursor mock that supports fetchone, fetchall, and `async for`."""

    def __init__(self, rows=None, description=None):
        self._rows = list(rows or [])
        self.description = list(description or [])

    async def fetchone(self):
        return self._rows[0] if self._rows else None

    async def fetchall(self):
        return self._rows

    def __aiter__(self):
        return self._async_gen()

    async def _async_gen(self):
        for row in self._rows:
            yield row


def _make_cursor(rows=None, description=None):
    return _AsyncCursor(rows=rows, description=description)


def _make_conn(cursor=None):
    """Return a mock async psycopg connection."""
    conn = AsyncMock()
    cur = cursor or _make_cursor()
    conn.execute = AsyncMock(return_value=cur)
    conn.set_autocommit = AsyncMock()
    conn.close = AsyncMock()
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=False)
    return conn, cur


def _col(name):
    c = MagicMock()
    c.name = name
    return c


# ── _pg_row_to_dict ───────────────────────────────────────────────────────────


class TestPgRowToDict:
    def test_basic(self):
        cursor = MagicMock()
        cursor.description = [_col("id"), _col("name"), _col("score")]
        row = (1, "test", 0.9)
        result = er._pg_row_to_dict(row, cursor)
        assert result == {"id": 1, "name": "test", "score": 0.9}

    def test_empty_row(self):
        cursor = MagicMock()
        cursor.description = []
        assert er._pg_row_to_dict((), cursor) == {}


# ── _messages_from_run ────────────────────────────────────────────────────────


class TestMessagesFromRun:
    def test_top_level_messages(self):
        state = {"messages": [{"type": "ai", "content": "hi"}]}
        assert er._messages_from_run(state) == state["messages"]

    def test_nested_under_values(self):
        msgs = [{"type": "ai", "content": "nested"}]
        state = {"values": {"messages": msgs}}
        assert er._messages_from_run(state) == msgs

    def test_empty_state(self):
        assert er._messages_from_run({}) == []


# ── _detect_interrupt ─────────────────────────────────────────────────────────


class TestDetectInterrupt:
    def test_top_level_interrupt(self):
        assert er._detect_interrupt({"__interrupt__": [{"value": {}}]}) is True

    def test_no_interrupt(self):
        assert er._detect_interrupt({"messages": []}) is False

    def test_interrupt_in_values(self):
        assert er._detect_interrupt({"values": {"__interrupt__": True}}) is True

    def test_interrupt_in_next_nodes(self):
        assert er._detect_interrupt({"next": ["__interrupt__"]}) is True

    def test_next_nodes_no_interrupt(self):
        assert er._detect_interrupt({"next": ["some_node", "other_node"]}) is False

    def test_empty_next_list(self):
        assert er._detect_interrupt({"next": []}) is False


# ── _extract_from_messages ────────────────────────────────────────────────────


class TestExtractFromMessages:
    def test_ai_text_response(self):
        msgs = [{"type": "ai", "content": "Hello world"}]
        resp, tcs, ctxs = er._extract_from_messages(msgs)
        assert resp == "Hello world"
        assert tcs == []
        assert ctxs == []

    def test_ai_block_content(self):
        msgs = [
            {
                "type": "ai",
                "content": [
                    {"type": "text", "text": "block response"},
                    {"type": "image", "data": "..."},
                ],
            }
        ]
        resp, _, _ = er._extract_from_messages(msgs)
        assert resp == "block response"

    def test_tool_message_becomes_context(self):
        msgs = [{"type": "tool", "content": "tool result text"}]
        _, _, ctxs = er._extract_from_messages(msgs)
        assert ctxs == ["tool result text"]

    def test_internal_tools_excluded(self):
        msgs = [
            {
                "type": "ai",
                "content": "",
                "tool_calls": [
                    {"name": "write_todos", "args": {}},
                    {"name": "calculate_bmi", "args": {"height_cm": 175}},
                ],
            }
        ]
        _, tcs, _ = er._extract_from_messages(msgs)
        assert len(tcs) == 1
        assert tcs[0]["tool_name"] == "calculate_bmi"

    def test_skips_empty_ai_content(self):
        msgs = [
            {"type": "ai", "content": ""},
            {"type": "ai", "content": "final"},
        ]
        resp, _, _ = er._extract_from_messages(msgs)
        assert resp == "final"

    def test_empty_messages(self):
        assert er._extract_from_messages([]) == ("", [], [])

    def test_tool_empty_content_skipped(self):
        msgs = [{"type": "tool", "content": "   "}]
        _, _, ctxs = er._extract_from_messages(msgs)
        assert ctxs == []


# ── _extract_tool_calls_from_messages ─────────────────────────────────────────


class TestExtractToolCallsFromMessages:
    def test_tool_calls_list(self):
        msgs = [
            {
                "type": "ai",
                "tool_calls": [
                    {"name": "calculate_bmi", "args": {"height_cm": 175}},
                ],
            }
        ]
        result = er._extract_tool_calls_from_messages(msgs)
        assert len(result) == 1
        assert result[0]["tool_name"] == "calculate_bmi"

    def test_internal_tool_excluded(self):
        msgs = [
            {
                "type": "ai",
                "tool_calls": [
                    {"name": "write_todos", "args": {}},
                ],
            }
        ]
        assert er._extract_tool_calls_from_messages(msgs) == []

    def test_gemini_function_call_fallback(self):
        msgs = [
            {
                "type": "ai",
                "tool_calls": [],
                "additional_kwargs": {
                    "function_call": {
                        "name": "send_email",
                        "arguments": '{"to": "a@b.com"}',
                    },
                },
            }
        ]
        result = er._extract_tool_calls_from_messages(msgs)
        assert result[0]["tool_name"] == "send_email"
        assert result[0]["arguments"]["to"] == "a@b.com"

    def test_gemini_invalid_json_args_safe(self):
        msgs = [
            {
                "type": "ai",
                "tool_calls": [],
                "additional_kwargs": {
                    "function_call": {"name": "send_email", "arguments": "not-json"},
                },
            }
        ]
        result = er._extract_tool_calls_from_messages(msgs)
        assert result[0]["tool_name"] == "send_email"
        assert result[0]["arguments"] == {}

    def test_non_ai_messages_skipped(self):
        msgs = [{"type": "tool", "content": "result"}]
        assert er._extract_tool_calls_from_messages(msgs) == []


# ── _compute_config_hash ──────────────────────────────────────────────────────


class TestComputeConfigHash:
    def test_returns_16_char_hex(self, tmp_path):
        (tmp_path / "system.yaml").write_text("model: gemini")
        with patch.dict("os.environ", {"AGENT_CONFIG_DIR": str(tmp_path)}):
            result = er._compute_config_hash()
        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)

    def test_changes_with_content(self, tmp_path):
        f = tmp_path / "system.yaml"
        f.write_text("model: a")
        with patch.dict("os.environ", {"AGENT_CONFIG_DIR": str(tmp_path)}):
            h1 = er._compute_config_hash()
        f.write_text("model: b")
        with patch.dict("os.environ", {"AGENT_CONFIG_DIR": str(tmp_path)}):
            h2 = er._compute_config_hash()
        assert h1 != h2

    def test_excludes_evals_dir(self, tmp_path):
        (tmp_path / "system.yaml").write_text("model: a")
        with patch.dict("os.environ", {"AGENT_CONFIG_DIR": str(tmp_path)}):
            h1 = er._compute_config_hash()
        evals = tmp_path / "evals"
        evals.mkdir()
        (evals / "eval_cases.yaml").write_text("extra content")
        with patch.dict("os.environ", {"AGENT_CONFIG_DIR": str(tmp_path)}):
            h2 = er._compute_config_hash()
        assert h1 == h2

    def test_missing_dir_returns_hash(self):
        with patch.dict("os.environ", {"AGENT_CONFIG_DIR": "/nonexistent/path"}):
            result = er._compute_config_hash()
        assert len(result) == 16

    def test_ignores_non_config_extensions(self, tmp_path):
        (tmp_path / "system.yaml").write_text("data")
        with patch.dict("os.environ", {"AGENT_CONFIG_DIR": str(tmp_path)}):
            h1 = er._compute_config_hash()
        (tmp_path / "notes.txt").write_text("ignored")
        with patch.dict("os.environ", {"AGENT_CONFIG_DIR": str(tmp_path)}):
            h2 = er._compute_config_hash()
        assert h1 == h2


# ── _get_config_hash ──────────────────────────────────────────────────────────


class TestGetConfigHash:
    def test_prefers_env_var(self):
        with patch.dict("os.environ", {"AGENT_CONFIG_HASH": "abc123"}):
            assert er._get_config_hash() == "abc123"

    def test_computes_when_env_missing(self, tmp_path):
        with patch.dict(
            "os.environ", {"AGENT_CONFIG_HASH": "", "AGENT_CONFIG_DIR": str(tmp_path)}
        ):
            result = er._get_config_hash()
        assert len(result) == 16


# ── _require_eval_files ───────────────────────────────────────────────────────


class TestRequireEvalCases:
    async def test_raises_when_missing(self, tmp_path):
        from fastapi import HTTPException

        with patch(
            "deep_agent.aegra.eval_routes._has_postgres_dataset",
            AsyncMock(return_value=False),
        ):
            with patch.dict("os.environ", {"CONFIG_PATH": str(tmp_path)}):
                with pytest.raises(HTTPException) as exc_info:
                    await er._require_eval_files()
        assert exc_info.value.status_code == 400

    async def test_passes_when_present(self, tmp_path):
        evals_dir = tmp_path / "evals" / "lightspeed"
        evals_dir.mkdir(parents=True)
        (evals_dir / "eval_cases.yaml").write_text("cases: []")
        (evals_dir / "system.yaml").write_text("llm: {}")
        with patch(
            "deep_agent.aegra.eval_routes._has_postgres_dataset",
            AsyncMock(return_value=False),
        ):
            with patch.dict("os.environ", {"CONFIG_PATH": str(tmp_path)}):
                await er._require_eval_files()  # should not raise


# ── _ensure_evals_table ───────────────────────────────────────────────────────


class TestEnsureEvalsTable:
    async def test_executes_all_ddl_statements(self):
        conn, _ = _make_conn()
        with patch(
            "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
        ):
            await er._ensure_evals_table()
        # +1 for the SET statement_timeout call prepended before DDL statements
        assert conn.execute.call_count == len(er._EVALS_DDL_STATEMENTS) + 1
        conn.close.assert_called_once()

    async def test_closes_on_ddl_failure(self):
        conn, _ = _make_conn()
        conn.execute.side_effect = [Exception("DDL failed")]
        with patch(
            "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
        ):
            with pytest.raises(Exception, match="DDL failed"):
                await er._ensure_evals_table()
        conn.close.assert_called_once()

    async def test_unique_violation_is_ignored(self):
        """UniqueViolation (concurrent workers racing on index creation) should
        be caught and treated as success — the object already exists."""
        import psycopg.errors

        conn, _ = _make_conn()
        # statement_timeout call succeeds, first DDL raises UniqueViolation, rest succeed
        unique_err = psycopg.errors.UniqueViolation()
        conn.execute.side_effect = [AsyncMock(), unique_err] + [AsyncMock()] * (
            len(er._EVALS_DDL_STATEMENTS) - 1
        )
        with patch(
            "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
        ):
            # Should not raise
            await er._ensure_evals_table()
        conn.close.assert_called_once()

    async def test_ensure_once_idempotent(self):
        er._table_ensured = False
        conn, _ = _make_conn()
        with patch(
            "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
        ):
            await er._ensure_evals_table_once()
            await er._ensure_evals_table_once()
        # Second call must not hit DB; +1 for SET statement_timeout
        assert conn.execute.call_count == len(er._EVALS_DDL_STATEMENTS) + 1
        er._table_ensured = False  # restore


# ── _atomic_set_in_progress ───────────────────────────────────────────────────


class TestAtomicSetInProgress:
    async def test_returns_existing_when_in_progress(self):
        er._table_ensured = True
        cols = [_col("config_hash"), _col("eval_status")]
        existing_row = ("abc123", "in_progress")
        cursor = _make_cursor(rows=[existing_row], description=cols)
        conn, _ = _make_conn(cursor)

        with patch(
            "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
        ):
            doc, is_new = await er._atomic_set_in_progress("abc123", force=False)

        assert is_new is False
        assert doc["eval_status"] == "in_progress"
        er._table_ensured = False

    async def test_returns_none_false_when_force_and_existing(self):
        er._table_ensured = True
        cols = [_col("eval_status")]
        cursor = _make_cursor(rows=[("in_progress",)], description=cols)
        conn, _ = _make_conn(cursor)

        with patch(
            "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
        ):
            doc, is_new = await er._atomic_set_in_progress("abc123", force=True)

        assert doc is None
        assert is_new is False
        er._table_ensured = False

    async def test_inserts_new_row_when_no_existing(self):
        er._table_ensured = True
        cols = [
            _col("config_hash"),
            _col("eval_status"),
            _col("id"),
        ]
        new_row = ("abc123", "in_progress", 42)

        select_cursor = _make_cursor(rows=[None], description=cols)
        select_cursor.fetchone = AsyncMock(return_value=None)
        insert_cursor = _make_cursor(rows=[new_row], description=cols)

        conn = AsyncMock()
        conn.execute = AsyncMock(side_effect=[select_cursor, insert_cursor])
        conn.__aenter__ = AsyncMock(return_value=conn)
        conn.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
        ):
            doc, is_new = await er._atomic_set_in_progress("abc123", force=False)

        assert is_new is True
        assert doc["id"] == 42  # preserved for threading through eval run
        er._table_ensured = False


# ── eval_status endpoint ──────────────────────────────────────────────────────


class TestEvalStatus:
    async def test_returns_no_dataset_when_none_configured(self, tmp_path):
        with patch(
            "deep_agent.aegra.eval_routes._has_postgres_dataset",
            AsyncMock(return_value=False),
        ):
            with patch.dict("os.environ", {"CONFIG_PATH": str(tmp_path)}):
                result = await er.eval_status()
        assert result["eval_status"] == "no_dataset"
        assert "dataset" in result["message"].lower()

    async def test_returns_not_started_when_no_rows(self):
        er._table_ensured = True
        cursor = _make_cursor(rows=[None])
        cursor.fetchone = AsyncMock(return_value=None)
        conn, _ = _make_conn(cursor)

        with patch(
            "deep_agent.aegra.eval_routes._has_postgres_dataset",
            AsyncMock(return_value=True),
        ):
            with patch(
                "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
            ):
                result = await er.eval_status()

        assert result["eval_status"] == "not_started"
        er._table_ensured = False

    async def test_returns_latest_row(self):
        er._table_ensured = True
        cols = [_col("eval_status"), _col("eval_score"), _col("id")]
        row = ("completed", 0.95, 7)
        cursor = _make_cursor(rows=[row], description=cols)
        conn, _ = _make_conn(cursor)
        # first execute is UPDATE (stale cleanup), second is SELECT
        conn.execute = AsyncMock(side_effect=[cursor, cursor])

        with patch(
            "deep_agent.aegra.eval_routes._has_postgres_dataset",
            AsyncMock(return_value=True),
        ):
            with patch(
                "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
            ):
                result = await er.eval_status()

        assert result["eval_status"] == "completed"
        assert result["eval_score"] == 0.95
        assert "id" not in result
        er._table_ensured = False


# ── eval_results endpoint ─────────────────────────────────────────────────────


class TestEvalResults:
    async def test_404_when_no_results(self):
        from fastapi import HTTPException

        er._table_ensured = True
        cursor = _make_cursor(rows=[None])
        cursor.fetchone = AsyncMock(return_value=None)
        conn, _ = _make_conn(cursor)

        mock_request = MagicMock()
        mock_request.query_params = {}

        with patch(
            "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
        ):
            with pytest.raises(HTTPException) as exc:
                await er.eval_results(mock_request)
        assert exc.value.status_code == 404
        er._table_ensured = False

    async def test_returns_completed_result(self):
        er._table_ensured = True
        cols = [_col("eval_status"), _col("eval_score"), _col("id")]
        row = ("completed", 0.88, 3)
        cursor = _make_cursor(rows=[row], description=cols)
        conn, _ = _make_conn(cursor)

        mock_request = MagicMock()
        mock_request.query_params = {}

        with patch(
            "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
        ):
            result = await er.eval_results(mock_request)

        assert result["eval_status"] == "completed"
        assert "id" not in result
        er._table_ensured = False

    async def test_filters_by_completed_at_param(self):
        er._table_ensured = True
        cols = [_col("eval_status"), _col("id")]
        cursor = _make_cursor(rows=[("completed", 1)], description=cols)
        conn, _ = _make_conn(cursor)

        mock_request = MagicMock()
        mock_request.query_params = {"completed_at": "2025-01-01T00:00:00"}

        with patch(
            "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
        ):
            result = await er.eval_results(mock_request)

        assert result["eval_status"] == "completed"
        er._table_ensured = False


# ── eval_history endpoint ─────────────────────────────────────────────────────


class TestEvalHistory:
    async def test_returns_runs_and_total(self):
        er._table_ensured = True
        now = datetime(2025, 1, 1, tzinfo=UTC)
        cols = [
            _col("eval_score"),
            _col("pass"),
            _col("fail"),
            _col("error"),
            _col("config_hash"),
            _col("created_at"),
            _col("completed_at"),
            _col("total_count"),
        ]
        rows = [
            (0.9, 9, 1, 0, "abc", now, now, 3),
            (0.8, 8, 2, 0, "def", now, now, 3),
        ]
        cursor = _make_cursor(rows=rows, description=cols)
        conn, _ = _make_conn(cursor)

        mock_request = MagicMock()
        mock_request.query_params = {"limit": "10"}

        with patch(
            "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
        ):
            result = await er.eval_history(mock_request)

        assert result["total"] == 3
        assert len(result["runs"]) == 2
        assert "total_count" not in result["runs"][0]
        assert isinstance(result["runs"][0]["created_at"], str)
        er._table_ensured = False

    async def test_limit_capped_at_100(self):
        er._table_ensured = True
        cursor = _make_cursor(rows=[], description=[])
        conn, _ = _make_conn(cursor)

        mock_request = MagicMock()
        mock_request.query_params = {"limit": "9999"}

        with patch(
            "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
        ):
            await er.eval_history(mock_request)

        # Query should have been called with limit=100, not 9999
        call_args = conn.execute.call_args[0][1]
        assert call_args[-1] == 100
        er._table_ensured = False


# ── eval_trends endpoint ──────────────────────────────────────────────────────


class TestEvalTrends:
    async def test_aggregates_by_metric(self):
        er._table_ensured = True
        now = datetime(2025, 1, 1, tzinfo=UTC)
        by_metric = {"custom:tool_eval": {"pass_rate": 0.9}}
        rows = [(by_metric, 0.9, now)]
        cursor = _make_cursor(rows=rows)
        conn, _ = _make_conn(cursor)

        mock_request = MagicMock()
        mock_request.query_params = {}

        with patch(
            "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
        ):
            result = await er.eval_trends(mock_request)

        assert "custom:tool_eval" in result["metrics"]
        assert result["metrics"]["custom:tool_eval"][0]["pass_rate"] == 0.9
        assert len(result["overall"]) == 1
        er._table_ensured = False

    async def test_handles_null_by_metric(self):
        er._table_ensured = True
        now = datetime(2025, 1, 1, tzinfo=UTC)
        rows = [(None, 0.5, now)]
        cursor = _make_cursor(rows=rows)
        conn, _ = _make_conn(cursor)

        mock_request = MagicMock()
        mock_request.query_params = {}

        with patch(
            "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
        ):
            result = await er.eval_trends(mock_request)

        assert result["metrics"] == {}
        assert len(result["overall"]) == 1
        er._table_ensured = False


# ── _fire_eval_run ────────────────────────────────────────────────────────────


class TestFireEvalRun:
    async def test_noop_when_no_url(self, caplog):
        with patch.dict("os.environ", {"EVAL_RUNNER_URL": ""}):
            # reload module-level constant
            with patch.object(er, "_EVAL_RUNNER_URL", ""):
                await er._fire_eval_run("hash123")
        assert "EVAL_RUNNER_URL not set" in caplog.text

    async def test_posts_to_eval_runner(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 202
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch.object(er, "_EVAL_RUNNER_URL", "http://eval:8099"):
            with patch(
                "deep_agent.aegra.eval_routes.httpx.AsyncClient",
                return_value=mock_client,
            ):
                await er._fire_eval_run("hash123", auth_token="tok")

        call_kwargs = mock_client.post.call_args
        assert "evals/run" in call_kwargs[0][0]
        assert call_kwargs[1]["json"]["config_hash"] == "hash123"
        assert call_kwargs[1]["headers"]["Authorization"] == "Bearer tok"

    async def test_bearer_prefix_not_doubled(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 202
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch.object(er, "_EVAL_RUNNER_URL", "http://eval:8099"):
            with patch(
                "deep_agent.aegra.eval_routes.httpx.AsyncClient",
                return_value=mock_client,
            ):
                await er._fire_eval_run("hash123", auth_token="Bearer already")

        headers = mock_client.post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer already"

    async def test_logs_warning_on_http_error(self, caplog):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=Exception("connection refused"))

        with patch.object(er, "_EVAL_RUNNER_URL", "http://eval:8099"):
            with patch(
                "deep_agent.aegra.eval_routes.httpx.AsyncClient",
                return_value=mock_client,
            ):
                await er._fire_eval_run("hash123")

        assert "eval_runner_call_failed" in caplog.text


# ── trigger_eval endpoint ─────────────────────────────────────────────────────


class TestTriggerEval:
    async def test_503_when_eval_runner_url_not_set(self, tmp_path):
        from fastapi import HTTPException

        mock_request = MagicMock()
        with patch.object(er, "_EVAL_RUNNER_URL", ""):
            with patch(
                "deep_agent.aegra.eval_routes._require_eval_files",
                new_callable=AsyncMock,
            ):
                with pytest.raises(HTTPException) as exc:
                    await er.trigger_eval(mock_request)
        assert exc.value.status_code == 503
        assert "EVAL_RUNNER_URL" in exc.value.detail

    async def test_400_when_no_eval_cases(self, tmp_path):
        from fastapi import HTTPException

        mock_request = MagicMock()
        with patch.object(er, "_EVAL_RUNNER_URL", "http://eval:8099"):
            with patch.dict("os.environ", {"CONFIG_PATH": str(tmp_path)}):
                with patch(
                    "deep_agent.aegra.eval_routes._has_postgres_dataset",
                    AsyncMock(return_value=False),
                ):
                    with pytest.raises(HTTPException) as exc:
                        await er.trigger_eval(mock_request)
        assert exc.value.status_code == 400

    async def test_returns_cached_when_exists(self, tmp_path):
        evals_dir = tmp_path / "evals" / "lightspeed"
        evals_dir.mkdir(parents=True)
        (evals_dir / "eval_cases.yaml").write_text("cases: []")
        (evals_dir / "system.yaml").write_text("llm: {}")

        er._table_ensured = True
        cols = [_col("eval_status"), _col("eval_score"), _col("id")]
        cached_row = ("completed", 0.9, 5)
        cursor = _make_cursor(rows=[cached_row], description=cols)
        conn, _ = _make_conn(cursor)

        mock_request = MagicMock()
        mock_request.headers = {}

        with patch.object(er, "_EVAL_RUNNER_URL", "http://eval:8099"):
            with patch.dict(
                "os.environ",
                {"CONFIG_PATH": str(tmp_path), "AGENT_CONFIG_HASH": "deadbeef"},
            ):
                with patch(
                    "deep_agent.aegra.eval_routes._pg_conn",
                    AsyncMock(return_value=conn),
                ):
                    result = await er.trigger_eval(mock_request)

        assert result["cached"] is True
        er._table_ensured = False

    async def test_queues_new_run_when_not_cached(self, tmp_path):
        evals_dir = tmp_path / "evals" / "lightspeed"
        evals_dir.mkdir(parents=True)
        (evals_dir / "eval_cases.yaml").write_text("cases: []")
        (evals_dir / "system.yaml").write_text("llm: {}")

        er._table_ensured = True

        # cache check returns nothing, atomic insert returns new doc
        cache_cursor = _make_cursor(rows=[None])
        cache_cursor.fetchone = AsyncMock(return_value=None)

        cols = [_col("eval_status"), _col("id")]
        new_row = ("in_progress", 10)
        insert_cursor = _make_cursor(rows=[new_row], description=cols)

        select_for_existing = _make_cursor(rows=[None])
        select_for_existing.fetchone = AsyncMock(return_value=None)

        conn = AsyncMock()
        conn.__aenter__ = AsyncMock(return_value=conn)
        conn.__aexit__ = AsyncMock(return_value=False)
        conn.execute = AsyncMock(
            side_effect=[
                cache_cursor,  # SELECT completed cached
                select_for_existing,  # SELECT in_progress check
                insert_cursor,  # INSERT new row
            ]
        )

        mock_request = MagicMock()
        mock_request.headers = {"authorization": "Bearer tok"}

        with patch(
            "deep_agent.aegra.eval_routes._require_eval_files", new_callable=AsyncMock
        ):
            with patch.dict(
                "os.environ",
                {"CONFIG_PATH": str(tmp_path), "AGENT_CONFIG_HASH": "deadbeef"},
            ):
                with patch(
                    "deep_agent.aegra.eval_routes._pg_conn",
                    AsyncMock(return_value=conn),
                ):
                    with patch(
                        "deep_agent.aegra.eval_routes._fire_eval_run",
                        new_callable=AsyncMock,
                        return_value=None,
                    ):
                        with patch.object(er, "_EVAL_RUNNER_URL", "http://eval:8099"):
                            result = await er.trigger_eval(mock_request)

        assert result["eval_status"] == "in_progress"
        assert result["queued"] is True
        er._table_ensured = False

    async def test_returns_in_progress_when_already_running(self, tmp_path):
        evals_dir = tmp_path / "evals" / "lightspeed"
        evals_dir.mkdir(parents=True)
        (evals_dir / "eval_cases.yaml").write_text("cases: []")
        (evals_dir / "system.yaml").write_text("llm: {}")

        er._table_ensured = True

        cache_cursor = _make_cursor(rows=[None])
        cache_cursor.fetchone = AsyncMock(return_value=None)

        cols = [_col("eval_status"), _col("id")]
        existing_row = ("in_progress", 9)
        existing_cursor = _make_cursor(rows=[existing_row], description=cols)

        conn = AsyncMock()
        conn.__aenter__ = AsyncMock(return_value=conn)
        conn.__aexit__ = AsyncMock(return_value=False)
        conn.execute = AsyncMock(side_effect=[cache_cursor, existing_cursor])

        mock_request = MagicMock()
        mock_request.headers = {}

        with patch.object(er, "_EVAL_RUNNER_URL", "http://eval:8099"):
            with patch.dict(
                "os.environ",
                {"CONFIG_PATH": str(tmp_path), "AGENT_CONFIG_HASH": "deadbeef"},
            ):
                with patch(
                    "deep_agent.aegra.eval_routes._pg_conn",
                    AsyncMock(return_value=conn),
                ):
                    result = await er.trigger_eval(mock_request)

        assert result["eval_status"] == "in_progress"
        assert "queued" not in result
        er._table_ensured = False


# ── force_trigger_eval endpoint ───────────────────────────────────────────────


class TestForceTriggerEval:
    async def test_503_when_eval_runner_url_not_set(self, tmp_path):
        from fastapi import HTTPException

        mock_request = MagicMock()
        with patch.object(er, "_EVAL_RUNNER_URL", ""):
            with pytest.raises(HTTPException) as exc:
                await er.force_trigger_eval(mock_request)
        assert exc.value.status_code == 503
        assert "EVAL_RUNNER_URL" in exc.value.detail

    async def test_400_when_no_eval_cases(self, tmp_path):
        from fastapi import HTTPException

        mock_request = MagicMock()
        with patch.object(er, "_EVAL_RUNNER_URL", "http://eval:8099"):
            with patch.dict("os.environ", {"CONFIG_PATH": str(tmp_path)}):
                with patch(
                    "deep_agent.aegra.eval_routes._has_postgres_dataset",
                    AsyncMock(return_value=False),
                ):
                    with pytest.raises(HTTPException) as exc:
                        await er.force_trigger_eval(mock_request)
        assert exc.value.status_code == 400

    async def test_force_queues_new_run(self, tmp_path):
        evals_dir = tmp_path / "evals" / "lightspeed"
        evals_dir.mkdir(parents=True)
        (evals_dir / "eval_cases.yaml").write_text("cases: []")
        (evals_dir / "system.yaml").write_text("llm: {}")

        er._table_ensured = True

        cols = [_col("eval_status"), _col("id")]
        new_row = ("in_progress", 11)
        select_cursor = _make_cursor(rows=[None])
        select_cursor.fetchone = AsyncMock(return_value=None)
        insert_cursor = _make_cursor(rows=[new_row], description=cols)

        conn = AsyncMock()
        conn.__aenter__ = AsyncMock(return_value=conn)
        conn.__aexit__ = AsyncMock(return_value=False)
        conn.execute = AsyncMock(side_effect=[select_cursor, insert_cursor])

        mock_request = MagicMock()
        mock_request.headers = {}

        with patch(
            "deep_agent.aegra.eval_routes._require_eval_files", new_callable=AsyncMock
        ):
            with patch.dict(
                "os.environ",
                {"CONFIG_PATH": str(tmp_path), "AGENT_CONFIG_HASH": "deadbeef"},
            ):
                with patch(
                    "deep_agent.aegra.eval_routes._pg_conn",
                    AsyncMock(return_value=conn),
                ):
                    with patch(
                        "deep_agent.aegra.eval_routes._fire_eval_run",
                        new_callable=AsyncMock,
                        return_value=None,
                    ):
                        with patch.object(er, "_EVAL_RUNNER_URL", "http://eval:8099"):
                            result = await er.force_trigger_eval(mock_request)

        assert result["eval_status"] == "in_progress"
        assert result["forced"] is True
        er._table_ensured = False


# ── _pg_conn ──────────────────────────────────────────────────────────────────


class TestPgConn:
    async def test_returns_connection(self):
        mock_conn = AsyncMock()
        mock_settings = MagicMock()
        mock_settings.database_uri = "postgresql://localhost/test"

        mock_psycopg = MagicMock()
        mock_psycopg.AsyncConnection.connect = AsyncMock(return_value=mock_conn)

        with (
            patch.dict("sys.modules", {"psycopg": mock_psycopg}),
            patch("deep_agent.src.settings.settings", mock_settings),
        ):
            conn = await er._pg_conn()

        mock_psycopg.AsyncConnection.connect.assert_awaited_once_with(
            "postgresql://localhost/test"
        )
        assert conn is mock_conn


# ── get_thread_tool_calls endpoint ────────────────────────────────────────────


class TestGetThreadToolCalls:
    async def test_returns_tool_calls(self):
        expected = [{"tool_name": "calculate_bmi", "arguments": {"height_cm": 175}}]
        mock_request = MagicMock()
        with (
            patch(
                "deep_agent.aegra.eval_routes._verify_thread_ownership",
                new_callable=AsyncMock,
            ),
            patch(
                "deep_agent.aegra.eval_routes._collect_subagent_tool_calls_from_postgres",
                new=AsyncMock(return_value=expected),
            ),
        ):
            result = await er.get_thread_tool_calls("thread-123", mock_request)
        assert result["thread_id"] == "thread-123"
        assert result["tool_calls"] == expected

    async def test_returns_empty_on_no_calls(self):
        mock_request = MagicMock()
        with (
            patch(
                "deep_agent.aegra.eval_routes._verify_thread_ownership",
                new_callable=AsyncMock,
            ),
            patch(
                "deep_agent.aegra.eval_routes._collect_subagent_tool_calls_from_postgres",
                new=AsyncMock(return_value=[]),
            ),
        ):
            result = await er.get_thread_tool_calls("thread-abc", mock_request)
        assert result["tool_calls"] == []


class TestVerifyThreadOwnership:
    async def test_raises_403_when_no_sub(self):
        from fastapi import HTTPException

        mock_request = MagicMock()
        with patch("deep_agent.aegra.eval_routes._extract_sub", return_value=None):
            with pytest.raises(HTTPException) as exc:
                await er._verify_thread_ownership("thread-1", mock_request)
        assert exc.value.status_code == 403

    async def test_raises_403_when_not_owned(self):
        from fastapi import HTTPException

        mock_request = MagicMock()
        cursor = _make_cursor(rows=[None])
        cursor.fetchone = AsyncMock(return_value=None)
        conn, _ = _make_conn(cursor)

        with (
            patch("deep_agent.aegra.eval_routes._extract_sub", return_value="user-a"),
            patch(
                "deep_agent.aegra.eval_routes._pg_conn",
                AsyncMock(return_value=conn),
            ),
        ):
            with pytest.raises(HTTPException) as exc:
                await er._verify_thread_ownership("thread-1", mock_request)
        assert exc.value.status_code == 403

    async def test_passes_when_owned(self):
        mock_request = MagicMock()
        cursor = _make_cursor(rows=[(1,)])
        conn, _ = _make_conn(cursor)

        with (
            patch("deep_agent.aegra.eval_routes._extract_sub", return_value="user-a"),
            patch(
                "deep_agent.aegra.eval_routes._pg_conn",
                AsyncMock(return_value=conn),
            ),
        ):
            await er._verify_thread_ownership("thread-1", mock_request)

    async def test_raises_403_on_db_error(self):
        from fastapi import HTTPException

        mock_request = MagicMock()
        with (
            patch("deep_agent.aegra.eval_routes._extract_sub", return_value="user-a"),
            patch(
                "deep_agent.aegra.eval_routes._pg_conn",
                AsyncMock(side_effect=Exception("db down")),
            ),
        ):
            with pytest.raises(HTTPException) as exc:
                await er._verify_thread_ownership("thread-1", mock_request)
        assert exc.value.status_code == 403


# ── _collect_subagent_tool_calls_via_remote_graph ─────────────────────────────


class TestCollectSubagentViaRemoteGraph:
    async def test_returns_tool_calls_from_subgraph(self):
        mock_snapshot = MagicMock()
        mock_task = MagicMock()
        mock_substate = MagicMock()
        mock_substate.values = {
            "messages": [
                {
                    "type": "ai",
                    "tool_calls": [{"name": "calculate_bmi", "args": {"h": 175}}],
                },
            ]
        }
        mock_substate.tasks = []
        mock_task.state = mock_substate
        mock_snapshot.tasks = [mock_task]

        mock_graph = AsyncMock()
        mock_graph.aget_state = AsyncMock(return_value=mock_snapshot)

        with patch("langgraph.pregel.remote.RemoteGraph", return_value=mock_graph):
            result = await er._collect_subagent_tool_calls_via_remote_graph("thread-x")

        assert any(tc["tool_name"] == "calculate_bmi" for tc in result)

    async def test_falls_back_to_postgres_on_error(self):
        with patch(
            "langgraph.pregel.remote.RemoteGraph",
            side_effect=Exception("SDK unavailable"),
        ):
            with patch(
                "deep_agent.aegra.eval_routes._collect_subagent_tool_calls_from_postgres",
                new=AsyncMock(
                    return_value=[{"tool_name": "fallback_tool", "arguments": {}}]
                ),
            ):
                result = await er._collect_subagent_tool_calls_via_remote_graph(
                    "thread-y"
                )

        assert result[0]["tool_name"] == "fallback_tool"


# ── eval_run endpoint ─────────────────────────────────────────────────────────


def _make_http_client(run_state: dict, interrupt_state: dict | None = None):
    """Build a mock httpx.AsyncClient for eval_run tests."""
    thread_resp = MagicMock()
    thread_resp.raise_for_status = MagicMock()
    thread_resp.json = MagicMock(return_value={"thread_id": "t-abc"})

    run_resp = MagicMock()
    run_resp.raise_for_status = MagicMock()
    run_resp.json = MagicMock(return_value=run_state)

    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.post = AsyncMock(side_effect=[thread_resp, run_resp])
    return client


class TestEvalRun:
    async def test_basic_run_no_interrupt(self):
        run_state = {
            "messages": [
                {"type": "ai", "content": "Your BMI is 22.9", "tool_calls": []},
            ]
        }
        client = _make_http_client(run_state)

        with patch(
            "deep_agent.aegra.eval_routes.httpx.AsyncClient", return_value=client
        ):
            with patch(
                "deep_agent.aegra.eval_routes._collect_subagent_tool_calls_via_remote_graph",
                new=AsyncMock(return_value=[]),
            ):
                result = await er.eval_run(er.EvalRunRequest(query="What is my BMI?"))

        assert result.response == "Your BMI is 22.9"
        assert result.was_interrupted is False
        assert result.pre_approval_response is None
        assert result.tool_calls == []

    async def test_run_with_tool_calls(self):
        run_state = {
            "messages": [
                {
                    "type": "ai",
                    "content": "Calculating...",
                    "tool_calls": [
                        {"name": "calculate_bmi", "args": {"height_cm": 175}}
                    ],
                },
                {"type": "tool", "content": "22.9 Normal"},
                {"type": "ai", "content": "Your BMI is 22.9"},
            ]
        }
        client = _make_http_client(run_state)

        with patch(
            "deep_agent.aegra.eval_routes.httpx.AsyncClient", return_value=client
        ):
            with patch(
                "deep_agent.aegra.eval_routes._collect_subagent_tool_calls_via_remote_graph",
                new=AsyncMock(return_value=[]),
            ):
                result = await er.eval_run(er.EvalRunRequest(query="BMI?"))

        assert result.response == "Your BMI is 22.9"
        assert any(tc["tool_name"] == "calculate_bmi" for tc in result.tool_calls)
        assert "22.9 Normal" in result.contexts

    async def test_run_with_hitl_auto_approve(self):
        pre_approval_state = {
            "__interrupt__": [{"value": {}}],
            "messages": [{"type": "ai", "content": "Awaiting your approval"}],
        }
        post_approval_state = {
            "messages": [{"type": "ai", "content": "Email sent successfully"}],
        }

        thread_resp = MagicMock()
        thread_resp.raise_for_status = MagicMock()
        thread_resp.json = MagicMock(return_value={"thread_id": "t-hitl"})

        pre_resp = MagicMock()
        pre_resp.raise_for_status = MagicMock()
        pre_resp.json = MagicMock(return_value=pre_approval_state)

        post_resp = MagicMock()
        post_resp.raise_for_status = MagicMock()
        post_resp.json = MagicMock(return_value=post_approval_state)

        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.post = AsyncMock(side_effect=[thread_resp, pre_resp, post_resp])

        with patch(
            "deep_agent.aegra.eval_routes.httpx.AsyncClient", return_value=client
        ):
            with patch(
                "deep_agent.aegra.eval_routes._collect_subagent_tool_calls_via_remote_graph",
                new=AsyncMock(return_value=[]),
            ):
                result = await er.eval_run(
                    er.EvalRunRequest(query="Send email", auto_approve=True)
                )

        assert result.was_interrupted is True
        assert result.pre_approval_response == "Awaiting your approval"
        assert result.response == "Email sent successfully"

    async def test_run_uses_provided_conversation_id(self):
        run_state = {"messages": [{"type": "ai", "content": "ok", "tool_calls": []}]}
        client = _make_http_client(run_state)

        with patch(
            "deep_agent.aegra.eval_routes.httpx.AsyncClient", return_value=client
        ):
            with patch(
                "deep_agent.aegra.eval_routes._collect_subagent_tool_calls_via_remote_graph",
                new=AsyncMock(return_value=[]),
            ):
                result = await er.eval_run(
                    er.EvalRunRequest(query="q", conversation_id="my-conv-id")
                )

        assert result.conversation_id == "my-conv-id"

    async def test_run_generates_conversation_id_when_absent(self):
        run_state = {"messages": [{"type": "ai", "content": "ok", "tool_calls": []}]}
        client = _make_http_client(run_state)

        with patch(
            "deep_agent.aegra.eval_routes.httpx.AsyncClient", return_value=client
        ):
            with patch(
                "deep_agent.aegra.eval_routes._collect_subagent_tool_calls_via_remote_graph",
                new=AsyncMock(return_value=[]),
            ):
                result = await er.eval_run(er.EvalRunRequest(query="q"))

        assert result.conversation_id  # not empty


# ── Eval token refresh / internal auth ───────────────────────────────────────


class TestWriteEvalRedis:
    def test_noop_when_refresh_disabled(self):
        """_write_eval_redis does nothing when EVAL_TOKEN_REFRESH_ENABLED is false."""
        with patch.object(er, "_EVAL_TOKEN_REFRESH_ENABLED", False):
            # Should return early without touching Redis
            with patch(
                "deep_agent.aegra.redis.cache_set",
                side_effect=AssertionError("should not call"),
            ):
                er._write_eval_redis("user123", "tok")

    def test_noop_when_sub_empty(self):
        """_write_eval_redis does nothing when sub is empty string."""
        with patch.object(er, "_EVAL_TOKEN_REFRESH_ENABLED", True):
            with patch(
                "deep_agent.aegra.redis.cache_set",
                side_effect=AssertionError("should not call"),
            ):
                er._write_eval_redis("", "tok")

    def test_writes_active_and_sub_keys(self):
        """When enabled and sub is set, writes active + trigger_sub keys."""
        with patch.object(er, "_EVAL_TOKEN_REFRESH_ENABLED", True):
            with (
                patch("deep_agent.aegra.redis.cache_set") as mock_set,
                patch("deep_agent.aegra.mcp_crypto.encrypt_secret", return_value=None),
            ):
                er._write_eval_redis("u1", "")
        keys = [call.args[0] for call in mock_set.call_args_list]
        assert any("eval:active:u1" in k for k in keys)
        assert any("eval:trigger_sub" in k for k in keys)

    def test_encrypts_refresh_token_when_present(self):
        """Refresh token is encrypted and stored when non-empty."""
        with patch.object(er, "_EVAL_TOKEN_REFRESH_ENABLED", True):
            with (
                patch("deep_agent.aegra.redis.cache_set") as mock_set,
                patch(
                    "deep_agent.aegra.mcp_crypto.encrypt_secret", return_value="enc-tok"
                ) as mock_enc,
            ):
                er._write_eval_redis("u1", "my-refresh")
        mock_enc.assert_called_once_with("my-refresh")
        keys = [call.args[0] for call in mock_set.call_args_list]
        assert any("eval:refresh:u1" in k for k in keys)


class TestInternalCleanupEndpoint:
    async def test_returns_disabled_when_refresh_off(self):
        """Returns {status: disabled} immediately when token refresh is not enabled."""
        mock_request = MagicMock()
        with patch.object(er, "_EVAL_TOKEN_REFRESH_ENABLED", False):
            result = await er.cleanup_eval_redis(mock_request)
        assert result == {"status": "disabled"}

    async def test_401_when_token_missing(self):
        """Returns 401 when X-Internal-Token header is absent."""
        from fastapi import HTTPException

        mock_request = MagicMock()
        mock_request.headers = {}
        with (
            patch.object(er, "_EVAL_TOKEN_REFRESH_ENABLED", True),
            patch.object(er, "_EVAL_INTERNAL_TOKEN", "secret"),
        ):
            with pytest.raises(HTTPException) as exc:
                await er.cleanup_eval_redis(mock_request)
        assert exc.value.status_code == 401

    async def test_401_when_token_wrong(self):
        """Returns 401 when X-Internal-Token does not match."""
        from fastapi import HTTPException

        mock_request = MagicMock()
        mock_request.headers = {"x-internal-token": "wrong"}
        with (
            patch.object(er, "_EVAL_TOKEN_REFRESH_ENABLED", True),
            patch.object(er, "_EVAL_INTERNAL_TOKEN", "correct"),
        ):
            with pytest.raises(HTTPException) as exc:
                await er.cleanup_eval_redis(mock_request)
        assert exc.value.status_code == 401

    async def test_200_with_correct_token(self):
        """Returns 200 when correct token is provided."""
        mock_request = MagicMock()
        mock_request.headers = {"x-internal-token": "secret"}
        mock_request.json = AsyncMock(return_value={})
        with (
            patch.object(er, "_EVAL_TOKEN_REFRESH_ENABLED", True),
            patch.object(er, "_EVAL_INTERNAL_TOKEN", "secret"),
            patch("deep_agent.aegra.redis.cache_get", return_value=None),
            patch("deep_agent.aegra.redis.cache_delete"),
        ):
            result = await er.cleanup_eval_redis(mock_request)
        assert "status" in result


class TestRequireBearer:
    async def test_returns_token_when_valid(self):
        creds = MagicMock()
        creds.credentials = "my-token"
        result = er._require_bearer(creds)
        assert result == "my-token"

    async def test_raises_401_when_none(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc:
            er._require_bearer(None)
        assert exc.value.status_code == 401

    async def test_raises_401_when_empty_credentials(self):
        from fastapi import HTTPException

        creds = MagicMock()
        creds.credentials = ""
        with pytest.raises(HTTPException) as exc:
            er._require_bearer(creds)
        assert exc.value.status_code == 401


class TestCheckMcpAuth:
    async def test_empty_when_no_user_identity(self):
        mock_request = MagicMock()
        mock_request.headers = {}
        with patch("deep_agent.aegra.eval_routes._extract_sub", return_value=None):
            result = await er._check_mcp_auth(mock_request)
        assert result == []

    async def test_empty_when_no_dcr_servers(self):
        mock_request = MagicMock()
        mock_request.headers = {"authorization": "Bearer tok"}
        mock_config = MagicMock()
        mock_config.get_mcp_servers.return_value = {
            "s1": {"enabled": True, "auth": False, "auth_mode": "sso"},
        }
        with (
            patch("deep_agent.aegra.eval_routes._extract_sub", return_value="user1"),
            patch(
                "deep_agent.src.agent.config.agent_config",
                mock_config,
            ),
            patch(
                "deep_agent.src.settings.settings",
                MagicMock(
                    database_uri="postgresql://x",
                    agent_deployment_id="agent1",
                ),
            ),
        ):
            result = await er._check_mcp_auth(mock_request)
        assert result == []

    async def test_returns_missing_when_token_none(self):
        mock_request = MagicMock()
        mock_request.headers = {"authorization": "Bearer tok"}
        mock_store = AsyncMock()
        mock_store.get_token = AsyncMock(return_value=None)
        mock_config = MagicMock()
        mock_config.get_mcp_servers.return_value = {
            "mcp1": {
                "enabled": True,
                "auth": True,
                "auth_mode": "dcr",
                "url": "http://mcp1",
            },
        }
        with (
            patch("deep_agent.aegra.eval_routes._extract_sub", return_value="user1"),
            patch(
                "deep_agent.src.agent.config.agent_config",
                mock_config,
            ),
            patch(
                "deep_agent.aegra.mcp_token_store.McpTokenStore",
                return_value=mock_store,
            ),
            patch(
                "deep_agent.src.settings.settings",
                MagicMock(
                    database_uri="postgresql://x",
                    agent_deployment_id="agent1",
                ),
            ),
        ):
            result = await er._check_mcp_auth(mock_request)
        assert len(result) == 1
        assert result[0]["name"] == "mcp1"

    async def test_returns_missing_when_token_expiring_soon(self):
        from datetime import datetime, timedelta, timezone

        mock_request = MagicMock()
        mock_request.headers = {"authorization": "Bearer tok"}
        expiring_token = MagicMock()
        expiring_token.expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
        mock_store = AsyncMock()
        mock_store.get_token = AsyncMock(return_value=expiring_token)
        mock_config = MagicMock()
        mock_config.get_mcp_servers.return_value = {
            "mcp1": {
                "enabled": True,
                "auth": True,
                "auth_mode": "oauth",
                "url": "http://mcp1",
            },
        }
        with (
            patch("deep_agent.aegra.eval_routes._extract_sub", return_value="user1"),
            patch(
                "deep_agent.src.agent.config.agent_config",
                mock_config,
            ),
            patch(
                "deep_agent.aegra.mcp_token_store.McpTokenStore",
                return_value=mock_store,
            ),
            patch(
                "deep_agent.src.settings.settings",
                MagicMock(
                    database_uri="postgresql://x",
                    agent_deployment_id="agent1",
                ),
            ),
            patch.object(er, "_EVAL_TOKEN_MIN_TTL_MINUTES", 30),
        ):
            result = await er._check_mcp_auth(mock_request)
        assert len(result) == 1
        assert result[0]["name"] == "mcp1"

    async def test_returns_empty_when_token_valid(self):
        from datetime import datetime, timedelta, timezone

        mock_request = MagicMock()
        mock_request.headers = {"authorization": "Bearer tok"}
        valid_token = MagicMock()
        valid_token.expires_at = datetime.now(timezone.utc) + timedelta(hours=2)
        mock_store = AsyncMock()
        mock_store.get_token = AsyncMock(return_value=valid_token)
        mock_config = MagicMock()
        mock_config.get_mcp_servers.return_value = {
            "mcp1": {
                "enabled": True,
                "auth": True,
                "auth_mode": "dcr",
                "url": "http://mcp1",
            },
        }
        with (
            patch("deep_agent.aegra.eval_routes._extract_sub", return_value="user1"),
            patch(
                "deep_agent.src.agent.config.agent_config",
                mock_config,
            ),
            patch(
                "deep_agent.aegra.mcp_token_store.McpTokenStore",
                return_value=mock_store,
            ),
            patch(
                "deep_agent.src.settings.settings",
                MagicMock(
                    database_uri="postgresql://x",
                    agent_deployment_id="agent1",
                ),
            ),
            patch.object(er, "_EVAL_TOKEN_MIN_TTL_MINUTES", 30),
        ):
            result = await er._check_mcp_auth(mock_request)
        assert result == []


class TestExtractSub:
    def test_returns_sub_from_valid_bearer(self):
        mock_request = MagicMock()
        mock_request.headers = {"authorization": "Bearer some-jwt"}
        with patch(
            "deep_agent.aegra.auth._decode_sub_unverified", return_value="user-abc"
        ):
            result = er._extract_sub(mock_request)
        assert result == "user-abc"

    def test_returns_none_when_no_auth_header(self):
        mock_request = MagicMock()
        mock_request.headers = MagicMock()
        mock_request.headers.get = MagicMock(return_value="")
        result = er._extract_sub(mock_request)
        assert result is None

    def test_returns_none_when_not_bearer(self):
        mock_request = MagicMock()
        mock_request.headers = MagicMock()
        mock_request.headers.get = MagicMock(return_value="Basic abc123")
        result = er._extract_sub(mock_request)
        assert result is None


class TestMarkEvalError:
    async def test_updates_postgres(self):
        now = datetime(2025, 1, 1, tzinfo=UTC)
        conn, _ = _make_conn()
        with patch(
            "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
        ):
            await er._mark_eval_error("hash1", "some error", now)
        conn.execute.assert_called_once()

    async def test_swallows_exception(self, caplog):
        now = datetime(2025, 1, 1, tzinfo=UTC)
        with patch(
            "deep_agent.aegra.eval_routes._pg_conn",
            AsyncMock(side_effect=Exception("pg down")),
        ):
            await er._mark_eval_error("hash1", "err", now)
        assert "Could not mark eval as error" in caplog.text


class TestQueueEvalRun:
    async def test_success_path(self):
        mock_request = MagicMock()
        mock_request.headers = {"authorization": "Bearer tok", "x-refresh-token": "rt"}
        record = {"created_at": datetime(2025, 1, 1, tzinfo=UTC)}
        with (
            patch(
                "deep_agent.aegra.eval_routes._fire_eval_run",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("deep_agent.aegra.eval_routes._extract_sub", return_value="user1"),
            patch("deep_agent.aegra.eval_routes._write_eval_redis"),
        ):
            result = await er._queue_eval_run("hash1", mock_request, record)
        assert result["eval_status"] == "in_progress"
        assert result["queued"] is True
        assert "forced" not in result

    async def test_success_forced(self):
        mock_request = MagicMock()
        mock_request.headers = {"authorization": "", "x-refresh-token": ""}
        record = {"created_at": datetime(2025, 1, 1, tzinfo=UTC)}
        with (
            patch(
                "deep_agent.aegra.eval_routes._fire_eval_run",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("deep_agent.aegra.eval_routes._extract_sub", return_value=None),
            patch("deep_agent.aegra.eval_routes._write_eval_redis"),
        ):
            result = await er._queue_eval_run(
                "hash1", mock_request, record, forced=True
            )
        assert result["forced"] is True

    async def test_error_rolls_back_and_raises(self):
        from fastapi import HTTPException

        mock_request = MagicMock()
        mock_request.headers = {"authorization": "Bearer tok", "x-refresh-token": ""}
        record = {"created_at": datetime(2025, 1, 1, tzinfo=UTC)}
        with (
            patch(
                "deep_agent.aegra.eval_routes._fire_eval_run",
                new_callable=AsyncMock,
                return_value="runner unreachable",
            ),
            patch("deep_agent.aegra.eval_routes._extract_sub", return_value="u1"),
            patch("deep_agent.aegra.eval_routes._write_eval_redis"),
            patch(
                "deep_agent.aegra.eval_routes._mark_eval_error",
                new_callable=AsyncMock,
            ) as mock_mark,
        ):
            with pytest.raises(HTTPException) as exc:
                await er._queue_eval_run("hash1", mock_request, record)
        assert exc.value.status_code == 503
        mock_mark.assert_called_once()


class TestCollectAgentModels:
    def test_reads_prompt_md_model(self, tmp_path):
        (tmp_path / "PROMPT.md").write_text("---\nmodel: gpt-4\n---\nContent")
        with patch.dict("os.environ", {"CONFIG_PATH": str(tmp_path)}):
            result = er._collect_agent_models()
        assert any(m["model"] == "gpt-4" for m in result)
        assert any(m["default"] is True for m in result)

    def test_reads_subagent_models(self, tmp_path):
        (tmp_path / "PROMPT.md").write_text("---\nmodel: gpt-4\n---\nContent")
        sub_dir = tmp_path / "subagents"
        sub_dir.mkdir()
        (sub_dir / "analyst.md").write_text("---\nmodel: claude-3\n---\nContent")
        with patch.dict("os.environ", {"CONFIG_PATH": str(tmp_path)}):
            result = er._collect_agent_models()
        assert len(result) == 2
        assert any(m["model"] == "claude-3" for m in result)

    def test_deduplicates_models(self, tmp_path):
        (tmp_path / "PROMPT.md").write_text("---\nmodel: gpt-4\n---\nContent")
        sub_dir = tmp_path / "subagents"
        sub_dir.mkdir()
        (sub_dir / "a.md").write_text("---\nmodel: gpt-4\n---\nContent")
        with patch.dict("os.environ", {"CONFIG_PATH": str(tmp_path)}):
            result = er._collect_agent_models()
        assert len(result) == 1

    def test_handles_missing_files(self, tmp_path):
        with patch.dict("os.environ", {"CONFIG_PATH": str(tmp_path)}):
            result = er._collect_agent_models()
        assert result == []


class TestGetEvalModels:
    async def test_returns_models(self):
        with patch(
            "deep_agent.aegra.eval_routes._collect_agent_models",
            return_value=[
                {"model": "gpt-4", "source": "orchestrator", "default": True}
            ],
        ):
            result = await er.get_eval_models()
        assert "models" in result
        assert len(result["models"]) == 1


class TestUpsertDataset:
    async def test_upserts_dataset(self):
        er._datasets_table_ensured = True
        conn, _ = _make_conn()
        with patch(
            "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
        ):
            body = er.DatasetUpsertRequest(
                cases=[{"query": "test", "expected": "ok"}],
                judge_model="gpt-4",
            )
            result = await er.upsert_dataset(body)
        assert result["status"] == "ok"
        assert result["case_count"] == 1
        assert conn.execute.call_count == 2
        er._datasets_table_ensured = False


class TestGetDataset:
    async def test_returns_dataset(self):
        er._datasets_table_ensured = True
        now = datetime(2025, 1, 1, tzinfo=UTC)
        dataset = {"cases": [{"query": "hi"}]}
        cursor = _make_cursor(
            rows=[(dataset, "gpt-4", now)],
        )
        conn, _ = _make_conn(cursor)
        with patch(
            "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
        ):
            result = await er.get_dataset()
        assert result["dataset"] == dataset
        assert result["judge_model"] == "gpt-4"
        er._datasets_table_ensured = False

    async def test_404_when_no_dataset(self):
        from fastapi import HTTPException

        er._datasets_table_ensured = True
        cursor = _make_cursor(rows=[])
        conn, _ = _make_conn(cursor)
        with patch(
            "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
        ):
            with pytest.raises(HTTPException) as exc:
                await er.get_dataset()
        assert exc.value.status_code == 404
        er._datasets_table_ensured = False

    async def test_parses_string_dataset(self):
        er._datasets_table_ensured = True
        now = datetime(2025, 1, 1, tzinfo=UTC)
        dataset_str = '{"cases": [{"query": "hi"}]}'
        cursor = _make_cursor(
            rows=[(dataset_str, None, now)],
        )
        conn, _ = _make_conn(cursor)
        with patch(
            "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
        ):
            result = await er.get_dataset()
        assert result["dataset"]["cases"][0]["query"] == "hi"
        er._datasets_table_ensured = False


class TestHasPostgresDataset:
    async def test_returns_true_when_exists(self):
        er._datasets_table_ensured = True
        cursor = _make_cursor(rows=[(1,)])
        conn, _ = _make_conn(cursor)
        with patch(
            "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
        ):
            result = await er._has_postgres_dataset()
        assert result is True
        er._datasets_table_ensured = False

    async def test_returns_false_when_empty(self):
        er._datasets_table_ensured = True
        cursor = _make_cursor(rows=[])
        conn, _ = _make_conn(cursor)
        with patch(
            "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
        ):
            result = await er._has_postgres_dataset()
        assert result is False
        er._datasets_table_ensured = False

    async def test_returns_false_on_exception(self):
        er._datasets_table_ensured = True
        with patch(
            "deep_agent.aegra.eval_routes._pg_conn",
            AsyncMock(side_effect=Exception("pg down")),
        ):
            result = await er._has_postgres_dataset()
        assert result is False
        er._datasets_table_ensured = False


class TestEnsureDatasetsTableOnce:
    async def test_idempotent(self):
        er._datasets_table_ensured = False
        conn, _ = _make_conn()
        with patch(
            "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
        ):
            await er._ensure_datasets_table_once()
            first_count = conn.execute.call_count
            await er._ensure_datasets_table_once()
            assert conn.execute.call_count == first_count
        er._datasets_table_ensured = False


class TestRunDdlOnce:
    async def test_runs_ddl_and_sets_flag(self):
        er._datasets_table_ensured = False
        conn, _ = _make_conn()
        with patch(
            "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
        ):
            await er._run_ddl_once(
                "CREATE TABLE test (id INT)",
                ["ALTER TABLE test ADD COLUMN name TEXT"],
                "_datasets_table_ensured",
                "test",
            )
        assert er._datasets_table_ensured is True
        er._datasets_table_ensured = False

    async def test_does_not_set_flag_on_failure(self):
        er._datasets_table_ensured = False
        conn, _ = _make_conn()
        conn.execute = AsyncMock(side_effect=Exception("DDL error"))
        with patch(
            "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
        ):
            with pytest.raises(Exception, match="DDL error"):
                await er._run_ddl_once(
                    "CREATE TABLE bad",
                    [],
                    "_datasets_table_ensured",
                    "test",
                )
        assert er._datasets_table_ensured is False

    async def test_skips_when_flag_already_set(self):
        er._datasets_table_ensured = True
        conn, _ = _make_conn()
        with patch(
            "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
        ):
            await er._run_ddl_once(
                "CREATE TABLE test (id INT)",
                [],
                "_datasets_table_ensured",
                "test",
            )
        conn.execute.assert_not_called()
        er._datasets_table_ensured = False

    async def test_migration_failure_logged_not_raised(self):
        er._datasets_table_ensured = False
        conn, _ = _make_conn()
        ok_result = AsyncMock()
        conn.execute = AsyncMock(
            side_effect=[ok_result, ok_result, Exception("migration err")]
        )
        with patch(
            "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
        ):
            await er._run_ddl_once(
                "CREATE TABLE test (id INT)",
                ["ALTER OK", "ALTER FAIL"],
                "_datasets_table_ensured",
                "test",
            )
        assert er._datasets_table_ensured is True
        er._datasets_table_ensured = False


class TestEvalStatusUndefinedTable:
    async def test_returns_no_dataset_on_undefined_table(self):
        import psycopg.errors

        er._table_ensured = True
        conn = AsyncMock()
        conn.__aenter__ = AsyncMock(return_value=conn)
        conn.__aexit__ = AsyncMock(return_value=False)
        conn.execute = AsyncMock(side_effect=psycopg.errors.UndefinedTable())

        with (
            patch(
                "deep_agent.aegra.eval_routes._has_postgres_dataset",
                AsyncMock(return_value=True),
            ),
            patch(
                "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
            ),
        ):
            result = await er.eval_status()
        assert result["eval_status"] == "no_dataset"
        assert er._table_ensured is False
        er._table_ensured = False


class TestEvalResultsUndefinedTable:
    async def test_404_on_undefined_table(self):
        import psycopg.errors
        from fastapi import HTTPException

        er._table_ensured = True
        conn = AsyncMock()
        conn.__aenter__ = AsyncMock(return_value=conn)
        conn.__aexit__ = AsyncMock(return_value=False)
        conn.execute = AsyncMock(side_effect=psycopg.errors.UndefinedTable())

        mock_request = MagicMock()
        mock_request.query_params = {}

        with patch(
            "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
        ):
            with pytest.raises(HTTPException) as exc:
                await er.eval_results(mock_request)
        assert exc.value.status_code == 404
        assert er._table_ensured is False
        er._table_ensured = False


class TestEvalHistoryUndefinedTable:
    async def test_returns_empty_on_undefined_table(self):
        import psycopg.errors

        er._table_ensured = True
        conn = AsyncMock()
        conn.__aenter__ = AsyncMock(return_value=conn)
        conn.__aexit__ = AsyncMock(return_value=False)
        conn.execute = AsyncMock(side_effect=psycopg.errors.UndefinedTable())

        mock_request = MagicMock()
        mock_request.query_params = {"limit": "10"}

        with patch(
            "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
        ):
            result = await er.eval_history(mock_request)
        assert result == {"runs": [], "total": 0}
        assert er._table_ensured is False
        er._table_ensured = False


class TestEvalTrendsErrors:
    async def test_invalid_limit_returns_400(self):
        from fastapi import HTTPException

        mock_request = MagicMock()
        mock_request.query_params = {"limit": "not-a-number"}
        with pytest.raises(HTTPException) as exc:
            await er.eval_trends(mock_request)
        assert exc.value.status_code == 400

    async def test_returns_empty_on_undefined_table(self):
        import psycopg.errors

        er._table_ensured = True
        conn = AsyncMock()
        conn.__aenter__ = AsyncMock(return_value=conn)
        conn.__aexit__ = AsyncMock(return_value=False)
        conn.execute = AsyncMock(side_effect=psycopg.errors.UndefinedTable())

        mock_request = MagicMock()
        mock_request.query_params = {"limit": "10"}

        with patch(
            "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
        ):
            result = await er.eval_trends(mock_request)
        assert result == {"metrics": {}, "overall": []}
        assert er._table_ensured is False
        er._table_ensured = False


class TestFireEvalRunErrorStatus:
    async def test_returns_error_on_4xx_status(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch.object(er, "_EVAL_RUNNER_URL", "http://eval:8099"):
            with patch(
                "deep_agent.aegra.eval_routes.httpx.AsyncClient",
                return_value=mock_client,
            ):
                result = await er._fire_eval_run("hash1")
        assert result is not None
        assert "400" in result

    async def test_returns_error_on_5xx_status(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch.object(er, "_EVAL_RUNNER_URL", "http://eval:8099"):
            with patch(
                "deep_agent.aegra.eval_routes.httpx.AsyncClient",
                return_value=mock_client,
            ):
                result = await er._fire_eval_run("hash1")
        assert result is not None
        assert "500" in result

    async def test_sends_internal_token_header(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 202
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with (
            patch.object(er, "_EVAL_RUNNER_URL", "http://eval:8099"),
            patch.object(er, "_EVAL_INTERNAL_TOKEN", "int-tok"),
            patch(
                "deep_agent.aegra.eval_routes.httpx.AsyncClient",
                return_value=mock_client,
            ),
        ):
            await er._fire_eval_run("hash1")
        headers = mock_client.post.call_args[1]["headers"]
        assert headers["X-Internal-Token"] == "int-tok"


class TestTriggerEvalMcpAuth:
    async def test_returns_403_when_mcp_auth_required(self, tmp_path):
        evals_dir = tmp_path / "evals" / "lightspeed"
        evals_dir.mkdir(parents=True)
        (evals_dir / "eval_cases.yaml").write_text("cases: []")

        er._table_ensured = True
        cache_cursor = _make_cursor(rows=[])
        conn, _ = _make_conn(cache_cursor)

        mock_request = MagicMock()
        mock_request.headers = {"authorization": "Bearer tok"}

        with (
            patch.object(er, "_EVAL_RUNNER_URL", "http://eval:8099"),
            patch(
                "deep_agent.aegra.eval_routes._require_eval_files",
                new_callable=AsyncMock,
            ),
            patch.dict(
                "os.environ",
                {"CONFIG_PATH": str(tmp_path), "AGENT_CONFIG_HASH": "deadbeef"},
            ),
            patch(
                "deep_agent.aegra.eval_routes._pg_conn",
                AsyncMock(return_value=conn),
            ),
            patch(
                "deep_agent.aegra.eval_routes._check_mcp_auth",
                new_callable=AsyncMock,
                return_value=[{"name": "mcp1", "connect_url": "/mcp/mcp1/connect"}],
            ),
        ):
            result = await er.trigger_eval(mock_request)

        assert result.status_code == 403
        body = json.loads(result.body)
        assert len(body["auth_required"]) == 1
        er._table_ensured = False


class TestForceTriggerEvalMcpAuth:
    async def test_returns_403_when_mcp_auth_required(self):
        mock_request = MagicMock()
        mock_request.headers = {"authorization": "Bearer tok"}

        with (
            patch.object(er, "_EVAL_RUNNER_URL", "http://eval:8099"),
            patch(
                "deep_agent.aegra.eval_routes._check_mcp_auth",
                new_callable=AsyncMock,
                return_value=[{"name": "mcp1", "connect_url": "/mcp/mcp1/connect"}],
            ),
        ):
            result = await er.force_trigger_eval(mock_request)

        assert result.status_code == 403
        body = json.loads(result.body)
        assert len(body["auth_required"]) == 1


class TestEvalHistoryInvalidLimit:
    async def test_returns_400_on_bad_limit(self):
        from fastapi import HTTPException

        mock_request = MagicMock()
        mock_request.query_params = {"limit": "abc"}
        with pytest.raises(HTTPException) as exc:
            await er.eval_history(mock_request)
        assert exc.value.status_code == 400


class TestAtomicSetInProgressRaceCondition:
    async def test_returns_none_false_when_insert_and_recheck_both_fail(self):
        er._table_ensured = True
        select_cursor = _make_cursor(rows=[])
        select_cursor.fetchone = AsyncMock(return_value=None)

        insert_cursor = _make_cursor(rows=[])
        insert_cursor.fetchone = AsyncMock(return_value=None)

        recheck_cursor = _make_cursor(rows=[])
        recheck_cursor.fetchone = AsyncMock(return_value=None)

        conn = AsyncMock()
        conn.__aenter__ = AsyncMock(return_value=conn)
        conn.__aexit__ = AsyncMock(return_value=False)
        conn.execute = AsyncMock(
            side_effect=[select_cursor, insert_cursor, recheck_cursor]
        )

        with patch(
            "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
        ):
            doc, is_new = await er._atomic_set_in_progress("abc123", force=False)

        assert doc is None
        assert is_new is False
        er._table_ensured = False

    async def test_returns_existing_on_insert_race(self):
        er._table_ensured = True
        select_cursor = _make_cursor(rows=[])
        select_cursor.fetchone = AsyncMock(return_value=None)

        insert_cursor = _make_cursor(rows=[])
        insert_cursor.fetchone = AsyncMock(return_value=None)

        cols = [_col("config_hash"), _col("eval_status")]
        race_row = ("abc123", "in_progress")
        recheck_cursor = _make_cursor(rows=[race_row], description=cols)

        conn = AsyncMock()
        conn.__aenter__ = AsyncMock(return_value=conn)
        conn.__aexit__ = AsyncMock(return_value=False)
        conn.execute = AsyncMock(
            side_effect=[select_cursor, insert_cursor, recheck_cursor]
        )

        with patch(
            "deep_agent.aegra.eval_routes._pg_conn", AsyncMock(return_value=conn)
        ):
            doc, is_new = await er._atomic_set_in_progress("abc123", force=False)

        assert doc is not None
        assert doc["eval_status"] == "in_progress"
        assert is_new is False
        er._table_ensured = False


class TestCleanupEvalRedisWithSub:
    async def test_cleans_up_sub_keys(self):
        mock_request = MagicMock()
        mock_request.headers = {"x-internal-token": "secret"}
        with (
            patch.object(er, "_EVAL_TOKEN_REFRESH_ENABLED", True),
            patch.object(er, "_EVAL_INTERNAL_TOKEN", "secret"),
            patch("deep_agent.aegra.redis.cache_get", return_value="user123"),
            patch("deep_agent.aegra.redis.cache_delete") as mock_delete,
        ):
            result = await er.cleanup_eval_redis(mock_request)
        assert result["status"] == "ok"
        delete_keys = [call.args[0] for call in mock_delete.call_args_list]
        assert "eval:trigger_sub" in delete_keys
        assert "eval:active:user123" in delete_keys
        assert "eval:access:user123" in delete_keys
        assert "eval:refresh:user123" in delete_keys


class TestRequireEvalFilesProductionNoDataset:
    async def test_raises_with_config_hint_when_file_exists(self, tmp_path):
        from fastapi import HTTPException

        evals_dir = tmp_path / "evals" / "lightspeed"
        evals_dir.mkdir(parents=True)
        (evals_dir / "eval_cases.yaml").write_text("cases: []")

        with (
            patch(
                "deep_agent.aegra.eval_routes._has_postgres_dataset",
                AsyncMock(return_value=False),
            ),
            patch.dict(
                "os.environ",
                {"CONFIG_PATH": str(tmp_path), "ENVIRONMENT": "production"},
            ),
        ):
            with pytest.raises(HTTPException) as exc:
                await er._require_eval_files()
        assert exc.value.status_code == 400
        assert "Dataset UI" in exc.value.detail

    async def test_passes_when_postgres_dataset_exists(self):
        with patch(
            "deep_agent.aegra.eval_routes._has_postgres_dataset",
            AsyncMock(return_value=True),
        ):
            await er._require_eval_files()
