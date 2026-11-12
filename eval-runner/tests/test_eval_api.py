"""Tests for eval_api.py — FastAPI endpoints and helper functions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import eval_api
import pytest
import yaml
from fastapi import Request
from fastapi.testclient import TestClient

# ── _score_from_counts ────────────────────────────────────────────────────────


def test_score_from_counts_all_pass() -> None:
    status, score = eval_api._score_from_counts(10, 0, 0)
    assert status == "passed"
    assert score == 1.0


def test_score_from_counts_mixed() -> None:
    status, score = eval_api._score_from_counts(7, 3, 0)
    assert status == "failed"
    assert score == pytest.approx(0.7)


def test_score_from_counts_all_error() -> None:
    status, score = eval_api._score_from_counts(0, 0, 5)
    assert status == "error"
    assert score == 0.0


def test_score_from_counts_some_error_with_pass() -> None:
    # Errors excluded from denominator: score = pass/(pass+fail) = 5/8
    status, score = eval_api._score_from_counts(5, 3, 2)
    assert status == "failed"
    assert score == pytest.approx(5 / 8)


def test_score_from_counts_zero_total() -> None:
    status, score = eval_api._score_from_counts(0, 0, 0)
    assert status == "error"
    assert score == 0.0


# ── _resolve_config_dir ───────────────────────────────────────────────────────


def test_resolve_config_dir_uses_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AGENT_CONFIG_DIR", str(tmp_path))
    result = eval_api._resolve_config_dir()
    assert result == str(tmp_path)


def test_resolve_config_dir_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AGENT_CONFIG_DIR", raising=False)
    with patch.object(Path, "is_dir", return_value=False):
        result = eval_api._resolve_config_dir()
    assert result == "/agent-config"


# ── _find_eval_files ──────────────────────────────────────────────────────────


@pytest.fixture
def cases_file(tmp_path: Path) -> Generator[Path, None, None]:
    """Write a minimal eval_cases.yaml and patch EVAL_CASES_PATH."""
    cases = [
        {
            "conversation_group_id": "c1",
            "tag": "tool_use",
            "turns": [{"query": "Q1", "turn_id": "t1"}],
        },
        {
            "conversation_group_id": "c2",
            "tag": "hitl",
            "turns": [{"query": "Q2", "turn_id": "t1"}],
        },
    ]
    p = tmp_path / "eval_cases.yaml"
    p.write_text(yaml.dump(cases))
    orig = eval_api.EVAL_CASES_PATH
    eval_api.EVAL_CASES_PATH = p
    yield p
    eval_api.EVAL_CASES_PATH = orig


def test_find_eval_files_specific_pattern(cases_file: Path) -> None:
    files = eval_api._find_eval_files("tool_use", None)
    try:
        assert len(files) == 1
        assert files[0].exists()
        content = yaml.safe_load(files[0].read_text())
        assert all(c["tag"] == "tool_use" for c in content)
    finally:
        for f in files:
            f.unlink(missing_ok=True)


def test_find_eval_files_all_patterns_splits_by_tag(cases_file: Path) -> None:
    files = eval_api._find_eval_files(None, None)
    try:
        assert len(files) == 2  # one per tag
    finally:
        for f in files:
            f.unlink(missing_ok=True)


def test_find_eval_files_pattern_not_found_raises(cases_file: Path) -> None:
    with pytest.raises(FileNotFoundError, match="multi_agent"):
        eval_api._find_eval_files("multi_agent", None)


def test_find_eval_files_missing_cases_raises(tmp_path: Path) -> None:
    orig = eval_api.EVAL_CASES_PATH
    eval_api.EVAL_CASES_PATH = tmp_path / "nonexistent.yaml"
    try:
        with pytest.raises(FileNotFoundError):
            eval_api._find_eval_files(None, None)
    finally:
        eval_api.EVAL_CASES_PATH = orig


def test_find_eval_files_no_tags_writes_single_temp_file(tmp_path: Path) -> None:
    cases = [
        {"conversation_group_id": "c1", "turns": [{"query": "Q", "turn_id": "t1"}]}
    ]
    p = tmp_path / "eval_cases.yaml"
    p.write_text(yaml.dump(cases))
    orig = eval_api.EVAL_CASES_PATH
    eval_api.EVAL_CASES_PATH = p
    try:
        files = eval_api._find_eval_files(None, None)
        assert len(files) == 1
        assert files[0].exists()
    finally:
        eval_api.EVAL_CASES_PATH = orig
        for f in files:
            f.unlink(missing_ok=True)


# ── _get_system_yaml_content ──────────────────────────────────────────────────


@pytest.fixture
def system_yaml_file(tmp_path: Path) -> Generator[Path, None, None]:
    cfg = {
        "storage": [
            {
                "type": "postgres",
                "host": "old-host",
                "port": 5432,
                "database": "mydb",
                "user": "u",
                "password": "old-pw",
            }
        ]
    }
    p = tmp_path / "system.yaml"
    p.write_text(yaml.dump(cfg))
    orig = eval_api.EVAL_SYSTEM_CONFIG
    eval_api.EVAL_SYSTEM_CONFIG = p
    yield p
    eval_api.EVAL_SYSTEM_CONFIG = orig


def test_get_system_yaml_injects_postgres_host(
    system_yaml_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("POSTGRES_HOST", "injected-host")
    content = eval_api._get_system_yaml_content()
    parsed = yaml.safe_load(content)
    assert parsed["storage"][0]["host"] == "injected-host"


def test_get_system_yaml_injects_password(
    system_yaml_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("POSTGRES_PASSWORD", "newsecret")
    content = eval_api._get_system_yaml_content()
    parsed = yaml.safe_load(content)
    assert parsed["storage"][0]["password"] == "newsecret"


def test_get_system_yaml_injects_port_db_user(
    system_yaml_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("POSTGRES_PORT", "5433")
    monkeypatch.setenv("POSTGRES_DB", "evaldb")
    monkeypatch.setenv("POSTGRES_USER", "evaluser")
    content = eval_api._get_system_yaml_content()
    parsed = yaml.safe_load(content)
    backend = parsed["storage"][0]
    assert backend["port"] == 5433
    assert backend["database"] == "evaldb"
    assert backend["user"] == "evaluser"


def test_get_system_yaml_missing_file_auto_generates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # When EVAL_SYSTEM_CONFIG doesn't exist, content is auto-generated from
    # the stored judge model / PROMPT.md — no FileNotFoundError raised.
    from unittest.mock import patch

    orig = eval_api.EVAL_SYSTEM_CONFIG
    eval_api.EVAL_SYSTEM_CONFIG = tmp_path / "nonexistent.yaml"
    try:
        with (
            patch(
                "eval_api._read_prompt_model_spec",
                return_value=("gemini-2.5-pro", None),
            ),
            patch("eval_api._detect_provider", return_value="vertex"),
        ):
            content = eval_api._get_system_yaml_content(judge_model=None)
        assert "gemini-2.5-pro" in content
        assert "judge" in content
    finally:
        eval_api.EVAL_SYSTEM_CONFIG = orig


def test_system_yaml_path_writes_temp_file(system_yaml_file: Path) -> None:
    path = eval_api._system_yaml_path()
    try:
        assert path.exists()
        content = yaml.safe_load(path.read_text())
        assert content["storage"][0]["type"] == "postgres"
    finally:
        path.unlink(missing_ok=True)


# ── _compute_config_hash ──────────────────────────────────────────────────────


def test_compute_config_hash_returns_16_chars(tmp_path: Path) -> None:
    h = eval_api._compute_config_hash(str(tmp_path))
    assert len(h) == 16


def test_compute_config_hash_excludes_evals_dir(tmp_path: Path) -> None:
    (tmp_path / "system.yaml").write_text("model: gpt4")
    h1 = eval_api._compute_config_hash(str(tmp_path))
    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "eval_cases.yaml").write_text("- query: hi")
    h2 = eval_api._compute_config_hash(str(tmp_path))
    assert h1 == h2


# ── _run_eval_pattern_sync ────────────────────────────────────────────────────


def test_run_eval_pattern_sync_returns_exit_code(tmp_path: Path) -> None:
    with patch("eval_api.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        rc = eval_api._run_eval_pattern_sync(
            tmp_path / "cases.yaml",
            tmp_path / "system.yaml",
            tmp_path / "output",
        )
    assert rc == 0


def test_run_eval_pattern_sync_non_zero_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with patch("eval_api.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1)
        eval_api._run_eval_pattern_sync(
            tmp_path / "cases.yaml",
            tmp_path / "system.yaml",
            tmp_path / "output",
        )
    assert "Non-zero exit" in caplog.text


def test_run_eval_pattern_sync_injects_auth_token(tmp_path: Path) -> None:
    with patch("eval_api.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        eval_api._run_eval_pattern_sync(
            tmp_path / "cases.yaml",
            tmp_path / "system.yaml",
            tmp_path / "output",
            auth_token="mytoken",
        )
    env_passed = mock_run.call_args[1]["env"]
    assert env_passed["AGENT_AUTH_TOKEN"] == "mytoken"


@pytest.mark.asyncio
async def test_run_eval_pattern_delegates_to_executor(tmp_path: Path) -> None:
    with patch("eval_api._run_eval_pattern_sync", return_value=0) as mock_sync:
        rc = await eval_api._run_eval_pattern(
            tmp_path / "cases.yaml",
            tmp_path / "system.yaml",
            tmp_path / "output",
            auth_token="tok",
        )
    assert rc == 0
    mock_sync.assert_called_once_with(
        tmp_path / "cases.yaml",
        tmp_path / "system.yaml",
        tmp_path / "output",
        "tok",
    )


# ── _extract_token ────────────────────────────────────────────────────────────


def _make_request(headers: dict[str, str]) -> Request:
    raw_headers = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    scope = {"type": "http", "headers": raw_headers}
    return Request(scope)


def test_extract_token_bearer() -> None:
    req = _make_request({"Authorization": "Bearer abc123"})
    assert eval_api._extract_token(req) == "abc123"


def test_extract_token_non_bearer_returns_fallback() -> None:
    orig = eval_api.AGENT_AUTH_TOKEN
    eval_api.AGENT_AUTH_TOKEN = "fallback-token"
    try:
        req = _make_request({})
        assert eval_api._extract_token(req) == "fallback-token"
    finally:
        eval_api.AGENT_AUTH_TOKEN = orig


def test_extract_token_bearer_with_extra_whitespace() -> None:
    req = _make_request({"Authorization": "Bearer   trimmed  "})
    assert eval_api._extract_token(req) == "trimmed"


# ── _run_eval (background coroutine) ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_eval_completes_with_db_summary(
    cases_file: Path, system_yaml_file: Path, tmp_path: Path
) -> None:
    orig_output = eval_api.EVAL_OUTPUT_DIR
    eval_api.EVAL_OUTPUT_DIR = tmp_path / "eval_output"
    eval_api._status = {"state": "idle", "run_id": None}
    eval_api._latest_result = None

    db_data: dict[str, Any] = {
        "summary": {"summary_stats": {"overall": {"PASS": 3, "FAIL": 1, "ERROR": 0}}},
        "ls_run_ids": ["ls-1"],
    }

    with patch("eval_api._run_eval_pattern", new_callable=AsyncMock, return_value=0):
        with patch("eval_api.load_results_since", return_value=db_data):
            with patch("eval_api.write_eval_result") as mock_write:
                with patch("eval_api.fetch_dataset", return_value=(None, None)):
                    await eval_api._run_eval(
                        "tool_use", auth_token="user-tok", run_id="test-run-1"
                    )

    assert eval_api._status["state"] == "completed"
    result: dict[str, Any] | None = eval_api._latest_result
    assert result is not None
    assert result["pass"] == 3
    assert result["fail"] == 1
    assert result["eval_score"] == pytest.approx(0.75)
    mock_write.assert_called_once()
    eval_api.EVAL_OUTPUT_DIR = orig_output


@pytest.mark.asyncio
async def test_run_eval_setup_failure_sets_error(
    cases_file: Path, tmp_path: Path
) -> None:
    orig_cases = eval_api.EVAL_CASES_PATH
    eval_api.EVAL_CASES_PATH = tmp_path / "missing.yaml"
    eval_api._status = {"state": "idle", "run_id": None}

    with patch("eval_api.fetch_dataset", return_value=(None, None)):
        await eval_api._run_eval(None, run_id="test-run-2")

    assert eval_api._status["state"] == "error"
    eval_api.EVAL_CASES_PATH = orig_cases


@pytest.mark.asyncio
async def test_run_eval_write_failure_still_completes(
    cases_file: Path, system_yaml_file: Path, tmp_path: Path
) -> None:
    orig_output = eval_api.EVAL_OUTPUT_DIR
    eval_api.EVAL_OUTPUT_DIR = tmp_path / "eval_output"

    with patch("eval_api._run_eval_pattern", new_callable=AsyncMock, return_value=0):
        with patch("eval_api.load_results_since", return_value={}):
            with patch(
                "eval_api.write_eval_result", side_effect=RuntimeError("db down")
            ):
                with patch("eval_api.fetch_dataset", return_value=(None, None)):
                    await eval_api._run_eval(None, run_id="test-run-3")

    assert eval_api._status["state"] == "completed"
    eval_api.EVAL_OUTPUT_DIR = orig_output


# ── FastAPI endpoints ─────────────────────────────────────────────────────────


@pytest.fixture
def api_client(tmp_path: Path) -> Generator[TestClient, None, None]:
    """TestClient with mocked DB, minimal config files, reset global state."""
    cases = [
        {
            "conversation_group_id": "c1",
            "tag": "tool_use",
            "turns": [{"query": "Q", "turn_id": "t1"}],
        }
    ]
    (tmp_path / "eval_cases.yaml").write_text(yaml.dump(cases))
    (tmp_path / "system.yaml").write_text(yaml.dump({"storage": []}))

    orig_cases = eval_api.EVAL_CASES_PATH
    orig_system = eval_api.EVAL_SYSTEM_CONFIG
    eval_api.EVAL_CASES_PATH = tmp_path / "eval_cases.yaml"
    eval_api.EVAL_SYSTEM_CONFIG = tmp_path / "system.yaml"
    eval_api._status = {"state": "idle", "run_id": None}
    eval_api._latest_result = None

    with patch("eval_api._EVAL_INTERNAL_TOKEN", "test-token"):
        with patch("eval_api.ensure_table"):
            with TestClient(
                eval_api.app, headers={"x-internal-token": "test-token"}
            ) as client:
                yield client

    eval_api.EVAL_CASES_PATH = orig_cases
    eval_api.EVAL_SYSTEM_CONFIG = orig_system
    eval_api._status = {"state": "idle", "run_id": None}
    eval_api._latest_result = None


def test_health_returns_ok(api_client: TestClient) -> None:
    resp = api_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_get_status_idle(api_client: TestClient) -> None:
    resp = api_client.get("/evals/status")
    assert resp.status_code == 200
    assert resp.json()["state"] == "idle"


def test_get_latest_results_404_when_none(api_client: TestClient) -> None:
    resp = api_client.get("/evals/results")
    assert resp.status_code == 404


def test_get_latest_results_returns_data(api_client: TestClient) -> None:
    eval_api._latest_result = {"run_id": "r1", "eval_score": 0.9}
    resp = api_client.get("/evals/results")
    assert resp.status_code == 200
    assert resp.json()["run_id"] == "r1"


def test_run_all_returns_202(api_client: TestClient) -> None:
    with patch("eval_api._run_eval"):
        resp = api_client.post("/evals/run", json={})
    assert resp.status_code == 202
    assert resp.json()["status"] == "started"
    assert resp.json()["pattern"] == "all"


def test_trigger_run_id_propagates_to_status_and_background_task(
    api_client: TestClient,
) -> None:
    """run_id in 202 response must match _status and the _run_eval argument."""
    captured: dict[str, Any] = {}

    async def capture_run_eval(*args: Any, **kwargs: Any) -> None:
        captured["args"] = args

    with patch("eval_api._run_eval", side_effect=capture_run_eval):
        resp = api_client.post("/evals/run", json={})

    assert resp.status_code == 202
    response_run_id = resp.json()["run_id"]
    assert response_run_id
    assert eval_api._status["state"] == "running"
    assert eval_api._status["run_id"] == response_run_id
    # args: (pattern, config_hash, auth_token, run_id, eval_row_id)
    assert captured["args"][3] == response_run_id


def test_run_all_with_body_fields(api_client: TestClient) -> None:
    with patch("eval_api._run_eval") as mock_run_eval:
        resp = api_client.post(
            "/evals/run",
            json={"config_hash": "abc"},
        )
    assert resp.status_code == 202
    assert "run_id" in resp.json()
    mock_run_eval.assert_called_once()
    pattern, config_hash, auth_token, run_id, eval_row_id = mock_run_eval.call_args[0]
    assert pattern is None
    assert config_hash == "abc"
    assert auth_token == ""
    assert run_id == resp.json()["run_id"]
    assert eval_row_id is None


def test_run_all_409_when_running(api_client: TestClient) -> None:
    eval_api._status = {"state": "running", "run_id": "existing"}
    resp = api_client.post("/evals/run", json={})
    assert resp.status_code == 409


def test_run_all_accepts_request_without_files(api_client: TestClient) -> None:
    # File-existence guards were removed — the trigger schedules the run regardless.
    # The background task handles missing files via Postgres fallback.
    with patch("eval_api._run_eval"):
        resp = api_client.post("/evals/run", json={})
    assert resp.status_code == 202


def test_run_pattern_valid(api_client: TestClient) -> None:
    with patch("eval_api._run_eval"):
        resp = api_client.post("/evals/run/tool_use")
    assert resp.status_code == 202


def test_run_pattern_invalid_returns_400(api_client: TestClient) -> None:
    resp = api_client.post("/evals/run/not_a_pattern")
    assert resp.status_code == 400


def test_get_run_results_found(api_client: TestClient) -> None:
    rows = [
        {
            "conversation_group_id": "c1",
            "result": "PASS",
            "score": 1.0,
            "turn_id": "t1",
            "metric_identifier": "m",
            "reason": "ok",
        }
    ]
    with patch("eval_api.get_results_by_run_id", return_value=rows):
        resp = api_client.get("/evals/results/run1")
    assert resp.status_code == 200
    assert resp.json()["run_id"] == "run1"
    assert len(resp.json()["results"]) == 1


def test_get_run_results_404_when_empty(api_client: TestClient) -> None:
    with patch("eval_api.get_results_by_run_id", return_value=[]):
        resp = api_client.get("/evals/results/nonexistent")
    assert resp.status_code == 404


def test_get_run_results_500_on_db_error(api_client: TestClient) -> None:
    with patch("eval_api.get_results_by_run_id", side_effect=Exception("db down")):
        resp = api_client.get("/evals/results/run1")
    assert resp.status_code == 500


def test_mcp_stub_returns_error(api_client: TestClient) -> None:
    resp = api_client.post("/mcp")
    assert resp.status_code == 200
    assert "error" in resp.json()


class TestProviderResolution:
    """Provider must follow the agent's resolution, not a name guess.

    The runner cannot import the agent's resolver — it ships as its own
    package with only src/ in the image — so it reads the same
    models/*.yaml the agent resolves against and translates the name into
    the evaluation framework's vocabulary.
    """

    def test_openai_compatible_model_does_not_fall_through_to_vertex(
        self, tmp_path: Path
    ) -> None:
        """A Granite/Mistral model used to get a Vertex client and always fail."""
        from eval_api import _detect_provider

        assert _detect_provider("granite-3.3-8b-instruct") == "hosted_vllm"
        assert _detect_provider("mistralai/Mistral-7B-Instruct") == "hosted_vllm"

    def test_known_prefixes_are_unchanged(self) -> None:
        from eval_api import _detect_provider

        assert _detect_provider("gemini-2.5-pro") == "vertex"
        assert _detect_provider("claude-sonnet-4-6") == "anthropic_vertex"
        assert _detect_provider("gpt-4o") == "openai"

    def test_declared_provider_is_translated_not_passed_through(
        self, tmp_path: Path
    ) -> None:
        """models/*.yaml says vertex_ai; the framework only knows vertex."""
        import eval_api

        models = tmp_path / "models"
        models.mkdir()
        (models / "m.yaml").write_text(
            "provider: vertex_ai\nmodel_id: some-tuned-model\n"
        )
        with patch.object(eval_api, "_agent_config_dir", str(tmp_path)):
            assert eval_api._detect_provider("some-tuned-model") == "vertex"

    def test_declaration_wins_over_name_inference(self, tmp_path: Path) -> None:
        """A gemini-named model declared openai-compatible follows the file."""
        import eval_api

        models = tmp_path / "models"
        models.mkdir()
        (models / "m.yaml").write_text(
            "provider: openai-compatibility\nmodel_id: gemini-via-gateway\n"
        )
        with patch.object(eval_api, "_agent_config_dir", str(tmp_path)):
            assert eval_api._detect_provider("gemini-via-gateway") == "hosted_vllm"

    def test_unreadable_models_dir_falls_back_to_inference(
        self, tmp_path: Path
    ) -> None:
        """A config problem must not make provider resolution a hard failure."""
        import eval_api

        with patch.object(eval_api, "_agent_config_dir", str(tmp_path / "missing")):
            assert eval_api._detect_provider("gemini-2.5-pro") == "vertex"


class TestModelNameFromFrontmatter:
    def test_bare_string_form(self) -> None:
        from eval_api import _model_name

        assert _model_name("granite-3.3-8b-instruct") == "granite-3.3-8b-instruct"

    def test_object_form_used_to_pin_a_provider(self) -> None:
        """str() on this mapping yielded "{'provider': ..., 'name': ...}"."""
        from eval_api import _model_name

        raw = {"provider": "openai-compatibility", "name": "granite-3.3-8b-instruct"}
        assert _model_name(raw) == "granite-3.3-8b-instruct"

    def test_unusable_values_yield_none(self) -> None:
        from eval_api import _model_name

        assert _model_name(None) is None
        assert _model_name("   ") is None
        assert _model_name({"provider": "openai"}) is None


class TestGatewayServedGeminiInFrontmatter:
    """A Gemini name says nothing about who serves it.

    gemini-2.5-pro behind an OpenAI-compatible gateway matches the
    `gemini-` prefix and would be sent to Vertex unless the pinned
    provider is carried through rather than re-inferred from the name.
    """

    def _prompt(self, tmp_path: Path, body: str) -> Path:
        (tmp_path / "PROMPT.md").write_text(f"---\n{body}\n---\nContent")
        return tmp_path

    def test_inline_provider_overrides_the_name_prefix(self, tmp_path: Path) -> None:
        import eval_api

        self._prompt(
            tmp_path,
            "model:\n  provider: openai-compatibility\n  name: gemini-2.5-pro",
        )
        with patch.object(eval_api, "_agent_config_dir", str(tmp_path)):
            name, pinned = eval_api._read_prompt_model_spec()
            assert name == "gemini-2.5-pro"
            assert eval_api._detect_provider(name, pinned) == "hosted_vllm"

    def test_bare_gemini_still_resolves_to_vertex(self, tmp_path: Path) -> None:
        import eval_api

        self._prompt(tmp_path, "model: gemini-2.5-pro")
        with patch.object(eval_api, "_agent_config_dir", str(tmp_path)):
            name, pinned = eval_api._read_prompt_model_spec()
            assert name is not None
            assert pinned is None
            assert eval_api._detect_provider(name, pinned) == "vertex"

    def test_models_yaml_declaration_also_wins(self, tmp_path: Path) -> None:
        """The other place a gateway-served Gemini can be declared."""
        import eval_api

        models = tmp_path / "models"
        models.mkdir()
        (models / "g.yaml").write_text(
            "provider: openai-compatibility\nmodel_id: gemini-2.5-pro\n"
        )
        with patch.object(eval_api, "_agent_config_dir", str(tmp_path)):
            assert eval_api._detect_provider("gemini-2.5-pro") == "hosted_vllm"

    def test_read_prompt_model_still_returns_just_the_name(
        self, tmp_path: Path
    ) -> None:
        """Existing callers of the name-only helper are unaffected."""
        import eval_api

        self._prompt(
            tmp_path,
            "model:\n  provider: openai-compatibility\n  name: gemini-2.5-pro",
        )
        with patch.object(eval_api, "_agent_config_dir", str(tmp_path)):
            assert eval_api._read_prompt_model() == "gemini-2.5-pro"


# ── _ProbeAccessFilter ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _default_exclude_paths() -> Generator[None, None, None]:
    """Pin LOG_EXCLUDE_PATHS to the code default for these tests.

    The module reads it from the environment at import time, and eval-runner's
    compose loads the repo-root `.env` — so a developer who sets
    LOG_EXCLUDE_PATHS (including to empty, the documented way to re-enable
    probe logging) would otherwise see these tests fail for no reason.
    """
    with patch.object(eval_api, "LOG_EXCLUDE_PATHS", ("/health", "/metrics")):
        yield


def _access_record(path: str, status: int) -> logging.LogRecord:
    """Build a record shaped like uvicorn's access log call."""
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("127.0.0.1:1", "GET", path, "1.1", status),
        exc_info=None,
    )


@pytest.mark.parametrize("path", ["/health", "/metrics"])
def test_probe_filter_drops_successful_probes(path: str) -> None:
    """Probes at 2xx are ~960 lines/hour and carry no information."""
    assert eval_api._ProbeAccessFilter().filter(_access_record(path, 200)) is False


@pytest.mark.parametrize("status", [404, 500, 503])
def test_probe_filter_keeps_failing_probes(status: int) -> None:
    """A failing probe must stay visible — this is the point of the filter."""
    assert (
        eval_api._ProbeAccessFilter().filter(_access_record("/health", status)) is True
    )


def test_probe_filter_keeps_real_traffic() -> None:
    assert (
        eval_api._ProbeAccessFilter().filter(_access_record("/evals/run", 200)) is True
    )


def test_probe_filter_ignores_query_string() -> None:
    assert (
        eval_api._ProbeAccessFilter().filter(_access_record("/health?v=1", 200))
        is False
    )


def test_probe_filter_passes_through_unexpected_record_shapes() -> None:
    """Non-access records on the same logger must not be swallowed."""
    rec = logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="something else",
        args=None,
        exc_info=None,
    )
    assert eval_api._ProbeAccessFilter().filter(rec) is True


def test_probe_filter_passes_through_non_string_path() -> None:
    """Defensive: a record with the right arity but wrong types must not be dropped."""
    rec = _access_record("/health", 200)
    rec.args = ("127.0.0.1:1", "GET", None, "1.1", "200")
    assert eval_api._ProbeAccessFilter().filter(rec) is True


@pytest.mark.parametrize("path", ["/health/ready", "/healthz", "/metrics/raw"])
def test_probe_filter_matches_by_prefix_like_aegra(path: str) -> None:
    """Shared LOG_EXCLUDE_PATHS must mean the same thing here as in template-agent.

    aegra's logger_middleware uses path.startswith(); eval-runner reads the same
    repo-root .env, so exact matching would make one variable mean two things.
    """
    assert eval_api._ProbeAccessFilter().filter(_access_record(path, 200)) is False


def test_probe_filter_prefix_match_still_keeps_failures() -> None:
    assert (
        eval_api._ProbeAccessFilter().filter(_access_record("/health/ready", 503))
        is True
    )


def test_probe_filter_is_registered_on_uvicorn_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without this the filter exists but never runs.

    Asserted as an invariant rather than a fact, because registration happens
    at import under `if LOG_EXCLUDE_PATHS:` — a developer with the variable set
    empty legitimately has no filter attached.
    """
    import importlib
    import os

    monkeypatch.setenv("LOG_EXCLUDE_PATHS", "/health,/metrics")
    logging.getLogger("uvicorn.access").filters.clear()
    importlib.reload(eval_api)
    try:
        assert eval_api.LOG_EXCLUDE_PATHS == ("/health", "/metrics")
        assert any(
            isinstance(f, eval_api._ProbeAccessFilter)
            for f in logging.getLogger("uvicorn.access").filters
        )
    finally:
        monkeypatch.delenv("LOG_EXCLUDE_PATHS", raising=False)
        os.environ.pop("LOG_EXCLUDE_PATHS", None)
        importlib.reload(eval_api)
