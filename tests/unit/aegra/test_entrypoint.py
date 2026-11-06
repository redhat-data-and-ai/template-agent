"""Unit tests for container entrypoint config validation."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from deep_agent.aegra.entrypoint import main, start_server, validate_config_mount


@pytest.fixture
def log():
    """Capture the entrypoint's structured logger.

    Boot output goes through structlog so it lands in the log pipeline as JSON
    like every other line; asserting on the logger rather than on stdout keeps
    these tests independent of how the renderer is configured.
    """
    with patch("deep_agent.aegra.entrypoint._logger") as factory:
        yield factory.return_value


class TestLoggerBootstrap:
    """``_logger`` must configure structlog before the first line is emitted."""

    def test_configures_logging_once_and_caches(self, monkeypatch):
        import deep_agent.aegra.entrypoint as ep

        monkeypatch.setattr(ep, "_LOGGER", None)
        fake_setup = MagicMock()
        fake_module = MagicMock(setup_logging=fake_setup)
        monkeypatch.setitem(sys.modules, "aegra_api", MagicMock())
        monkeypatch.setitem(sys.modules, "aegra_api.utils", MagicMock())
        monkeypatch.setitem(sys.modules, "aegra_api.utils.setup_logging", fake_module)

        first = ep._logger()
        second = ep._logger()

        fake_setup.assert_called_once()
        assert first is second


class TestValidateConfigMount:
    def _write_valid_config(self, tmp_path):
        (tmp_path / "PROMPT.md").write_text(
            "---\nname: test\nmodel: gpt-4\n---\nPrompt body.\n"
        )
        (tmp_path / "mcp.json").write_text(
            json.dumps({"mcpServers": {"test": {"url": "http://localhost"}}})
        )

    def test_exits_when_config_path_missing(self, tmp_path, log):
        missing = tmp_path / "nonexistent"
        with patch("deep_agent.aegra.entrypoint.CONFIG_PATH", missing):
            with pytest.raises(SystemExit, match="1"):
                validate_config_mount()
        log.error.assert_called_once()

    def test_exits_when_prompt_md_missing(self, tmp_path, log):
        (tmp_path / "mcp.json").write_text("{}")
        with patch("deep_agent.aegra.entrypoint.CONFIG_PATH", tmp_path):
            with pytest.raises(SystemExit, match="1"):
                validate_config_mount()
        assert log.error.call_args.kwargs["missing_files"] == ["PROMPT.md"]

    def test_exits_when_mcp_json_missing(self, tmp_path, log):
        (tmp_path / "PROMPT.md").write_text("---\nname: test\nmodel: gpt-4\n---\n")
        with patch("deep_agent.aegra.entrypoint.CONFIG_PATH", tmp_path):
            with pytest.raises(SystemExit, match="1"):
                validate_config_mount()
        assert log.error.call_args.kwargs["missing_files"] == ["mcp.json"]

    def test_passes_with_valid_config(self, tmp_path, log):
        self._write_valid_config(tmp_path)
        with patch("deep_agent.aegra.entrypoint.CONFIG_PATH", tmp_path):
            validate_config_mount()
        log.error.assert_not_called()
        assert log.info.call_args_list[-1].kwargs["mcp_servers"] == ["test"]

    def test_warns_on_invalid_mcp_json(self, tmp_path, log):
        (tmp_path / "PROMPT.md").write_text("---\nname: test\nmodel: gpt-4\n---\n")
        (tmp_path / "mcp.json").write_text("{ invalid json //")
        with patch("deep_agent.aegra.entrypoint.CONFIG_PATH", tmp_path):
            validate_config_mount()
        assert log.warning.call_args.args[0] == "Invalid mcp.json"

    def test_warns_when_mcp_json_unreadable(self, tmp_path, log):
        """An unreadable mcp.json warns but does not abort the boot."""
        self._write_valid_config(tmp_path)
        with patch("deep_agent.aegra.entrypoint.CONFIG_PATH", tmp_path):
            with patch("builtins.open", side_effect=OSError("permission denied")):
                validate_config_mount()
        assert log.warning.call_args.args[0] == "Could not read mcp.json"

    def test_boot_output_is_structured_not_printed(self, tmp_path, log, capsys):
        """Nothing may reach stdout/stderr directly — it would bypass JSON rendering.

        Regression guard for the reviewer finding: print() here emitted plain
        text into a pipeline where every other line is JSON.
        """
        self._write_valid_config(tmp_path)
        with patch("deep_agent.aegra.entrypoint.CONFIG_PATH", tmp_path):
            validate_config_mount()
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""


class TestMain:
    def test_validates_config_before_starting_the_server(self):
        """A bad config mount must abort before uvicorn binds the port."""
        call_order = []
        with (
            patch(
                "deep_agent.aegra.entrypoint.validate_config_mount",
                side_effect=lambda: call_order.append("validate"),
            ) as validate,
            patch(
                "deep_agent.aegra.entrypoint.start_server",
                side_effect=lambda: call_order.append("serve"),
            ) as serve,
        ):
            main()
        validate.assert_called_once()
        serve.assert_called_once()
        assert call_order == ["validate", "serve"]


class TestStartServer:
    """``start_server`` must not let uvicorn install its own access logger.

    Aegra's ``StructLogMiddleware`` already emits a structured access line per
    request and honours ``LOG_EXCLUDE_PATHS``. Aegra silences ``uvicorn.access``
    in ``setup_logging()``, but ``uvicorn.run`` re-applies uvicorn's own
    ``LOGGING_CONFIG`` afterwards and reinstates it — so every request, health
    probes included, gets logged twice unless ``access_log=False`` is passed.
    """

    def _run_start_server(self, monkeypatch):
        fake_uvicorn = MagicMock()
        fake_aegra_main = MagicMock()
        monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
        monkeypatch.setitem(sys.modules, "aegra_api", MagicMock())
        monkeypatch.setitem(sys.modules, "aegra_api.main", fake_aegra_main)
        start_server()
        fake_uvicorn.run.assert_called_once()
        return fake_uvicorn.run.call_args

    def test_disables_uvicorn_access_log(self, monkeypatch, log):
        _, kwargs = self._run_start_server(monkeypatch)
        assert kwargs["access_log"] is False

    def test_keeps_uvicorn_error_logging_at_info(self, monkeypatch, log):
        """``uvicorn.error`` still reports startup, shutdown and bad requests."""
        _, kwargs = self._run_start_server(monkeypatch)
        assert kwargs["log_level"] == "info"

    def test_suppresses_uvicorn_log_config(self, monkeypatch, log):
        """Without this, uvicorn re-attaches plain-text handlers to uvicorn.error.

        Protocol-level failures ("Invalid HTTP request received.") and the
        startup banner would then reach the log pipeline as unparseable text
        while every other line is JSON.
        """
        _, kwargs = self._run_start_server(monkeypatch)
        assert kwargs["log_config"] is None

    def test_honours_host_and_port_env(self, monkeypatch, log):
        monkeypatch.setenv("AGENT_HOST", "127.0.0.1")
        monkeypatch.setenv("AGENT_PORT", "6001")
        _, kwargs = self._run_start_server(monkeypatch)
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == 6001
