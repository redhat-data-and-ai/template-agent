"""Unit tests for structured logging utility."""

from unittest.mock import patch

from deep_agent.utils.pylogger import (
    _inject_request_context,
    bind_request_context,
    clear_request_context,
    force_reconfigure_all_loggers,
    get_python_logger,
    get_uvicorn_log_config,
)


class TestGetPythonLogger:
    def test_returns_bound_logger(self):
        logger = get_python_logger("INFO")
        assert logger is not None

    def test_idempotent(self):
        a = get_python_logger("DEBUG")
        b = get_python_logger("DEBUG")
        assert a is not None
        assert b is not None


class TestForceReconfigure:
    def test_reconfigures(self):
        force_reconfigure_all_loggers("WARNING")
        logger = get_python_logger()
        assert logger is not None


class TestRequestContext:
    def setup_method(self):
        clear_request_context()

    def teardown_method(self):
        clear_request_context()

    def test_bind_and_inject(self):
        bind_request_context(
            trace_id="req-123",
            user_id="user-456",
            thread_id="thread-789",
        )
        event: dict = {"event": "test"}
        result = _inject_request_context(None, "info", event)
        assert result["trace_id"] == "req-123"
        assert result["user_id"] == "user-456"
        assert result["thread_id"] == "thread-789"
        assert result["service"] == "template-agent"

    def test_inject_without_bind(self):
        event: dict = {"event": "test"}
        result = _inject_request_context(None, "info", event)
        assert "trace_id" not in result
        assert "user_id" not in result
        assert "service" in result

    def test_clear_resets(self):
        bind_request_context(trace_id="req-x")
        clear_request_context()
        event: dict = {"event": "test"}
        result = _inject_request_context(None, "info", event)
        assert "trace_id" not in result

    def test_partial_bind(self):
        bind_request_context(user_id="u1")
        event: dict = {"event": "test"}
        result = _inject_request_context(None, "info", event)
        assert result["user_id"] == "u1"
        assert "trace_id" not in result


class TestConsoleRenderer:
    def test_json_format_default(self):
        with patch("deep_agent.utils.pylogger.LOG_FORMAT", "json"):
            from deep_agent.utils.pylogger import _get_renderer

            renderer = _get_renderer()
            assert "JSON" in type(renderer).__name__

    def test_console_format(self):
        with patch("deep_agent.utils.pylogger.LOG_FORMAT", "console"):
            from deep_agent.utils.pylogger import _get_renderer

            renderer = _get_renderer()
            assert "Console" in type(renderer).__name__


class TestUvicornLogConfig:
    def test_returns_valid_config(self):
        config = get_uvicorn_log_config("INFO")
        assert config["version"] == 1
        assert "formatters" in config
        assert "handlers" in config
        assert "loggers" in config

    def test_respects_log_level(self):
        config = get_uvicorn_log_config("DEBUG")
        assert config["loggers"][""]["level"] == "DEBUG"
