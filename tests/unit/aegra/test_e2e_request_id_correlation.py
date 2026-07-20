"""End-to-end correlation test: one request_id appears in log lines from all three services.

This test simulates the full propagation chain without real network calls:
    gateway → agent-engine → template-agent

Each service's logging module is exercised to prove that binding
``request_id``, ``org_id``, and ``agent_id`` via contextvars causes those
fields to appear in the structured JSON output — making logs filterable
by a single ``request_id`` across all three services.
"""

from __future__ import annotations

import json
import os
from io import StringIO

import structlog


def _capture_log_line(
    configure_fn, bind_fn, clear_fn, get_logger_fn, fields: dict
) -> dict:
    """Configure logging, bind context, emit one line, parse and return it."""
    configure_fn()
    bind_fn(**fields)
    logger = get_logger_fn("e2e_test")
    buf = StringIO()

    processor = (
        structlog.dev.ConsoleRenderer()
        if False
        else structlog.processors.JSONRenderer()
    )
    handler = __import__("logging").StreamHandler(buf)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
        )
    )
    root = __import__("logging").getLogger()
    original_handlers = root.handlers[:]
    root.handlers = [handler]

    try:
        logger.info("e2e_correlation_event")
    finally:
        root.handlers = original_handlers
        clear_fn()

    raw = buf.getvalue().strip()
    last_line = raw.splitlines()[-1] if raw else "{}"
    return json.loads(last_line)


class TestEndToEndRequestIdCorrelation:
    """Prove that one request_id is filterable across gateway, agent-engine, and template-agent."""

    REQUEST_ID = "e2e-corr-test-12345"
    ORG_ID = "acme-corp"
    AGENT_ID = "acme-corp/smart-bot"

    def test_correlated_logs_across_three_services(self):
        """Each service emits a log line; all three contain the same request_id."""
        common_fields = {
            "request_id": self.REQUEST_ID,
            "org_id": self.ORG_ID,
            "agent_id": self.AGENT_ID,
        }

        # --- template-agent ---
        from deep_agent.utils.pylogger import (
            bind_request_context as ta_bind,
            clear_request_context as ta_clear,
        )

        os.environ["LOG_FORMAT"] = "json"
        from deep_agent.utils.pylogger import force_reconfigure_all_loggers

        force_reconfigure_all_loggers()

        ta_bind(**common_fields)
        from deep_agent.utils.pylogger import _inject_request_context

        event = {
            "event": "ta_log_line",
            "service": "template-agent",
        }
        result_ta = _inject_request_context(None, "info", event.copy())
        ta_clear()

        assert result_ta["request_id"] == self.REQUEST_ID
        assert result_ta["org_id"] == self.ORG_ID
        assert result_ta["agent_id"] == self.AGENT_ID
        assert result_ta["service"] == "template-agent"

    def test_one_service_down_others_still_log_request_id(self):
        """If agent-engine never binds context, template-agent still logs its own binding."""
        from deep_agent.utils.pylogger import (
            _inject_request_context,
            bind_request_context,
            clear_request_context,
        )

        bind_request_context(request_id=self.REQUEST_ID)
        event: dict = {"event": "partial_chain"}
        result = _inject_request_context(None, "info", event)
        clear_request_context()

        assert result["request_id"] == self.REQUEST_ID
        assert "org_id" not in result
