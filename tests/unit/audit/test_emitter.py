"""Unit tests for platform audit emitter."""

import json
from io import StringIO
from unittest.mock import patch

import pytest

from deep_agent.src.audit.context import bind_audit_context, clear_audit_context
from deep_agent.src.audit.emitter import emit_audit_event


@pytest.fixture(autouse=True)
def _clear_context():
    clear_audit_context()
    yield
    clear_audit_context()


class TestEmitAuditEventDisabled:
    def test_noop_when_disabled(self):
        with patch("deep_agent.src.audit.emitter.is_audit_enabled", return_value=False):
            with patch(
                "deep_agent.src.audit.emitter.sys.stdout", new_callable=StringIO
            ) as out:
                emit_audit_event("llm_call", model="test")
                assert out.getvalue() == ""


class TestEmitAuditEventEnabled:
    def test_emits_envelope(self):
        bind_audit_context(user="alice@example.com", org="acme", trace_id="trace-1")
        with patch("deep_agent.src.audit.emitter.is_audit_enabled", return_value=True):
            with patch(
                "deep_agent.src.audit.emitter.sys.stdout", new_callable=StringIO
            ) as out:
                emit_audit_event("llm_call", model="gemini", phase="start")
                record = json.loads(out.getvalue().strip())
                assert record["event"] == "platform.audit"
                assert record["audit_event_type"] == "llm_call"
                assert record["user"] == "alice@example.com"
                assert record["org"] == "acme"
                assert record["trace_id"] == "trace-1"
                assert record["details"]["model"] == "gemini"
                assert record["logger"] == "platform.audit"
                assert record["level"] == "info"

    def test_buffers_on_emit_failure(self):
        with patch("deep_agent.src.audit.emitter.is_audit_enabled", return_value=True):
            with patch("deep_agent.src.audit.emitter.sys.stdout") as mock_stdout:
                mock_stdout.write.side_effect = RuntimeError("sink down")
                with patch("deep_agent.src.audit.emitter.enqueue") as mock_enqueue:
                    emit_audit_event("llm_call", model="gemini")
                    mock_enqueue.assert_called_once()
                    envelope = mock_enqueue.call_args.args[0]
                    assert envelope["audit_event_type"] == "llm_call"
