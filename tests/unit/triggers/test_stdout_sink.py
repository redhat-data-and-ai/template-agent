"""Unit tests for the stdout output sink."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from deep_agent.src.triggers.sinks.protocol import TriggerResult
from deep_agent.src.triggers.sinks.stdout import StdoutSink
from deep_agent.src.triggers.sources.protocol import TriggerEvent


def _make_event(**overrides) -> TriggerEvent:
    defaults = {
        "name": "test-event",
        "payload": {"key": "value"},
        "source": "unit-test",
    }
    defaults.update(overrides)
    return TriggerEvent(**defaults)


def _make_result(**overrides) -> TriggerResult:
    defaults = {
        "event": _make_event(),
        "output": {"answer": 42},
        "duration_ms": 123.4,
        "success": True,
    }
    defaults.update(overrides)
    return TriggerResult(**defaults)


class TestStdoutSink:
    """Test StdoutSink writes JSON to stdout."""

    async def test_emit_writes_json_to_stdout(self):
        sink = StdoutSink()
        result = _make_result()

        mock_stdout = MagicMock()
        with patch("deep_agent.src.triggers.sinks.stdout.sys.stdout", mock_stdout):
            await sink.emit(result)

        mock_stdout.write.assert_called_once()
        mock_stdout.flush.assert_called_once()

        written = mock_stdout.write.call_args[0][0]
        assert written.endswith("\n")

    async def test_emit_output_is_valid_json_with_event_data(self):
        sink = StdoutSink()
        result = _make_result()

        mock_stdout = MagicMock()
        with patch("deep_agent.src.triggers.sinks.stdout.sys.stdout", mock_stdout):
            await sink.emit(result)

        written = mock_stdout.write.call_args[0][0]
        line = written.rstrip("\n")
        parsed = json.loads(line)

        assert parsed["success"] is True
        assert parsed["duration_ms"] == 123.4
        assert parsed["event"]["name"] == "test-event"
        assert parsed["event"]["payload"] == {"key": "value"}
        assert parsed["output"] == {"answer": 42}

    async def test_close_is_noop(self):
        sink = StdoutSink()
        await sink.close()  # must not raise
