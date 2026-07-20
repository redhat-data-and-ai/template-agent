"""Unit tests for the file output sink."""

from __future__ import annotations

import json
from pathlib import Path

from deep_agent.src.triggers.sinks.file import FileSink
from deep_agent.src.triggers.sinks.protocol import TriggerResult
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
        "duration_ms": 100.0,
        "success": True,
    }
    defaults.update(overrides)
    return TriggerResult(**defaults)


class TestFileSink:
    """Test FileSink appends JSONL to a file."""

    async def test_emit_creates_parent_dirs_and_appends_jsonl(self, tmp_path: Path):
        nested = tmp_path / "subdir" / "deep" / "output.jsonl"
        sink = FileSink(str(nested))

        await sink.emit(_make_result())
        await sink.close()

        assert nested.exists()
        lines = nested.read_text().strip().splitlines()
        assert len(lines) == 1

        parsed = json.loads(lines[0])
        assert parsed["success"] is True
        assert parsed["event"]["name"] == "test-event"

    async def test_multiple_emits_append_multiple_lines(self, tmp_path: Path):
        output_file = tmp_path / "output.jsonl"
        sink = FileSink(str(output_file))

        await sink.emit(_make_result(output="first"))
        await sink.emit(_make_result(output="second"))
        await sink.emit(_make_result(output="third"))
        await sink.close()

        lines = output_file.read_text().strip().splitlines()
        assert len(lines) == 3

        outputs = [json.loads(line)["output"] for line in lines]
        assert outputs == ["first", "second", "third"]

    async def test_close_closes_file_handle(self, tmp_path: Path):
        output_file = tmp_path / "output.jsonl"
        sink = FileSink(str(output_file))

        await sink.emit(_make_result())
        assert sink._handle is not None

        await sink.close()
        assert sink._handle is None

    async def test_emit_after_close_reopens_handle(self, tmp_path: Path):
        output_file = tmp_path / "output.jsonl"
        sink = FileSink(str(output_file))

        await sink.emit(_make_result(output="before-close"))
        await sink.close()
        assert sink._handle is None

        await sink.emit(_make_result(output="after-close"))
        assert sink._handle is not None
        await sink.close()

        lines = output_file.read_text().strip().splitlines()
        assert len(lines) == 2

        outputs = [json.loads(line)["output"] for line in lines]
        assert outputs == ["before-close", "after-close"]
