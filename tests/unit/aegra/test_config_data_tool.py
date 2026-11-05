"""Tests for the sandboxed config data reader."""

from pathlib import Path

import pytest

from deep_agent.aegra import config_data_tool as tool


def _read(filename: str) -> str:
    return tool.read_config_data.invoke({"filename": filename})


def _use_dirs(
    monkeypatch: pytest.MonkeyPatch, data: Path, classification: Path
) -> None:
    monkeypatch.setattr(tool, "_DATA_DIR", data)
    monkeypatch.setattr(tool, "classification_dir", lambda: classification)


def test_prefers_classification_then_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    classification = data / "classification"
    data.mkdir()
    classification.mkdir()
    (data / "logic.yaml").write_text("root: true\n", encoding="utf-8")
    (classification / "logic.yaml").write_text(
        "classification: true\n", encoding="utf-8"
    )
    (data / "only_root.yaml").write_text("root-only\n", encoding="utf-8")
    (data / "subdir").mkdir()
    _use_dirs(monkeypatch, data, classification)

    assert "classification: true" in _read("logic.yaml")
    assert _read("only_root.yaml") == "root-only\n"
    missing = _read("nope.yaml")
    assert "not found" in missing
    assert "logic.yaml" in missing
    assert "only_root.yaml" in missing
    assert "Only filenames" in _read("../logic.yaml")


def test_missing_directories_list_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_dirs(
        monkeypatch,
        tmp_path / "missing",
        tmp_path / "missing" / "classification",
    )
    assert "(none)" in _read("logic.yaml")


def test_rejects_oversized_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    classification = data / "classification"
    data.mkdir()
    classification.mkdir()
    (classification / "big.yaml").write_text("abcdef", encoding="utf-8")
    _use_dirs(monkeypatch, data, classification)
    monkeypatch.setattr(tool, "_MAX_SIZE_BYTES", 1)
    assert "exceeding" in _read("big.yaml")


def test_rejects_non_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "folder"
    folder.mkdir()
    monkeypatch.setattr(tool, "_resolve_data_file", lambda _name: folder)
    assert "not a regular file" in _read("folder")


def test_reports_read_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "data"
    classification = data / "classification"
    data.mkdir()
    classification.mkdir()
    (classification / "logic.yaml").write_text("ok\n", encoding="utf-8")
    _use_dirs(monkeypatch, data, classification)

    def _boom(self: Path, *args: object, **kwargs: object) -> str:
        raise OSError("disk")

    monkeypatch.setattr(Path, "read_text", _boom)
    assert "Error reading 'logic.yaml'" in _read("logic.yaml")
