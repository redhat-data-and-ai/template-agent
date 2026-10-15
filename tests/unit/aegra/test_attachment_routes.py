"""Unit tests for the thread-attachment endpoints.

These cover the parts that decide whether a file is accepted, named
safely and written to the right thread. The extraction engines
themselves are exercised in ``tests/unit/attachments``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import deep_agent.aegra.attachment_routes as ar


class TestSuffixOrFile:
    def test_returns_the_extension(self):
        assert ar.suffix_or_file("report.PDF") == "pdf"

    def test_falls_back_to_binary_without_one(self):
        # Reported to the user, so "binary" beats an empty string.
        assert ar.suffix_or_file("Makefile") == "binary"


class TestOrchestratorModel:
    def test_reads_the_configured_model(self):
        cfg = MagicMock()
        cfg.get_orchestrator_config.return_value = {"model": "gemini-9-ultra"}
        with patch("deep_agent.src.agent.config.agent_config", cfg):
            assert ar._orchestrator_model() == "gemini-9-ultra"

    def test_unwraps_a_model_mapping(self):
        cfg = MagicMock()
        cfg.get_orchestrator_config.return_value = {"model": {"name": "claude-x"}}
        with patch("deep_agent.src.agent.config.agent_config", cfg):
            assert ar._orchestrator_model() == "claude-x"

    def test_falls_back_when_config_raises(self):
        cfg = MagicMock()
        cfg.get_orchestrator_config.side_effect = RuntimeError("no config")
        with patch("deep_agent.src.agent.config.agent_config", cfg):
            assert ar._orchestrator_model() == "gemini-3.5-flash-lite"


class TestConfigReaders:
    """A broken config block must not take the endpoint down with it."""

    def test_config_reads_the_attachments_section(self):
        cfg = MagicMock()
        cfg.get_raw_section.return_value = {"max_files_per_request": 3}
        with patch("deep_agent.src.agent.config.agent_config", cfg):
            assert ar._config().max_files_per_request == 3

    def test_config_defaults_when_the_section_is_broken(self):
        cfg = MagicMock()
        cfg.get_raw_section.side_effect = ValueError("bad yaml")
        with patch("deep_agent.src.agent.config.agent_config", cfg):
            assert ar._config() == ar.AttachmentsConfig()

    def test_extraction_defaults_when_the_section_is_broken(self):
        cfg = MagicMock()
        cfg.get_raw_section.side_effect = ValueError("bad yaml")
        with patch("deep_agent.src.agent.config.agent_config", cfg):
            assert ar._extraction_config() == ar.ExtractionConfig()


def _upload(chunks: list[bytes], filename: str = "f.txt") -> MagicMock:
    upload = MagicMock()
    upload.filename = filename
    upload.read = AsyncMock(side_effect=[*chunks, b""])
    return upload


class TestReadBounded:
    """content-length is a claim; the bytes read are the fact."""

    @pytest.mark.asyncio
    async def test_reads_everything_under_the_limit(self):
        assert await ar._read_bounded(_upload([b"ab", b"cd"]), 10) == b"abcd"

    @pytest.mark.asyncio
    async def test_413_once_the_limit_is_passed(self):
        with pytest.raises(HTTPException) as exc:
            await ar._read_bounded(_upload([b"x" * 8, b"x" * 8]), 10)
        assert exc.value.status_code == 413

    @pytest.mark.asyncio
    async def test_stops_reading_at_the_limit_rather_than_buffering_it_all(self):
        # The point of chunking: a client declaring 1 KB and sending a
        # gigabyte must not get a gigabyte into memory first.
        upload = _upload([b"x" * 8] * 100)
        with pytest.raises(HTTPException):
            await ar._read_bounded(upload, 10)
        assert upload.read.await_count == 2


def _runs_response(status_code: int, body) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = body
    return response


class TestAwaitIdleThread:
    """A run already executing checkpoints state loaded before the
    attachment write, silently discarding the file."""

    @pytest.mark.asyncio
    async def test_returns_when_no_run_is_active(self):
        client = MagicMock()
        client.get = AsyncMock(
            return_value=_runs_response(200, [{"status": "success"}])
        )
        await ar._await_idle_thread(client, "t1")
        assert client.get.await_count == 1

    @pytest.mark.asyncio
    async def test_accepts_a_wrapped_runs_list(self):
        client = MagicMock()
        client.get = AsyncMock(
            return_value=_runs_response(200, {"runs": [{"status": "error"}]})
        )
        await ar._await_idle_thread(client, "t1")
        assert client.get.await_count == 1

    @pytest.mark.asyncio
    async def test_waits_then_proceeds_once_the_run_finishes(self):
        client = MagicMock()
        client.get = AsyncMock(
            side_effect=[
                _runs_response(200, [{"status": "running"}]),
                _runs_response(200, [{"status": "success"}]),
            ]
        )
        with patch("asyncio.sleep", new=AsyncMock()):
            await ar._await_idle_thread(client, "t1")
        assert client.get.await_count == 2

    @pytest.mark.asyncio
    async def test_proceeds_when_the_run_list_cannot_be_read(self):
        # Cannot tell, so do not block the upload forever.
        client = MagicMock()
        client.get = AsyncMock(side_effect=RuntimeError("socket closed"))
        await ar._await_idle_thread(client, "t1")

    @pytest.mark.asyncio
    async def test_proceeds_on_a_non_200(self):
        client = MagicMock()
        client.get = AsyncMock(return_value=_runs_response(503, {}))
        await ar._await_idle_thread(client, "t1")

    @pytest.mark.asyncio
    async def test_gives_up_on_a_wedged_run_rather_than_hanging(self):
        client = MagicMock()
        client.get = AsyncMock(
            return_value=_runs_response(200, [{"status": "running"}])
        )
        ticks = iter([0.0, 1.0, ar._RUN_WAIT_SECONDS + 1])
        with (
            patch("asyncio.sleep", new=AsyncMock()),
            patch("time.monotonic", side_effect=lambda: next(ticks)),
        ):
            await ar._await_idle_thread(client, "t1")


def _request(auth: str | None = None) -> MagicMock:
    request = MagicMock()
    request.headers = {"Authorization": auth} if auth else {}
    return request


class _FakeClient:
    """Minimal stand-in for the httpx client the routes open."""

    def __init__(self, responses: dict[str, MagicMock]):
        self._responses = responses
        self.requested: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, path: str):
        self.requested.append(path)
        return self._responses[path]


def _state(values: dict) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"values": values}
    return response


class TestExtractionStatus:
    """A misconfigured extractor should be answerable by a readiness
    check rather than discovered at upload time, in front of a user."""

    @pytest.mark.asyncio
    async def test_reports_a_reachable_engine(self):
        with (
            patch.object(ar, "authenticated_user_id", new=AsyncMock(return_value="u1")),
            patch.object(
                ar,
                "_extraction_config",
                return_value=ar.ExtractionConfig(engine="docling", fallback=True),
            ),
            patch(
                "deep_agent.src.attachments.engines.probe",
                new=AsyncMock(return_value=(True, "200 OK")),
            ),
        ):
            body = await ar.extraction_status(_request())

        assert body["engine"] == "docling"
        assert body["reachable"] is True
        assert body["detail"] == "200 OK"
        assert body["handles"]  # a remote engine advertises what it takes

    @pytest.mark.asyncio
    async def test_reports_an_unreachable_engine_without_raising(self):
        with (
            patch.object(ar, "authenticated_user_id", new=AsyncMock(return_value="u1")),
            patch.object(
                ar,
                "_extraction_config",
                return_value=ar.ExtractionConfig(engine="tika"),
            ),
            patch(
                "deep_agent.src.attachments.engines.probe",
                new=AsyncMock(return_value=(False, "connection refused")),
            ),
        ):
            body = await ar.extraction_status(_request())

        assert body["reachable"] is False
        assert body["detail"] == "connection refused"

    @pytest.mark.asyncio
    async def test_builtin_advertises_no_suffix_list(self):
        # "handles" describes what a *remote* engine takes; the
        # built-in parsers are asked per file.
        with (
            patch.object(ar, "authenticated_user_id", new=AsyncMock(return_value="u1")),
            patch.object(ar, "_extraction_config", return_value=ar.ExtractionConfig()),
            patch(
                "deep_agent.src.attachments.engines.probe",
                new=AsyncMock(return_value=(True, "")),
            ),
        ):
            body = await ar.extraction_status(_request())

        assert body["engine"] == "builtin"
        assert body["handles"] == []


class TestListAttachments:
    @pytest.mark.asyncio
    async def test_lists_only_files_under_the_mount_path(self):
        config = ar.AttachmentsConfig()
        prefix = config.mount_path.rstrip("/") + "/"
        client = _FakeClient(
            {
                "/threads/t1/state": _state(
                    {
                        "files": {
                            f"{prefix}notes.txt": "hello",
                            "/elsewhere/secret.txt": "not an attachment",
                        }
                    }
                )
            }
        )
        with (
            patch.object(ar, "authenticated_user_id", new=AsyncMock(return_value="u1")),
            patch.object(ar, "_config", return_value=config),
            patch("httpx.AsyncClient", return_value=client),
        ):
            body = await ar.list_attachments("t1", _request("Bearer tok"))

        assert [a["filename"] for a in body["attachments"]] == ["notes.txt"]
        assert body["attachments"][0]["chars"] == len("hello")

    @pytest.mark.asyncio
    async def test_counts_characters_of_a_dict_shaped_entry(self):
        config = ar.AttachmentsConfig()
        prefix = config.mount_path.rstrip("/") + "/"
        client = _FakeClient(
            {
                "/threads/t1/state": _state(
                    {"files": {f"{prefix}a.md": {"content": "abc"}}}
                )
            }
        )
        with (
            patch.object(ar, "authenticated_user_id", new=AsyncMock(return_value="u1")),
            patch.object(ar, "_config", return_value=config),
            patch("httpx.AsyncClient", return_value=client),
        ):
            body = await ar.list_attachments("t1", _request())

        assert body["attachments"][0]["chars"] == 3

    @pytest.mark.asyncio
    async def test_empty_when_the_thread_has_no_files(self):
        client = _FakeClient({"/threads/t1/state": _state({})})
        with (
            patch.object(ar, "authenticated_user_id", new=AsyncMock(return_value="u1")),
            patch.object(ar, "_config", return_value=ar.AttachmentsConfig()),
            patch("httpx.AsyncClient", return_value=client),
        ):
            body = await ar.list_attachments("t1", _request())

        assert body["attachments"] == []

    @pytest.mark.asyncio
    async def test_404_when_the_thread_is_not_there(self):
        missing = MagicMock()
        missing.status_code = 404
        client = _FakeClient({"/threads/t1/state": missing})
        with (
            patch.object(ar, "authenticated_user_id", new=AsyncMock(return_value="u1")),
            patch.object(ar, "_config", return_value=ar.AttachmentsConfig()),
            patch("httpx.AsyncClient", return_value=client),
            pytest.raises(HTTPException) as exc,
        ):
            await ar.list_attachments("t1", _request())
        assert exc.value.status_code == 404


class TestUploadGuards:
    @pytest.mark.asyncio
    async def test_404_when_attachments_are_disabled(self):
        with (
            patch.object(ar, "authenticated_user_id", new=AsyncMock(return_value="u1")),
            patch.object(
                ar, "_config", return_value=ar.AttachmentsConfig(enabled=False)
            ),
            patch.object(ar, "_extraction_config", return_value=ar.ExtractionConfig()),
            pytest.raises(HTTPException) as exc,
        ):
            await ar.upload_attachments("t1", _request(), files=[_upload([b"x"])])
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_400_over_the_per_request_file_count(self):
        with (
            patch.object(ar, "authenticated_user_id", new=AsyncMock(return_value="u1")),
            patch.object(
                ar,
                "_config",
                return_value=ar.AttachmentsConfig(max_files_per_request=1),
            ),
            patch.object(ar, "_extraction_config", return_value=ar.ExtractionConfig()),
            pytest.raises(HTTPException) as exc,
        ):
            await ar.upload_attachments(
                "t1", _request(), files=[_upload([b"a"]), _upload([b"b"])]
            )
        assert exc.value.status_code == 400
