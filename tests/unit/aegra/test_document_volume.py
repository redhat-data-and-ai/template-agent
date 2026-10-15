"""Volume accounting and path safety for the documents endpoint.

The capacity figure is the one number here that must not be wrong:
it is what the UI warns on and what the upload path enforces against.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import deep_agent.aegra.document_routes as dr


class TestCapacityBytes:
    def test_defaults_when_unset(self):
        with patch.dict("os.environ", {}, clear=True):
            assert dr._capacity_bytes() == dr._DEFAULT_CAPACITY_BYTES

    def test_reads_the_override(self):
        with patch.dict("os.environ", {"DOCUMENTS_MAX_BYTES": "1024"}):
            assert dr._capacity_bytes() == 1024

    def test_a_malformed_override_falls_back_rather_than_crashing(self):
        # A typo in an optional limit must not take the endpoint down.
        with patch.dict("os.environ", {"DOCUMENTS_MAX_BYTES": "lots"}):
            assert dr._capacity_bytes() == dr._DEFAULT_CAPACITY_BYTES


class TestUsedBytes:
    def test_zero_for_a_volume_that_is_not_mounted(self, tmp_path):
        assert dr._used_bytes(tmp_path / "missing") == 0

    def test_sums_nested_files(self, tmp_path):
        (tmp_path / "a.txt").write_bytes(b"12345")
        nested = tmp_path / "sub"
        nested.mkdir()
        (nested / "b.txt").write_bytes(b"123")
        assert dr._used_bytes(tmp_path) == 8

    def test_skips_a_file_that_vanishes_mid_walk(self, tmp_path):
        (tmp_path / "a.txt").write_bytes(b"12345")
        real_stat = Path.stat

        def _stat(self, *args, **kwargs):
            if self.name == "a.txt":
                raise OSError("vanished")
            return real_stat(self, *args, **kwargs)

        with patch.object(Path, "stat", _stat):
            assert dr._used_bytes(tmp_path) == 0


class TestSafeTarget:
    def test_resolves_a_plain_name(self, tmp_path):
        assert dr._safe_target(tmp_path, "notes.md").parent == tmp_path.resolve()

    @pytest.mark.parametrize("name", ["../escape.txt", "../../etc/passwd"])
    def test_strips_traversal_to_the_basename(self, tmp_path, name):
        # Path(name).name drops the traversal, so the file lands on
        # the volume rather than outside it.
        assert dr._safe_target(tmp_path, name).parent == tmp_path.resolve()


class TestListDocuments:
    @pytest.mark.asyncio
    async def test_lists_files_with_usage(self, tmp_path):
        (tmp_path / "a.txt").write_bytes(b"12345")
        with (
            patch.object(dr, "authenticated_user_id", new=AsyncMock(return_value="u1")),
            patch.object(dr, "_root", return_value=tmp_path),
        ):
            body = await dr.list_documents(MagicMock())

        assert [f["path"] for f in body["documents"]] == ["a.txt"]
        assert body["usage"]["used_bytes"] == 5

    @pytest.mark.asyncio
    async def test_hides_dotfiles(self, tmp_path):
        (tmp_path / ".keep").write_bytes(b"")
        (tmp_path / "real.txt").write_bytes(b"x")
        with (
            patch.object(dr, "authenticated_user_id", new=AsyncMock(return_value="u1")),
            patch.object(dr, "_root", return_value=tmp_path),
        ):
            body = await dr.list_documents(MagicMock())

        assert [f["path"] for f in body["documents"]] == ["real.txt"]

    @pytest.mark.asyncio
    async def test_empty_when_the_volume_is_not_mounted(self, tmp_path):
        with (
            patch.object(dr, "authenticated_user_id", new=AsyncMock(return_value="u1")),
            patch.object(dr, "_root", return_value=tmp_path / "missing"),
        ):
            body = await dr.list_documents(MagicMock())

        assert body["documents"] == []

    @pytest.mark.asyncio
    async def test_requires_a_developer(self, tmp_path):
        # The volume is shared with no per-user partitioning, and what
        # is on it is what the model reads.
        auth = AsyncMock(side_effect=HTTPException(status_code=403, detail="nope"))
        with (
            patch.object(dr, "authenticated_user_id", new=auth),
            pytest.raises(HTTPException) as exc,
        ):
            await dr.list_documents(MagicMock())
        assert exc.value.status_code == 403
        assert auth.await_args.kwargs["developer_only"] is True


class TestReadDocument:
    @pytest.mark.asyncio
    async def test_reads_a_file(self, tmp_path):
        (tmp_path / "notes.md").write_text("hello")
        with (
            patch.object(dr, "authenticated_user_id", new=AsyncMock(return_value="u1")),
            patch.object(dr, "_root", return_value=tmp_path),
        ):
            body = await dr.read_document("notes.md", MagicMock())
        assert body["content"] == "hello"

    @pytest.mark.asyncio
    async def test_404_for_a_file_that_is_not_there(self, tmp_path):
        with (
            patch.object(dr, "authenticated_user_id", new=AsyncMock(return_value="u1")),
            patch.object(dr, "_root", return_value=tmp_path),
            pytest.raises(HTTPException) as exc,
        ):
            await dr.read_document("missing.md", MagicMock())
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_400_for_a_directory(self, tmp_path):
        (tmp_path / "sub").mkdir()
        with (
            patch.object(dr, "authenticated_user_id", new=AsyncMock(return_value="u1")),
            patch.object(dr, "_root", return_value=tmp_path),
            pytest.raises(HTTPException) as exc,
        ):
            await dr.read_document("sub", MagicMock())
        assert exc.value.status_code == 400


class TestDeleteDocument:
    @pytest.mark.asyncio
    async def test_deletes_and_reports_usage(self, tmp_path):
        (tmp_path / "gone.txt").write_bytes(b"12345")
        with (
            patch.object(dr, "authenticated_user_id", new=AsyncMock(return_value="u1")),
            patch.object(dr, "_root", return_value=tmp_path),
        ):
            body = await dr.delete_document("gone.txt", MagicMock())

        assert body["deleted"] == "gone.txt"
        assert body["usage"]["used_bytes"] == 0
        assert not (tmp_path / "gone.txt").exists()

    @pytest.mark.asyncio
    async def test_404_for_a_file_that_is_not_there(self, tmp_path):
        with (
            patch.object(dr, "authenticated_user_id", new=AsyncMock(return_value="u1")),
            patch.object(dr, "_root", return_value=tmp_path),
            pytest.raises(HTTPException) as exc,
        ):
            await dr.delete_document("missing.txt", MagicMock())
        assert exc.value.status_code == 404


class TestWriteDocument:
    @pytest.mark.asyncio
    async def test_415_for_an_unreadable_type(self, tmp_path):
        # The agent reads these as text; a type it cannot read is a
        # silent no-op rather than a stored file.
        with (
            patch.object(dr, "authenticated_user_id", new=AsyncMock(return_value="u1")),
            patch.object(dr, "_root", return_value=tmp_path),
            pytest.raises(HTTPException) as exc,
        ):
            await dr.write_document(
                "payload.bin", dr.DocumentBody(content="x"), MagicMock()
            )
        assert exc.value.status_code == 415
