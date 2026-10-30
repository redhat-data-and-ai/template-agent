"""Unit tests for the PVC data ingestion endpoint."""

import hashlib
import io
import json
import os
import stat
import tarfile
import threading
import time
import zipfile
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from deep_agent.aegra import upload_routes
from deep_agent.aegra.upload_routes import LIVE_DIR, upload_router


@pytest.fixture
def data_root(tmp_path):
    """Writable DATA_PATH for the duration of a test."""
    root = tmp_path / "data"
    root.mkdir()
    with patch.object(upload_routes.settings, "DATA_PATH", str(root)):
        yield root


@pytest.fixture
def client(data_root):
    """TestClient with the feature enabled and auth stubbed out."""
    app = FastAPI()
    app.include_router(upload_router)
    with (
        patch.object(upload_routes.settings, "UPLOAD_ENABLED", True),
        patch.object(
            upload_routes,
            "check_ldap_role",
            new_callable=AsyncMock,
            return_value="dev-user",
        ),
    ):
        yield TestClient(app)


def _zip(entries, symlinks=None):
    """Build an in-memory zip. entries maps name -> content."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in entries.items():
            zf.writestr(name, content)
        for name, target in (symlinks or {}).items():
            info = zipfile.ZipInfo(name)
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            zf.writestr(info, target)
    return buf.getvalue()


def _tar(entries):
    """Build an in-memory tar. entries maps name -> content."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, content in entries.items():
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _live(data_root):
    """Return {relative path: content} for the published tree."""
    live = data_root / LIVE_DIR
    if not live.is_dir():
        return {}
    return {
        str(p.relative_to(live)): p.read_text() for p in live.rglob("*") if p.is_file()
    }


def _scratch(data_root):
    """Any staging/old directories left behind."""
    return [p.name for p in data_root.iterdir() if p.name.startswith(".")]


class TestGating:
    def test_disabled_returns_404(self, data_root):
        app = FastAPI()
        app.include_router(upload_router)
        with patch.object(upload_routes.settings, "UPLOAD_ENABLED", False):
            res = TestClient(app).post(
                "/upload", files={"files": ("a.md", b"x", "text/markdown")}
            )
        assert res.status_code == 404

    def test_auth_failure_propagates(self, data_root):
        from fastapi import HTTPException

        app = FastAPI()
        app.include_router(upload_router)
        with (
            patch.object(upload_routes.settings, "UPLOAD_ENABLED", True),
            patch.object(
                upload_routes,
                "check_ldap_role",
                new_callable=AsyncMock,
                side_effect=HTTPException(status_code=403, detail="nope"),
            ),
        ):
            res = TestClient(app).post(
                "/upload", files={"files": ("a.md", b"x", "text/markdown")}
            )
        assert res.status_code == 403


class TestArchiveReplace:
    def test_zip_publishes_tree(self, client, data_root):
        payload = _zip({"docs/a.md": "A", "docs/b.txt": "B"})
        res = client.post("/upload", files={"files": ("x.zip", payload)})

        assert res.status_code == 200, res.text
        body = res.json()
        assert body["mode"] == "replace"
        assert body["files_written"] == 2
        assert _live(data_root) == {"docs/a.md": "A", "docs/b.txt": "B"}
        assert _scratch(data_root) == []

    def test_tar_publishes_tree(self, client, data_root):
        res = client.post("/upload", files={"files": ("x.tar", _tar({"a.md": "A"}))})

        assert res.status_code == 200, res.text
        assert res.json()["mode"] == "replace"
        assert _live(data_root) == {"a.md": "A"}

    def test_second_archive_replaces_wholesale(self, client, data_root):
        client.post("/upload", files={"files": ("1.zip", _zip({"old.md": "O"}))})
        client.post("/upload", files={"files": ("2.zip", _zip({"new.md": "N"}))})

        assert _live(data_root) == {"new.md": "N"}

    def test_top_level_directory_is_preserved(self, client, data_root):
        """No --strip-components behaviour: the caller's folder name survives."""
        payload = _zip({"llm-foundry/GUIDANCE.md": "G"})
        client.post("/upload", files={"files": ("x.zip", payload)})

        assert _live(data_root) == {"llm-foundry/GUIDANCE.md": "G"}

    def test_archive_named_zip_that_is_not_one_is_rejected(self, client, data_root):
        res = client.post("/upload", files={"files": ("x.zip", b"not an archive")})

        assert res.status_code == 400
        assert "not a valid" in res.json()["detail"]
        assert _live(data_root) == {}


class TestLooseFileMerge:
    def test_single_file_merges_without_disturbing_tree(self, client, data_root):
        client.post("/upload", files={"files": ("x.zip", _zip({"keep.md": "K"}))})

        res = client.post("/upload", files={"files": ("add.md", b"A")})

        assert res.status_code == 200, res.text
        assert res.json()["mode"] == "merge"
        assert _live(data_root) == {"keep.md": "K", "add.md": "A"}

    def test_multiple_files_in_one_request(self, client, data_root):
        res = client.post(
            "/upload",
            files=[("files", ("a.md", b"A")), ("files", ("b.md", b"B"))],
        )

        assert res.status_code == 200, res.text
        assert res.json()["files_written"] == 2
        assert _live(data_root) == {"a.md": "A", "b.md": "B"}

    def test_same_name_overwrites_in_place(self, client, data_root):
        client.post("/upload", files={"files": ("x.zip", _zip({"a.md": "OLD"}))})
        client.post("/upload", files={"files": ("a.md", b"NEW")})

        assert _live(data_root) == {"a.md": "NEW"}

    def test_overwrite_does_not_corrupt_via_hardlink(self, client, data_root):
        """Merge stages hardlinks; overwriting must unlink, not write through."""
        client.post(
            "/upload", files={"files": ("x.zip", _zip({"a.md": "OLD", "b.md": "B"}))}
        )
        before = (data_root / LIVE_DIR / "b.md").stat().st_ino

        client.post("/upload", files={"files": ("a.md", b"NEW")})

        assert _live(data_root) == {"a.md": "NEW", "b.md": "B"}
        # Untouched file keeps its inode — the copy really was by link.
        assert (data_root / LIVE_DIR / "b.md").stat().st_ino == before

    def test_archive_alongside_loose_files_is_rejected(self, client, data_root):
        """Mixing a snapshot and an incremental update has no coherent meaning."""
        client.post("/upload", files={"files": ("x.zip", _zip({"keep.md": "K"}))})

        res = client.post(
            "/upload",
            files=[
                ("files", ("a.md", b"A")),
                ("files", ("proj.zip", _zip({"inner.md": "I"}))),
            ],
        )

        assert res.status_code == 400
        assert "must be uploaded on its own" in res.json()["detail"]
        # Nothing partially applied: a.md did not land either.
        assert _live(data_root) == {"keep.md": "K"}

    def test_tar_alongside_loose_files_is_rejected(self, client, data_root):
        res = client.post(
            "/upload",
            files=[
                ("files", ("a.md", b"A")),
                ("files", ("proj.tar", _tar({"inner.md": "I"}))),
            ],
        )

        assert res.status_code == 400
        assert "must be uploaded on its own" in res.json()["detail"]

    def test_two_archives_rejected(self, client, data_root):
        payload = _zip({"a.md": "A"})
        res = client.post(
            "/upload",
            files=[("files", ("one.zip", payload)), ("files", ("two.zip", payload))],
        )

        assert res.status_code == 400
        assert "must be uploaded on its own" in res.json()["detail"]

    def test_detection_is_by_content_not_name(self, client, data_root):
        """A real archive named .md is still caught; a .md named .zip is not here."""
        res = client.post(
            "/upload",
            files=[
                ("files", ("a.md", b"A")),
                ("files", ("sneaky.md", _zip({"inner.md": "I"}))),
            ],
        )

        assert res.status_code == 400
        assert "must be uploaded on its own" in res.json()["detail"]

    def test_nested_filename_is_honoured(self, client, data_root):
        res = client.post("/upload", files={"files": ("docs/deep/a.md", b"A")})

        assert res.status_code == 200, res.text
        assert _live(data_root) == {"docs/deep/a.md": "A"}


class TestPathTraversal:
    def test_zip_slip_rejected(self, client, data_root):
        payload = _zip({"../../etc/passwd": "bad", "ok.md": "A"})
        res = client.post("/upload", files={"files": ("x.zip", payload)})

        assert res.status_code == 400
        assert "traversal" in res.json()["detail"]
        assert _live(data_root) == {}

    def test_loose_file_traversal_rejected(self, client, data_root):
        res = client.post(
            "/upload", files={"files": ("../../app/config/agent/PROMPT.md", b"bad")}
        )

        assert res.status_code == 400
        assert "traversal" in res.json()["detail"]

    def test_absolute_path_rejected(self, client, data_root):
        res = client.post(
            "/upload", files={"files": ("x.zip", _zip({"/etc/x.md": "b"}))}
        )

        assert res.status_code == 400

    def test_directory_entries_are_skipped_not_rejected(self, client, data_root):
        """git archive emits an entry per subdirectory; these must not 400."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("docs/", "")  # directory entry, name ends in "/"
            zf.writestr("docs/a.md", "A")
        res = client.post("/upload", files={"files": ("x.zip", buf.getvalue())})

        assert res.status_code == 200, res.text
        assert res.json()["files_written"] == 1
        assert _live(data_root) == {"docs/a.md": "A"}

    def test_directories_are_not_counted_as_skipped_files(self, client, data_root):
        """A repo archive holds hundreds of dir entries; counting them as
        skipped files reads as data loss."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for d in ("a/", "a/b/", "a/b/c/", "docs/"):
                zf.writestr(d, "")
            zf.writestr("a/b/c/deep.md", "D")
            # Archives are unfiltered, so the one genuinely skipped entry
            # has to be a non-regular member rather than a bad extension.
            info = zipfile.ZipInfo("link.md")
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            zf.writestr(info, "/etc/passwd")

        res = client.post("/upload", files={"files": ("x.zip", buf.getvalue())})

        assert res.status_code == 200, res.text
        body = res.json()
        assert body["files_written"] == 1
        # 4 directories ignored entirely; only the symlink counts as skipped.
        assert body["files_skipped"] == 1
        assert _live(data_root) == {"a/b/c/deep.md": "D"}

    def test_control_characters_rejected(self, client, data_root):
        """Archive parsers truncate at NUL, but a multipart filename arrives intact."""
        for bad in ("evil\x00.md", "evil\n.md", "evil\r.md"):
            res = client.post("/upload", files={"files": (bad, b"x")})
            assert res.status_code == 400, f"{bad!r} -> {res.status_code}"

    def test_percent_encoded_traversal_rejected(self, client, data_root):
        res = client.post("/upload", files={"files": ("%2e%2e/%2e%2e/tmp/x.md", b"x")})

        assert res.status_code == 400
        assert "percent-encoded" in res.json()["detail"]

    def test_backslash_path_rejected(self, client, data_root):
        res = client.post("/upload", files={"files": ("..\\..\\tmp\\win.md", b"x")})

        assert res.status_code == 400
        assert "backslash" in res.json()["detail"]

    def test_overlong_name_is_400_not_500(self, client, data_root):
        """ENAMETOOLONG would be a 500 that also leaks the staging path."""
        res = client.post("/upload", files={"files": ("a" * 300 + ".md", b"x")})

        assert res.status_code == 400
        detail = res.json()["detail"]
        assert "exceeds" in detail
        assert ".staging-" not in detail and "/var/" not in detail

    def test_server_errors_do_not_leak_internal_paths(self, client, data_root):
        """A 500 must not hand filesystem layout back to the caller."""
        with patch.object(
            upload_routes, "_publish", side_effect=OSError("boom /secret/staging/path")
        ):
            res = client.post(
                "/upload", files={"files": ("x.zip", _zip({"a.md": "A"}))}
            )

        assert res.status_code == 500
        assert res.json()["detail"] == "Failed to publish upload"
        assert "/secret/" not in res.text

    def test_symlink_entry_is_skipped(self, client, data_root):
        payload = _zip({"ok.md": "A"}, symlinks={"link.md": "/etc/passwd"})
        res = client.post("/upload", files={"files": ("x.zip", payload)})

        assert res.status_code == 200, res.text
        assert res.json()["files_skipped"] == 1
        assert _live(data_root) == {"ok.md": "A"}


class TestCaps:
    def test_oversize_single_file_rejected(self, client, data_root):
        with patch.object(upload_routes.settings, "UPLOAD_MAX_FILE_BYTES", 10):
            res = client.post(
                "/upload", files={"files": ("x.zip", _zip({"a.md": "x" * 50}))}
            )

        assert res.status_code == 400
        assert "per-file cap" in res.json()["detail"]
        assert _live(data_root) == {}

    def test_cumulative_cap_rejected(self, client, data_root):
        payload = _zip({"a.md": "x" * 40, "b.md": "y" * 40})
        with patch.object(upload_routes.settings, "UPLOAD_MAX_EXTRACTED_BYTES", 50):
            res = client.post("/upload", files={"files": ("x.zip", payload)})

        assert res.status_code == 400
        assert "total size" in res.json()["detail"]

    def test_merge_cap_counts_existing_tree(self, client, data_root):
        """Repeated merges must not grow the tree past the cap 5MB at a time."""
        client.post("/upload", files={"files": ("x.zip", _zip({"a.md": "x" * 40}))})

        with patch.object(upload_routes.settings, "UPLOAD_MAX_EXTRACTED_BYTES", 50):
            res = client.post("/upload", files={"files": ("b.md", b"y" * 40)})

        assert res.status_code == 400
        assert "total size" in res.json()["detail"]
        # The pre-existing tree survived the rejection.
        assert _live(data_root) == {"a.md": "x" * 40}

    def test_merge_overwrite_does_not_double_count(self, client, data_root):
        """Replacing a file in place costs only the difference, not both copies."""
        client.post("/upload", files={"files": ("x.zip", _zip({"f.md": "a" * 100}))})

        with patch.object(upload_routes.settings, "UPLOAD_MAX_EXTRACTED_BYTES", 150):
            res = client.post("/upload", files={"files": ("f.md", b"b" * 100)})

        # Resulting tree is still 100 bytes, well under the cap.
        assert res.status_code == 200, res.text
        assert _live(data_root) == {"f.md": "b" * 100}

    def test_merge_growth_past_cap_still_rejected(self, client, data_root):
        """The relaxed accounting must not let the tree grow past the cap."""
        client.post("/upload", files={"files": ("x.zip", _zip({"f.md": "a" * 100}))})

        with patch.object(upload_routes.settings, "UPLOAD_MAX_EXTRACTED_BYTES", 150):
            res = client.post("/upload", files={"files": ("g.md", b"c" * 100)})

        assert res.status_code == 400
        assert "total size" in res.json()["detail"]
        assert _live(data_root) == {"f.md": "a" * 100}

    def test_duplicate_archive_entries_counted_once(self, client, data_root):
        """A zip carrying the same name twice must not double-count it."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("f.md", "a" * 100)
            zf.writestr("f.md", "b" * 100)
        with patch.object(upload_routes.settings, "UPLOAD_MAX_EXTRACTED_BYTES", 150):
            res = client.post("/upload", files={"files": ("x.zip", buf.getvalue())})

        assert res.status_code == 200, res.text
        assert _live(data_root) == {"f.md": "b" * 100}

    def test_file_count_cap_counts_skipped_entries(self, client, data_root):
        payload = _zip({"a.png": "1", "b.png": "2", "c.md": "3"})
        with patch.object(upload_routes.settings, "UPLOAD_MAX_FILE_COUNT", 2):
            res = client.post("/upload", files={"files": ("x.zip", payload)})

        assert res.status_code == 400
        assert "entries" in res.json()["detail"]

    def test_tar_headers_are_read_lazily(self, client, data_root):
        """A compressed tar of many empty entries must not be read whole.

        `getmembers()` loads every header before the count cap is checked,
        so a ~1MB gzip tar of 300k entries costs ~131MB of RSS. Iterating
        the TarFile yields one header at a time, so the cap stops the read
        a few entries in.
        """
        import tarfile as _tarfile

        buf = io.BytesIO()
        with _tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for i in range(20_000):
                info = _tarfile.TarInfo(f"d{i // 1000}/f{i}.md")
                info.size = 0
                tf.addfile(info)

        seen = []
        real_next = _tarfile.TarFile.next

        def counting_next(self):
            member = real_next(self)
            if member is not None:
                seen.append(member)
            return member

        with patch.object(_tarfile.TarFile, "next", counting_next):
            res = client.post(
                "/upload", files={"files": ("bomb.tar.gz", buf.getvalue())}
            )

        assert res.status_code == 400
        assert "exceeds" in res.json()["detail"]
        # Stopped at the cap rather than walking all 20k headers.
        assert len(seen) <= upload_routes.settings.UPLOAD_MAX_FILE_COUNT + 5, (
            f"read {len(seen)} headers; the count cap should have stopped it"
        )

    def test_body_over_archive_cap_rejected(self, client, data_root):
        with patch.object(upload_routes.settings, "UPLOAD_MAX_ARCHIVE_BYTES", 10):
            res = client.post("/upload", files={"files": ("a.md", b"x" * 100)})

        assert res.status_code == 413

    def test_archive_cap_defaults_to_10mb(self):
        """The uploaded .zip/.tar itself is capped at 10MB."""
        from deep_agent.src.settings import Settings

        assert Settings().UPLOAD_MAX_ARCHIVE_BYTES == 10 * 1024 * 1024

    def test_archive_at_exactly_the_cap_is_accepted(self, client, data_root):
        """The cap is inclusive: <= 10MB passes, one byte more does not."""
        payload = _zip({"a.md": "x" * 40})
        with patch.object(
            upload_routes.settings, "UPLOAD_MAX_ARCHIVE_BYTES", len(payload)
        ):
            ok = client.post("/upload", files={"files": ("x.zip", payload)})
        assert ok.status_code == 200, ok.text

        with patch.object(
            upload_routes.settings, "UPLOAD_MAX_ARCHIVE_BYTES", len(payload) - 1
        ):
            too_big = client.post("/upload", files={"files": ("x.zip", payload)})
        assert too_big.status_code == 413


class TestExtensionAllowlist:
    """The allowlist applies to loose files only.

    An archive is a snapshot the caller deliberately assembled, so it is
    taken whole -- filtering it would silently deliver less than they sent.
    Loose files are ad-hoc, so the allowlist guards against stray binaries.
    """

    def test_archive_is_taken_whole_without_filtering(self, client, data_root):
        payload = _zip(
            {
                "a.md": "A",
                "logo.png": "binary",
                "run.exe": "x",
                "Makefile": "no extension",
                ".gitignore": "dotfile",
            }
        )
        res = client.post("/upload", files={"files": ("x.zip", payload)})

        assert res.status_code == 200, res.text
        body = res.json()
        assert body["files_written"] == 5
        assert body["files_skipped"] == 0
        assert set(_live(data_root)) == {
            "a.md",
            "logo.png",
            "run.exe",
            "Makefile",
            ".gitignore",
        }

    def test_tar_is_also_taken_whole(self, client, data_root):
        res = client.post(
            "/upload",
            files={"files": ("x.tar", _tar({"a.md": "A", "logo.png": "bin"}))},
        )

        assert res.status_code == 200, res.text
        assert res.json()["files_written"] == 2
        assert set(_live(data_root)) == {"a.md", "logo.png"}

    def test_loose_file_disallowed_extension_is_skipped_not_fatal(
        self, client, data_root
    ):
        res = client.post(
            "/upload",
            files=[
                ("files", ("a.md", b"A")),
                ("files", ("logo.png", b"binary")),
                ("files", ("run.exe", b"x")),
            ],
        )

        assert res.status_code == 200, res.text
        body = res.json()
        assert body["files_written"] == 1
        assert body["files_skipped"] == 2
        assert _live(data_root) == {"a.md": "A"}

    def test_loose_dotfiles_are_treated_as_extensionless(self, client, data_root):
        res = client.post(
            "/upload",
            files=[("files", (".gitignore", b"x")), ("files", ("a.md", b"A"))],
        )

        assert res.json()["files_skipped"] == 1
        assert _live(data_root) == {"a.md": "A"}

    def test_extension_match_is_case_insensitive(self, client, data_root):
        """Loose files only -- archives are unfiltered either way."""
        res = client.post("/upload", files={"files": ("GUIDANCE.MD", b"G")})

        assert res.status_code == 200, res.text
        assert _live(data_root) == {"GUIDANCE.MD": "G"}

    def test_zero_permitted_loose_files_rejected(self, client, data_root):
        """A loose upload filtered to nothing must not blank a good tree."""
        client.post("/upload", files={"files": ("x.zip", _zip({"keep.md": "K"}))})

        res = client.post("/upload", files={"files": ("a.png", b"x")})

        assert res.status_code == 400
        assert "no permitted files" in res.json()["detail"]
        assert _live(data_root) == {"keep.md": "K"}

    def test_empty_archive_rejected(self, client, data_root):
        """Archives are unfiltered, so only a genuinely empty one is refused."""
        client.post("/upload", files={"files": ("x.zip", _zip({"keep.md": "K"}))})

        res = client.post("/upload", files={"files": ("y.zip", _zip({}))})

        assert res.status_code == 400
        assert _live(data_root) == {"keep.md": "K"}


class TestReupload:
    def test_identical_archive_is_idempotent(self, client, data_root):
        payload = _zip({"proj/a.md": "A", "proj/b.md": "B"})
        first = client.post("/upload", files={"files": ("x.zip", payload)}).json()
        before = _live(data_root)

        second = client.post("/upload", files={"files": ("x.zip", payload)}).json()

        assert second == first
        assert _live(data_root) == before
        assert _scratch(data_root) == []

    def test_replace_drops_files_absent_from_new_archive(self, client, data_root):
        client.post(
            "/upload",
            files={"files": ("a.zip", _zip({"keep.md": "v1", "only-in-a.md": "x"}))},
        )

        client.post(
            "/upload",
            files={"files": ("b.zip", _zip({"keep.md": "v2", "only-in-b.md": "y"}))},
        )

        assert _live(data_root) == {"keep.md": "v2", "only-in-b.md": "y"}

    def test_failed_merge_leaves_hardlinked_originals_untouched(
        self, client, data_root
    ):
        """Staging is a hardlink copy; cleaning it up must not touch the tree."""
        client.post("/upload", files={"files": ("x.zip", _zip({"keep.md": "v1"}))})
        live_file = data_root / LIVE_DIR / "keep.md"
        inode_before = live_file.stat().st_ino

        res = client.post("/upload", files={"files": ("../escape.md", b"bad")})

        assert res.status_code == 400
        assert live_file.read_text() == "v1"
        assert live_file.stat().st_ino == inode_before
        assert _scratch(data_root) == []

    def test_repeated_merges_accumulate(self, client, data_root):
        client.post("/upload", files={"files": ("x.zip", _zip({"base.md": "B"}))})
        client.post("/upload", files={"files": ("one.md", b"1")})
        client.post("/upload", files={"files": ("two.md", b"2")})

        assert _live(data_root) == {"base.md": "B", "one.md": "1", "two.md": "2"}
        assert _scratch(data_root) == []

    def test_archive_after_merges_replaces_everything(self, client, data_root):
        client.post("/upload", files={"files": ("x.zip", _zip({"base.md": "B"}))})
        client.post("/upload", files={"files": ("extra.md", b"E")})

        client.post("/upload", files={"files": ("y.zip", _zip({"fresh.md": "F"}))})

        assert _live(data_root) == {"fresh.md": "F"}


class TestMalformedArchives:
    """A damaged archive is a bad request, not a server fault.

    Sniffing magic bytes only proves the header is plausible, so the real
    failure surfaces part-way through the read as BadZipFile, TarError,
    EOFError, zlib.error and friends -- none of them OSError. Unhandled,
    they become a 500 with no audit trail.
    """

    @pytest.fixture
    def audit(self):
        with patch.object(upload_routes, "emit_audit_event") as spy:
            yield spy

    def _corrupt_zip(self):
        """A zip whose directory is valid but whose payload is garbage.

        Corrupt only the compressed bytes, located from the local header
        rather than guessed at a fixed offset: `is_zipfile` must still
        return True (it reads the end-of-central-directory record) so the
        failure lands during the read, which is the case under test. A
        blind smash at a fixed offset damages headers on some Python
        versions and the file is then rejected earlier, for a different
        reason.
        """
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("a.md", "hello world" * 200)
        raw = bytearray(buf.getvalue())

        with zipfile.ZipFile(io.BytesIO(bytes(raw))) as zf:
            info = zf.infolist()[0]
        # Local header: 30 fixed bytes, then the name and extra fields.
        start = info.header_offset + 30 + len(info.filename) + len(info.extra)
        end = start + info.compress_size
        raw[start:end] = b"\x00" * (end - start)
        return bytes(raw)

    def _truncated_targz(self):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            info = tarfile.TarInfo("a.md")
            data = b"x" * 5000
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        return buf.getvalue()[: len(buf.getvalue()) // 2]

    def _truncated_tar(self):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            info = tarfile.TarInfo("a.md")
            data = b"x" * 9000
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        return buf.getvalue()[:2048]

    def test_corrupt_zip_is_400_not_500(self, client, data_root, audit):
        res = client.post("/upload", files={"files": ("x.zip", self._corrupt_zip())})

        assert res.status_code == 400
        assert "could not be read" in res.json()["detail"]
        assert audit.call_args[0][0] == "data_upload_rejected"

    def test_truncated_gzip_tar_is_400_not_500(self, client, data_root, audit):
        """is_tarfile() itself raises EOFError here, before the read begins."""
        res = client.post(
            "/upload", files={"files": ("x.tar.gz", self._truncated_targz())}
        )

        assert res.status_code == 400
        assert "could not be read" in res.json()["detail"]
        assert audit.call_args[0][0] == "data_upload_rejected"

    def test_truncated_tar_is_400_not_500(self, client, data_root, audit):
        res = client.post("/upload", files={"files": ("x.tar", self._truncated_tar())})

        assert res.status_code == 400
        assert audit.call_args[0][0] == "data_upload_rejected"

    def test_a_damaged_archive_leaves_the_previous_tree_intact(self, client, data_root):
        client.post("/upload", files={"files": ("good.zip", _zip({"keep.md": "K"}))})

        res = client.post("/upload", files={"files": ("x.zip", self._corrupt_zip())})

        assert res.status_code == 400
        assert _live(data_root) == {"keep.md": "K"}
        assert _scratch(data_root) == []


class TestLocking:
    def test_lock_released_after_success(self, client, data_root):
        res = client.post("/upload", files={"files": ("x.zip", _zip({"a.md": "A"}))})

        assert res.status_code == 200, res.text
        assert not (data_root / upload_routes._LOCK_NAME).exists()

    def test_lock_released_after_rejection(self, client, data_root):
        """A failed upload must not leave ingestion wedged."""
        res = client.post("/upload", files={"files": ("x.zip", b"not an archive")})

        assert res.status_code == 400
        assert not (data_root / upload_routes._LOCK_NAME).exists()
        # And the next upload still works.
        ok = client.post("/upload", files={"files": ("y.zip", _zip({"a.md": "A"}))})
        assert ok.status_code == 200, ok.text

    def test_held_lock_returns_409(self, client, data_root):
        (data_root / upload_routes._LOCK_NAME).write_text("99999\n")

        res = client.post("/upload", files={"files": ("x.zip", _zip({"a.md": "A"}))})

        assert res.status_code == 409
        assert "another upload is in progress" in res.json()["detail"]

    def test_stale_lock_is_broken(self, client, data_root):
        """A lock orphaned by a killed pod must not wedge uploads forever."""
        lock = data_root / upload_routes._LOCK_NAME
        lock.write_text("99999\n")
        stale = time.time() - 10_000
        os.utime(lock, (stale, stale))

        res = client.post("/upload", files={"files": ("x.zip", _zip({"a.md": "A"}))})

        assert res.status_code == 200, res.text
        assert _live(data_root) == {"a.md": "A"}
        assert not lock.exists()

    def test_concurrent_uploads_never_collide(self, client, data_root):
        """One wins, the other gets 409 — never a 500 or a leaked directory."""
        payload_a = _zip({"a.md": "A" * 500})
        payload_b = _zip({"b.md": "B" * 500})
        results = []

        def push(name, payload):
            results.append(
                client.post("/upload", files={"files": (name, payload)}).status_code
            )

        threads = [
            threading.Thread(target=push, args=("a.zip", payload_a)),
            threading.Thread(target=push, args=("b.zip", payload_b)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sorted(results) in ([200, 200], [200, 409])
        assert 500 not in results
        assert _scratch(data_root) == []
        # Whichever won, the tree is exactly one upload's content.
        assert _live(data_root) in ({"a.md": "A" * 500}, {"b.md": "B" * 500})

    def test_stale_lock_loser_does_not_also_acquire(self, data_root):
        """Of several callers breaking the same stale lock, only one wins.

        Deterministic rather than raced: os.rename is made to fail exactly
        as it would if another caller had stolen the entry a moment
        earlier. Breaking the lock by unlinking the path cannot express
        that failure -- the unlink simply succeeds and both callers
        proceed, which is the bug.
        """
        lock = data_root / upload_routes._LOCK_NAME
        lock.write_text("99999\n")
        stale = time.time() - 10_000
        os.utime(lock, (stale, stale))

        real_rename = os.rename

        def someone_else_got_there_first(src, dst):
            if str(src) == str(lock):
                raise FileNotFoundError(2, "No such file or directory", str(src))
            return real_rename(src, dst)

        with patch.object(upload_routes.os, "rename", someone_else_got_there_first):
            with pytest.raises(upload_routes.UploadBusy):
                with upload_routes._upload_lock(data_root):
                    pytest.fail("acquired a lock another caller had already stolen")

    def test_stealing_a_lock_replaced_since_the_age_check(self, data_root):
        """rename() takes whatever sits at the path, not what we judged stale.

        If a faster breaker already swapped in its own fresh lock, the
        rename grabs that one and both uploads proceed. Verify the inode
        is checked after the rename, the live lock is put back, and we
        stand down.
        """
        lock = data_root / upload_routes._LOCK_NAME
        lock.write_text("stale-token\n1\n")
        stale = time.time() - 10_000
        os.utime(lock, (stale, stale))

        real_rename = os.rename

        def swap_in_a_fresh_lock_first(src, dst):
            if str(src) == str(lock):
                # Another breaker got here between our stat and our rename.
                # It renames the stale entry aside, exactly as the real
                # code does, so the stale inode stays allocated and the
                # replacement is guaranteed a different one. Unlinking
                # instead would let ext4 recycle the number and the test
                # would depend on the filesystem.
                real_rename(lock, data_root / "taken-by-other-breaker")
                lock.write_text("fresh-holder-token\n2\n")
            return real_rename(src, dst)

        with patch.object(upload_routes.os, "rename", swap_in_a_fresh_lock_first):
            with pytest.raises(upload_routes.UploadBusy):
                with upload_routes._upload_lock(data_root):
                    pytest.fail("acquired alongside the live holder")

        # The live holder's lock was restored, not destroyed.
        assert lock.exists(), "the live holder's lock was consumed"
        assert lock.read_text().startswith("fresh-holder-token")
        assert [p.name for p in data_root.iterdir() if ".stale-" in p.name] == []

    def test_release_does_not_delete_a_successors_lock(self, data_root):
        """If our lock was stolen as stale, releasing must leave the new one.

        Identity is the token written into the file, not the inode: ext4
        recycles inode numbers, so a lock unlinked and recreated can reuse
        the same st_ino and a successor's lock would look like our own.
        """
        lock = data_root / upload_routes._LOCK_NAME

        with upload_routes._upload_lock(data_root):
            # Simulate a stale-breaker stealing ours and installing its own.
            lock.unlink()
            lock.write_text("a-different-token\n12345\n")

        assert lock.exists(), "release deleted the successor's lock"
        assert lock.read_text().startswith("a-different-token")
        lock.unlink()

    def test_release_removes_our_own_lock(self, data_root):
        lock = data_root / upload_routes._LOCK_NAME
        with upload_routes._upload_lock(data_root):
            assert lock.exists()
        assert not lock.exists()

    def test_no_stale_sidecar_files_left_behind(self, data_root):
        lock = data_root / upload_routes._LOCK_NAME
        lock.write_text("99999\n")
        stale = time.time() - 10_000
        os.utime(lock, (stale, stale))

        with upload_routes._upload_lock(data_root):
            pass

        leftovers = [p.name for p in data_root.iterdir() if ".stale-" in p.name]
        assert leftovers == []

    def test_orphans_are_swept(self, client, data_root):
        """Scratch left by a crashed upload is cleaned on the next one."""
        (data_root / f"{upload_routes._STAGING_PREFIX}dead").mkdir()
        (data_root / f"{upload_routes._OLD_PREFIX}dead").mkdir()

        res = client.post("/upload", files={"files": ("x.zip", _zip({"a.md": "A"}))})

        assert res.status_code == 200, res.text
        assert _scratch(data_root) == []


class TestDelete:
    def test_requires_feature_flag(self, data_root):
        app = FastAPI()
        app.include_router(upload_router)
        with patch.object(upload_routes.settings, "UPLOAD_ENABLED", False):
            res = TestClient(app).delete("/upload")
        assert res.status_code == 404

    def test_requires_developer_role(self, data_root):
        from fastapi import HTTPException

        app = FastAPI()
        app.include_router(upload_router)
        with (
            patch.object(upload_routes.settings, "UPLOAD_ENABLED", True),
            patch.object(
                upload_routes,
                "check_ldap_role",
                new_callable=AsyncMock,
                side_effect=HTTPException(status_code=403, detail="nope"),
            ) as auth,
        ):
            res = TestClient(app).delete("/upload")

        assert res.status_code == 403
        # Gated as developer-only, same as upload.
        assert auth.await_args.kwargs["developer_only"] is True

    def test_clears_the_tree(self, client, data_root):
        client.post(
            "/upload",
            files={"files": ("x.zip", _zip({"proj/a.md": "A", "proj/b.md": "BB"}))},
        )

        res = client.delete("/upload")

        assert res.status_code == 200, res.text
        assert res.json() == {"files_deleted": 2, "bytes_deleted": 3}
        assert _live(data_root) == {}
        assert _scratch(data_root) == []

    def test_is_idempotent(self, client, data_root):
        client.post("/upload", files={"files": ("x.zip", _zip({"a.md": "A"}))})
        client.delete("/upload")

        res = client.delete("/upload")

        assert res.status_code == 200, res.text
        assert res.json() == {"files_deleted": 0, "bytes_deleted": 0}

    def test_upload_works_after_delete(self, client, data_root):
        client.post("/upload", files={"files": ("x.zip", _zip({"old.md": "O"}))})
        client.delete("/upload")

        res = client.post("/upload", files={"files": ("y.zip", _zip({"new.md": "N"}))})

        assert res.status_code == 200, res.text
        assert _live(data_root) == {"new.md": "N"}

    def test_returns_409_while_an_upload_holds_the_lock(self, client, data_root):
        (data_root / upload_routes._LOCK_NAME).write_text("99999\n")

        res = client.delete("/upload")

        assert res.status_code == 409


class TestPermissions:
    def test_written_files_and_dirs_are_group_writable(self, client, data_root):
        """Data outlives the writer; the container UID may differ next start."""
        client.post("/upload", files={"files": ("x.zip", _zip({"proj/a.md": "A"}))})

        live = data_root / LIVE_DIR
        assert (live / "proj" / "a.md").stat().st_mode & 0o060 == 0o060
        assert (live / "proj").stat().st_mode & 0o070 == 0o070
        assert live.stat().st_mode & 0o070 == 0o070

    def test_merge_falls_back_to_copy_when_hardlink_denied(self, client, data_root):
        """fs.protected_hardlinks can refuse a link to a foreign-owned file."""
        client.post("/upload", files={"files": ("x.zip", _zip({"keep.md": "K"}))})

        real_link = os.link
        calls = {"n": 0}

        def denied(src, dst, **kw):
            calls["n"] += 1
            raise PermissionError("Operation not permitted")

        with patch.object(upload_routes.os, "link", side_effect=denied):
            res = client.post("/upload", files={"files": ("add.md", b"A")})

        assert calls["n"] > 0, "hardlink path was not exercised"
        assert res.status_code == 200, res.text
        # Merge still produced the right tree, just via copy.
        assert _live(data_root) == {"keep.md": "K", "add.md": "A"}
        assert real_link is os.link  # patch scoped, not leaked


class TestAudit:
    @pytest.fixture
    def audit(self):
        with patch.object(upload_routes, "emit_audit_event") as spy:
            yield spy

    def test_upload_is_audited_with_user(self, client, data_root, audit):
        client.post("/upload", files={"files": ("x.zip", _zip({"a.md": "A"}))})

        audit.assert_called_once()
        event_type, kwargs = audit.call_args[0][0], audit.call_args[1]
        assert event_type == "data_upload"
        assert kwargs["user"] == "dev-user"
        assert kwargs["mode"] == "replace"
        assert kwargs["files_written"] == 1

    def test_delete_is_audited_with_user(self, client, data_root, audit):
        client.post("/upload", files={"files": ("x.zip", _zip({"a.md": "A"}))})
        audit.reset_mock()

        client.delete("/upload")

        audit.assert_called_once()
        event_type, kwargs = audit.call_args[0][0], audit.call_args[1]
        assert event_type == "data_delete"
        assert kwargs["user"] == "dev-user"
        assert kwargs["files_deleted"] == 1

    def test_rejected_upload_is_audited(self, client, data_root, audit):
        """A refused push onto shared storage is worth seeing a pattern of."""
        client.post("/upload", files={"files": ("a.png", b"x")})

        audit.assert_called_once()
        event_type, kwargs = audit.call_args[0][0], audit.call_args[1]
        assert event_type == "data_upload_rejected"
        assert kwargs["user"] == "dev-user"
        assert "no permitted files" in kwargs["reason"]

    def test_unauthorized_request_emits_no_audit_event(self, data_root, audit):
        """Nothing happened, so nothing to audit."""
        from fastapi import HTTPException

        app = FastAPI()
        app.include_router(upload_router)
        with (
            patch.object(upload_routes.settings, "UPLOAD_ENABLED", True),
            patch.object(
                upload_routes,
                "check_ldap_role",
                new_callable=AsyncMock,
                side_effect=HTTPException(status_code=403, detail="nope"),
            ),
        ):
            TestClient(app).delete("/upload")

        audit.assert_not_called()


class TestAtomicity:
    def test_failure_leaves_previous_tree_intact(self, client, data_root):
        client.post("/upload", files={"files": ("x.zip", _zip({"keep.md": "K"}))})
        before = _live(data_root)

        bad = _zip({"ok.md": "A", "../escape.md": "bad"})
        res = client.post("/upload", files={"files": ("y.zip", bad)})

        assert res.status_code == 400
        assert _live(data_root) == before
        assert _scratch(data_root) == []

    def test_failed_swap_restores_old_tree(self, client, data_root):
        client.post("/upload", files={"files": ("x.zip", _zip({"keep.md": "K"}))})
        before = _live(data_root)

        real_rename = os.rename
        calls = {"n": 0}

        def flaky(src, dst):
            calls["n"] += 1
            if calls["n"] == 2:  # the staging -> live rename
                raise OSError("simulated ENOSPC")
            return real_rename(src, dst)

        with patch.object(upload_routes.os, "rename", side_effect=flaky):
            res = client.post(
                "/upload", files={"files": ("y.zip", _zip({"new.md": "N"}))}
            )

        assert res.status_code == 500
        assert _live(data_root) == before

    def test_no_uuid_or_current_in_agent_visible_paths(self, client, data_root):
        client.post(
            "/upload", files={"files": ("x.zip", _zip({"llm-foundry/a.md": "A"}))}
        )

        for rel in _live(data_root):
            assert "current" not in rel
            assert ".staging-" not in rel and ".old-" not in rel


class TestChecksum:
    """What gets verified follows the payload type.

    An archive is checked as transmitted -- one digest over the .zip/.tar
    itself, before it is unpacked. Individually uploaded files are checked
    one by one. Members inside an archive are never digested.
    """

    @pytest.fixture
    def strict_client(self, client):
        """Client with UPLOAD_CHECKSUM_REQUIRED on."""
        with patch.object(upload_routes.settings, "UPLOAD_CHECKSUM_REQUIRED", True):
            yield client

    @staticmethod
    def _sha(payload):
        if isinstance(payload, str):
            payload = payload.encode()
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def _manifest(cls, entries, prefix=True):
        return json.dumps(
            {
                name: ("sha256:" if prefix else "") + cls._sha(content)
                for name, content in entries.items()
            }
        )

    # ── Archives: the zipped artifact as sent ─────────────────────────

    def test_archive_matching_its_digest_publishes(self, client, data_root):
        payload = _zip({"docs/a.md": "A", "docs/b.txt": "B"})
        res = client.post(
            "/upload",
            files={"files": ("x.zip", payload)},
            data={"checksum": f"sha256:{self._sha(payload)}"},
        )

        assert res.status_code == 200, res.text
        body = res.json()
        assert body["checksum_verified"] is True
        assert body["files_written"] == 2
        # Members are deliberately not digested: only the archive was.
        assert body["files_verified"] == 0
        assert _live(data_root) == {"docs/a.md": "A", "docs/b.txt": "B"}

    def test_bare_digest_without_prefix_is_accepted(self, client, data_root):
        payload = _zip({"a.md": "A"})
        res = client.post(
            "/upload",
            files={"files": ("x.zip", payload)},
            data={"checksum": self._sha(payload)},
        )

        assert res.status_code == 200, res.text

    def test_corrupted_archive_is_rejected(self, client, data_root):
        """One trailing byte: the content is intact, the artifact is not."""
        payload = _zip({"a.md": "A"})
        res = client.post(
            "/upload",
            files={"files": ("x.zip", payload + b"\x00")},
            data={"checksum": self._sha(payload)},
        )

        assert res.status_code == 400
        assert "checksum mismatch" in res.json()["detail"]
        assert _live(data_root) == {}
        assert _scratch(data_root) == []

    def test_digest_of_contents_instead_of_archive_is_rejected(self, client, data_root):
        """A zip's digest is not the digest of what it holds."""
        res = client.post(
            "/upload",
            files={"files": ("x.zip", _zip({"a.md": "A"}))},
            data={"checksum": self._sha("A")},
        )

        assert res.status_code == 400
        assert "checksum mismatch" in res.json()["detail"]

    def test_tar_archive_is_verified(self, client, data_root):
        payload = _tar({"a.md": "A"})
        res = client.post(
            "/upload",
            files={"files": ("x.tar", payload)},
            data={"checksum": self._sha(payload)},
        )

        assert res.status_code == 200, res.text
        assert _live(data_root) == {"a.md": "A"}

    def test_per_file_manifest_is_refused_for_an_archive(self, client, data_root):
        """Members are not digested, so accepting this would be a lie."""
        res = client.post(
            "/upload",
            files={"files": ("x.zip", _zip({"a.md": "A"}))},
            data={"checksums": self._manifest({"a.md": "A"})},
        )

        assert res.status_code == 400
        assert "individually uploaded files" in res.json()["detail"]
        assert _live(data_root) == {}

    def test_archive_is_rejected_before_it_is_unpacked(self, client, data_root):
        """A bad digest must not cost an extraction."""
        payload = _zip({"a.md": "A"})
        with patch.object(upload_routes, "_populate") as populate:
            res = client.post(
                "/upload",
                files={"files": ("x.zip", payload)},
                data={"checksum": self._sha("wrong")},
            )

        assert res.status_code == 400
        assert "checksum mismatch" in res.json()["detail"]
        populate.assert_not_called()

    # ── Individually uploaded files ───────────────────────────────────

    def test_single_file_verified_by_checksum(self, client, data_root):
        res = client.post(
            "/upload",
            files={"files": ("notes.md", b"hello")},
            data={"checksum": f"sha256:{self._sha('hello')}"},
        )

        assert res.status_code == 200, res.text
        assert res.json()["checksum_verified"] is True
        assert _live(data_root) == {"notes.md": "hello"}

    def test_single_file_with_a_wrong_digest_is_rejected(self, client, data_root):
        res = client.post(
            "/upload",
            files={"files": ("notes.md", b"hello")},
            data={"checksum": self._sha("goodbye")},
        )

        assert res.status_code == 400
        assert "checksum mismatch" in res.json()["detail"]
        assert _live(data_root) == {}

    def test_several_files_each_verified(self, client, data_root):
        res = client.post(
            "/upload",
            files=[("files", ("a.md", b"A")), ("files", ("b.md", b"B"))],
            data={"checksums": self._manifest({"a.md": "A", "b.md": "B"})},
        )

        assert res.status_code == 200, res.text
        assert res.json()["files_verified"] == 2
        assert _live(data_root) == {"a.md": "A", "b.md": "B"}

    def test_one_bad_digest_rejects_the_whole_request(self, client, data_root):
        """Rejects inside _ingest, after staging was built and seeded.

        Unlike the payload-digest mismatches, this path has already created
        a .staging-* directory and hardlink-copied the live tree into it,
        so it is the one that can leak scratch or a half-published tree.
        """
        client.post("/upload", files={"files": ("x.zip", _zip({"keep.md": "K"}))})
        before = _live(data_root)

        res = client.post(
            "/upload",
            files=[("files", ("a.md", b"A")), ("files", ("b.md", b"B"))],
            data={"checksums": self._manifest({"a.md": "A", "b.md": "WRONG"})},
        )

        assert res.status_code == 400
        assert "b.md" in res.json()["detail"]
        assert _live(data_root) == before
        assert _scratch(data_root) == []

    def test_undeclared_file_is_rejected(self, client, data_root):
        res = client.post(
            "/upload",
            files=[("files", ("a.md", b"A")), ("files", ("b.md", b"B"))],
            data={"checksums": self._manifest({"a.md": "A"})},
        )

        assert res.status_code == 400
        assert "b.md" in res.json()["detail"]
        assert _live(data_root) == {}

    def test_allowlist_filtered_file_says_so(self, client, data_root):
        """Not "was not published" -- that sends the caller hunting for a
        truncated upload when it was the server that filtered the file."""
        res = client.post(
            "/upload",
            files=[("files", ("a.md", b"A")), ("files", ("b.exe", b"B"))],
            data={"checksums": self._manifest({"a.md": "A", "b.exe": "B"})},
        )

        assert res.status_code == 400
        detail = res.json()["detail"]
        assert "b.exe" in detail
        assert "UPLOAD_ALLOWED_EXTENSIONS" in detail
        assert _live(data_root) == {}

    def test_declared_file_that_was_not_sent_is_rejected(self, client, data_root):
        res = client.post(
            "/upload",
            files={"files": ("a.md", b"A")},
            data={"checksums": self._manifest({"a.md": "A", "missing.md": "M"})},
        )

        assert res.status_code == 400
        assert "missing.md" in res.json()["detail"]
        assert _live(data_root) == {}

    def test_names_differing_only_by_whitespace_do_not_share_a_digest(
        self, strict_client, data_root
    ):
        """Regression: " a.md" and "a.md" are two files, not one.

        Folding them to one key let the second entry's digest vouch for the
        first, so a file nobody declared was published with a 200 — and
        UPLOAD_CHECKSUM_REQUIRED did not stop it, because a manifest *was*
        present. Digests are now keyed by where bytes actually landed.
        """
        res = strict_client.post(
            "/upload",
            files=[("files", ("a.md", b"A")), ("files", (" a.md", b"EVIL"))],
            data={"checksums": self._manifest({"a.md": "EVIL"})},
        )

        assert res.status_code == 400
        assert "not listed in checksums" in res.json()["detail"]
        assert _live(data_root) == {}
        assert _scratch(data_root) == []

    def test_a_whitespace_name_can_still_be_declared(self, client, data_root):
        """The fix must not make a legal filename undeclarable."""
        res = client.post(
            "/upload",
            files={"files": (" a.md", b"A")},
            data={"checksums": self._manifest({" a.md": "A"})},
        )

        assert res.status_code == 200, res.text
        assert _live(data_root) == {" a.md": "A"}

    def test_checksum_is_refused_for_several_files(self, client, data_root):
        """No single stream to digest, so this would verify nothing."""
        res = client.post(
            "/upload",
            files=[("files", ("a.md", b"A")), ("files", ("b.md", b"B"))],
            data={"checksum": self._sha("A")},
        )

        assert res.status_code == 400
        assert "single uploaded file" in res.json()["detail"]
        assert _live(data_root) == {}

    def test_merge_leaves_the_existing_tree_alone(self, client, data_root):
        client.post("/upload", files={"files": ("x.zip", _zip({"keep.md": "K"}))})

        res = client.post(
            "/upload",
            files={"files": ("add.md", b"A")},
            data={"checksum": self._sha("A")},
        )

        assert res.status_code == 200, res.text
        assert _live(data_root) == {"keep.md": "K", "add.md": "A"}

    def test_mismatch_leaves_the_previous_tree_intact(self, client, data_root):
        client.post("/upload", files={"files": ("x.zip", _zip({"keep.md": "K"}))})
        before = _live(data_root)

        res = client.post(
            "/upload",
            files={"files": ("y.zip", _zip({"new.md": "N"}))},
            data={"checksum": self._sha("nope")},
        )

        assert res.status_code == 400
        assert _live(data_root) == before
        assert _scratch(data_root) == []

    # ── Malformed input ───────────────────────────────────────────────

    def test_unsupported_algorithm_is_rejected(self, client, data_root):
        res = client.post(
            "/upload",
            files={"files": ("x.zip", _zip({"a.md": "A"}))},
            data={"checksum": f"md5:{'a' * 32}"},
        )

        assert res.status_code == 400
        assert "unsupported checksum algorithm" in res.json()["detail"]

    def test_malformed_manifest_is_rejected(self, client, data_root):
        res = client.post(
            "/upload",
            files={"files": ("a.md", b"A")},
            data={"checksums": "{not json"},
        )

        assert res.status_code == 400
        assert "not valid JSON" in res.json()["detail"]

    def test_checksum_sent_as_a_file_part_is_rejected(self, client, data_root):
        res = client.post(
            "/upload",
            files=[
                ("files", ("a.md", b"A")),
                ("checksum", ("c.txt", b"deadbeef")),
            ],
        )

        assert res.status_code == 400
        assert "text field" in res.json()["detail"]

    # ── Audit ─────────────────────────────────────────────────────────

    def test_mismatch_is_audited(self, client, data_root):
        with patch.object(upload_routes, "emit_audit_event") as audit:
            client.post(
                "/upload",
                files={"files": ("a.md", b"A")},
                data={"checksum": self._sha("B")},
            )

        audit.assert_called_once()
        event_type, kwargs = audit.call_args[0][0], audit.call_args[1]
        assert event_type == "data_upload_rejected"
        assert kwargs["user"] == "dev-user"
        assert "checksum mismatch" in kwargs["reason"]

    def test_verification_is_recorded_on_success(self, client, data_root):
        payload = _zip({"a.md": "A"})
        with patch.object(upload_routes, "emit_audit_event") as audit:
            client.post(
                "/upload",
                files={"files": ("x.zip", payload)},
                data={"checksum": self._sha(payload)},
            )

        assert audit.call_args[1]["checksum_verified"] is True

    # ── The flag ──────────────────────────────────────────────────────

    def test_nothing_sent_is_fine_while_the_flag_is_off(self, client, data_root):
        res = client.post("/upload", files={"files": ("x.zip", _zip({"a.md": "A"}))})

        assert res.status_code == 200, res.text
        assert res.json()["checksum_verified"] is False
        assert res.json()["files_verified"] == 0

    def test_archive_without_a_digest_is_rejected_when_required(
        self, strict_client, data_root
    ):
        res = strict_client.post(
            "/upload", files={"files": ("x.zip", _zip({"a.md": "A"}))}
        )

        assert res.status_code == 400
        assert "checksum" in res.json()["detail"]
        assert _live(data_root) == {}

    def test_files_without_digests_are_rejected_when_required(
        self, strict_client, data_root
    ):
        res = strict_client.post(
            "/upload",
            files=[("files", ("a.md", b"A")), ("files", ("b.md", b"B"))],
        )

        assert res.status_code == 400
        assert "checksums" in res.json()["detail"]
        assert _live(data_root) == {}

    def test_single_loose_file_without_a_digest_is_rejected_when_required(
        self, strict_client, data_root
    ):
        """The one-file branch of the requirement, distinct from the archive."""
        res = strict_client.post("/upload", files={"files": ("a.md", b"A")})

        assert res.status_code == 400
        assert "checksum" in res.json()["detail"]
        assert _live(data_root) == {}

    def test_policy_refusals_are_audited(self, strict_client, data_root):
        """An operator turning the flag on needs to see it firing."""
        with patch.object(upload_routes, "emit_audit_event") as audit:
            strict_client.post(
                "/upload", files={"files": ("x.zip", _zip({"a.md": "A"}))}
            )

        audit.assert_called_once()
        assert audit.call_args[0][0] == "data_upload_rejected"
        assert "checksum" in audit.call_args[1]["reason"]

    def test_single_file_may_satisfy_the_requirement_either_way(
        self, strict_client, data_root
    ):
        """A lone file is its own payload, so both spellings cover it."""
        by_checksum = strict_client.post(
            "/upload",
            files={"files": ("a.md", b"A")},
            data={"checksum": self._sha("A")},
        )
        assert by_checksum.status_code == 200, by_checksum.text

        by_manifest = strict_client.post(
            "/upload",
            files={"files": ("b.md", b"B")},
            data={"checksums": self._manifest({"b.md": "B"})},
        )
        assert by_manifest.status_code == 200, by_manifest.text

    def test_good_archive_still_passes_when_required(self, strict_client, data_root):
        payload = _zip({"a.md": "A"})
        res = strict_client.post(
            "/upload",
            files={"files": ("x.zip", payload)},
            data={"checksum": self._sha(payload)},
        )

        assert res.status_code == 200, res.text
        assert _live(data_root) == {"a.md": "A"}


class TestAuthBeforeBodyParsing:
    """The body must not be read until the flag and authorization pass.

    FastAPI parses a declared ``files: list[UploadFile] = File(...)``
    parameter *before* it solves dependencies, so no dependency, decorator
    or security scheme can run first. The route therefore takes a raw
    Request and calls request.form() itself, after the checks. Otherwise an
    unauthenticated caller gets their body spooled to temp disk and only
    then receives a 404 or 401. See adrs/upload-data-pvc.md.
    """

    def _app(self):
        app = FastAPI()
        app.include_router(upload_router)
        return app

    def test_route_declares_no_file_parameter(self):
        """A declared File() param would reintroduce parse-before-auth."""
        import inspect

        sig = inspect.signature(upload_routes.upload)
        assert list(sig.parameters) == ["request"], (
            "a declared body parameter makes FastAPI parse before auth"
        )

    def test_disabled_feature_does_not_read_the_body(self, data_root):
        consumed = []

        def body():
            yield (
                b'--b\r\nContent-Disposition: form-data; name="files";'
                b' filename="x.md"\r\n\r\n'
            )
            for _ in range(8):
                consumed.append(1)
                yield b"x" * 65536
            yield b"\r\n--b--\r\n"

        with patch.object(upload_routes.settings, "UPLOAD_ENABLED", False):
            res = TestClient(self._app(), raise_server_exceptions=False).post(
                "/upload",
                content=body(),
                headers={"Content-Type": "multipart/form-data; boundary=b"},
            )

        assert res.status_code == 404

    def test_unauthorized_does_not_read_the_body(self, data_root):
        from fastapi import HTTPException

        parsed = []
        real_form = upload_routes.Request.form

        def tracking_form(self, **kw):
            parsed.append(1)
            return real_form(self, **kw)

        async def deny(*a, **k):
            raise HTTPException(status_code=403, detail="denied")

        with (
            patch.object(upload_routes.settings, "UPLOAD_ENABLED", True),
            patch.object(upload_routes, "check_ldap_role", side_effect=deny),
            patch.object(upload_routes.Request, "form", tracking_form),
        ):
            res = TestClient(self._app(), raise_server_exceptions=False).post(
                "/upload", files={"files": ("x.md", b"hello")}
            )

        assert res.status_code == 403
        assert parsed == [], "the body was parsed despite authorization failing"
