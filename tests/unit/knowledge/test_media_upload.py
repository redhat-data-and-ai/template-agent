"""Cloud Storage path for media too large to inline.

Not the Files API: `client.files.upload` exists only against AI Studio
and fails on Vertex with "This method is only supported in the Gemini
Developer client". On Vertex the model reads a `gs://` URI instead.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from deep_agent.src.knowledge.media_upload import (
    MAX_FILE_API_BYTES,
    MediaUploadError,
    delete_media,
    upload_media,
)


class TestWithoutABucket:
    def test_it_says_what_is_missing_rather_than_failing_obscurely(self, monkeypatch):
        """ "upload failed" would send someone looking at the file."""
        monkeypatch.delenv("MEDIA_BUCKET", raising=False)
        with pytest.raises(MediaUploadError) as exc:
            upload_media("talk.mp4", b"x" * 1024)
        assert "MEDIA_BUCKET" in str(exc.value)

    def test_it_also_names_the_inline_limit(self, monkeypatch):
        monkeypatch.delenv("MEDIA_BUCKET", raising=False)
        with pytest.raises(MediaUploadError, match="14 MB"):
            upload_media("talk.mp4", b"x" * 1024)


class TestWithABucket:
    def _storage(self):
        client = MagicMock()
        blob = MagicMock()
        client.bucket.return_value.blob.return_value = blob
        return client, blob

    def test_it_returns_a_gs_uri(self, monkeypatch):
        monkeypatch.setenv("MEDIA_BUCKET", "my-bucket")
        client, _ = self._storage()
        with patch(
            "deep_agent.src.knowledge.media_upload._client", return_value=client
        ):
            uri = upload_media("talk.mp4", b"video")
        assert uri.startswith("gs://my-bucket/agent-media/")
        assert uri.endswith("/talk.mp4")

    def test_two_uploads_of_the_same_name_do_not_collide(self, monkeypatch):
        """Otherwise one user's recording overwrites another's."""
        monkeypatch.setenv("MEDIA_BUCKET", "my-bucket")
        client, _ = self._storage()
        with patch(
            "deep_agent.src.knowledge.media_upload._client", return_value=client
        ):
            first = upload_media("talk.mp4", b"a")
            second = upload_media("talk.mp4", b"b")
        assert first != second

    def test_the_content_type_is_set(self, monkeypatch):
        """The model routes on mime type, not on the extension."""
        monkeypatch.setenv("MEDIA_BUCKET", "my-bucket")
        client, blob = self._storage()
        with patch(
            "deep_agent.src.knowledge.media_upload._client", return_value=client
        ):
            upload_media("talk.mp4", b"video")
        assert blob.upload_from_string.call_args.kwargs["content_type"] == "video/mp4"

    def test_a_storage_failure_names_the_bucket(self, monkeypatch):
        monkeypatch.setenv("MEDIA_BUCKET", "my-bucket")
        client, blob = self._storage()
        blob.upload_from_string.side_effect = RuntimeError("denied")
        with patch(
            "deep_agent.src.knowledge.media_upload._client", return_value=client
        ):
            with pytest.raises(MediaUploadError, match="gs://my-bucket"):
                upload_media("talk.mp4", b"video")


class TestLimits:
    def test_an_absurdly_large_file_is_refused_before_uploading(self, monkeypatch):
        monkeypatch.setenv("MEDIA_BUCKET", "my-bucket")
        with pytest.raises(MediaUploadError, match="exceeds"):
            upload_media("huge.mp4", b"x" * (MAX_FILE_API_BYTES + 1))

    def test_the_ceiling_is_far_above_the_inline_limit(self):
        """The point of this path is that 17 MB is ordinary."""
        from deep_agent.src.knowledge.transcribe import MAX_INLINE_BYTES

        assert MAX_FILE_API_BYTES > 100 * MAX_INLINE_BYTES


class TestCleanup:
    def test_deletion_failures_are_swallowed(self):
        """A bucket lifecycle rule is the real cleanup; this is tidiness."""
        with patch(
            "deep_agent.src.knowledge.media_upload._client",
            side_effect=RuntimeError("no network"),
        ):
            delete_media("gs://bucket/agent-media/x/a.mp4")  # must not raise

    def test_a_non_gs_uri_is_ignored(self):
        delete_media("https://example.com/a.mp4")  # must not raise
