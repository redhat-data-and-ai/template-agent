"""Tests for the record/replay fixture itself.

A cassette that silently lets requests reach the network is worse than
no cassette: the suite looks deterministic and is not. These prove the
failure modes.
"""

from pathlib import Path

import httpx
import pytest

from tests.support.cassette import (
    Cassette,
    CassetteMissError,
    CassetteTransport,
    _redact_body,
    _redact_headers,
    cassette_mode,
    request_key,
)


class TestRequestKey:
    def test_same_request_same_key(self):
        a = request_key("POST", "https://api/x", b'{"p":1}')
        b = request_key("POST", "https://api/x", b'{"p":1}')
        assert a == b

    def test_different_body_different_key(self):
        """Two prompts to one endpoint must not collide."""
        a = request_key("POST", "https://api/x", b'{"prompt":"hi"}')
        b = request_key("POST", "https://api/x", b'{"prompt":"bye"}')
        assert a != b

    def test_different_method_different_key(self):
        assert request_key("GET", "https://api/x", b"") != request_key(
            "POST", "https://api/x", b""
        )


class TestRedaction:
    @pytest.mark.parametrize(
        "header", ["Authorization", "X-API-Key", "Cookie", "X-Refresh-Token"]
    )
    def test_credential_headers_are_stripped(self, header: str):
        """A recorded Authorization header is a committed credential."""
        out = _redact_headers({header: "s3cret", "Accept": "application/json"})
        assert out[header] == "[REDACTED]"
        assert out["Accept"] == "application/json"

    def test_secret_body_fields_are_stripped(self):
        out = _redact_body(b'{"api_key":"sk-live-xyz","model":"gemini"}')
        assert "sk-live-xyz" not in out
        assert "gemini" in out

    def test_nested_secrets_are_stripped(self):
        out = _redact_body(b'{"auth":{"client_secret":"shh"},"n":1}')
        assert "shh" not in out

    def test_non_json_body_survives(self):
        assert _redact_body(b"plain text") == "plain text"


class TestReplay:
    @pytest.mark.asyncio
    async def test_replays_a_recorded_response(self, tmp_path: Path):
        path = tmp_path / "c.json"
        cassette = Cassette(path, "replay")
        key = request_key("POST", "https://api/v1/chat", b'{"p":1}')
        cassette.record(key, {"status": 200, "headers": {}, "body": '{"ok":true}'})

        async with httpx.AsyncClient(transport=CassetteTransport(cassette)) as client:
            response = await client.post("https://api/v1/chat", content=b'{"p":1}')

        assert response.status_code == 200
        assert response.json() == {"ok": True}

    @pytest.mark.asyncio
    async def test_unrecorded_request_raises_rather_than_escaping(self, tmp_path: Path):
        """The whole point: replay must never reach the network."""
        cassette = Cassette(tmp_path / "empty.json", "replay")

        async with httpx.AsyncClient(transport=CassetteTransport(cassette)) as client:
            with pytest.raises(CassetteMissError) as excinfo:
                await client.post("https://api/v1/chat", content=b"{}")

        assert "re-record" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_repeated_requests_replay_in_order(self, tmp_path: Path):
        """A retry should get the retried response, not the first again."""
        cassette = Cassette(tmp_path / "c.json", "replay")
        key = request_key("POST", "https://api/x", b"{}")
        cassette.record(key, {"status": 500, "headers": {}, "body": "boom"})
        cassette.record(key, {"status": 200, "headers": {}, "body": "ok"})

        async with httpx.AsyncClient(transport=CassetteTransport(cassette)) as client:
            first = await client.post("https://api/x", content=b"{}")
            second = await client.post("https://api/x", content=b"{}")

        assert first.status_code == 500
        assert second.status_code == 200

    @pytest.mark.asyncio
    async def test_exhausted_sequence_repeats_the_last(self, tmp_path: Path):
        cassette = Cassette(tmp_path / "c.json", "replay")
        key = request_key("GET", "https://api/x", b"")
        cassette.record(key, {"status": 200, "headers": {}, "body": "a"})

        async with httpx.AsyncClient(transport=CassetteTransport(cassette)) as client:
            await client.get("https://api/x")
            again = await client.get("https://api/x")

        assert again.text == "a"


class TestPersistence:
    def test_round_trips_through_disk(self, tmp_path: Path):
        path = tmp_path / "c.json"
        writing = Cassette(path, "record")
        writing.record("K", {"status": 200, "headers": {}, "body": "x"})
        writing.save()

        assert Cassette(path, "replay").recorded_count == 1

    def test_absent_cassette_is_empty_not_an_error(self, tmp_path: Path):
        assert Cassette(tmp_path / "nope.json", "replay").recorded_count == 0


class TestMode:
    def test_replay_is_the_default(self, monkeypatch):
        """CI must never record; recording silently hits the real API."""
        monkeypatch.delenv("CASSETTE_MODE", raising=False)
        assert cassette_mode() == "replay"

    def test_record_requires_an_explicit_opt_in(self, monkeypatch):
        monkeypatch.setenv("CASSETTE_MODE", "record")
        assert cassette_mode() == "record"

    def test_any_other_value_is_replay(self, monkeypatch):
        monkeypatch.setenv("CASSETTE_MODE", "yes-please")
        assert cassette_mode() == "replay"


class TestFixtureWiring:
    @pytest.mark.asyncio
    async def test_fixture_yields_a_bound_client(self, llm_cassette):
        async with llm_cassette() as client:
            with pytest.raises(CassetteMissError):
                await client.get("https://api.example/unrecorded")
