"""Raise diff coverage for headless modules above the 80% gate."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from deep_agent.aegra import invoke_routes
from deep_agent.aegra.invoke_routes import (
    _forward_headers,
    approval_resume,
    rewrite_keepalive,
    router,
    run_body,
)
from deep_agent.headless import __main__ as headless_main
from deep_agent.headless.auth import (
    _Callback,
    _pkce_pair,
    _token_error,
    access_token,
    browser_token,
    client_credentials_token,
    exchange_authorization_code,
    password_grant_token,
)
from deep_agent.headless.interrupts import (
    _interrupt_value,
    count_actions,
    invoke_resume_body,
)
from deep_agent.headless.invoke import HeadlessInvoker, _http_error, _read_stream
from deep_agent.headless.settings import HeadlessSettings
from deep_agent.headless.stream import (
    _text_from_content,
    _unwrap_update,
    end_status,
    iter_sse_events,
    message_texts,
    thread_id_from_events,
)
from deep_agent.headless.workflow import WorkflowError, apply_workflow


def _settings(**kwargs: Any) -> HeadlessSettings:
    base = dict(
        _env_file=None,
        sso_client_id="ask-data-ui",
        sso_client_secret="secret",
        sso_client_secret_prod="prod-secret",
        sso_sa_username="sa",
        sso_sa_password_prod="pw",
        sso_token_url_prod="https://auth.example/token",
        sso_issuer_url="https://auth.example/realms/EmployeeIDP",
        agent_url="https://agent.example",
        agent_message="hello",
        agent_max_resumes=2,
        agent_timeout=5,
        headless_ssl_verify=False,
    )
    base.update(kwargs)
    return HeadlessSettings(**base)


# --- stream -----------------------------------------------------------------


def test_iter_sse_skips_bad_json_and_reads_events() -> None:
    lines = [
        "event: updates",
        "data: not-json",
        "event: end",
        'data: {"status": "success"}',
    ]
    events = list(iter_sse_events(lines))
    assert events == [("end", {"status": "success"})]


def test_unwrap_update_and_thread_id_helpers() -> None:
    assert _unwrap_update(["meta", {"thread_id": "t1"}]) == {"thread_id": "t1"}
    assert _unwrap_update("x") == {}
    assert thread_id_from_events([("metadata", {"thread_id": "t9"})]) == "t9"
    assert thread_id_from_events([("metadata", {})]) is None
    assert thread_id_from_events([], "hdr") == "hdr"


def test_message_texts_from_list_content() -> None:
    events = [
        (
            "updates",
            {
                "model": {
                    "messages": [
                        {
                            "type": "ai",
                            "content": [
                                {"type": "text", "text": "hi"},
                                " there",
                            ],
                        },
                        {"role": "user", "content": "skip"},
                        {"type": "ai", "content": 12},
                    ]
                }
            },
        )
    ]
    assert message_texts(events) == ["hi\nthere"]
    assert _text_from_content("  x  ") == "x"
    assert _text_from_content(None) == ""


def test_end_status_reads_last_end_event() -> None:
    assert (
        end_status(
            [
                ("end", {"status": "interrupted"}),
                ("end", {"status": "success"}),
            ]
        )
        == "success"
    )


# --- interrupts / settings / workflow ---------------------------------------


def test_interrupt_value_and_count_non_dict() -> None:
    assert _interrupt_value("not-json") == "not-json"
    assert _interrupt_value({"value": "{bad"}) == "{bad"
    assert count_actions([{"value": "plain"}]) == 1
    assert invoke_resume_body("t", 1)["thread_id"] == "t"


def test_settings_authorization_and_token_urls() -> None:
    blank = HeadlessSettings(
        _env_file=None, sso_issuer_url="", sso_authorization_url=""
    )
    assert blank.authorization_url() == ""
    assert blank.token_url() == ""
    issuer = HeadlessSettings(
        _env_file=None,
        sso_issuer_url="https://auth.example/realms/x",
        sso_authorization_url="",
        sso_token_url="",
    )
    assert issuer.authorization_url().endswith("/protocol/openid-connect/auth")
    assert issuer.token_url().endswith("/protocol/openid-connect/token")
    assert (
        HeadlessSettings(
            _env_file=None, agent_url="https://a.example/invoke"
        ).invoke_url()
        == "https://a.example/invoke"
    )
    assert (
        HeadlessSettings(_env_file=None).aif_invoke_url()
        == HeadlessSettings(_env_file=None).invoke_url()
    )


def test_workflow_unknown_and_missing_destination() -> None:
    with pytest.raises(WorkflowError, match="Unknown workflow"):
        apply_workflow(_settings(), "lab")
    with pytest.raises(WorkflowError, match="remote agent"):
        apply_workflow(HeadlessSettings(_env_file=None), "preprod")


# --- auth -------------------------------------------------------------------


def test_pkce_and_token_error_shapes() -> None:
    verifier, challenge = _pkce_pair()
    assert verifier and challenge and verifier != challenge

    class Resp:
        status_code = 400

        def json(self) -> dict[str, str]:
            return {"error": "invalid", "error_description": "bad"}

    assert "invalid" in _token_error(Resp())  # type: ignore[arg-type]

    class Bad:
        status_code = 500

        def json(self) -> list[str]:
            raise ValueError("no")

    assert _token_error(Bad()) == "HTTP 500"  # type: ignore[arg-type]

    class NonDict:
        status_code = 401

        def json(self) -> list[str]:
            return ["x"]

    assert _token_error(NonDict()) == "HTTP 401"  # type: ignore[arg-type]


def test_exchange_authorization_code_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    missing = HeadlessSettings(_env_file=None, sso_client_id="", sso_issuer_url="")
    assert exchange_authorization_code(missing, "c", "v") is None

    class Ok:
        status_code = 200

        def json(self) -> dict[str, str]:
            return {"access_token": "tok"}

    monkeypatch.setattr(
        "deep_agent.headless.auth.httpx.post",
        lambda *a, **k: Ok(),
    )
    assert exchange_authorization_code(_settings(environment="prod"), "c", "v") == "tok"

    class Fail:
        status_code = 400

        def json(self) -> dict[str, str]:
            return {"error": "nope"}

    monkeypatch.setattr(
        "deep_agent.headless.auth.httpx.post",
        lambda *a, **k: Fail(),
    )
    assert exchange_authorization_code(_settings(environment="prod"), "c", "v") is None

    def boom(*a: Any, **k: Any) -> Any:
        raise httpx.ConnectError("down")

    monkeypatch.setattr("deep_agent.headless.auth.httpx.post", boom)
    assert exchange_authorization_code(_settings(environment="prod"), "c", "v") is None

    class Empty:
        status_code = 200

        def json(self) -> dict[str, object]:
            return {"access_token": None}

    monkeypatch.setattr(
        "deep_agent.headless.auth.httpx.post",
        lambda *a, **k: Empty(),
    )
    assert exchange_authorization_code(_settings(environment="prod"), "c", "v") is None


def test_password_and_client_creds_and_access_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert password_grant_token(HeadlessSettings(_env_file=None)) is None
    assert client_credentials_token(HeadlessSettings(_env_file=None)) is None

    class Ok:
        status_code = 200

        def json(self) -> dict[str, str]:
            return {"access_token": "sa"}

    monkeypatch.setattr(
        "deep_agent.headless.auth.httpx.post",
        lambda *a, **k: Ok(),
    )
    assert (
        password_grant_token(_settings(environment="prod", auth_mode="password"))
        == "sa"
    )
    assert (
        client_credentials_token(
            _settings(auth_mode="client-creds", sso_client_secret="secret")
        )
        == "sa"
    )

    class Fail:
        status_code = 401

        def json(self) -> dict[str, str]:
            return {"error": "denied"}

    monkeypatch.setattr(
        "deep_agent.headless.auth.httpx.post",
        lambda *a, **k: Fail(),
    )
    assert password_grant_token(_settings(environment="prod")) is None
    assert client_credentials_token(_settings(sso_client_secret="secret")) is None

    def boom(*a: Any, **k: Any) -> Any:
        raise httpx.ConnectError("x")

    monkeypatch.setattr("deep_agent.headless.auth.httpx.post", boom)
    assert password_grant_token(_settings(environment="prod")) is None
    assert client_credentials_token(_settings(sso_client_secret="secret")) is None

    class Empty:
        status_code = 200

        def json(self) -> dict[str, object]:
            return {}

    monkeypatch.setattr(
        "deep_agent.headless.auth.httpx.post",
        lambda *a, **k: Empty(),
    )
    assert password_grant_token(_settings(environment="prod")) is None
    assert client_credentials_token(_settings(sso_client_secret="secret")) is None

    monkeypatch.setattr("deep_agent.headless.auth.password_grant_token", lambda s: "p")
    monkeypatch.setattr(
        "deep_agent.headless.auth.client_credentials_token", lambda s: "c"
    )
    monkeypatch.setattr("deep_agent.headless.auth.browser_token", lambda s: "b")
    assert access_token(_settings(auth_mode="password")) == "p"
    assert access_token(_settings(auth_mode="client-creds")) == "c"
    assert access_token(_settings(auth_mode="sso")) == "b"
    assert access_token(_settings(auth_mode="nope")) is None


def test_browser_token_missing_config_and_callback_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert browser_token(HeadlessSettings(_env_file=None)) is None
    assert (
        browser_token(
            HeadlessSettings(
                _env_file=None,
                sso_client_id="id",
                sso_issuer_url="https://auth.example/realms/x",
                sso_client_secret="",
                sso_client_secret_prod="",
                environment="prod",
            )
        )
        is None
    )

    monkeypatch.setattr(
        "deep_agent.headless.auth._Callback.start",
        lambda self: "Port busy",
    )
    monkeypatch.setattr(
        "deep_agent.headless.auth._token_from_agent_callback",
        lambda *a, **k: "via-agent",
    )
    monkeypatch.setattr("deep_agent.headless.auth.webbrowser.open", lambda url: None)
    got = browser_token(
        _settings(
            environment="prod",
            sso_authorization_url="https://auth.example/auth",
            sso_client_secret_prod="prod-secret",
        )
    )
    assert got == "via-agent"

    cb = _Callback(0)

    def fail_bind(self: _Callback) -> str | None:
        return "Port 1 is already in use. Stop the chat UI and run this command again."

    # exercise wait/stop with a fake server slot
    cb.code = "abc"
    cb.wait(timeout=1)
    cb.stop()


def test_token_from_agent_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    from deep_agent.headless.auth import _token_from_agent_callback

    class Ready:
        status_code = 200

    class Pending:
        status_code = 200

        def json(self) -> dict[str, str]:
            return {"code": "the-code"}

    monkeypatch.setattr(
        "deep_agent.headless.auth.httpx.post",
        lambda *a, **k: Ready(),
    )
    monkeypatch.setattr(
        "deep_agent.headless.auth.httpx.get",
        lambda *a, **k: Pending(),
    )
    monkeypatch.setattr("deep_agent.headless.auth.webbrowser.open", lambda u: None)
    monkeypatch.setattr(
        "deep_agent.headless.auth.exchange_authorization_code",
        lambda s, c, v: f"tok-{c}",
    )
    settings = _settings(
        environment="prod",
        sso_callback_port=8765,
        sso_authorization_url="https://auth.example/auth",
    )
    assert (
        _token_from_agent_callback(
            settings, "https://auth.example/auth?x=1", "st", "ver"
        )
        == "tok-the-code"
    )

    monkeypatch.setattr(
        "deep_agent.headless.auth.httpx.post",
        lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("x")),
    )
    assert (
        _token_from_agent_callback(settings, "https://auth.example/auth", "st", "ver")
        is None
    )


def test_browser_token_own_callback_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCallback:
        def __init__(self, port: int) -> None:
            self.code = "code"
            self.error = None

        def start(self) -> str | None:
            return None

        def wait(self, timeout: int = 180) -> None:
            return None

        def stop(self) -> None:
            return None

    monkeypatch.setattr("deep_agent.headless.auth._Callback", FakeCallback)
    monkeypatch.setattr("deep_agent.headless.auth.webbrowser.open", lambda u: None)
    monkeypatch.setattr(
        "deep_agent.headless.auth.exchange_authorization_code",
        lambda s, c, v: "user-tok",
    )
    assert (
        browser_token(
            _settings(
                environment="prod",
                sso_authorization_url="https://auth.example/auth",
                sso_client_secret_prod="prod-secret",
            )
        )
        == "user-tok"
    )

    class NoCode(FakeCallback):
        def __init__(self, port: int) -> None:
            self.code = None
            self.error = "timed out"

    monkeypatch.setattr("deep_agent.headless.auth._Callback", NoCode)
    assert (
        browser_token(
            _settings(
                environment="prod",
                sso_authorization_url="https://auth.example/auth",
                sso_client_secret_prod="prod-secret",
            )
        )
        is None
    )


# --- invoke client ----------------------------------------------------------


def test_http_error_helpers() -> None:
    class R:
        status_code = 400

        def json(self) -> dict[str, str]:
            return {"detail": "nope"}

    assert _http_error(R()) == "HTTP 400: nope"  # type: ignore[arg-type]

    class Bad:
        status_code = 500

        def json(self) -> dict[str, str]:
            raise ValueError("x")

    assert _http_error(Bad()) == "HTTP 500"  # type: ignore[arg-type]

    class ListBody:
        status_code = 502

        def json(self) -> list[str]:
            return ["x"]

    assert _http_error(ListBody()) == "HTTP 502"  # type: ignore[arg-type]

    class Empty:
        status_code = 503

        def json(self) -> dict[str, str]:
            return {}

    assert _http_error(Empty()) == "HTTP 503"  # type: ignore[arg-type]


def test_invoker_unknown_target_and_after_turn() -> None:
    inv = HeadlessInvoker(_settings(headless_target="weird"), "tok")
    assert inv.run() == 1

    success = [("end", {"status": "success"})]
    assert inv._after_turn("t", success) == 0

    err = [("end", {"status": "error"})]
    assert inv._after_turn("t", err) == 1

    mcp = [
        (
            "updates",
            {
                "__interrupt__": [
                    {"value": {"type": "mcp_auth_required", "mcp_name": "atlan"}}
                ]
            },
        ),
        ("end", {"status": "interrupted"}),
    ]
    assert inv._after_turn("t", mcp) == 2

    paused = [
        (
            "updates",
            {"__interrupt__": [{"value": {"action_requests": [{"name": "x"}]}}]},
        ),
        ("end", {"status": "interrupted"}),
    ]
    assert inv._after_turn(None, paused) == 1
    assert inv._after_turn("t", paused) is None


def test_invoker_threads_and_invoke_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    finished = [
        ("updates", {"model": {"messages": [{"type": "ai", "content": "done"}]}}),
        ("end", {"status": "success"}),
    ]
    paused = [
        (
            "updates",
            {"__interrupt__": [{"value": {"action_requests": [{"name": "tool"}]}}]},
        ),
        ("end", {"status": "interrupted"}),
    ]

    # threads happy path
    streams = iter([("t1", paused), ("t1", finished)])

    def fake_read(client: Any, url: str, body: dict[str, Any], headers: dict[str, str]):
        return next(streams)

    class Created:
        status_code = 200

        def json(self) -> dict[str, str]:
            return {"thread_id": "t1"}

    class Client:
        def __enter__(self) -> Client:
            return self

        def __exit__(self, *a: Any) -> None:
            return None

        def post(self, *a: Any, **k: Any) -> Created:
            return Created()

    monkeypatch.setattr("deep_agent.headless.invoke._read_stream", fake_read)
    monkeypatch.setattr("deep_agent.headless.invoke.httpx.Client", lambda **k: Client())
    threads = HeadlessInvoker(
        _settings(headless_target="threads", agent_url="https://agent.example"),
        "tok",
    )
    assert threads.run() == 0

    # create thread failure
    class BadCreated:
        status_code = 500

        def json(self) -> dict[str, str]:
            return {"detail": "no"}

    class BadClient(Client):
        def post(self, *a: Any, **k: Any) -> BadCreated:
            return BadCreated()

    monkeypatch.setattr(
        "deep_agent.headless.invoke.httpx.Client", lambda **k: BadClient()
    )
    assert (
        HeadlessInvoker(
            _settings(headless_target="threads", agent_url="https://agent.example"),
            "tok",
        ).run()
        == 1
    )

    # invoke path with resume then success
    streams2 = iter([(None, paused), ("hdr", finished)])

    def fake_read2(
        client: Any, url: str, body: dict[str, Any], headers: dict[str, str]
    ):
        return next(streams2)

    monkeypatch.setattr("deep_agent.headless.invoke._read_stream", fake_read2)
    monkeypatch.setattr("deep_agent.headless.invoke.httpx.Client", lambda **k: Client())
    # inject thread id from events when header missing on first pause
    paused_with_thread = [
        ("metadata", {"thread_id": "tid-1"}),
        (
            "updates",
            {"__interrupt__": [{"value": {"action_requests": [{"name": "tool"}]}}]},
        ),
        ("end", {"status": "interrupted"}),
    ]
    streams3 = iter([(None, paused_with_thread), ("hdr", finished)])

    def fake_read3(
        client: Any, url: str, body: dict[str, Any], headers: dict[str, str]
    ):
        return next(streams3)

    monkeypatch.setattr("deep_agent.headless.invoke._read_stream", fake_read3)
    assert HeadlessInvoker(_settings(headless_target="invoke"), "tok").run() == 0

    # HTTP error on invoke
    def boom_client(**k: Any) -> Any:
        raise httpx.ConnectError("down")

    monkeypatch.setattr("deep_agent.headless.invoke.httpx.Client", boom_client)
    assert HeadlessInvoker(_settings(headless_target="invoke"), "tok").run() == 1
    assert (
        HeadlessInvoker(
            _settings(headless_target="threads", agent_url="https://agent.example"),
            "tok",
        ).run()
        == 1
    )


def test_read_stream_success_and_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class OkResponse:
        status_code = 200
        headers = {"X-Thread-ID": "th-1"}

        def iter_lines(self):
            yield "event: end"
            yield 'data: {"status": "interrupted"}'
            yield "event: end"
            yield 'data: {"status": "error"}'

        def read(self) -> bytes:
            return b""

        def __enter__(self) -> OkResponse:
            return self

        def __exit__(self, *a: Any) -> None:
            return None

    class Client:
        def stream(self, *a: Any, **k: Any) -> OkResponse:
            return OkResponse()

    header, events = _read_stream(Client(), "http://x/invoke", {}, {})  # type: ignore[arg-type]
    assert header == "th-1"
    assert end_status(events) == "error"

    class BadResponse:
        status_code = 500
        headers: dict[str, str] = {}
        request = httpx.Request("POST", "http://x")

        def read(self) -> bytes:
            return b"{}"

        def json(self) -> dict[str, str]:
            return {"detail": "fail"}

        def __enter__(self) -> BadResponse:
            return self

        def __exit__(self, *a: Any) -> None:
            return None

    class BadClient:
        def stream(self, *a: Any, **k: Any) -> BadResponse:
            return BadResponse()

    with pytest.raises(httpx.HTTPStatusError):
        _read_stream(BadClient(), "http://x/invoke", {}, {})  # type: ignore[arg-type]


def test_invoker_stops_after_max_resumes(monkeypatch: pytest.MonkeyPatch) -> None:
    paused = [
        (
            "updates",
            {"__interrupt__": [{"value": {"action_requests": [{"name": "tool"}]}}]},
        ),
        ("end", {"status": "interrupted"}),
    ]

    def always_pause(*a: Any, **k: Any):
        return ("tid", paused)

    class Client:
        def __enter__(self) -> Client:
            return self

        def __exit__(self, *a: Any) -> None:
            return None

    monkeypatch.setattr("deep_agent.headless.invoke._read_stream", always_pause)
    monkeypatch.setattr("deep_agent.headless.invoke.httpx.Client", lambda **k: Client())
    assert (
        HeadlessInvoker(
            _settings(headless_target="invoke", agent_max_resumes=0), "tok"
        ).run()
        == 3
    )


# --- __main__ ---------------------------------------------------------------


def test_main_happy_and_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        headless_main,
        "HeadlessSettings",
        lambda: _settings(agent_url="https://agent.example"),
    )
    monkeypatch.setattr(headless_main, "access_token", lambda s: "tok")
    monkeypatch.setattr(
        headless_main,
        "HeadlessInvoker",
        lambda s, t: type("I", (), {"run": lambda self: 0})(),
    )
    assert headless_main.main(["preprod", "hi"]) == 0

    monkeypatch.setattr(headless_main, "access_token", lambda s: None)
    assert headless_main.main(["preprod"]) == 1

    monkeypatch.setattr(
        headless_main,
        "HeadlessSettings",
        lambda: HeadlessSettings(_env_file=None),
    )
    assert headless_main.main(["prod"]) == 1


# --- invoke_routes ----------------------------------------------------------


def test_run_body_and_approval_and_keepalive_and_headers() -> None:
    assert run_body({"input": {"x": 1}, "command": {"resume": {}}})["input"] == {"x": 1}
    assert run_body({"message": "  "}) is None
    assert run_body({}) is None

    paused = [
        (
            "updates",
            {"__interrupt__": [{"value": {"action_requests": [{"name": "t"}]}}]},
        ),
        ("end", {"status": "interrupted"}),
    ]
    assert approval_resume(paused) is not None
    assert approval_resume([("end", {"status": "success"})]) is None

    pending = bytearray()
    assert b"thinking" in rewrite_keepalive(pending, b": heartbeat", final=True)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": [(b"authorization", b"Bearer abc")],
        "client": ("127.0.0.1", 123),
        "server": ("test", 80),
    }
    req = Request(scope)
    assert _forward_headers(req)["Authorization"] == "Bearer abc"


def test_sso_callback_routes() -> None:
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    assert (
        client.post("/auth/callback/oidc/expect", content=b"not-json").status_code
        == 400
    )
    assert (
        client.post("/auth/callback/oidc/expect", json={"state": ""}).status_code == 400
    )
    assert client.post("/auth/callback/oidc/expect", json={"state": "s1"}).json() == {
        "ok": True
    }

    bad = client.get("/auth/callback/oidc", params={"state": "wrong"})
    assert bad.status_code == 400
    ok = client.get("/auth/callback/oidc", params={"state": "s1", "code": "abc"})
    assert ok.status_code == 200
    assert "Signed in" in ok.text
    pending = client.get("/auth/callback/oidc/pending")
    assert pending.json()["code"] == "abc"
    # second read clears
    assert client.get("/auth/callback/oidc/pending").json()["code"] is None

    client.post("/auth/callback/oidc/expect", json={"state": "s2"})
    err = client.get(
        "/auth/callback/oidc", params={"state": "s2", "error": "access_denied"}
    )
    assert err.status_code == 400
    assert client.get("/auth/callback/oidc/pending").json()["error"] == "access_denied"


def test_invoke_route_validation_and_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    assert client.post("/invoke", content=b"x").status_code == 400
    assert client.post("/invoke", json={"message": ""}).status_code == 400

    class StreamResp:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        async def aiter_bytes(self):
            payload = (
                b'event: updates\ndata: {"model":{"messages":'
                b'[{"type":"ai","content":"hi"}]}}\n\n'
                b'event: end\ndata: {"status":"success"}\n\n'
            )
            yield payload

        async def aclose(self) -> None:
            return None

    class Created:
        status_code = 201

        def json(self) -> dict[str, str]:
            return {"thread_id": "thread-1"}

    class FakeAsyncClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            self._opened = False

        async def post(self, *a: Any, **k: Any) -> Created:
            return Created()

        async def aclose(self) -> None:
            return None

        def build_request(self, *a: Any, **k: Any) -> MagicMock:
            return MagicMock()

        async def send(self, *a: Any, **k: Any) -> StreamResp:
            return StreamResp()

    monkeypatch.setattr(invoke_routes.httpx, "AsyncClient", FakeAsyncClient)
    response = client.post(
        "/invoke",
        json={"message": "hi"},
        headers={"Authorization": "Bearer t"},
    )
    assert response.status_code == 200
    assert response.headers.get("x-thread-id") == "thread-1"
    assert b"hi" in response.content or b"success" in response.content


def test_invoke_route_thread_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    class BoomClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        async def post(self, *a: Any, **k: Any) -> Any:
            raise httpx.ConnectError("down")

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(invoke_routes.httpx, "AsyncClient", BoomClient)
    assert client.post("/invoke", json={"message": "hi"}).status_code == 502

    class BadStatus:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        async def post(self, *a: Any, **k: Any) -> Any:
            return type("R", (), {"status_code": 500, "json": lambda self: {}})()

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(invoke_routes.httpx, "AsyncClient", BadStatus)
    assert client.post("/invoke", json={"message": "hi"}).status_code == 500

    class NoThread:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        async def post(self, *a: Any, **k: Any) -> Any:
            return type(
                "R",
                (),
                {"status_code": 200, "json": lambda self: {}},
            )()

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(invoke_routes.httpx, "AsyncClient", NoThread)
    assert client.post("/invoke", json={"message": "hi"}).status_code == 502


def test_invoke_route_open_run_failure_and_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    class FailOpen:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        async def aclose(self) -> None:
            return None

        def build_request(self, *a: Any, **k: Any) -> MagicMock:
            return MagicMock()

        async def send(self, *a: Any, **k: Any) -> Any:
            raise httpx.ConnectError("fail")

    monkeypatch.setattr(invoke_routes.httpx, "AsyncClient", FailOpen)
    assert (
        client.post(
            "/invoke", json={"message": "hi", "thread_id": "already"}
        ).status_code
        == 502
    )

    class PauseThenOk:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        def __init__(self, kind: str) -> None:
            self.kind = kind

        async def aiter_bytes(self):
            if self.kind == "pause":
                yield (
                    b"event: updates\ndata: "
                    + json.dumps(
                        {
                            "__interrupt__": [
                                {"value": {"action_requests": [{"name": "t"}]}}
                            ]
                        }
                    ).encode()
                    + b"\n\n"
                    + b'event: end\ndata: {"status":"interrupted"}\n\n'
                )
            else:
                yield b'event: end\ndata: {"status":"success"}\n\n'

        async def aclose(self) -> None:
            return None

    calls = {"n": 0}

    class ResumeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        async def aclose(self) -> None:
            return None

        def build_request(self, *a: Any, **k: Any) -> MagicMock:
            return MagicMock()

        async def send(self, *a: Any, **k: Any) -> PauseThenOk:
            calls["n"] += 1
            return PauseThenOk("pause" if calls["n"] == 1 else "ok")

    monkeypatch.setattr(invoke_routes.httpx, "AsyncClient", ResumeClient)
    response = client.post("/invoke", json={"message": "hi", "thread_id": "tid"})
    assert response.status_code == 200
    assert calls["n"] >= 2


def test_open_run_builds_request(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    class Client:
        def build_request(self, *a: Any, **k: Any) -> str:
            return "req"

        async def send(self, req: str, stream: bool = False) -> str:
            assert req == "req"
            assert stream is True
            return "resp"

    result = asyncio.run(
        invoke_routes._open_run(Client(), "http://base", "tid", {"x": 1}, {})  # type: ignore[arg-type]
    )
    assert result == "resp"
