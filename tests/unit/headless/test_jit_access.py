"""DICE JIT grant and revoke follow the Dataverse test suite."""

from __future__ import annotations

import httpx

from deep_agent.headless.jit_access import JITAccessError, JITSession
from deep_agent.headless.settings import HeadlessSettings


def _settings(**overrides: str) -> HeadlessSettings:
    values = {
        "environment": "prod",
        "dice_jit_trigger_token": "trigger-token",
        "dice_project_id": "106670",
    }
    values.update(overrides)
    return HeadlessSettings(_env_file=None, **values)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_missing_token_skips_gitlab(capsys) -> None:
    session = JITSession(_settings(dice_jit_trigger_token=""))
    assert session.grant() is True
    session.revoke()
    assert "was not requested" in capsys.readouterr().out


def test_grant_posts_action_and_environment() -> None:
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, str(request.url)))
        if request.method == "POST":
            body = request.content.decode()
            assert "variables%5BJIT_DVA_ACTION%5D=grant" in body
            assert "variables%5BJIT_DVA_ENV%5D=prod" in body
            assert "trigger-token" in body
            return httpx.Response(201, json={"id": 42})
        return httpx.Response(200, json={"status": "success"})

    session = JITSession(_settings(), client=_client(handler), poll_interval=0)
    assert session.grant() is True
    assert seen[0] == (
        "POST",
        "https://gitlab.cee.redhat.com/api/v4/projects/106670/trigger/pipeline",
    )


def test_revoke_sends_revoke_after_a_grant() -> None:
    actions: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            body = request.content.decode()
            if "JIT_DVA_ACTION%5D=revoke" in body or "JIT_DVA_ACTION=revoke" in body:
                actions.append("revoke")
            else:
                actions.append("grant")
            return httpx.Response(201, json={"id": 7})
        return httpx.Response(200, json={"status": "success"})

    session = JITSession(_settings(), client=_client(handler), poll_interval=0)
    assert session.grant() is True
    session.revoke()
    assert actions == ["grant", "revoke"]


def test_failed_grant_still_asks_for_revoke() -> None:
    actions: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            body = request.content.decode()
            actions.append("revoke" if "revoke" in body else "grant")
            return httpx.Response(403, text="denied trigger-token")
        return httpx.Response(200, json={"status": "success"})

    session = JITSession(_settings(), client=_client(handler), poll_interval=0)
    assert session.grant() is False
    session.revoke()
    assert actions[0] == "grant"
    assert "revoke" in actions


def test_unknown_environment_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(201, json={"id": 1})

    session = JITSession(
        _settings(environment="lab"),
        client=_client(handler),
        poll_interval=0,
    )
    assert session.grant() is False


def test_poll_timeout_is_a_jit_error() -> None:
    clock = {"now": 0.0}

    def monotonic() -> float:
        return clock["now"]

    def sleep(seconds: float) -> None:
        clock["now"] += seconds

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(201, json={"id": 9})
        return httpx.Response(200, json={"status": "running"})

    session = JITSession(
        _settings(),
        client=_client(handler),
        sleep=sleep,
        monotonic=monotonic,
        poll_interval=10,
        poll_timeout=20,
    )
    try:
        session._run("grant", 1)
    except JITAccessError as exc:
        assert "did not finish" in str(exc)
    else:
        raise AssertionError("expected a timeout")
