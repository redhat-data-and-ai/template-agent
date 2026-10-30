"""Headless runner covers local and AI Factory calls, and both auth shapes."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from deep_agent.headless.interrupts import (
    aif_resume_body,
    count_actions,
    find_mcp_auth,
    local_resume_body,
    local_start_body,
)
from deep_agent.headless.settings import HeadlessSettings
from deep_agent.headless.stream import (
    interrupts_from_events,
    message_texts,
    parse_sse_lines,
    thread_id_from_events,
)


def test_password_grant_uses_the_prod_service_account() -> None:
    settings = HeadlessSettings(
        _env_file=None,
        environment="prod",
        sso_client_id="ask-data-ui",
        sso_client_secret="unsuffixed",
        sso_client_secret_prod="prod-secret",
        sso_sa_username="dataverse-agent-sa",
        sso_sa_password_prod="'quoted-password'",
        sso_token_url_prod="https://auth.example/token",
        sso_issuer_url="https://auth.example/realms/EmployeeIDP",
    )
    account = settings.service_account()
    assert account["username"] == "dataverse-agent-sa"
    assert account["password"] == "quoted-password"
    assert account["client_secret"] == "prod-secret"
    assert account["token_url"] == "https://auth.example/token"


def test_local_workflow_uses_browser_sso() -> None:
    from deep_agent.headless.workflow import apply_workflow

    settings = apply_workflow(
        HeadlessSettings(
            _env_file=None,
            environment="preprod",
            agent_port=5002,
            sso_issuer_url="https://auth.stage.redhat.com/auth/realms/EmployeeIDP",
        ),
        "local",
    )
    assert settings.auth_mode == "sso"
    assert settings.headless_target == "invoke"
    assert settings.environment == "preprod"
    assert settings.invoke_url() == "http://127.0.0.1:5002/invoke"


def test_deployed_workflows_use_service_account_password_grant() -> None:
    from deep_agent.headless.workflow import apply_workflow

    for name in ("preprod", "prod"):
        settings = apply_workflow(
            HeadlessSettings(
                _env_file=None,
                agent_url="https://agent.example",
            ),
            name,
        )
        assert settings.auth_mode == "password"
        assert settings.environment == name


def test_local_workflow_matches_prod_issuer_to_prod_sa() -> None:
    from deep_agent.headless.workflow import apply_workflow

    settings = apply_workflow(
        HeadlessSettings(
            _env_file=None,
            environment="preprod",
            sso_issuer_url="https://auth.redhat.com/auth/realms/EmployeeIDP",
        ),
        "local",
    )
    assert settings.environment == "prod"


def test_prod_workflow_uses_the_service_account() -> None:
    from deep_agent.headless.workflow import apply_workflow

    settings = apply_workflow(
        HeadlessSettings(
            _env_file=None,
            agent_base_url="https://aif.example",
            agent_org="aurora",
            agent_name="dva",
        ),
        "prod",
    )
    assert settings.environment == "prod"
    assert settings.auth_mode == "password"
    assert settings.headless_target == "invoke"
    assert (
        settings.invoke_url()
        == "https://aif.example/api/engine/v1/agents/aurora/dva/invoke"
    )


def test_prod_workflow_refuses_the_laptop_agent() -> None:
    from deep_agent.headless.workflow import WorkflowError, apply_workflow

    try:
        apply_workflow(HeadlessSettings(_env_file=None), "prod")
    except WorkflowError as exc:
        assert "remote agent" in str(exc)
    else:
        raise AssertionError("expected a workflow error")


def test_preprod_workflow_can_call_a_remote_agent_url() -> None:
    from deep_agent.headless.workflow import apply_workflow

    settings = apply_workflow(
        HeadlessSettings(_env_file=None, agent_url="https://agent.example"),
        "preprod",
    )
    assert settings.environment == "preprod"
    assert settings.headless_target == "invoke"
    assert settings.invoke_url() == "https://agent.example/invoke"


def test_sso_exchange_sends_the_environment_client_secret(monkeypatch) -> None:
    from deep_agent.headless.auth import exchange_authorization_code

    captured: dict[str, object] = {}

    class _Response:
        status_code = 200

        def json(self) -> dict[str, str]:
            return {"access_token": "user-token"}

    def fake_post(url: str, data: dict[str, str], timeout: float) -> _Response:
        captured["url"] = url
        captured["data"] = data
        return _Response()

    monkeypatch.setattr("deep_agent.headless.auth.httpx.post", fake_post)
    settings = HeadlessSettings(
        _env_file=None,
        environment="prod",
        sso_client_id="ask-data-ui",
        sso_client_secret_prod="prod-secret",
        sso_token_url_prod="https://auth.example/token",
    )
    assert exchange_authorization_code(settings, "code", "verifier") == "user-token"
    sent = captured["data"]
    assert isinstance(sent, dict)
    assert sent["client_secret"] == "prod-secret"
    assert captured["url"] == "https://auth.example/token"


def test_issuer_urls_are_derived_when_endpoints_are_blank() -> None:
    settings = HeadlessSettings(
        _env_file=None,
        sso_issuer_url="https://auth.example/auth/realms/EmployeeIDP/",
        sso_authorization_url="",
        sso_token_url="",
    )

    assert settings.authorization_url().endswith("/protocol/openid-connect/auth")
    assert settings.token_url().endswith("/protocol/openid-connect/token")


def test_aif_engine_url_uses_org_and_name() -> None:
    settings = HeadlessSettings(
        _env_file=None,
        workflow="preprod",
        agent_base_url="https://aif.example",
        agent_org="aurora",
        agent_name="dva",
    )
    assert (
        settings.invoke_url()
        == "https://aif.example/api/engine/v1/agents/aurora/dva/invoke"
    )


def test_direct_agent_url_wins_over_aif_engine() -> None:
    settings = HeadlessSettings(
        _env_file=None,
        workflow="prod",
        agent_url="https://my-agent.example",
        agent_base_url="https://aif.example",
        agent_org="aurora",
        agent_name="dva",
    )
    assert settings.invoke_url() == "https://my-agent.example/invoke"


def test_chat_url_gains_an_invoke_suffix() -> None:
    settings = HeadlessSettings(
        _env_file=None,
        agent_url="https://agent.example/preprod/aurora/dva/chat",
    )
    assert (
        settings.invoke_url() == "https://agent.example/preprod/aurora/dva/chat/invoke"
    )


def test_local_invoke_url_uses_agent_port() -> None:
    settings = HeadlessSettings(_env_file=None, workflow="local", agent_port=5002)
    assert settings.invoke_url() == "http://127.0.0.1:5002/invoke"

    from_port = HeadlessSettings(
        _env_file=None,
        workflow="local",
        agent_port=8081,
    )
    assert from_port.invoke_url() == "http://127.0.0.1:8081/invoke"

    explicit = HeadlessSettings(
        _env_file=None,
        workflow="local",
        agent_url="http://127.0.0.1:8080",
    )
    assert explicit.invoke_url() == "http://127.0.0.1:8080/invoke"


def test_invoke_message_becomes_a_run_body() -> None:
    from deep_agent.aegra.invoke_routes import run_body

    started = run_body({"message": "who is dpundir's manager"})
    assert started is not None
    assert started["input"]["messages"][0]["content"] == "who is dpundir's manager"

    resumed = run_body(
        {
            "thread_id": "thread-1",
            "command": {"resume": {"decisions": [{"type": "approve"}]}},
            "assistant_id": "agent",
        }
    )
    assert resumed is not None
    assert resumed["command"]["resume"]["decisions"] == [{"type": "approve"}]
    assert run_body({}) is None


def test_invoke_keepalive_says_thinking() -> None:
    from deep_agent.aegra.invoke_routes import rewrite_keepalive

    pending = bytearray()
    first = rewrite_keepalive(pending, b"event: end\n: heart")
    second = rewrite_keepalive(pending, b"beat\r\n\r\n", final=True)
    assert first + second == b"event: end\n: thinking\r\n\r\n"


def test_invoke_approves_a_tool_pause_and_stops_on_mcp_login() -> None:
    from deep_agent.aegra.invoke_routes import approval_resume

    paused = [
        (
            "updates",
            {
                "__interrupt__": [
                    {"value": {"action_requests": [{"name": "read_file"}]}}
                ]
            },
        ),
        ("end", {"status": "interrupted"}),
    ]
    resume = approval_resume(paused)
    assert resume is not None
    assert resume["command"]["resume"]["decisions"] == [{"type": "approve"}]

    finished = paused + [("end", {"status": "success"})]
    assert approval_resume(finished) is None
    assert (
        approval_resume(
            [
                (
                    "updates",
                    {
                        "__interrupt__": [
                            {
                                "value": {
                                    "type": "mcp_auth_required",
                                    "mcp_name": "atlan-mcp",
                                }
                            }
                        ]
                    },
                ),
                ("end", {"status": "interrupted"}),
            ]
        )
        is None
    )


def test_explicit_token_url_wins_over_issuer() -> None:
    settings = HeadlessSettings(
        _env_file=None,
        sso_issuer_url="https://auth.example/realms/EmployeeIDP",
        sso_token_url="https://auth.example/token",
    )
    assert settings.token_url() == "https://auth.example/token"


def test_mcp_auth_is_detected_as_json_string_and_as_dict() -> None:
    encoded = json.dumps(
        {"type": "mcp_auth_required", "mcp_name": "atlan-mcp", "message": "Connect"}
    )
    from_string = find_mcp_auth([{"value": encoded}])
    from_dict = find_mcp_auth(
        [{"value": {"type": "mcp_auth_required", "mcp_name": "atlan-mcp"}}]
    )

    assert from_string is not None and from_string["mcp_name"] == "atlan-mcp"
    assert from_dict is not None and from_dict["mcp_name"] == "atlan-mcp"
    assert (
        find_mcp_auth([{"value": {"action_requests": [{"name": "fluent_search"}]}}])
        is None
    )


def test_hitl_action_count_and_both_resume_bodies() -> None:
    interrupts = [
        {
            "value": {
                "action_requests": [{"name": "fluent_search"}, {"name": "execute_sql"}]
            }
        }
    ]
    assert count_actions(interrupts) == 2

    local = local_resume_body(2)
    aif = aif_resume_body("thread-1", 2)

    assert local["command"]["resume"]["decisions"] == [
        {"type": "approve"},
        {"type": "approve"},
    ]
    assert aif["thread_id"] == "thread-1"
    assert aif["assistant_id"] == "agent"
    assert aif["stream_mode"] == "updates"
    start = local_start_body("hello")
    assert start["input"]["messages"][0]["content"] == "hello"


def test_sse_parser_reads_local_updates_and_aif_messages() -> None:
    lines = [
        "event: metadata",
        'data: {"run_id": "r", "thread_id": "t-9"}',
        "",
        "event: updates",
        'data: {"model": {"messages": [{"type": "ai", "content": "catalog answer"}]}}',
        "",
        "event: updates",
        'data: {"__interrupt__": [{"value": {"action_requests": [{"name": "fluent_search"}]}}]}',
    ]
    events = parse_sse_lines(lines)

    assert thread_id_from_events(events) == "t-9"
    assert message_texts(events) == ["catalog answer"]
    assert len(interrupts_from_events(events)) == 1


def test_interrupts_from_events_keeps_only_the_latest_pause() -> None:
    events = [
        (
            "updates",
            {
                "__interrupt__": [
                    {"value": {"action_requests": [{"name": "read_file"}]}},
                    {"value": {"action_requests": [{"name": "task"}]}},
                ]
            },
        ),
        ("end", {"status": "interrupted"}),
        (
            "updates",
            {"__interrupt__": [{"value": {"action_requests": [{"name": "task"}]}}]},
        ),
        ("end", {"status": "interrupted"}),
    ]
    latest = interrupts_from_events(events)
    assert len(latest) == 1
    assert count_actions(latest) == 1


def test_callback_reports_a_busy_port_before_opening_a_browser() -> None:
    from deep_agent.headless.auth import _Callback

    occupied = HTTPServer(("localhost", 0), BaseHTTPRequestHandler)
    port = int(occupied.server_address[1])
    try:
        message = _Callback(port).start()
    finally:
        occupied.server_close()

    assert message is not None
    assert "already in use" in message


def test_aif_invoker_auto_approves_tool_pauses(monkeypatch) -> None:
    from deep_agent.headless.invoke import HeadlessInvoker
    from deep_agent.headless.workflow import apply_workflow

    settings = apply_workflow(
        HeadlessSettings(
            _env_file=None,
            agent_base_url="https://aif.example",
            agent_org="aurora",
            agent_name="dva-test",
            agent_message="who is dpundir's manager",
            agent_max_resumes=3,
        ),
        "preprod",
    )
    invoker = HeadlessInvoker(settings, "sa-token")
    bodies: list[dict] = []

    paused = [
        (
            "updates",
            {
                "__interrupt__": [
                    {"value": {"action_requests": [{"name": "fluent_search"}]}}
                ]
            },
        ),
        ("end", {"status": "interrupted"}),
    ]
    finished = [
        (
            "updates",
            {"model": {"messages": [{"type": "ai", "content": "done"}]}},
        ),
        ("end", {"status": "success"}),
    ]
    streams = iter(
        [
            ("thread-1", paused),
            ("thread-1", finished),
        ]
    )

    def fake_read(_client, _url, body, _headers):
        bodies.append(body)
        return next(streams)

    monkeypatch.setattr("deep_agent.headless.invoke._read_stream", fake_read)
    monkeypatch.setattr(
        "deep_agent.headless.invoke.httpx.Client",
        lambda **_kwargs: type(
            "C",
            (),
            {
                "__enter__": lambda self: self,
                "__exit__": lambda self, *_a: None,
            },
        )(),
    )

    assert invoker.run() == 0
    assert bodies[0] == {"message": "who is dpundir's manager"}
    assert bodies[1]["command"]["resume"]["decisions"] == [{"type": "approve"}]
    assert bodies[1]["thread_id"] == "thread-1"
    assert bodies[1]["assistant_id"] == "agent"
