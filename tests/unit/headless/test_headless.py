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


def test_issuer_urls_are_derived_when_endpoints_are_blank() -> None:
    settings = HeadlessSettings(
        _env_file=None,
        sso_issuer_url="https://auth.example/auth/realms/EmployeeIDP/",
        sso_authorization_url="",
        sso_token_url="",
    )

    assert settings.authorization_url().endswith("/protocol/openid-connect/auth")
    assert settings.token_url().endswith("/protocol/openid-connect/token")


def test_aif_invoke_url_uses_org_and_name() -> None:
    settings = HeadlessSettings(
        _env_file=None,
        agent_base_url="https://aif.example",
        agent_org="aurora",
        agent_name="dva",
    )
    assert (
        settings.aif_invoke_url()
        == "https://aif.example/api/engine/v1/agents/aurora/dva/invoke"
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
