"""Cross-repo contract for MCP static-credential auth (RHITAIF-894).

agent-engine materializes config/agent/mcp.json; this service consumes it.
The key names below are the contract between the two.

Counterpart:
  agent-engine  src/ai_agent_engine/materialize/build.py
                _build_single_mcp_config / _MCP_VAULT_SECRET_TYPES
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from deep_agent.aegra.mcp import (
    _API_KEY_LOCATIONS,
    _build_server_config,
    _resolve_connection_token,
)


def _entry(**extra):
    return {
        "url": "https://x.example.com/mcp",
        "transport": "streamable_http",
        "auth": True,
        "ssl_verify": True,
        **extra,
    }


def test_accepted_key_locations_match_the_builder():
    """Must equal registry MCP_API_KEY_LOCATIONS and the UI dropdown."""
    assert _API_KEY_LOCATIONS == ("header", "query")


@pytest.mark.asyncio
async def test_api_key_mode_reads_the_env_var_agent_engine_injects():
    entry = _entry(auth_mode="api_key", auth_env_var="MCP_MY_MCP_BEARER_TOKEN")
    with patch.dict(os.environ, {"MCP_MY_MCP_BEARER_TOKEN": "tok-abc"}):
        token = await _resolve_connection_token("m", entry, None, None)
    assert token == "tok-abc"


@pytest.mark.asyncio
async def test_api_key_mode_never_falls_back_to_sso():
    """AC: 'SSO is not used as a fallback' for bearer-configured MCPs."""
    entry = _entry(auth_mode="api_key", auth_env_var="MCP_UNSET_VAR")
    with patch.dict(os.environ, {}, clear=True):
        token = await _resolve_connection_token("m", entry, "sso-token", "user-1")
    assert token is None


@pytest.mark.asyncio
async def test_sso_mode_still_uses_the_sso_token():
    """Regression guard: existing SSO passthrough MCPs are unaffected."""
    entry = _entry(auth_mode="sso")
    token = await _resolve_connection_token("m", entry, "sso-token", "user-1")
    assert token == "sso-token"


def test_bearer_shape_from_agent_engine_sends_authorization_bearer():
    cfg = _build_server_config(
        _entry(auth_mode="bearer", auth_env_var="MCP_M_BEARER_TOKEN"), "tok"
    )
    assert cfg["headers"]["Authorization"] == "Bearer tok"


def test_api_key_shape_from_agent_engine_uses_configured_placement():
    cfg = _build_server_config(
        _entry(
            auth_mode="api_key",
            auth_env_var="MCP_M_API_KEY",
            api_key_name="X-API-Key",
            api_key_location="header",
        ),
        "k",
    )
    assert cfg["headers"]["X-API-Key"] == "k"
    assert "Authorization" not in cfg["headers"]


# --- non-string scheme and duplicate query keys (CodeRabbit, PR #399) ---


def test_query_placement_replaces_an_existing_key():
    """A stale key already in the URL must not win on first-value servers."""
    from deep_agent.aegra.mcp import _with_query_param

    assert (
        _with_query_param("https://x.example.com/mcp?apikey=old", "apikey", "new")
        == "https://x.example.com/mcp?apikey=new"
    )


def test_query_placement_preserves_other_params():
    from deep_agent.aegra.mcp import _with_query_param

    assert (
        _with_query_param("https://x.example.com/mcp?v=2&apikey=old", "apikey", "new")
        == "https://x.example.com/mcp?v=2&apikey=new"
    )


def test_query_placement_on_configured_url_with_stale_key():
    cfg = _build_server_config(
        _entry(
            url="https://x.example.com/mcp?apikey=stale",
            auth_mode="api_key",
            auth_env_var="MCP_X_API_KEY",
            api_key_name="apikey",
            api_key_location="query",
        ),
        "fresh",
    )
    assert cfg["url"] == "https://x.example.com/mcp?apikey=fresh"
    assert "stale" not in cfg["url"]


# --- query-placed credentials must not reach the logs (PR #399 review) ---


def test_redact_url_query_masks_every_value():
    from deep_agent.aegra.mcp import redact_url_query

    assert (
        redact_url_query("https://x.example.com/mcp?apikey=s3cret&v=2")
        == "https://x.example.com/mcp?apikey=REDACTED&v=REDACTED"
    )


def test_redact_url_query_keeps_host_and_path():
    from deep_agent.aegra.mcp import redact_url_query

    out = redact_url_query("https://x.example.com/mcp?apikey=s3cret")
    assert "x.example.com/mcp" in out
    assert "s3cret" not in out


def test_redact_url_query_passes_through_a_bare_url():
    from deep_agent.aegra.mcp import redact_url_query

    assert redact_url_query("https://x.example.com/mcp") == "https://x.example.com/mcp"


def test_http_client_log_filter_masks_request_urls():
    """httpx logs the full URL at INFO; the filter must scrub it."""
    import logging

    from deep_agent.utils.pylogger import RedactUrlQueryFilter

    record = logging.LogRecord(
        name="httpx",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='HTTP Request: POST https://x.example.com/mcp?apikey=s3cret "200 OK"',
        args=(),
        exc_info=None,
    )
    assert RedactUrlQueryFilter().filter(record) is True
    assert "s3cret" not in record.msg
    assert "apikey=REDACTED" in record.msg


def test_http_client_log_filter_masks_lazy_args():
    import logging

    from deep_agent.utils.pylogger import RedactUrlQueryFilter

    record = logging.LogRecord(
        name="httpx",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="calling %s",
        args=("https://x.example.com/mcp?apikey=s3cret",),
        exc_info=None,
    )
    RedactUrlQueryFilter().filter(record)
    assert "s3cret" not in record.getMessage()


def test_log_filter_masks_non_string_url_objects():
    """httpx passes Request.url as httpx.URL, not str, via lazy %s."""
    import logging

    import httpx

    from deep_agent.utils.pylogger import RedactUrlQueryFilter

    record = logging.LogRecord(
        name="httpx",
        level=logging.INFO,
        pathname="f.py",
        lineno=1,
        msg='HTTP Request: %s %s "%s %d %s"',
        args=(
            "POST",
            httpx.URL("https://x.example.com/mcp?apikey=s3cret"),
            "HTTP/1.1",
            200,
            "OK",
        ),
        exc_info=None,
    )
    RedactUrlQueryFilter().filter(record)
    msg = record.getMessage()
    assert "s3cret" not in msg
    assert "apikey=REDACTED" in msg


def test_log_filter_masks_exception_tracebacks():
    """httpx.HTTPStatusError embeds the raw URL in its message."""
    import logging
    import sys

    import httpx

    from deep_agent.utils.pylogger import RedactUrlQueryFilter

    request = httpx.Request("POST", "https://x.example.com/mcp?apikey=s3cret")
    try:
        httpx.Response(500, request=request).raise_for_status()
    except httpx.HTTPStatusError:
        record = logging.LogRecord(
            name="deep_agent.aegra.mcp",
            level=logging.ERROR,
            pathname="f.py",
            lineno=1,
            msg="connection failed",
            args=(),
            exc_info=sys.exc_info(),
        )
        RedactUrlQueryFilter().filter(record)
        assert record.exc_text is not None
        assert "s3cret" not in record.exc_text
        assert "apikey=REDACTED" in record.exc_text


# --- basic token auth type ---


def test_basic_mode_sends_authorization_basic():
    cfg = _build_server_config(
        _entry(auth_mode="basic", auth_env_var="MCP_M_BASIC_TOKEN"), "tok"
    )
    assert cfg["headers"]["Authorization"] == "Basic tok"


def test_legacy_api_key_without_name_still_sends_bearer():
    """Agents deployed before the mode split keep working until redeployed."""
    cfg = _build_server_config(
        _entry(auth_mode="api_key", auth_env_var="MCP_M_BEARER_TOKEN"), "tok"
    )
    assert cfg["headers"]["Authorization"] == "Bearer tok"


@pytest.mark.asyncio
async def test_basic_mode_reads_the_injected_env_var():
    entry = _entry(auth_mode="basic", auth_env_var="MCP_M_BASIC_TOKEN")
    with patch.dict(os.environ, {"MCP_M_BASIC_TOKEN": "b64tok"}):
        token = await _resolve_connection_token("m", entry, None, None)
    assert token == "b64tok"


@pytest.mark.asyncio
async def test_basic_mode_never_falls_back_to_sso():
    entry = _entry(auth_mode="basic", auth_env_var="MCP_UNSET")
    with patch.dict(os.environ, {}, clear=True):
        token = await _resolve_connection_token("m", entry, "sso-token", "user-1")
    assert token is None


# --- the shipped sample config must stay valid against the runtime ---


def _load_sample_mcp_json() -> dict:
    """Parse config/agent/mcp.json, which is JSONC (allows // comments)."""
    import json
    import pathlib
    import re

    path = pathlib.Path(__file__).parents[3] / "config" / "agent" / "mcp.json"
    text = re.sub(r"^\s*//.*$", "", path.read_text(), flags=re.M)
    return json.loads(text)["mcpServers"]


def test_sample_config_covers_every_static_credential_mode():
    servers = _load_sample_mcp_json()
    modes = {
        name: cfg.get("auth_mode")
        for name, cfg in servers.items()
        if cfg.get("auth_mode") in ("api_key", "bearer", "basic")
    }
    assert modes == {
        "template-mcp-server-bearer": "bearer",
        "template-mcp-server-basic": "basic",
        "template-mcp-server-api-key": "api_key",
        "template-mcp-server-api-key-query": "api_key",
    }


def test_sample_static_credential_entries_declare_an_env_var():
    for name, cfg in _load_sample_mcp_json().items():
        if cfg.get("auth_mode") in ("api_key", "bearer", "basic"):
            assert cfg.get("auth_env_var"), name


def test_sample_entries_produce_the_documented_headers():
    """Each sample, fed a token, must yield what its comment claims."""
    servers = _load_sample_mcp_json()

    bearer = _build_server_config(servers["template-mcp-server-bearer"], "tok")
    assert bearer["headers"]["Authorization"] == "Bearer tok"

    basic = _build_server_config(servers["template-mcp-server-basic"], "tok")
    assert basic["headers"]["Authorization"] == "Basic tok"

    api_key = _build_server_config(servers["template-mcp-server-api-key"], "tok")
    assert api_key["headers"]["X-API-Key"] == "tok"
    assert "Authorization" not in api_key["headers"]

    query = _build_server_config(servers["template-mcp-server-api-key-query"], "tok")
    assert query["url"].endswith("?apikey=tok")
    assert "Authorization" not in query["headers"]


def test_sample_api_key_locations_are_all_supported():
    for name, cfg in _load_sample_mcp_json().items():
        loc = cfg.get("api_key_location")
        if loc is not None:
            assert loc in _API_KEY_LOCATIONS, f"{name} uses {loc}"


def test_host_proxy_blocks_basic_with_no_credential():
    """Regression: basic must be guarded like api_key, not fall through."""
    import asyncio

    from fastapi import HTTPException

    from deep_agent.aegra.mcp_host import _resolve_bearer

    entry = _entry(auth_mode="basic", auth_env_var="MCP_UNSET")
    with patch.dict(os.environ, {}, clear=True):
        try:
            asyncio.run(_resolve_bearer("m", entry, user_id="u", sso_token=None))
        except HTTPException as exc:
            assert exc.status_code == 500
            assert "basic" in str(exc.detail)
        else:
            raise AssertionError("expected the host proxy to block basic")


def test_host_proxy_still_blocks_api_key_with_no_credential():
    import asyncio

    from fastapi import HTTPException

    from deep_agent.aegra.mcp_host import _resolve_bearer

    entry = _entry(auth_mode="api_key", auth_env_var="MCP_UNSET", api_key_name="X-Key")
    with patch.dict(os.environ, {}, clear=True):
        try:
            asyncio.run(_resolve_bearer("m", entry, user_id="u", sso_token=None))
        except HTTPException as exc:
            assert exc.status_code == 500
            assert "api_key" in str(exc.detail)
        else:
            raise AssertionError("expected the host proxy to block api_key")


def test_every_static_credential_mode_is_distinct():
    """bearer / basic / api_key are three separate runtime modes."""
    schemes = {}
    for mode in ("bearer", "basic"):
        cfg = _build_server_config(_entry(auth_mode=mode, auth_env_var="V"), "tok")
        schemes[mode] = cfg["headers"]["Authorization"]
    assert schemes == {"bearer": "Bearer tok", "basic": "Basic tok"}

    api_key = _build_server_config(
        _entry(auth_mode="api_key", auth_env_var="V", api_key_name="X-Key"), "tok"
    )
    assert api_key["headers"] == {"X-Key": "tok"}


def test_redact_url_query_handles_an_unparseable_url():
    """A malformed URL must not raise inside a log path."""
    from deep_agent.aegra.mcp import redact_url_query

    assert redact_url_query("http://[") == "<unparseable-url>"


def test_log_filter_masks_dict_style_args():
    """logging supports %(name)s formatting with a dict for args."""
    import logging

    from deep_agent.utils.pylogger import RedactUrlQueryFilter

    record = logging.LogRecord(
        name="httpx",
        level=logging.INFO,
        pathname="f.py",
        lineno=1,
        msg="calling %(url)s",
        # logging accepts the mapping form wrapped in a tuple and unwraps it,
        # which is how record.args ends up a dict.
        args=({"url": "https://x.example.com/mcp?apikey=s3cret"},),
        exc_info=None,
    )
    RedactUrlQueryFilter().filter(record)
    assert "s3cret" not in record.getMessage()
    assert "apikey=REDACTED" in record.getMessage()


def test_log_filter_masks_preformatted_exc_text():
    """A formatter may have already rendered exc_info into exc_text."""
    import logging

    from deep_agent.utils.pylogger import RedactUrlQueryFilter

    record = logging.LogRecord(
        name="httpx",
        level=logging.ERROR,
        pathname="f.py",
        lineno=1,
        msg="boom",
        args=(),
        exc_info=None,
    )
    record.exc_text = "Traceback: url 'https://x.example.com/mcp?apikey=s3cret'"
    RedactUrlQueryFilter().filter(record)
    assert "s3cret" not in record.exc_text
    assert "apikey=REDACTED" in record.exc_text


def test_legacy_api_key_without_name_does_not_warn_per_request(caplog):
    """Still works for pre-split agents, but silently.

    The config-shape warning belongs in the loader, which runs once —
    _build_server_config runs on every connection and host-proxy request.
    """
    import logging

    from deep_agent.aegra.mcp import _build_server_config

    with caplog.at_level(logging.WARNING):
        cfg = _build_server_config(_entry(auth_mode="api_key", auth_env_var="V"), "tok")
    assert cfg["headers"]["Authorization"] == "Bearer tok"
    assert caplog.text == ""


def test_loader_reports_nameless_api_key_once(caplog):
    """The same condition is reported at load time instead."""
    import logging

    from deep_agent.src.agent.config import AgentConfig

    with caplog.at_level(logging.ERROR):
        AgentConfig._validate_mcp_server(
            "m", {"auth_mode": "api_key", "auth_env_var": "V"}
        )
    assert "requires 'api_key_name'" in caplog.text


def test_redaction_keeps_structlog_json_valid():
    """The filter runs on rendered JSON; it must not eat delimiters."""
    import json

    from deep_agent.utils.pylogger import _mask_url_queries

    cases = [
        {"event": "calling https://x.example.com/mcp?apikey=s3cret", "logger": "m"},
        {"event": "https://x/mcp?k=v", "next": "field"},
        {"event": "not reachable (https://x/mcp?apikey=s3cret) — skipped"},
        {"event": "multi https://x/mcp?a=1&b=2 done"},
    ]
    for payload in cases:
        masked = _mask_url_queries(json.dumps(payload))
        json.loads(masked)  # must not raise
        assert "s3cret" not in masked
        assert "REDACTED" in masked


def test_redaction_preserves_trailing_punctuation():
    from deep_agent.utils.pylogger import _mask_url_queries

    assert (
        _mask_url_queries("see (https://x/mcp?k=v) now")
        == "see (https://x/mcp?k=REDACTED) now"
    )
    assert _mask_url_queries('"https://x/mcp?k=v"') == '"https://x/mcp?k=REDACTED"'


# --- review follow-ups ---


def test_placement_fields_are_rejected_for_bearer_and_basic(caplog):
    """Validator and builder must agree: placement is api_key-only."""
    import logging

    from deep_agent.src.agent.config import AgentConfig

    for mode in ("bearer", "basic"):
        caplog.clear()
        with caplog.at_level(logging.ERROR):
            AgentConfig._validate_mcp_server(
                "m",
                {
                    "auth_mode": mode,
                    "auth_env_var": "V",
                    "api_key_name": "k",
                    "api_key_location": "query",
                },
            )
        assert "only valid with auth_mode 'api_key'" in caplog.text, mode


def test_unnormalised_auth_mode_is_normalised_not_silently_ignored(caplog):
    import logging

    from deep_agent.src.agent.config import AgentConfig

    cfg = {"auth_mode": " Bearer ", "auth_env_var": "V"}
    with caplog.at_level(logging.ERROR):
        AgentConfig._validate_mcp_server("m", cfg)
    assert cfg["auth_mode"] == "bearer"
    assert caplog.text == ""


def test_unknown_auth_mode_disables_the_server(caplog):
    import logging

    from deep_agent.src.agent.config import AgentConfig

    cfg = {"auth_mode": "magic", "enabled": True}
    with caplog.at_level(logging.ERROR):
        AgentConfig._validate_mcp_server("m", cfg)
    assert cfg["enabled"] is False


def test_api_key_name_charset_is_validated(caplog):
    import logging

    from deep_agent.src.agent.config import AgentConfig

    with caplog.at_level(logging.ERROR):
        AgentConfig._validate_mcp_server(
            "m",
            {"auth_mode": "api_key", "auth_env_var": "V", "api_key_name": "bad name"},
        )
    assert "not a valid HTTP header" in caplog.text


def test_query_param_preserves_existing_encoding():
    """Round-tripping would rewrite ?flag as flag= and %20 as +."""
    from deep_agent.aegra.mcp import _with_query_param

    assert _with_query_param("https://x/mcp?flag", "k", "v") == "https://x/mcp?flag&k=v"
    assert (
        _with_query_param("https://x/mcp?a=x%20y", "k", "v")
        == "https://x/mcp?a=x%20y&k=v"
    )


def test_redaction_masks_values_containing_stop_characters():
    from deep_agent.utils.pylogger import _mask_url_queries

    assert _mask_url_queries("?apikey=SEC,RET") == "?apikey=REDACTED"
    assert "SECRET" not in _mask_url_queries("?q=1&amp;apikey=SECRET")


def test_redactor_is_idempotent_across_reconfigures():
    import logging

    from deep_agent.utils.pylogger import _attach_url_redactor_to_root_handlers

    handler = logging.StreamHandler()
    logging.getLogger().addHandler(handler)
    try:
        for _ in range(3):
            _attach_url_redactor_to_root_handlers()
        assert len(handler.filters) == 1
    finally:
        logging.getLogger().removeHandler(handler)


def test_basic_rejects_a_plaintext_credential(caplog):
    import asyncio
    import logging
    import os
    from unittest.mock import patch

    from deep_agent.aegra.mcp import _resolve_connection_token

    entry = {"auth_mode": "basic", "auth_env_var": "V"}
    with caplog.at_level(logging.ERROR), patch.dict(os.environ, {"V": "user:pass"}):
        asyncio.run(_resolve_connection_token("m", entry, None, None))
    assert "base64-encoded" in caplog.text
