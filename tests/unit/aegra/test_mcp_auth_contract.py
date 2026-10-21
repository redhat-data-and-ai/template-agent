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
    assert _API_KEY_LOCATIONS == ("header", "query", "body")


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
    """agent-engine omits api_key_name for bearer; default placement applies."""
    cfg = _build_server_config(
        _entry(auth_mode="api_key", auth_env_var="MCP_M_BEARER_TOKEN"), "tok"
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


# --- api_key_scheme: Basic vs Bearer (RHITAIF-894) ---


def test_basic_scheme_sends_authorization_basic():
    """Langfuse case: Authorization: Basic <base64>."""
    cfg = _build_server_config(
        _entry(
            auth_mode="api_key",
            auth_env_var="MCP_LANGFUSE_API_KEY",
            api_key_name="Authorization",
            api_key_location="header",
            api_key_scheme="basic",
        ),
        "YmFzZTY0OnRva2Vu",
    )
    assert cfg["headers"]["Authorization"] == "Basic YmFzZTY0OnRva2Vu"


def test_bearer_scheme_sends_authorization_bearer():
    cfg = _build_server_config(
        _entry(
            auth_mode="api_key",
            auth_env_var="MCP_X_API_KEY",
            api_key_name="Authorization",
            api_key_scheme="bearer",
        ),
        "tok",
    )
    assert cfg["headers"]["Authorization"] == "Bearer tok"


def test_none_scheme_sends_the_raw_key():
    cfg = _build_server_config(
        _entry(
            auth_mode="api_key",
            auth_env_var="MCP_X_API_KEY",
            api_key_name="X-API-Key",
            api_key_scheme="none",
        ),
        "raw-key",
    )
    assert cfg["headers"]["X-API-Key"] == "raw-key"


def test_absent_scheme_defaults_to_raw_key():
    """Configs written before auth_scheme existed must not change behaviour."""
    cfg = _build_server_config(
        _entry(
            auth_mode="api_key",
            auth_env_var="MCP_X_API_KEY",
            api_key_name="X-API-Key",
        ),
        "raw-key",
    )
    assert cfg["headers"]["X-API-Key"] == "raw-key"


def test_unknown_scheme_falls_back_to_raw_key():
    cfg = _build_server_config(
        _entry(
            auth_mode="api_key",
            auth_env_var="MCP_X_API_KEY",
            api_key_name="X-API-Key",
            api_key_scheme="digest",
        ),
        "raw-key",
    )
    assert cfg["headers"]["X-API-Key"] == "raw-key"


# --- scheme set without a key name (CodeRabbit, PR #399) ---


def test_basic_scheme_without_name_uses_authorization_header():
    """Previously sent 'Bearer <key>' when 'Basic' was requested."""
    cfg = _build_server_config(
        _entry(
            auth_mode="api_key",
            auth_env_var="MCP_X_API_KEY",
            api_key_scheme="basic",
        ),
        "k",
    )
    assert cfg["headers"]["Authorization"] == "Basic k"


def test_bearer_scheme_without_name_is_unchanged():
    cfg = _build_server_config(
        _entry(
            auth_mode="api_key",
            auth_env_var="MCP_X_API_KEY",
            api_key_scheme="bearer",
        ),
        "k",
    )
    assert cfg["headers"]["Authorization"] == "Bearer k"


def test_no_scheme_and_no_name_keeps_default_path():
    cfg = _build_server_config(
        _entry(auth_mode="api_key", auth_env_var="MCP_X_API_KEY"), "k"
    )
    assert cfg["headers"]["Authorization"] == "Bearer k"


def test_explicit_name_still_wins_over_the_default():
    cfg = _build_server_config(
        _entry(
            auth_mode="api_key",
            auth_env_var="MCP_X_API_KEY",
            api_key_name="X-API-Key",
            api_key_scheme="none",
        ),
        "k",
    )
    assert cfg["headers"]["X-API-Key"] == "k"
    assert "Authorization" not in cfg["headers"]


# --- non-string scheme and duplicate query keys (CodeRabbit, PR #399) ---


def test_non_string_scheme_falls_back_to_raw_key():
    """A numeric scheme in mcp.json must not raise during connection setup."""
    from deep_agent.aegra.mcp import _apply_api_key_scheme

    assert _apply_api_key_scheme("k", 123) == "k"
    assert _apply_api_key_scheme("k", True) == "k"
    assert _apply_api_key_scheme("k", None) == "k"


def test_non_string_scheme_in_server_config_does_not_raise():
    cfg = _build_server_config(
        _entry(
            auth_mode="api_key",
            auth_env_var="MCP_X_API_KEY",
            api_key_name="X-API-Key",
            api_key_scheme=123,
        ),
        "k",
    )
    assert cfg["headers"]["X-API-Key"] == "k"


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
