"""Unit tests for MCP helper functions (token refresh, error classification)."""

import base64
import json
import time
from unittest.mock import AsyncMock, patch

import pytest

from deep_agent.aegra.mcp import (
    _is_auth_error,
    _is_connection_error,
    _jwt_exp,
    refresh_access_token,
)


class TestJwtExp:
    def _make_jwt(self, exp: float) -> str:
        header = base64.urlsafe_b64encode(b'{"alg":"HS256"}').rstrip(b"=").decode()
        payload = (
            base64.urlsafe_b64encode(json.dumps({"exp": exp, "sub": "user"}).encode())
            .rstrip(b"=")
            .decode()
        )
        return f"{header}.{payload}.fakesig"

    def test_extracts_exp(self):
        future = time.time() + 3600
        token = self._make_jwt(future)
        assert abs(_jwt_exp(token) - future) < 1

    def test_returns_zero_on_bad_token(self):
        assert _jwt_exp("not.a.jwt") == 0.0
        assert _jwt_exp("") == 0.0
        assert _jwt_exp("single_segment") == 0.0

    def test_returns_zero_when_no_exp(self):
        header = base64.urlsafe_b64encode(b'{"alg":"HS256"}').rstrip(b"=").decode()
        payload = (
            base64.urlsafe_b64encode(json.dumps({"sub": "user"}).encode())
            .rstrip(b"=")
            .decode()
        )
        token = f"{header}.{payload}.sig"
        assert _jwt_exp(token) == 0.0


class TestIsAuthError:
    def test_401_in_message(self):
        exc = Exception("HTTP 401 Unauthorized")
        assert _is_auth_error(exc) is True

    def test_403_in_message(self):
        exc = Exception("403 Forbidden")
        assert _is_auth_error(exc) is True

    def test_non_auth_error(self):
        exc = Exception("Connection refused")
        assert _is_auth_error(exc) is False

    def test_nested_cause(self):
        inner = Exception("Unauthorized")
        outer = Exception("wrapper")
        outer.__cause__ = inner
        assert _is_auth_error(outer) is True


class TestIsConnectionError:
    def test_connection_refused(self):
        exc = Exception("Connection refused")
        assert _is_connection_error(exc) is True

    def test_connect_error(self):
        exc = Exception("ConnectError: failed to connect")
        assert _is_connection_error(exc) is True

    def test_attempts_failed(self):
        exc = Exception("All connection attempts failed")
        assert _is_connection_error(exc) is True

    def test_non_connection_error(self):
        exc = Exception("Invalid JSON response")
        assert _is_connection_error(exc) is False


class TestRefreshAccessToken:
    def _make_jwt(self, exp: float) -> str:
        header = base64.urlsafe_b64encode(b'{"alg":"HS256"}').rstrip(b"=").decode()
        payload = (
            base64.urlsafe_b64encode(json.dumps({"exp": exp, "sub": "user"}).encode())
            .rstrip(b"=")
            .decode()
        )
        return f"{header}.{payload}.fakesig"

    @pytest.mark.asyncio
    async def test_returns_token_if_still_valid(self):
        token = self._make_jwt(time.time() + 3600)
        result = await refresh_access_token(token, "refresh_token")
        assert result == token

    @pytest.mark.asyncio
    async def test_returns_original_if_no_refresh_token(self):
        token = self._make_jwt(time.time() - 60)
        result = await refresh_access_token(token, None)
        assert result == token

    @pytest.mark.asyncio
    async def test_returns_original_if_no_token_endpoint(self):
        token = self._make_jwt(time.time() - 60)
        with patch(
            "deep_agent.src.infrastructure.mcp._get_token_endpoint",
            return_value="",
        ):
            result = await refresh_access_token(token, "refresh_tok")
            assert result == token
