"""Tests for request_auth module - token extraction from request headers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from template_agent.src.request_auth import (
    MISSING_AUTH_DETAIL,
    access_token_from_request,
)


class TestAccessTokenFromRequest:
    """Tests for access_token_from_request function."""

    def test_x_token_header(self):
        """X-Token header is extracted correctly."""
        request = MagicMock()
        request.headers = {"X-Token": "my-token-123"}
        assert access_token_from_request(request) == "my-token-123"

    def test_x_token_header_with_whitespace(self):
        """X-Token header is stripped of whitespace."""
        request = MagicMock()
        request.headers = {"X-Token": "  my-token-123  "}
        assert access_token_from_request(request) == "my-token-123"

    def test_authorization_bearer_header(self):
        """Authorization: Bearer header is extracted correctly."""
        request = MagicMock()
        request.headers = {"authorization": "Bearer my-bearer-token"}
        assert access_token_from_request(request) == "my-bearer-token"

    def test_authorization_bearer_lowercase(self):
        """Authorization header with lowercase 'bearer' is handled."""
        request = MagicMock()
        request.headers = {"authorization": "bearer my-bearer-token"}
        assert access_token_from_request(request) == "my-bearer-token"

    def test_authorization_bearer_mixed_case(self):
        """Authorization header with mixed case 'Bearer' is handled."""
        request = MagicMock()
        request.headers = {"authorization": "BEARER my-bearer-token"}
        assert access_token_from_request(request) == "my-bearer-token"

    def test_authorization_bearer_with_extra_whitespace(self):
        """Authorization header with extra whitespace is handled."""
        request = MagicMock()
        request.headers = {"authorization": "  Bearer   my-bearer-token  "}
        assert access_token_from_request(request) == "my-bearer-token"

    def test_x_token_takes_precedence_over_authorization(self):
        """X-Token header takes precedence over Authorization header."""
        request = MagicMock()
        request.headers = {
            "X-Token": "x-token-value",
            "authorization": "Bearer bearer-value",
        }
        assert access_token_from_request(request) == "x-token-value"

    def test_no_auth_headers_returns_none(self):
        """Returns None when no auth headers present."""
        request = MagicMock()
        request.headers = {}
        assert access_token_from_request(request) is None

    def test_empty_authorization_returns_none(self):
        """Returns None when Authorization header is empty."""
        request = MagicMock()
        request.headers = {"authorization": ""}
        assert access_token_from_request(request) is None

    def test_authorization_without_bearer_returns_none(self):
        """Returns None when Authorization header doesn't use Bearer scheme."""
        request = MagicMock()
        request.headers = {"authorization": "Basic somebase64"}
        assert access_token_from_request(request) is None

    def test_authorization_bearer_only_returns_none(self):
        """Returns None when Authorization header is just 'Bearer' without token."""
        request = MagicMock()
        request.headers = {"authorization": "Bearer"}
        assert access_token_from_request(request) is None

    def test_missing_auth_detail_constant(self):
        """Verify MISSING_AUTH_DETAIL constant is set correctly."""
        assert "X-Token" in MISSING_AUTH_DETAIL
        assert "Authorization" in MISSING_AUTH_DETAIL or "Bearer" in MISSING_AUTH_DETAIL
