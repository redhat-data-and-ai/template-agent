"""Unit tests for LDAP config and service layer."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from deep_agent.src.ldap.config import LdapSettings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_URL = "ldaps://ldap.test.example.com"
_BIND_DN = "uid=svc,ou=users,dc=test,dc=example,dc=com"
_IPA_BIND_DN = "uid=svc,cn=users,cn=accounts,dc=ipa,dc=example,dc=com"
_IPA_GROUP_BASE = "cn=groups,cn=accounts,dc=ipa,dc=example,dc=com"
_PASSWORD = "test-password"


def _make_settings(**overrides) -> LdapSettings:
    defaults = dict(
        LDAP_URL=_URL,
        LDAP_BIND_DN=_BIND_DN,
        LDAP_PASSWORD=_PASSWORD,
        LDAP_GROUP_SEARCH_BASE="ou=groups,dc=test,dc=example,dc=com",
        LDAP_TLS_VERIFY=False,
    )
    defaults.update(overrides)
    return LdapSettings(**defaults)


# ---------------------------------------------------------------------------
# LdapSettings — config.py
# ---------------------------------------------------------------------------


class TestDeriveBaseDn:
    def test_standard_hostname(self):
        s = LdapSettings(LDAP_URL="ldaps://ldap.example.com")
        assert s.derive_base_dn() == "dc=example,dc=com"

    def test_subdomain_uses_last_two_labels(self):
        s = LdapSettings(LDAP_URL="ldaps://ldap.sub.example.com")
        assert s.derive_base_dn() == "dc=example,dc=com"

    def test_empty_url_returns_empty(self):
        s = LdapSettings(LDAP_URL="")
        assert s.derive_base_dn() == ""

    def test_single_label_hostname_returns_empty(self):
        s = LdapSettings(LDAP_URL="ldaps://localhost")
        assert s.derive_base_dn() == ""


class TestGetBindDn:
    def test_ldap_bind_dn_override_wins(self):
        s = LdapSettings(
            LDAP_URL=_URL,
            LDAP_BASE_UID="svc",
            LDAP_BIND_DN=_IPA_BIND_DN,
        )
        assert s.get_bind_dn() == _IPA_BIND_DN

    def test_derived_from_base_uid_when_no_override(self):
        s = LdapSettings(
            LDAP_URL="ldaps://ldap.example.com", LDAP_BASE_UID="svc", LDAP_BIND_DN=""
        )
        assert s.get_bind_dn() == "uid=svc,ou=users,dc=example,dc=com"

    def test_empty_when_no_uid_and_no_override(self):
        s = LdapSettings(LDAP_URL=_URL, LDAP_BASE_UID="", LDAP_BIND_DN="")
        assert s.get_bind_dn() == ""

    def test_override_ignores_base_uid(self):
        s = LdapSettings(
            LDAP_URL=_URL, LDAP_BASE_UID="ignored", LDAP_BIND_DN=_IPA_BIND_DN
        )
        assert s.get_bind_dn() == _IPA_BIND_DN


class TestGetGroupSearchBase:
    def test_override_wins(self):
        s = LdapSettings(LDAP_URL=_URL, LDAP_GROUP_SEARCH_BASE=_IPA_GROUP_BASE)
        assert s.get_group_search_base() == _IPA_GROUP_BASE

    def test_default_derived_from_url(self):
        s = LdapSettings(LDAP_URL="ldaps://ldap.example.com", LDAP_GROUP_SEARCH_BASE="")
        assert (
            s.get_group_search_base() == "ou=adhoc,ou=managedGroups,dc=example,dc=com"
        )

    def test_empty_when_no_url(self):
        s = LdapSettings(LDAP_URL="", LDAP_GROUP_SEARCH_BASE="")
        assert s.get_group_search_base() == ""


# ---------------------------------------------------------------------------
# service.py helpers
# ---------------------------------------------------------------------------


def _reset_service_globals(svc):
    svc._ldap_conn = None
    svc._bind_failed = False
    svc._bind_failed_at = 0.0
    svc._memory_cache.clear()


@pytest.fixture()
def svc():
    import deep_agent.src.ldap.service as _svc

    _reset_service_globals(_svc)
    yield _svc
    _reset_service_globals(_svc)


# Patch ldap3 symbols where they are imported (inside _ensure_bound)
_LDAP3_CONN = "ldap3.Connection"
_LDAP3_SERVER = "ldap3.Server"
_LDAP3_TLS = "ldap3.Tls"


def _mock_cache(svc):
    """Patch cache layer to avoid Redis dependency in unit tests."""
    svc._memory_cache.clear()
    return patch.object(svc, "_cache_get", return_value=None), patch.object(
        svc, "_cache_set"
    )


# ---------------------------------------------------------------------------
# _ensure_bound
# ---------------------------------------------------------------------------


class TestEnsureBound:
    def test_returns_none_when_no_url(self, svc):
        with patch.object(svc, "ldap_settings", _make_settings(LDAP_URL="")):
            assert svc._ensure_bound() is None

    def test_returns_none_for_non_ldaps_url(self, svc):
        with patch.object(
            svc,
            "ldap_settings",
            _make_settings(LDAP_URL="ldap://ldap.test.example.com"),
        ):
            assert svc._ensure_bound() is None

    def test_successful_bind_returns_connection(self, svc):
        mock_conn = MagicMock()
        mock_conn.closed = False
        with (
            patch.object(svc, "ldap_settings", _make_settings()),
            patch(_LDAP3_CONN, return_value=mock_conn),
            patch(_LDAP3_SERVER),
            patch(_LDAP3_TLS),
        ):
            result = svc._ensure_bound()
        assert result is mock_conn
        assert svc._bind_failed is False

    def test_invalid_credentials_sets_30s_cooldown(self, svc):
        with (
            patch.object(svc, "ldap_settings", _make_settings()),
            patch(_LDAP3_CONN, side_effect=Exception("invalidCredentials")),
            patch(_LDAP3_SERVER),
            patch(_LDAP3_TLS),
        ):
            result = svc._ensure_bound()
        assert result is None
        assert svc._bind_failed is True
        assert svc._bind_failed_at > time.monotonic() + 25

    def test_account_locked_sets_60s_cooldown(self, svc):
        with (
            patch.object(svc, "ldap_settings", _make_settings()),
            patch(_LDAP3_CONN, side_effect=Exception("unwillingToPerform")),
            patch(_LDAP3_SERVER),
            patch(_LDAP3_TLS),
        ):
            result = svc._ensure_bound()
        assert result is None
        assert svc._bind_failed_at > time.monotonic() + 55

    def test_transient_error_retries_twice(self, svc):
        call_count = 0

        def fail_always(*a, **kw):
            nonlocal call_count
            call_count += 1
            raise Exception("connection refused")

        with (
            patch.object(svc, "ldap_settings", _make_settings()),
            patch(_LDAP3_CONN, side_effect=fail_always),
            patch(_LDAP3_SERVER),
            patch(_LDAP3_TLS),
        ):
            result = svc._ensure_bound()

        assert result is None
        assert call_count == 2
        assert svc._bind_failed is True
        assert svc._bind_failed_at <= time.monotonic() + 4

    def test_cooldown_blocks_subsequent_calls(self, svc):
        with (
            patch.object(svc, "ldap_settings", _make_settings()),
            patch(_LDAP3_CONN, side_effect=Exception("invalidCredentials")),
            patch(_LDAP3_SERVER),
            patch(_LDAP3_TLS),
        ):
            svc._ensure_bound()

        with patch(_LDAP3_CONN) as mock_cls:
            svc._ensure_bound()
        mock_cls.assert_not_called()

    def test_reuses_open_connection(self, svc):
        mock_conn = MagicMock()
        mock_conn.closed = False
        svc._ldap_conn = mock_conn

        with (
            patch.object(svc, "ldap_settings", _make_settings()),
            patch(_LDAP3_CONN) as mock_cls,
        ):
            result = svc._ensure_bound()

        mock_cls.assert_not_called()
        assert result is mock_conn

    def test_reconnects_when_connection_is_closed(self, svc):
        old_conn = MagicMock()
        old_conn.closed = True
        svc._ldap_conn = old_conn

        new_conn = MagicMock()
        new_conn.closed = False
        with (
            patch.object(svc, "ldap_settings", _make_settings()),
            patch(_LDAP3_CONN, return_value=new_conn),
            patch(_LDAP3_SERVER),
            patch(_LDAP3_TLS),
        ):
            result = svc._ensure_bound()
        assert result is new_conn


# ---------------------------------------------------------------------------
# _is_user_in_group_sync — member matching
# ---------------------------------------------------------------------------


def _make_entry(attr: str, values: list[str]) -> MagicMock:
    entry = MagicMock()
    for a in ["member", "uniqueMember", "memberUid"]:
        mock_val = MagicMock()
        mock_val.values = values if a == attr else []
        setattr(entry, a, mock_val)
    return entry


def _mock_bound_conn(svc, entries) -> MagicMock:
    mock_conn = MagicMock()
    mock_conn.closed = False
    mock_conn.search.return_value = True
    mock_conn.entries = entries
    mock_conn.result = {"result": 0, "description": "success"}
    svc._ldap_conn = mock_conn
    svc._bind_failed = False
    return mock_conn


class TestIsUserInGroupSync:
    def test_uid_direct_match(self, svc):
        entry = _make_entry("memberUid", ["testuser"])
        _mock_bound_conn(svc, [entry])
        with (
            patch.object(svc, "ldap_settings", _make_settings()),
            patch.object(svc, "_cache_get", return_value=None),
            patch.object(svc, "_cache_set"),
        ):
            assert svc._is_user_in_group_sync("testuser", "test-group") is True

    def test_ipa_full_dn_match(self, svc):
        ipa_dn = "uid=testuser,cn=users,cn=accounts,dc=ipa,dc=example,dc=com"
        entry = _make_entry("member", [ipa_dn])
        _mock_bound_conn(svc, [entry])
        with (
            patch.object(
                svc, "ldap_settings", _make_settings(LDAP_BIND_DN=_IPA_BIND_DN)
            ),
            patch.object(svc, "_cache_get", return_value=None),
            patch.object(svc, "_cache_set"),
        ):
            assert svc._is_user_in_group_sync("testuser", "test-group") is True

    def test_generic_ldap_dn_match(self, svc):
        entry = _make_entry(
            "member", ["uid=testuser,ou=users,dc=test,dc=example,dc=com"]
        )
        _mock_bound_conn(svc, [entry])
        with (
            patch.object(svc, "ldap_settings", _make_settings()),
            patch.object(svc, "_cache_get", return_value=None),
            patch.object(svc, "_cache_set"),
        ):
            assert svc._is_user_in_group_sync("testuser", "test-group") is True

    def test_case_insensitive_match(self, svc):
        entry = _make_entry("memberUid", ["TESTUSER"])
        _mock_bound_conn(svc, [entry])
        with (
            patch.object(svc, "ldap_settings", _make_settings()),
            patch.object(svc, "_cache_get", return_value=None),
            patch.object(svc, "_cache_set"),
        ):
            assert svc._is_user_in_group_sync("testuser", "test-group") is True

    def test_user_not_in_group(self, svc):
        entry = _make_entry("memberUid", ["other-user"])
        _mock_bound_conn(svc, [entry])
        with (
            patch.object(svc, "ldap_settings", _make_settings()),
            patch.object(svc, "_cache_get", return_value=None),
            patch.object(svc, "_cache_set"),
        ):
            assert svc._is_user_in_group_sync("testuser", "test-group") is False

    def test_returns_false_when_bind_fails(self, svc):
        with (
            patch.object(svc, "ldap_settings", _make_settings()),
            patch(_LDAP3_CONN, side_effect=Exception("invalidCredentials")),
            patch(_LDAP3_SERVER),
            patch(_LDAP3_TLS),
            patch.object(svc, "_cache_get", return_value=None),
        ):
            assert svc._is_user_in_group_sync("testuser", "test-group") is False

    def test_cache_hit_skips_ldap(self, svc):
        conn = _mock_bound_conn(svc, [])
        with (
            patch.object(svc, "ldap_settings", _make_settings()),
            patch.object(svc, "_cache_get", return_value=True),
            patch.object(svc, "_cache_set"),
        ):
            result = svc._is_user_in_group_sync("testuser", "test-group")
        assert result is True
        conn.search.assert_not_called()

    def test_result_is_cached_after_lookup(self, svc):
        entry = _make_entry("memberUid", ["testuser"])
        conn = _mock_bound_conn(svc, [entry])
        cache_set_mock = MagicMock()
        with (
            patch.object(svc, "ldap_settings", _make_settings()),
            patch.object(svc, "_cache_get", return_value=None),
            patch.object(svc, "_cache_set", cache_set_mock),
        ):
            svc._is_user_in_group_sync("testuser", "test-group")
        cache_set_mock.assert_called_once_with(
            "ldap:membership:testuser:test-group", True
        )
