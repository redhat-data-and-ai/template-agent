"""Unit tests for deep_agent.src.ldap.config."""

import os
from unittest.mock import patch

import pytest

from deep_agent.src.ldap.config import LdapSettings


class TestLdapSettings:
    def test_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            s = LdapSettings()
            assert s.LDAP_URL == ""
            assert s.LDAP_BASE_UID == ""
            assert s.LDAP_PASSWORD == ""
            assert s.LDAP_GROUP_SEARCH_BASE == ""
            assert s.LDAP_TLS_VERIFY is True
            assert s.LDAP_CACHE_TTL_SECONDS == 300

    def test_from_env(self):
        env = {
            "LDAP_URL": "ldaps://ldap.corp.example.com",
            "LDAP_BASE_UID": "svcacct",
            "LDAP_PASSWORD": "secret",
            "LDAP_TLS_VERIFY": "false",
            "LDAP_CACHE_TTL_SECONDS": "600",
        }
        with patch.dict(os.environ, env, clear=True):
            s = LdapSettings()
            assert s.LDAP_URL == "ldaps://ldap.corp.example.com"
            assert s.LDAP_BASE_UID == "svcacct"
            assert s.LDAP_PASSWORD == "secret"
            assert s.LDAP_TLS_VERIFY is False
            assert s.LDAP_CACHE_TTL_SECONDS == 600

    def test_password_hidden_in_repr(self):
        s = LdapSettings(LDAP_PASSWORD="supersecret")
        r = repr(s)
        assert "supersecret" not in r


class TestDeriveBaseDn:
    def test_standard_hostname(self):
        s = LdapSettings(LDAP_URL="ldaps://ldap.example.com")
        assert s.derive_base_dn() == "dc=example,dc=com"

    def test_deep_hostname_uses_last_two(self):
        s = LdapSettings(LDAP_URL="ldaps://ldap.corp.example.org")
        assert s.derive_base_dn() == "dc=example,dc=org"

    def test_empty_url(self):
        s = LdapSettings(LDAP_URL="")
        assert s.derive_base_dn() == ""

    def test_single_part_hostname(self):
        s = LdapSettings(LDAP_URL="ldaps://localhost")
        assert s.derive_base_dn() == ""

    def test_two_part_hostname(self):
        s = LdapSettings(LDAP_URL="ldap://example.com")
        assert s.derive_base_dn() == "dc=example,dc=com"


class TestGetBindDn:
    def test_simple_uid(self):
        s = LdapSettings(LDAP_URL="ldaps://ldap.example.com", LDAP_BASE_UID="svc")
        assert s.get_bind_dn() == "uid=svc,ou=users,dc=example,dc=com"

    def test_full_dn_passthrough(self):
        full_dn = "cn=admin,dc=example,dc=com"
        s = LdapSettings(LDAP_URL="ldaps://ldap.example.com", LDAP_BASE_UID=full_dn)
        assert s.get_bind_dn() == full_dn

    def test_empty_uid(self):
        s = LdapSettings(LDAP_BASE_UID="")
        assert s.get_bind_dn() == ""


class TestGetGroupSearchBase:
    def test_default_derivation(self):
        s = LdapSettings(LDAP_URL="ldaps://ldap.example.com")
        assert (
            s.get_group_search_base() == "ou=adhoc,ou=managedGroups,dc=example,dc=com"
        )

    def test_explicit_override(self):
        s = LdapSettings(
            LDAP_URL="ldaps://ldap.example.com",
            LDAP_GROUP_SEARCH_BASE="ou=groups,dc=custom,dc=com",
        )
        assert s.get_group_search_base() == "ou=groups,dc=custom,dc=com"

    def test_no_url_empty_base(self):
        s = LdapSettings(LDAP_URL="")
        assert s.get_group_search_base() == ""


class TestCacheTtlValidation:
    def test_too_low_raises(self):
        with pytest.raises(Exception):
            LdapSettings(LDAP_CACHE_TTL_SECONDS=5)

    def test_too_high_raises(self):
        with pytest.raises(Exception):
            LdapSettings(LDAP_CACHE_TTL_SECONDS=100000)

    def test_boundary_valid(self):
        s = LdapSettings(LDAP_CACHE_TTL_SECONDS=10)
        assert s.LDAP_CACHE_TTL_SECONDS == 10
