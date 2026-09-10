"""Unit tests for deep_agent.src.ldap.service."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import deep_agent.src.ldap.service as svc
from deep_agent.src.ldap.prompt_config import (
    GroupRoleMapping,
    PromptAccessConfig,
)


@pytest.fixture(autouse=True)
def _reset_module():
    """Reset module-level state between tests."""
    svc._ldap_conn = None
    svc._bind_failed = False
    svc._memory_cache.clear()
    svc._startup_warning_logged = False
    yield
    svc._ldap_conn = None
    svc._bind_failed = False
    svc._memory_cache.clear()
    svc._startup_warning_logged = False


def _make_entry(member=None, unique_member=None, member_uid=None):
    """Build a mock LDAP entry with the given attribute values."""
    entry = MagicMock()

    def _attr(values):
        if values is None:
            return None
        attr = MagicMock()
        attr.values = values
        return attr

    entry.member = _attr(member)
    entry.uniqueMember = _attr(unique_member)
    entry.memberUid = _attr(member_uid)
    return entry


# ── _escape_ldap_filter ──────────────────────────────────────────────────────


class TestEscapeLdapFilter:
    def test_plain_string(self):
        assert svc._escape_ldap_filter("team-owners") == "team-owners"

    def test_backslash(self):
        assert "\\5c" in svc._escape_ldap_filter("a\\b")

    def test_asterisk(self):
        assert "\\2a" in svc._escape_ldap_filter("a*b")

    def test_parentheses(self):
        result = svc._escape_ldap_filter("a(b)c")
        assert "\\28" in result
        assert "\\29" in result

    def test_null_byte(self):
        assert "\\00" in svc._escape_ldap_filter("a\x00b")

    def test_combined(self):
        result = svc._escape_ldap_filter("t*(e)st\\")
        assert "(" not in result
        assert ")" not in result
        assert "*" not in result


# ── _ensure_bound ────────────────────────────────────────────────────────────


class TestEnsureBound:
    def test_no_url_returns_none(self):
        with patch("deep_agent.src.ldap.service.ldap_settings") as ms:
            ms.LDAP_URL = ""
            assert svc._ensure_bound() is None

    def test_returns_existing_conn(self):
        mock_conn = MagicMock()
        svc._ldap_conn = mock_conn
        svc._bind_failed = False
        with patch("deep_agent.src.ldap.service.ldap_settings") as ms:
            ms.LDAP_URL = "ldaps://ldap.example.com"
            assert svc._ensure_bound() is mock_conn

    def test_bind_failure_returns_none(self):
        svc._ldap_conn = None
        svc._bind_failed = False

        ldap3_mock = MagicMock()
        ldap3_mock.Connection.side_effect = Exception("bind error")

        with patch("deep_agent.src.ldap.service.ldap_settings") as ms:
            ms.LDAP_URL = "ldaps://ldap.example.com"
            ms.LDAP_TLS_VERIFY = True
            ms.LDAP_PASSWORD = "pass"
            ms.get_bind_dn.return_value = "uid=svc,dc=example,dc=com"

            with patch.dict("sys.modules", {"ldap3": ldap3_mock}):
                result = svc._ensure_bound()
                assert result is None
                assert svc._bind_failed is True

    def test_successful_bind(self):
        mock_conn = MagicMock()
        ldap3_mock = MagicMock()
        ldap3_mock.Connection.return_value = mock_conn

        svc._ldap_conn = None
        svc._bind_failed = False

        with patch("deep_agent.src.ldap.service.ldap_settings") as ms:
            ms.LDAP_URL = "ldaps://ldap.example.com"
            ms.LDAP_TLS_VERIFY = True
            ms.LDAP_PASSWORD = "pass"
            ms.get_bind_dn.return_value = "uid=svc,dc=example,dc=com"

            with patch.dict("sys.modules", {"ldap3": ldap3_mock}):
                result = svc._ensure_bound()
                assert result is mock_conn
                assert svc._bind_failed is False

    def test_tls_verify_false(self):
        mock_conn = MagicMock()
        ldap3_mock = MagicMock()
        ldap3_mock.Connection.return_value = mock_conn

        svc._ldap_conn = None
        svc._bind_failed = False

        with patch("deep_agent.src.ldap.service.ldap_settings") as ms:
            ms.LDAP_URL = "ldaps://ldap.example.com"
            ms.LDAP_TLS_VERIFY = False
            ms.LDAP_PASSWORD = "pass"
            ms.get_bind_dn.return_value = "uid=svc,dc=example,dc=com"

            with patch.dict("sys.modules", {"ldap3": ldap3_mock}):
                result = svc._ensure_bound()
                assert result is mock_conn


# ── _cache_get / _cache_set ──────────────────────────────────────────────────


class TestCaching:
    def test_memory_cache_hit(self):
        svc._memory_cache["key1"] = (True, time.monotonic())
        with patch("deep_agent.src.ldap.service.ldap_settings") as ms:
            ms.LDAP_CACHE_TTL_SECONDS = 300
            with patch("deep_agent.aegra.redis.cache_get", return_value=None):
                assert svc._cache_get("key1") is True

    def test_memory_cache_expired(self):
        svc._memory_cache["key1"] = (True, time.monotonic() - 400)
        with patch("deep_agent.src.ldap.service.ldap_settings") as ms:
            ms.LDAP_CACHE_TTL_SECONDS = 300
            with patch("deep_agent.aegra.redis.cache_get", return_value=None):
                assert svc._cache_get("key1") is None

    def test_redis_cache_hit_true(self):
        with patch("deep_agent.aegra.redis.cache_get", return_value="1"):
            assert svc._cache_get("key1") is True

    def test_redis_cache_hit_false(self):
        with patch("deep_agent.aegra.redis.cache_get", return_value="0"):
            assert svc._cache_get("key1") is False

    def test_cache_miss(self):
        with patch("deep_agent.aegra.redis.cache_get", return_value=None):
            assert svc._cache_get("key1") is None

    def test_cache_set_writes_both(self):
        with (
            patch("deep_agent.aegra.redis.cache_set") as redis_set,
            patch("deep_agent.src.ldap.service.ldap_settings") as ms,
        ):
            ms.LDAP_CACHE_TTL_SECONDS = 300
            svc._cache_set("k", True)
            redis_set.assert_called_once_with("k", "1", 300)
            assert "k" in svc._memory_cache
            assert svc._memory_cache["k"][0] is True

    def test_cache_set_false_value(self):
        with (
            patch("deep_agent.aegra.redis.cache_set") as redis_set,
            patch("deep_agent.src.ldap.service.ldap_settings") as ms,
        ):
            ms.LDAP_CACHE_TTL_SECONDS = 300
            svc._cache_set("k", False)
            redis_set.assert_called_once_with("k", "0", 300)


# ── _is_user_in_group_sync ──────────────────────────────────────────────────


class TestIsUserInGroupSync:
    def test_cache_hit_returns_cached(self):
        svc._memory_cache["ldap:membership:alice:team"] = (True, time.monotonic())
        with patch("deep_agent.src.ldap.service.ldap_settings") as ms:
            ms.LDAP_CACHE_TTL_SECONDS = 300
            with patch("deep_agent.aegra.redis.cache_get", return_value=None):
                assert svc._is_user_in_group_sync("alice", "team") is True

    def test_no_conn_returns_false(self):
        with (
            patch.object(svc, "_ensure_bound", return_value=None),
            patch("deep_agent.aegra.redis.cache_get", return_value=None),
        ):
            assert svc._is_user_in_group_sync("alice", "team") is False

    def test_member_uid_match(self):
        entry = _make_entry(member_uid=["alice"])
        mock_conn = MagicMock()
        mock_conn.entries = [entry]
        ldap3_mock = MagicMock()

        with (
            patch.object(svc, "_ensure_bound", return_value=mock_conn),
            patch.object(svc, "_derive_base_dn", return_value="dc=example,dc=com"),
            patch("deep_agent.src.ldap.service.ldap_settings") as ms,
            patch("deep_agent.aegra.redis.cache_get", return_value=None),
            patch("deep_agent.aegra.redis.cache_set"),
            patch.dict("sys.modules", {"ldap3": ldap3_mock}),
        ):
            ms.get_group_search_base.return_value = "ou=groups,dc=example,dc=com"
            ms.LDAP_CACHE_TTL_SECONDS = 300
            assert svc._is_user_in_group_sync("alice", "team-owners") is True

    def test_member_dn_match(self):
        entry = _make_entry(member=["uid=bob,ou=users,dc=example,dc=com"])
        mock_conn = MagicMock()
        mock_conn.entries = [entry]
        ldap3_mock = MagicMock()

        with (
            patch.object(svc, "_ensure_bound", return_value=mock_conn),
            patch.object(svc, "_derive_base_dn", return_value="dc=example,dc=com"),
            patch("deep_agent.src.ldap.service.ldap_settings") as ms,
            patch("deep_agent.aegra.redis.cache_get", return_value=None),
            patch("deep_agent.aegra.redis.cache_set"),
            patch.dict("sys.modules", {"ldap3": ldap3_mock}),
        ):
            ms.get_group_search_base.return_value = "ou=groups,dc=example,dc=com"
            ms.LDAP_CACHE_TTL_SECONDS = 300
            assert svc._is_user_in_group_sync("bob", "team") is True

    def test_member_dn_prefix_match(self):
        entry = _make_entry(member=["uid=carol,ou=people,dc=other,dc=com"])
        mock_conn = MagicMock()
        mock_conn.entries = [entry]
        ldap3_mock = MagicMock()

        with (
            patch.object(svc, "_ensure_bound", return_value=mock_conn),
            patch.object(svc, "_derive_base_dn", return_value="dc=example,dc=com"),
            patch("deep_agent.src.ldap.service.ldap_settings") as ms,
            patch("deep_agent.aegra.redis.cache_get", return_value=None),
            patch("deep_agent.aegra.redis.cache_set"),
            patch.dict("sys.modules", {"ldap3": ldap3_mock}),
        ):
            ms.get_group_search_base.return_value = "ou=groups,dc=example,dc=com"
            ms.LDAP_CACHE_TTL_SECONDS = 300
            assert svc._is_user_in_group_sync("carol", "team") is True

    def test_no_match(self):
        entry = _make_entry(member=["uid=other,ou=users,dc=example,dc=com"])
        mock_conn = MagicMock()
        mock_conn.entries = [entry]
        ldap3_mock = MagicMock()

        with (
            patch.object(svc, "_ensure_bound", return_value=mock_conn),
            patch.object(svc, "_derive_base_dn", return_value="dc=example,dc=com"),
            patch("deep_agent.src.ldap.service.ldap_settings") as ms,
            patch("deep_agent.aegra.redis.cache_get", return_value=None),
            patch("deep_agent.aegra.redis.cache_set"),
            patch.dict("sys.modules", {"ldap3": ldap3_mock}),
        ):
            ms.get_group_search_base.return_value = "ou=groups,dc=example,dc=com"
            ms.LDAP_CACHE_TTL_SECONDS = 300
            assert svc._is_user_in_group_sync("alice", "team") is False

    def test_search_exception_returns_false(self):
        mock_conn = MagicMock()
        mock_conn.search.side_effect = Exception("timeout")
        ldap3_mock = MagicMock()

        with (
            patch.object(svc, "_ensure_bound", return_value=mock_conn),
            patch("deep_agent.src.ldap.service.ldap_settings") as ms,
            patch("deep_agent.aegra.redis.cache_get", return_value=None),
            patch.dict("sys.modules", {"ldap3": ldap3_mock}),
        ):
            ms.get_group_search_base.return_value = "ou=groups,dc=example,dc=com"
            assert svc._is_user_in_group_sync("alice", "team") is False
            assert svc._bind_failed is True

    def test_case_insensitive_match(self):
        entry = _make_entry(member_uid=["ALICE"])
        mock_conn = MagicMock()
        mock_conn.entries = [entry]
        ldap3_mock = MagicMock()

        with (
            patch.object(svc, "_ensure_bound", return_value=mock_conn),
            patch.object(svc, "_derive_base_dn", return_value="dc=example,dc=com"),
            patch("deep_agent.src.ldap.service.ldap_settings") as ms,
            patch("deep_agent.aegra.redis.cache_get", return_value=None),
            patch("deep_agent.aegra.redis.cache_set"),
            patch.dict("sys.modules", {"ldap3": ldap3_mock}),
        ):
            ms.get_group_search_base.return_value = "ou=groups,dc=example,dc=com"
            ms.LDAP_CACHE_TTL_SECONDS = 300
            assert svc._is_user_in_group_sync("alice", "team") is True

    def test_empty_entries(self):
        mock_conn = MagicMock()
        mock_conn.entries = []
        ldap3_mock = MagicMock()

        with (
            patch.object(svc, "_ensure_bound", return_value=mock_conn),
            patch.object(svc, "_derive_base_dn", return_value="dc=example,dc=com"),
            patch("deep_agent.src.ldap.service.ldap_settings") as ms,
            patch("deep_agent.aegra.redis.cache_get", return_value=None),
            patch("deep_agent.aegra.redis.cache_set"),
            patch.dict("sys.modules", {"ldap3": ldap3_mock}),
        ):
            ms.get_group_search_base.return_value = "ou=groups,dc=example,dc=com"
            ms.LDAP_CACHE_TTL_SECONDS = 300
            assert svc._is_user_in_group_sync("alice", "team") is False

    def test_unique_member_match(self):
        entry = _make_entry(unique_member=["uid=dave,ou=users,dc=example,dc=com"])
        mock_conn = MagicMock()
        mock_conn.entries = [entry]
        ldap3_mock = MagicMock()

        with (
            patch.object(svc, "_ensure_bound", return_value=mock_conn),
            patch.object(svc, "_derive_base_dn", return_value="dc=example,dc=com"),
            patch("deep_agent.src.ldap.service.ldap_settings") as ms,
            patch("deep_agent.aegra.redis.cache_get", return_value=None),
            patch("deep_agent.aegra.redis.cache_set"),
            patch.dict("sys.modules", {"ldap3": ldap3_mock}),
        ):
            ms.get_group_search_base.return_value = "ou=groups,dc=example,dc=com"
            ms.LDAP_CACHE_TTL_SECONDS = 300
            assert svc._is_user_in_group_sync("dave", "team") is True


# ── _resolve_user_role_sync ──────────────────────────────────────────────────


class TestResolveUserRoleSync:
    def test_highest_role_wins(self):
        mappings = [
            GroupRoleMapping(role="users", group="g1"),
            GroupRoleMapping(role="owners", group="g2"),
            GroupRoleMapping(role="builders", group="g3"),
        ]
        with patch.object(svc, "_is_user_in_group_sync", return_value=True):
            assert svc._resolve_user_role_sync("alice", mappings) == "owners"

    def test_no_membership_returns_denied(self):
        mappings = [GroupRoleMapping(role="owners", group="g1")]
        with patch.object(svc, "_is_user_in_group_sync", return_value=False):
            assert svc._resolve_user_role_sync("alice", mappings) == "denied"

    def test_single_match(self):
        mappings = [
            GroupRoleMapping(role="builders", group="g1"),
            GroupRoleMapping(role="admins", group="g2"),
        ]

        def check(uid, group):
            return group == "g1"

        with patch.object(svc, "_is_user_in_group_sync", side_effect=check):
            assert svc._resolve_user_role_sync("alice", mappings) == "builders"

    def test_unknown_role_treated_as_zero_priority(self):
        mappings = [
            GroupRoleMapping(role="unknown", group="g1"),
            GroupRoleMapping(role="users", group="g2"),
        ]
        with patch.object(svc, "_is_user_in_group_sync", return_value=True):
            assert svc._resolve_user_role_sync("alice", mappings) == "users"


# ── resolve_user_role (async) ────────────────────────────────────────────────


class TestResolveUserRole:
    @pytest.mark.asyncio
    async def test_no_groups_returns_users(self):
        config = PromptAccessConfig(groups=None)
        with patch.object(svc, "get_prompt_access_config", return_value=config):
            assert await svc.resolve_user_role("alice") == "users"

    @pytest.mark.asyncio
    async def test_no_ldap_url_private_returns_denied(self):
        config = PromptAccessConfig(
            groups=[GroupRoleMapping(role="owners", group="g1")],
            accessibility="private",
        )
        with (
            patch.object(svc, "get_prompt_access_config", return_value=config),
            patch("deep_agent.src.ldap.service.ldap_settings") as ms,
        ):
            ms.LDAP_URL = ""
            assert await svc.resolve_user_role("alice") == "denied"

    @pytest.mark.asyncio
    async def test_no_ldap_url_public_returns_users(self):
        config = PromptAccessConfig(
            groups=[GroupRoleMapping(role="owners", group="g1")],
            accessibility="public",
        )
        with (
            patch.object(svc, "get_prompt_access_config", return_value=config),
            patch("deep_agent.src.ldap.service.ldap_settings") as ms,
        ):
            ms.LDAP_URL = ""
            assert await svc.resolve_user_role("alice") == "users"

    @pytest.mark.asyncio
    async def test_public_denied_becomes_users(self):
        config = PromptAccessConfig(
            groups=[GroupRoleMapping(role="owners", group="g1")],
            accessibility="public",
        )
        with (
            patch.object(svc, "get_prompt_access_config", return_value=config),
            patch("deep_agent.src.ldap.service.ldap_settings") as ms,
            patch.object(svc, "_resolve_user_role_sync", return_value="denied"),
        ):
            ms.LDAP_URL = "ldaps://ldap.example.com"
            assert await svc.resolve_user_role("alice") == "users"

    @pytest.mark.asyncio
    async def test_private_denied_stays_denied(self):
        config = PromptAccessConfig(
            groups=[GroupRoleMapping(role="owners", group="g1")],
            accessibility="private",
        )
        with (
            patch.object(svc, "get_prompt_access_config", return_value=config),
            patch("deep_agent.src.ldap.service.ldap_settings") as ms,
            patch.object(svc, "_resolve_user_role_sync", return_value="denied"),
        ):
            ms.LDAP_URL = "ldaps://ldap.example.com"
            assert await svc.resolve_user_role("alice") == "denied"

    @pytest.mark.asyncio
    async def test_matched_role_returned(self):
        config = PromptAccessConfig(
            groups=[GroupRoleMapping(role="builders", group="g1")],
            accessibility="private",
        )
        with (
            patch.object(svc, "get_prompt_access_config", return_value=config),
            patch("deep_agent.src.ldap.service.ldap_settings") as ms,
            patch.object(svc, "_resolve_user_role_sync", return_value="builders"),
        ):
            ms.LDAP_URL = "ldaps://ldap.example.com"
            assert await svc.resolve_user_role("alice") == "builders"

    @pytest.mark.asyncio
    async def test_no_ldap_url_logs_warning_once(self):
        config = PromptAccessConfig(
            groups=[GroupRoleMapping(role="owners", group="g1")],
            accessibility="private",
        )
        with (
            patch.object(svc, "get_prompt_access_config", return_value=config),
            patch("deep_agent.src.ldap.service.ldap_settings") as ms,
        ):
            ms.LDAP_URL = ""
            await svc.resolve_user_role("alice")
            assert svc._startup_warning_logged is True
            await svc.resolve_user_role("bob")


# ── close_ldap ───────────────────────────────────────────────────────────────


class TestCloseLdap:
    def test_unbinds_and_clears(self):
        mock_conn = MagicMock()
        svc._ldap_conn = mock_conn
        svc._bind_failed = True
        svc._memory_cache["k"] = (True, 0)

        svc.close_ldap()

        mock_conn.unbind.assert_called_once()
        assert svc._ldap_conn is None
        assert svc._bind_failed is False
        assert len(svc._memory_cache) == 0

    def test_no_conn_still_clears(self):
        svc._ldap_conn = None
        svc._bind_failed = True
        svc._memory_cache["k"] = (True, 0)

        svc.close_ldap()

        assert svc._bind_failed is False
        assert len(svc._memory_cache) == 0

    def test_unbind_exception_swallowed(self):
        mock_conn = MagicMock()
        mock_conn.unbind.side_effect = Exception("already closed")
        svc._ldap_conn = mock_conn

        svc.close_ldap()
        assert svc._ldap_conn is None


# ── _derive_base_dn ─────────────────────────────────────────────────────────


class TestDeriveBaseDn:
    def test_delegates_to_settings(self):
        with patch("deep_agent.src.ldap.service.ldap_settings") as ms:
            ms.derive_base_dn.return_value = "dc=a,dc=b"
            assert svc._derive_base_dn() == "dc=a,dc=b"
