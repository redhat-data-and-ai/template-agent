"""Unit tests for deep_agent.src.ldap.service."""

from __future__ import annotations

import logging
import time
from contextlib import ExitStack, contextmanager
from unittest.mock import MagicMock, patch

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
    svc._bind_failed_at = 0.0
    svc._memory_cache.clear()
    svc._startup_warning_logged = False
    yield
    svc._ldap_conn = None
    svc._bind_failed = False
    svc._bind_failed_at = 0.0
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

    def test_non_ldaps_url_refuses_to_bind(self, caplog):
        """Plaintext ldap:// must be rejected before credentials are sent."""
        with (
            caplog.at_level(logging.ERROR, logger="deep_agent.src.ldap.service"),
            patch("deep_agent.src.ldap.service.ldap_settings") as ms,
        ):
            ms.LDAP_URL = "ldap://ldap.example.com"
            assert svc._ensure_bound() is None

        messages = [r.getMessage() for r in caplog.records]
        assert any("must use ldaps://" in m for m in messages)
        # The URL is redacted to its scheme, never logged whole.
        assert not any("ldap.example.com" in m for m in messages)

    def test_returns_existing_conn(self):
        mock_conn = MagicMock(closed=False)
        svc._ldap_conn = mock_conn
        svc._bind_failed = False
        with patch("deep_agent.src.ldap.service.ldap_settings") as ms:
            ms.LDAP_URL = "ldaps://ldap.example.com"
            assert svc._ensure_bound() is mock_conn

    def test_drops_closed_conn_and_rebinds(self):
        stale_conn = MagicMock(closed=True)
        svc._ldap_conn = stale_conn
        svc._bind_failed = False

        fresh_conn = MagicMock()
        ldap3_mock = MagicMock()
        ldap3_mock.Connection.return_value = fresh_conn

        with patch("deep_agent.src.ldap.service.ldap_settings") as ms:
            ms.LDAP_URL = "ldaps://ldap.example.com"
            ms.LDAP_TLS_VERIFY = True
            ms.LDAP_PASSWORD = "pass"
            ms.get_bind_dn.return_value = "uid=svc,dc=example,dc=com"

            with patch.dict("sys.modules", {"ldap3": ldap3_mock}):
                result = svc._ensure_bound()
                assert result is fresh_conn
                assert svc._ldap_conn is fresh_conn

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

    def test_evict_expired_drops_only_stale_entries(self):
        with patch("deep_agent.src.ldap.service.ldap_settings") as ms:
            ms.LDAP_CACHE_TTL_SECONDS = 300
            now = time.monotonic()
            svc._memory_cache["fresh"] = (True, now)
            svc._memory_cache["stale"] = (True, now - 600)

            svc._evict_expired()

        assert "fresh" in svc._memory_cache
        assert "stale" not in svc._memory_cache

    def test_cache_set_evicts_expired_when_full(self):
        """A full cache of stale entries is evicted, not cleared."""
        with (
            patch("deep_agent.aegra.redis.cache_set"),
            patch("deep_agent.src.ldap.service.ldap_settings") as ms,
        ):
            ms.LDAP_CACHE_TTL_SECONDS = 300
            stale = time.monotonic() - 600
            for i in range(svc._MEMORY_CACHE_MAX_SIZE):
                svc._memory_cache[f"old{i}"] = (True, stale)

            svc._cache_set("new", True)

        assert "new" in svc._memory_cache
        assert len(svc._memory_cache) == 1

    def test_cache_set_clears_when_eviction_frees_nothing(self):
        """A full cache of fresh entries has no expired entries to drop, so
        the whole cache is discarded rather than growing without bound."""
        with (
            patch("deep_agent.aegra.redis.cache_set"),
            patch("deep_agent.src.ldap.service.ldap_settings") as ms,
        ):
            ms.LDAP_CACHE_TTL_SECONDS = 300
            now = time.monotonic()
            for i in range(svc._MEMORY_CACHE_MAX_SIZE):
                svc._memory_cache[f"live{i}"] = (True, now)

            svc._cache_set("new", True)

        assert list(svc._memory_cache) == ["new"]


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

    def test_search_exception_retries_then_returns_false(self):
        mock_conn = MagicMock()
        mock_conn.search.side_effect = Exception("timeout")
        ldap3_mock = MagicMock()

        with (
            patch.object(svc, "_ensure_bound", return_value=mock_conn) as ensure_mock,
            patch("deep_agent.src.ldap.service.ldap_settings") as ms,
            patch("deep_agent.aegra.redis.cache_get", return_value=None),
            patch.dict("sys.modules", {"ldap3": ldap3_mock}),
        ):
            ms.get_group_search_base.return_value = "ou=groups,dc=example,dc=com"
            assert svc._is_user_in_group_sync("alice", "team") is False
            assert svc._bind_failed is True
            assert ensure_mock.call_count == 2

    def test_search_exception_recovers_on_retry(self):
        stale_conn = MagicMock()
        stale_conn.search.side_effect = Exception("connection reset")

        fresh_conn = MagicMock()
        entry = _make_entry(member_uid=["alice"])
        fresh_conn.entries = [entry]

        ldap3_mock = MagicMock()
        call_count = {"n": 0}

        def ensure_side_effect(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return stale_conn
            svc._bind_failed = False
            return fresh_conn

        with (
            patch.object(svc, "_ensure_bound", side_effect=ensure_side_effect),
            patch.object(svc, "_derive_base_dn", return_value="dc=example,dc=com"),
            patch("deep_agent.src.ldap.service.ldap_settings") as ms,
            patch("deep_agent.aegra.redis.cache_get", return_value=None),
            patch("deep_agent.aegra.redis.cache_set"),
            patch.dict("sys.modules", {"ldap3": ldap3_mock}),
        ):
            ms.get_group_search_base.return_value = "ou=groups,dc=example,dc=com"
            ms.LDAP_CACHE_TTL_SECONDS = 300
            assert svc._is_user_in_group_sync("alice", "team") is True

    def test_search_returns_false_fatal_error_triggers_retry_and_unbind(self):
        """Non-noSuchObject failures (e.g. operationsError) still retry and unbind."""
        mock_conn = MagicMock()
        mock_conn.search.return_value = False
        mock_conn.result = {"result": 1, "description": "operationsError"}
        ldap3_mock = MagicMock()

        with (
            patch.object(svc, "_ensure_bound", return_value=mock_conn) as ensure_mock,
            patch("deep_agent.src.ldap.service.ldap_settings") as ms,
            patch("deep_agent.aegra.redis.cache_get", return_value=None),
            patch.dict("sys.modules", {"ldap3": ldap3_mock}),
        ):
            ms.get_group_search_base.return_value = "ou=groups,dc=example,dc=com"
            assert svc._is_user_in_group_sync("alice", "team") is False
            assert svc._bind_failed is True
            assert ensure_mock.call_count == 2
            assert mock_conn.unbind.call_count == 2

    def test_search_returns_false_success_empty_skips_without_retry(self):
        """ldap3 returns False + result 0 when the group CN is missing; that is a miss, not a connection failure."""
        mock_conn = MagicMock()
        mock_conn.search.return_value = False
        mock_conn.result = {
            "result": 0,
            "description": "success",
            "dn": "",
            "message": "",
            "referrals": None,
            "type": "searchResDone",
        }
        ldap3_mock = MagicMock()

        with (
            patch.object(svc, "_ensure_bound", return_value=mock_conn) as ensure_mock,
            patch.object(svc, "_derive_base_dn", return_value="dc=example,dc=com"),
            patch("deep_agent.src.ldap.service.ldap_settings") as ms,
            patch("deep_agent.aegra.redis.cache_get", return_value=None),
            patch("deep_agent.aegra.redis.cache_set") as redis_set,
            patch.dict("sys.modules", {"ldap3": ldap3_mock}),
        ):
            ms.get_group_search_base.return_value = "ou=groups,dc=example,dc=com"
            ms.LDAP_CACHE_TTL_SECONDS = 300
            assert svc._is_user_in_group_sync("alice", "team") is False
            assert svc._bind_failed is False
            assert ensure_mock.call_count == 1
            mock_conn.unbind.assert_not_called()
            redis_set.assert_called_once_with("ldap:membership:alice:team", "0", 300)

    def test_search_returns_false_no_such_object_skips_without_retry_no_cache(self):
        """noSuchObject (code 32) returns False without destroying connection or caching."""
        mock_conn = MagicMock()
        mock_conn.search.return_value = False
        mock_conn.result = {"result": 32, "description": "noSuchObject"}
        ldap3_mock = MagicMock()

        with (
            patch.object(svc, "_ensure_bound", return_value=mock_conn) as ensure_mock,
            patch.object(svc, "_derive_base_dn", return_value="dc=example,dc=com"),
            patch("deep_agent.src.ldap.service.ldap_settings") as ms,
            patch("deep_agent.aegra.redis.cache_get", return_value=None),
            patch("deep_agent.aegra.redis.cache_set") as redis_set,
            patch.dict("sys.modules", {"ldap3": ldap3_mock}),
        ):
            ms.get_group_search_base.return_value = "ou=groups,dc=example,dc=com"
            ms.LDAP_CACHE_TTL_SECONDS = 300
            assert svc._is_user_in_group_sync("alice", "team") is False
            assert svc._bind_failed is False
            assert ensure_mock.call_count == 1
            mock_conn.unbind.assert_not_called()
            redis_set.assert_not_called()

    def test_search_returns_false_referral_skips_without_retry_no_cache(self):
        """Referral (code 10) returns False without destroying connection or caching."""
        mock_conn = MagicMock()
        mock_conn.search.return_value = False
        mock_conn.result = {"result": 10, "description": "referral"}
        ldap3_mock = MagicMock()

        with (
            patch.object(svc, "_ensure_bound", return_value=mock_conn) as ensure_mock,
            patch.object(svc, "_derive_base_dn", return_value="dc=example,dc=com"),
            patch("deep_agent.src.ldap.service.ldap_settings") as ms,
            patch("deep_agent.aegra.redis.cache_get", return_value=None),
            patch("deep_agent.aegra.redis.cache_set") as redis_set,
            patch.dict("sys.modules", {"ldap3": ldap3_mock}),
        ):
            ms.get_group_search_base.return_value = "ou=groups,dc=example,dc=com"
            ms.LDAP_CACHE_TTL_SECONDS = 300
            assert svc._is_user_in_group_sync("alice", "team") is False
            assert svc._bind_failed is False
            assert ensure_mock.call_count == 1
            mock_conn.unbind.assert_not_called()
            redis_set.assert_not_called()

    def test_search_returns_false_recovers_on_retry(self):
        bad_conn = MagicMock()
        bad_conn.search.return_value = False
        bad_conn.result = {"result": 1, "description": "operationsError"}

        good_conn = MagicMock()
        good_conn.search.return_value = True
        entry = _make_entry(member_uid=["alice"])
        good_conn.entries = [entry]

        ldap3_mock = MagicMock()
        call_count = {"n": 0}

        def ensure_side_effect(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return bad_conn
            svc._bind_failed = False
            return good_conn

        with (
            patch.object(svc, "_ensure_bound", side_effect=ensure_side_effect),
            patch.object(svc, "_derive_base_dn", return_value="dc=example,dc=com"),
            patch("deep_agent.src.ldap.service.ldap_settings") as ms,
            patch("deep_agent.aegra.redis.cache_get", return_value=None),
            patch("deep_agent.aegra.redis.cache_set"),
            patch.dict("sys.modules", {"ldap3": ldap3_mock}),
        ):
            ms.get_group_search_base.return_value = "ou=groups,dc=example,dc=com"
            ms.LDAP_CACHE_TTL_SECONDS = 300
            assert svc._is_user_in_group_sync("alice", "team") is True

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

    def test_single_member_value_is_coerced_to_a_list(self):
        """Some servers return a bare string instead of a list of members."""
        mock_conn = MagicMock()
        mock_conn.search.return_value = True
        mock_conn.entries = [_make_entry(member_uid="alice")]
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

    def test_unbind_failure_while_discarding_session_is_swallowed(self):
        """A dead session often cannot be unbound cleanly; that must not raise."""
        mock_conn = MagicMock()
        mock_conn.search.side_effect = Exception("session terminated by server")
        mock_conn.unbind.side_effect = Exception("socket already closed")
        ldap3_mock = MagicMock()

        with (
            patch.object(svc, "_ensure_bound", return_value=mock_conn),
            patch.object(svc, "_derive_base_dn", return_value="dc=example,dc=com"),
            patch("deep_agent.src.ldap.service.ldap_settings") as ms,
            patch("deep_agent.aegra.redis.cache_get", return_value=None),
            patch.dict("sys.modules", {"ldap3": ldap3_mock}),
        ):
            ms.get_group_search_base.return_value = "ou=groups,dc=example,dc=com"
            ms.LDAP_CACHE_TTL_SECONDS = 300
            assert svc._is_user_in_group_sync("alice", "team") is False

        assert mock_conn.unbind.call_count == 2


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

    def test_failed_group_does_not_block_other_groups(self):
        """If one group check raises, the user still gets their role from other groups."""
        mappings = [
            GroupRoleMapping(role="users", group="bad-group"),
            GroupRoleMapping(role="owners", group="good-group"),
        ]

        def check(uid, group):
            if group == "bad-group":
                raise Exception("LDAP timeout")
            return True

        with patch.object(svc, "_is_user_in_group_sync", side_effect=check):
            assert svc._resolve_user_role_sync("alice", mappings) == "owners"

    def test_all_groups_fail_returns_denied(self):
        """If all group checks raise, the user gets 'denied'."""
        mappings = [
            GroupRoleMapping(role="owners", group="g1"),
            GroupRoleMapping(role="users", group="g2"),
        ]
        with patch.object(
            svc, "_is_user_in_group_sync", side_effect=Exception("LDAP down")
        ):
            assert svc._resolve_user_role_sync("alice", mappings) == "denied"


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


# ── search retry log levels (RHITAIF-1927) ──────────────────────────────────


@contextmanager
def _ldap_env(caplog, *, conn=None, ensure=None, ldap3_mock=None):
    """Patch stack shared by the log-level tests below.

    Yields the patched ``ldap_settings`` mock so a test can add to it.
    Pass ``conn`` for a fixed connection, ``ensure`` for a per-attempt side
    effect, or neither to let the real ``_ensure_bound`` run.
    """
    assert conn is None or ensure is None, (
        "pass conn or ensure, not both — otherwise `ensure` is silently "
        "ignored and the test never exercises its retry scenario"
    )
    with ExitStack() as stack:
        enter = stack.enter_context
        enter(caplog.at_level(logging.DEBUG, logger="deep_agent.src.ldap.service"))
        if conn is not None:
            enter(patch.object(svc, "_ensure_bound", return_value=conn))
        elif ensure is not None:
            enter(patch.object(svc, "_ensure_bound", side_effect=ensure))
        enter(patch.object(svc, "_derive_base_dn", return_value="dc=example,dc=com"))
        ms = enter(patch("deep_agent.src.ldap.service.ldap_settings"))
        enter(patch("deep_agent.aegra.redis.cache_get", return_value=None))
        enter(patch("deep_agent.aegra.redis.cache_set"))
        enter(patch.dict("sys.modules", {"ldap3": ldap3_mock or MagicMock()}))
        ms.get_group_search_base.return_value = "ou=groups,dc=example,dc=com"
        ms.LDAP_CACHE_TTL_SECONDS = 300
        yield ms


class TestSearchRetryLogLevels:
    """A session reset the rebind recovers from must not be logged at ERROR."""

    def test_recovery_logs_warning_and_recovered_not_error(self, caplog):
        """First search raises, rebind succeeds: WARNING + recovered INFO, no ERROR."""
        stale_conn = MagicMock()
        stale_conn.search.side_effect = Exception("session terminated by server")

        fresh_conn = MagicMock()
        fresh_conn.entries = [_make_entry(member_uid=["alice"])]
        fresh_conn.result = {"result": 0, "description": "success"}

        call_count = {"n": 0}

        def ensure_side_effect(**_kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return stale_conn
            svc._bind_failed = False
            return fresh_conn

        with _ldap_env(caplog, ensure=ensure_side_effect):
            assert svc._is_user_in_group_sync("alice", "team") is True

        assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []

        retry = [r for r in caplog.records if "ldap_search_retry" in r.getMessage()]
        recovered = [
            r for r in caplog.records if "ldap_search_recovered" in r.getMessage()
        ]
        assert len(retry) == 1
        assert len(recovered) == 1
        # Levels, not just event names: an accidental demotion of the retry
        # record to INFO must fail here.
        assert retry[0].levelno == logging.WARNING
        assert recovered[0].levelno == logging.INFO

    def test_clean_first_attempt_logs_no_recovery(self, caplog):
        """A lookup that works first time must NOT claim it recovered.

        Pins the ``attempt > 0`` guard in ``_log_search_recovered``. Without
        it, every successful lookup is stamped 'completed after reconnect',
        which inverts the signal this logging exists to provide.
        """
        good_conn = MagicMock()
        good_conn.search.return_value = True
        good_conn.entries = [_make_entry(member_uid=["alice"])]
        good_conn.result = {"result": 0, "description": "success"}

        with _ldap_env(caplog, conn=good_conn):
            assert svc._is_user_in_group_sync("alice", "team") is True

        messages = [r.getMessage() for r in caplog.records]
        assert not any("ldap_search_recovered" in m for m in messages)
        assert not any("ldap_search_retry" in m for m in messages)
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    def test_fatal_result_code_recovery_logs_no_error(self, caplog):
        """Fatal result code (operationsError) then a clean retry: no ERROR.

        Code 1 is absent from ``_LDAP_NON_FATAL_RESULT_CODES``, so it is
        raised and retried — this is the retryable path, not a skip.
        """
        bad_conn = MagicMock()
        bad_conn.search.return_value = False
        bad_conn.result = {"result": 1, "description": "operationsError"}

        good_conn = MagicMock()
        good_conn.search.return_value = True
        good_conn.entries = [_make_entry(member_uid=["alice"])]
        good_conn.result = {"result": 0, "description": "success"}

        call_count = {"n": 0}

        def ensure_side_effect(**_kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return bad_conn
            svc._bind_failed = False
            return good_conn

        with _ldap_env(caplog, ensure=ensure_side_effect):
            assert svc._is_user_in_group_sync("alice", "team") is True

        assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
        recovered = [
            r for r in caplog.records if "ldap_search_recovered" in r.getMessage()
        ]
        assert len(recovered) == 1
        assert recovered[0].levelno == logging.INFO

    def test_truncated_result_on_retry_is_not_logged_as_recovered(self, caplog):
        """A partial result is not a completed lookup.

        ``ldap3.Connection.search()`` returns True whenever the search
        finished with at least one entry, so a server-side size limit yields
        entries *and* a nonzero result code (4, sizeLimitExceeded). Claiming
        the membership lookup "completed" there would be wrong.
        """
        stale_conn = MagicMock()
        stale_conn.search.side_effect = Exception("session terminated by server")

        truncated_conn = MagicMock()
        truncated_conn.search.return_value = True
        truncated_conn.entries = [_make_entry(member_uid=["alice"])]
        truncated_conn.result = {"result": 4, "description": "sizeLimitExceeded"}

        call_count = {"n": 0}

        def ensure_side_effect(**_kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return stale_conn
            svc._bind_failed = False
            return truncated_conn

        with _ldap_env(caplog, ensure=ensure_side_effect):
            assert svc._is_user_in_group_sync("alice", "team") is True

        messages = [r.getMessage() for r in caplog.records]
        assert any("ldap_search_retry" in m for m in messages)
        assert not any("ldap_search_recovered" in m for m in messages)

    def test_exhausted_retries_log_error(self, caplog):
        """Both attempts fail: exactly one ERROR, keeping the legacy phrase."""
        mock_conn = MagicMock()
        mock_conn.search.side_effect = Exception("timeout")

        with _ldap_env(caplog, conn=mock_conn):
            assert svc._is_user_in_group_sync("alice", "team") is False

        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(errors) == 1
        message = errors[0].getMessage()
        assert "ldap_search_exhausted" in message
        # Existing log queries key on this phrase; it must survive the rename.
        assert "LDAP search failed for group" in message
        assert any("ldap_search_retry" in r.getMessage() for r in caplog.records)

    def test_failed_rebind_logs_error_from_production_code(self, caplog):
        """Search fails, then the real rebind fails: ERROR, naming the group.

        ``_ensure_bound`` is deliberately NOT patched here. The bind failure
        must be emitted by ``service.py`` itself — a test that logs its own
        ERROR and then asserts on it proves nothing.
        """
        stale_conn = MagicMock()
        stale_conn.closed = False
        stale_conn.search.side_effect = Exception("session terminated by server")

        ldap3_mock = MagicMock()
        ldap3_mock.Connection.side_effect = Exception("invalidCredentials")

        svc._ldap_conn = stale_conn
        svc._bind_failed = False

        # No `conn` or `ensure`: the real _ensure_bound must run.
        with _ldap_env(caplog, ldap3_mock=ldap3_mock) as ms:
            ms.LDAP_URL = "ldaps://ldap.example.com"
            ms.LDAP_TLS_VERIFY = True
            ms.LDAP_PASSWORD = "secret"
            ms.get_bind_dn.return_value = "uid=svc,ou=users,dc=example,dc=com"
            assert svc._is_user_in_group_sync("alice", "team") is False

        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        # Every ERROR must originate in the production module, not this test.
        assert all(r.filename == "service.py" for r in errors), [
            (r.filename, r.getMessage()) for r in errors
        ]
        # Exactly one: a single failure must not be reported twice.
        assert len(errors) == 1, [r.getMessage() for r in errors]
        # _ensure_bound owns the bind-failure ERROR and classifies the cause.
        assert "LDAP bind failed" in errors[0].getMessage()

        # The group is still attributable: the retry warning immediately
        # before the ERROR names it.
        retry = [r for r in caplog.records if "ldap_search_retry" in r.getMessage()]
        assert len(retry) == 1
        assert retry[0].levelno == logging.WARNING
        assert "group=team" in retry[0].getMessage()

    def test_cold_start_bind_failure_is_not_reported_as_a_rebind(self, caplog):
        """No session ever existed: report a plain bind error, nothing else.

        Pins the ``last_attempt`` argument. If the caller always suppressed
        ``_ensure_bound``'s own ERROR, a first-ever bind failure would be
        reported as a session reset that never happened — and would be
        injected into the ``LDAP search failed for group`` query.
        """
        ldap3_mock = MagicMock()
        ldap3_mock.Connection.side_effect = Exception("invalidCredentials")

        svc._ldap_conn = None
        svc._bind_failed = False

        with _ldap_env(caplog, ldap3_mock=ldap3_mock) as ms:
            ms.LDAP_URL = "ldaps://ldap.example.com"
            ms.LDAP_TLS_VERIFY = True
            ms.LDAP_PASSWORD = "secret"
            ms.get_bind_dn.return_value = "uid=svc,ou=users,dc=example,dc=com"
            assert svc._is_user_in_group_sync("alice", "team") is False

        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(errors) == 1
        message = errors[0].getMessage()
        assert "LDAP bind failed" in message
        assert "ldap_search_exhausted" not in message
        assert "LDAP search failed for group" not in message

    def test_group_not_found_on_retry_logs_recovery(self, caplog):
        """Reconnect then result code 0: conclusive, so recovery is logged.

        Code 0 means the search completed and the group has no match. That is
        a real recovery, unlike codes 10/32/34 which stay inconclusive.
        """
        stale_conn = MagicMock()
        stale_conn.search.side_effect = Exception("session terminated by server")

        empty_conn = MagicMock()
        empty_conn.search.return_value = False
        empty_conn.result = {"result": 0, "description": "success"}

        call_count = {"n": 0}

        def ensure_side_effect(**_kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return stale_conn
            svc._bind_failed = False
            return empty_conn

        with _ldap_env(caplog, ensure=ensure_side_effect):
            assert svc._is_user_in_group_sync("alice", "team") is False

        assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
        recovered = [
            r for r in caplog.records if "ldap_search_recovered" in r.getMessage()
        ]
        assert len(recovered) == 1
        assert recovered[0].levelno == logging.INFO
