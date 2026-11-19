"""LDAP group membership queries with Redis caching.

Provides the core ``resolve_user_role()`` function that ties together
PROMPT.md group config, LDAP membership lookups, and the role hierarchy.
All LDAP I/O is synchronous (ldap3) and wrapped in ``asyncio.to_thread``
for use in the async auth layer.
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
from typing import Any

from deep_agent.src.ldap.config import ldap_settings
from deep_agent.src.ldap.prompt_config import (
    ROLE_HIERARCHY,
    GroupRoleMapping,
    get_prompt_access_config,
)
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_ldap_conn: Any = None
_bind_failed: bool = False
_bind_failed_at: float = 0.0
_ldap_lock = threading.Lock()
_memory_cache: dict[str, tuple[bool, float]] = {}
_MEMORY_CACHE_MAX_SIZE: int = 1024

_startup_warning_logged: bool = False

_LDAP_FILTER_ESCAPE = re.compile(r"([\\*\(\)\x00])")

_COOLDOWN_TRANSIENT: int = 3
_COOLDOWN_INVALID_CREDENTIALS: int = 30
_COOLDOWN_ACCOUNT_LOCKED: int = 60


def _escape_ldap_filter(value: str) -> str:
    """Escape special characters for use in an LDAP search filter (RFC 4515)."""
    return _LDAP_FILTER_ESCAPE.sub(
        lambda m: "\\" + format(ord(m.group(1)), "02x"), value
    )


def _log_search_recovered(group_cn: str, attempt: int) -> None:
    """Record that a retried search completed, so recovery is visible in logs."""
    if attempt > 0:
        logger.info(
            "ldap_search_recovered: group=%s attempt=%d/2 — "
            "membership lookup completed after reconnect",
            group_cn,
            attempt + 1,
        )


def _derive_base_dn() -> str:
    """Delegate to ldap_settings for the derived base DN."""
    return ldap_settings.derive_base_dn()


def _ensure_bound(last_attempt: bool = True) -> Any:
    """Ensure an LDAP connection is bound. Returns the connection or None."""
    global _ldap_conn, _bind_failed, _bind_failed_at

    if not ldap_settings.LDAP_URL:
        return None

    if not ldap_settings.LDAP_URL.startswith("ldaps://"):
        logger.error(
            "LDAP_URL must use ldaps:// (got %s) — refusing to send credentials in cleartext",
            ldap_settings.LDAP_URL.split("://")[0] + "://...",
        )
        return None

    if _bind_failed and time.monotonic() < _bind_failed_at:
        return None

    if _ldap_conn is not None and not _bind_failed:
        if not getattr(_ldap_conn, "closed", False):
            return _ldap_conn
        _ldap_conn = None

    import ssl

    from ldap3 import Connection, Server, Tls

    tls_validate = ssl.CERT_REQUIRED if ldap_settings.LDAP_TLS_VERIFY else ssl.CERT_NONE
    tls = Tls(validate=tls_validate)
    server = Server(
        ldap_settings.LDAP_URL,
        use_ssl=True,
        tls=tls,
        connect_timeout=10,
    )
    bind_dn = ldap_settings.get_bind_dn()
    logger.info(
        "LDAP attempting bind — url=%s group_search_base=%s",
        ldap_settings.LDAP_URL,
        ldap_settings.get_group_search_base(),
    )

    for bind_attempt in range(2):
        try:
            conn = Connection(
                server,
                user=bind_dn,
                password=ldap_settings.LDAP_PASSWORD,
                auto_bind=True,
                read_only=True,
            )
            _ldap_conn = conn
            _bind_failed = False
            logger.info("LDAP bound successfully to %s", ldap_settings.LDAP_URL)
            return conn
        except Exception as exc:
            exc_str = str(exc)
            if "invalidCredentials" in exc_str:
                cooldown = _COOLDOWN_INVALID_CREDENTIALS
                reason = "invalid credentials"
            elif "unwillingToPerform" in exc_str:
                cooldown = _COOLDOWN_ACCOUNT_LOCKED
                reason = "account locked"
            else:
                cooldown = _COOLDOWN_TRANSIENT
                reason = "connection error"

            is_auth_error = cooldown > 3
            is_last_bind = bind_attempt == 1 or is_auth_error

            if is_last_bind:
                if last_attempt or is_auth_error:
                    logger.error(
                        "LDAP bind failed: %s (retry in %ds)", reason, cooldown
                    )
                else:
                    logger.warning("LDAP bind failed: %s — retrying", reason)
                _bind_failed = True
                _bind_failed_at = time.monotonic() + cooldown
                _ldap_conn = None
                return None

            logger.warning(
                "LDAP bind transient error on attempt 1 — retrying immediately"
            )

    return None


def _cache_get(key: str) -> bool | None:
    """Read from Redis cache, falling back to in-memory cache."""
    from deep_agent.aegra.redis import cache_get

    val = cache_get(key)
    if val is not None:
        return val == "1"

    entry = _memory_cache.get(key)
    if entry is not None:
        result, ts = entry
        if time.monotonic() - ts < ldap_settings.LDAP_CACHE_TTL_SECONDS:
            return result

    return None


def _evict_expired() -> None:
    """Remove expired entries from the in-memory cache."""
    ttl = ldap_settings.LDAP_CACHE_TTL_SECONDS
    now = time.monotonic()
    expired = [k for k, (_, ts) in _memory_cache.items() if now - ts >= ttl]
    for k in expired:
        del _memory_cache[k]


def _cache_set(key: str, value: bool) -> None:
    """Write to Redis cache and in-memory fallback."""
    from deep_agent.aegra.redis import cache_set

    cache_set(key, "1" if value else "0", ldap_settings.LDAP_CACHE_TTL_SECONDS)

    if len(_memory_cache) >= _MEMORY_CACHE_MAX_SIZE:
        _evict_expired()
    if len(_memory_cache) >= _MEMORY_CACHE_MAX_SIZE:
        _memory_cache.clear()

    _memory_cache[key] = (value, time.monotonic())


def _is_user_in_group_sync(user_id: str, group_cn: str) -> bool:
    """Check if a user belongs to an LDAP group (synchronous)."""
    global _ldap_conn, _bind_failed

    cache_key = f"ldap:membership:{user_id}:{group_cn}"

    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # Non-fatal LDAP result codes that should not destroy the connection.
    # 0 = success with zero entries, 10 = referral, 32 = noSuchObject,
    # 34 = invalidDNSyntax.
    _LDAP_NON_FATAL_RESULT_CODES = {0, 10, 32, 34}
    # Only cache when the result is conclusive (code 0 = search succeeded,
    # group simply has no matching entries). Codes 10/32/34 are inconclusive
    # — the group may become reachable later.
    _LDAP_CACHEABLE_RESULT_CODES = {0}

    with _ldap_lock:
        search_base = ldap_settings.get_group_search_base()
        base_dn = _derive_base_dn()
        member_attrs = ["member", "uniqueMember", "memberUid"]

        for attempt in range(2):
            conn = _ensure_bound(last_attempt=attempt == 1)
            if conn is None:
                return False

            try:
                from ldap3 import SUBTREE

                safe_cn = _escape_ldap_filter(group_cn)
                search_ok = conn.search(
                    search_base,
                    f"(cn={safe_cn})",
                    search_scope=SUBTREE,
                    attributes=member_attrs,
                )

                if not search_ok:
                    result_code = conn.result.get("result", -1) if conn.result else -1
                    if result_code in _LDAP_NON_FATAL_RESULT_CODES:
                        if result_code in _LDAP_CACHEABLE_RESULT_CODES:
                            # Code 0 is conclusive — the search really did
                            # complete, the group simply has no match — so a
                            # retry that reaches here has genuinely recovered.
                            # Codes 10/32/34 are inconclusive, and get no
                            # recovery marker.
                            _log_search_recovered(group_cn, attempt)
                        logger.warning(
                            "LDAP group '%s' not found (result code %d: %s) — "
                            "skipping, other groups unaffected",
                            group_cn,
                            result_code,
                            conn.result.get("description", "unknown"),
                        )
                        if result_code in _LDAP_CACHEABLE_RESULT_CODES:
                            _cache_set(cache_key, False)
                        return False
                    raise RuntimeError(
                        f"LDAP search returned False (result: {conn.result})"
                    )

                found = False
                user_lower = user_id.lower()
                for entry in conn.entries:
                    for attr in member_attrs:
                        values = getattr(entry, attr, None)
                        if values is None:
                            continue
                        raw_values = (
                            values.values if hasattr(values, "values") else values
                        )
                        if not isinstance(raw_values, (list, tuple)):
                            raw_values = [raw_values]
                        for member_val in raw_values:
                            member_str = str(member_val).lower()
                            if (
                                member_str == user_lower
                                or member_str == f"uid={user_lower},ou=users,{base_dn}"
                                or member_str.startswith(f"uid={user_lower},")
                            ):
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break

                # Only a clean completion counts as recovery. ldap3 returns
                # True whenever the search finished with at least one entry,
                # so a truncated result (e.g. sizeLimitExceeded, code 4) also
                # lands here — entries present, result set incomplete.
                result_code = conn.result.get("result", -1) if conn.result else -1
                if result_code in _LDAP_CACHEABLE_RESULT_CODES:
                    _log_search_recovered(group_cn, attempt)
                _cache_set(cache_key, found)
                return found

            except Exception as exc:
                try:
                    conn.unbind()
                except Exception:
                    pass
                _ldap_conn = None
                _bind_failed = True

                if attempt == 0:
                    logger.warning(
                        "ldap_search_retry: group=%s attempt=1/2 error=%s — "
                        "discarding session, retrying with a fresh bind",
                        group_cn,
                        type(exc).__name__,
                    )
                    continue

                # Keeps the original "<group>: <error>" shape so existing log
                # queries and field extractions still match, but now only for
                # failures that the reconnect did not fix. The attempt count is
                # appended, not infixed, for that reason. Only the exception
                # type is logged, matching the redaction upstream applied here.
                logger.error(
                    "ldap_search_exhausted: LDAP search failed for group "
                    "%s: %s (after 2 attempts)",
                    group_cn,
                    type(exc).__name__,
                )
                return False

        return False


def _resolve_user_role_sync(
    user_id: str, group_mappings: list[GroupRoleMapping]
) -> str:
    """Resolve the highest-priority role from LDAP group membership (synchronous)."""
    highest_role: str | None = None
    highest_priority = 0

    for mapping in group_mappings:
        try:
            if _is_user_in_group_sync(user_id, mapping.group):
                priority = ROLE_HIERARCHY.get(mapping.role, 0)
                if priority > highest_priority:
                    highest_priority = priority
                    highest_role = mapping.role
        except Exception as exc:
            logger.error(
                "LDAP check failed for group '%s' (role '%s'), "
                "continuing with remaining groups: %s",
                mapping.group,
                mapping.role,
                exc,
            )

    return highest_role or "denied"


async def resolve_user_role(user_id: str) -> str | None:
    """Resolve a user's role from PROMPT.md groups + LDAP membership.

    Returns None when ENABLE_AUTH=false (dev mode, full access).
    Returns 'users' when no groups configured (chat only).
    Accessibility-aware: private + no match = 'denied', public + no match = 'users'.
    """
    global _startup_warning_logged

    config = get_prompt_access_config()

    if config.groups is None:
        return "users"

    is_public = config.accessibility == "public"

    if not ldap_settings.LDAP_URL:
        if not _startup_warning_logged:
            logger.warning(
                "PROMPT.md has groups defined but LDAP_URL is not set — "
                "denying all access (fail-closed)"
            )
            _startup_warning_logged = True
        return "users" if is_public else "denied"

    role = await asyncio.to_thread(_resolve_user_role_sync, user_id, config.groups)

    if role == "denied" and is_public:
        return "users"
    return role


def close_ldap() -> None:
    """Unbind the LDAP connection and clear the in-memory cache."""
    global _ldap_conn, _bind_failed
    with _ldap_lock:
        if _ldap_conn is not None:
            try:
                _ldap_conn.unbind()
            except Exception:
                pass
            _ldap_conn = None
        _bind_failed = False
    _memory_cache.clear()
    logger.info("LDAP client closed")
