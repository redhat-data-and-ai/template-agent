"""OIDC/SSO authentication handler for Aegra.

Validates JWT access tokens against an OIDC provider using JWKS
(JSON Web Key Set). Supports any OIDC-compliant SSO provider
(Keycloak, Okta, Azure AD, Auth0, etc.).

Features:
    - ENABLE_AUTH toggle for dev vs production
    - OIDC discovery OR explicit JWKS URI
    - Refresh token propagation (stored in auth user dict)
    - User ID encryption for observability privacy

Required env vars:
    ENABLE_AUTH: Enable/disable authentication (default: false)
    SSO_ISSUER_URL: OIDC issuer URL
    SSO_CLIENT_ID: OAuth2 client ID (used as expected audience)

Optional env vars:
    SSO_CLIENT_SECRET: OAuth2 client secret
    SSO_JWKS_URI: Explicit JWKS URI (skips OIDC discovery)
    SSO_JWT_AUDIENCE: Expected JWT audience (defaults to SSO_CLIENT_ID)
    ENABLE_USER_ID_ENCRYPTION: Encrypt user IDs in logs/traces (default: false)
    USER_ID_ENCRYPTION_KEY: 32-byte hex key for user ID encryption
"""

import asyncio
import hashlib
import hmac
import os
from typing import Any

import httpx
import jwt
from langgraph_sdk import Auth

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

auth = Auth()

ENABLE_AUTH = os.environ.get("ENABLE_AUTH", "true").lower() == "true"
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development").lower()

# Enforce auth in production - fail fast at startup
if ENVIRONMENT == "production" and not ENABLE_AUTH:
    raise RuntimeError(
        "ENABLE_AUTH=false is not permitted when ENVIRONMENT=production. "
        "Production deployments must use SSO authentication. "
        "Set ENABLE_AUTH=true and configure SSO_ISSUER_URL, SSO_CLIENT_ID, and SSO_CLIENT_SECRET."
    )

SSO_ISSUER_URL = os.environ.get("SSO_ISSUER_URL", "")
SSO_CLIENT_ID = os.environ.get("SSO_CLIENT_ID", "")
SSO_CLIENT_SECRET = os.environ.get("SSO_CLIENT_SECRET", "")
SSO_JWKS_URI = os.environ.get("SSO_JWKS_URI", "")
SSO_JWT_ALGORITHMS = os.environ.get("SSO_JWT_ALGORITHMS", "RS256,ES256").split(",")
SSO_JWT_AUDIENCE = os.environ.get("SSO_JWT_AUDIENCE", "")

DEV_USERNAME = os.environ.get("SSO_DEV_USERNAME", "John Doe")
DEV_USER_ID = os.environ.get("SSO_DEV_USER_ID", "dev-user")

ENABLE_USER_ID_ENCRYPTION = (
    os.environ.get("ENABLE_USER_ID_ENCRYPTION", "false").lower() == "true"
)
USER_ID_ENCRYPTION_KEY = os.environ.get("USER_ID_ENCRYPTION_KEY", "")

_jwks_client: jwt.PyJWKClient | None = None

# ── Eval token refresh (off by default) ──────────────────────────────────────
EVAL_TOKEN_REFRESH_ENABLED: bool = (
    os.environ.get("EVAL_TOKEN_REFRESH_ENABLED", "false").lower() == "true"
)
_EVAL_ACTIVE_TTL = (
    3600  # 60-min safety-net TTL; explicit cleanup via /evals/internal/cleanup
)
_EVAL_REFRESH_TTL = 3600
_EVAL_ACCESS_TTL = 270  # 5-min token − 30s MCP buffer


def _decode_sub_unverified(token: str) -> str | None:
    """Extract sub from a JWT without signature verification."""
    try:
        payload = jwt.decode(token, options={"verify_signature": False})
        sub = payload.get("sub")
        return str(sub) if sub else None
    except Exception:
        return None


async def _oidc_refresh(refresh_token: str) -> tuple[str, str]:
    """Exchange a refresh token for a new (access_token, refresh_token) pair.

    Shared by auth.py (expired-token path) and mcp.py (proactive refresh).
    Returns the new refresh token so callers can persist it — required when
    Keycloak refresh-token rotation is enabled (invalidates old RT on use).
    """
    token_url = f"{SSO_ISSUER_URL.rstrip('/')}/protocol/openid-connect/token"
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, pool=2.0)) as client:
        resp = await client.post(
            token_url,
            data={
                "grant_type": "refresh_token",
                "client_id": SSO_CLIENT_ID,
                "client_secret": SSO_CLIENT_SECRET,
                "refresh_token": refresh_token,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["access_token"], data.get("refresh_token", refresh_token)


def encrypt_user_id(user_id: str) -> str:
    """Deterministically encrypt a user ID for observability privacy.

    Uses HMAC-SHA256 with a secret key, producing a consistent hash
    so the same user always maps to the same encrypted ID.
    """
    if not ENABLE_USER_ID_ENCRYPTION or not USER_ID_ENCRYPTION_KEY:
        return user_id
    return hmac.new(
        USER_ID_ENCRYPTION_KEY.encode(), user_id.encode(), hashlib.sha256
    ).hexdigest()[:16]


def _resolve_jwks_uri() -> str:
    """Resolve JWKS URI from explicit config or OIDC discovery.

    Caches the resolved URI in ``_RESOLVED_JWKS_URI`` env var so that
    workers that re-import this module skip the HTTP discovery round-trip.
    """
    if SSO_JWKS_URI:
        logger.info("Using explicit SSO_JWKS_URI: %s", SSO_JWKS_URI)
        return SSO_JWKS_URI

    cached = os.environ.get("_RESOLVED_JWKS_URI", "")
    if cached:
        logger.debug("Using cached JWKS URI: %s", cached)
        return cached

    if not SSO_ISSUER_URL:
        raise RuntimeError("SSO_ISSUER_URL or SSO_JWKS_URI must be set")

    discovery_url = f"{SSO_ISSUER_URL.rstrip('/')}/.well-known/openid-configuration"
    logger.info("Discovering JWKS from: %s", discovery_url)
    resp = httpx.get(discovery_url, timeout=10)
    resp.raise_for_status()
    jwks_uri: str = resp.json()["jwks_uri"]
    os.environ["_RESOLVED_JWKS_URI"] = jwks_uri
    return jwks_uri


def _get_jwks_client() -> jwt.PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        try:
            jwks_uri = _resolve_jwks_uri()
            _jwks_client = jwt.PyJWKClient(jwks_uri, cache_keys=True, lifespan=3600)
        except Exception as e:
            logger.error("Failed to initialize JWKS client: %s", e)
            raise RuntimeError(f"JWKS initialization failed: {e}") from e
    return _jwks_client


def _decode_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT against the SSO provider's JWKS."""
    client = _get_jwks_client()
    signing_key = client.get_signing_key_from_jwt(token)

    decode_options: dict[str, Any] = {"require": ["exp", "iss", "sub"]}
    kwargs: dict[str, Any] = {
        "algorithms": [a.strip() for a in SSO_JWT_ALGORITHMS],
        "options": decode_options,
    }
    if SSO_JWT_AUDIENCE:
        kwargs["audience"] = SSO_JWT_AUDIENCE
    else:
        decode_options["verify_aud"] = False
    if SSO_ISSUER_URL:
        kwargs["issuer"] = SSO_ISSUER_URL

    result: dict[str, Any] = jwt.decode(token, signing_key.key, **kwargs)
    return result


def _build_dev_user() -> dict[str, Any]:
    """Build a dev-mode user identity when auth is disabled."""
    return {
        "identity": DEV_USER_ID,
        "display_name": DEV_USERNAME,
        "permissions": ["read", "write", "admin"],
        "is_authenticated": True,
        "email": "dev@localhost",
        "encrypted_id": encrypt_user_id(DEV_USER_ID),
    }


@auth.authenticate
async def authenticate(headers: dict) -> dict:
    """Validate the Bearer token from the Authorization header.

    When ENABLE_AUTH is false AND not in production, returns a dev user identity.
    Production always requires authentication.
    Extracts access_token and refresh_token for downstream propagation.
    """
    # Block auth bypass in production (defense-in-depth)
    if ENVIRONMENT == "production" and not ENABLE_AUTH:
        raise PermissionError(
            "Authentication bypass disabled in production. "
            "Set ENABLE_AUTH=true and configure SSO credentials."
        )

    if not ENABLE_AUTH:
        logger.warning("Auth bypass active (development mode)")
        return _build_dev_user()

    auth_header = headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise PermissionError("Missing or invalid Authorization header")

    access_token = auth_header[7:]
    refresh_token = headers.get("x-refresh-token", "")

    try:
        payload = _decode_token(access_token)
    except jwt.ExpiredSignatureError:
        # ── Expired token: refresh if this is an active eval run ─────────────
        if not EVAL_TOKEN_REFRESH_ENABLED:
            raise PermissionError("Token expired")

        from deep_agent.aegra.redis import (
            cache_get,
            cache_set,
            distributed_lock,
            get_redis_client,
        )

        if get_redis_client() is None:
            raise PermissionError("Token expired")

        sub = _decode_sub_unverified(access_token)
        if not sub:
            raise PermissionError("Token expired and sub unreadable")

        if not await asyncio.to_thread(cache_get, f"eval:active:{sub}"):
            raise PermissionError("Token expired — no active eval for this user")

        from deep_agent.aegra.mcp_crypto import decrypt_secret, encrypt_secret

        # Cache hit: another thread already refreshed — reuse without OIDC call
        enc_cached = await asyncio.to_thread(cache_get, f"eval:access:{sub}")
        if enc_cached:
            try:
                cached = decrypt_secret(enc_cached) or ""
                p = _decode_token(cached)
                enc_rt = await asyncio.to_thread(cache_get, f"eval:refresh:{sub}") or ""
                stored_rt = decrypt_secret(enc_rt) or "" if enc_rt else ""
                return _make_user(p, cached, stored_rt)
            except Exception:
                pass  # cached token also expired — fall through to lock path

        # Lock: only one thread calls OIDC; others poll the cache
        async with distributed_lock(
            f"eval:refresh_lock:{sub}", ttl_seconds=30, wait_seconds=12
        ) as state:
            if state == "held":
                enc_rt = await asyncio.to_thread(
                    lambda: cache_get(f"eval:refresh:{sub}") or ""
                )
                if not enc_rt:
                    raise PermissionError("Token expired — no refresh token stored")
                stored_rt = decrypt_secret(enc_rt) or ""
                if not stored_rt:
                    raise PermissionError(
                        "Token expired — refresh token could not be decrypted"
                    )
                try:
                    new_access, new_rt = await _oidc_refresh(stored_rt)
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 400:
                        raise PermissionError(
                            "Eval refresh token expired or rotated — re-trigger eval"
                        ) from exc
                    raise
                await asyncio.to_thread(
                    cache_set,
                    f"eval:access:{sub}",
                    encrypt_secret(new_access) or "",
                    _EVAL_ACCESS_TTL,
                )
                await asyncio.to_thread(
                    cache_set,
                    f"eval:refresh:{sub}",
                    encrypt_secret(new_rt) or "",
                    _EVAL_REFRESH_TTL,
                )
                logger.info("eval_token_refreshed")
                return _make_user(_decode_token(new_access), new_access, new_rt)
            else:
                # Lock loser: poll until winner writes the cache
                for _ in range(6):
                    await asyncio.sleep(0.5)
                    enc_polled = await asyncio.to_thread(
                        cache_get, f"eval:access:{sub}"
                    )
                    if enc_polled:
                        try:
                            polled = decrypt_secret(enc_polled) or ""
                            p = _decode_token(polled)
                            enc_rt = await asyncio.to_thread(
                                lambda: cache_get(f"eval:refresh:{sub}") or ""
                            )
                            stored_rt = decrypt_secret(enc_rt) or "" if enc_rt else ""
                            return _make_user(p, polled, stored_rt)
                        except Exception:
                            break
                raise PermissionError(
                    "Token expired — refresh in progress, please retry"
                )

    user_id = payload["sub"]

    # Valid token: keep Redis refresh token current while eval is active
    if EVAL_TOKEN_REFRESH_ENABLED and refresh_token:
        from deep_agent.aegra.mcp_crypto import encrypt_secret
        from deep_agent.aegra.redis import cache_get, cache_set, get_redis_client

        if get_redis_client() is not None:
            active = await asyncio.to_thread(cache_get, f"eval:active:{user_id}")
            if active:
                await asyncio.to_thread(
                    cache_set,
                    f"eval:refresh:{user_id}",
                    encrypt_secret(refresh_token) or "",
                    _EVAL_REFRESH_TTL,
                )

    return _make_user(payload, access_token, refresh_token)


def _make_user(
    payload: dict[str, Any], access_token: str, refresh_token: str
) -> dict[str, Any]:
    uid = payload["sub"]
    return {
        "identity": uid,
        "display_name": payload.get("name", payload.get("preferred_username", "")),
        "permissions": payload.get("realm_access", {}).get("roles", []),
        "is_authenticated": True,
        "email": payload.get("email", ""),
        "access_token": access_token,
        "refresh_token": refresh_token,
        "encrypted_id": encrypt_user_id(uid),
    }
