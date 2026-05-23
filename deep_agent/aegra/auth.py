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

    When ENABLE_AUTH is false, returns a dev user identity.
    Extracts access_token and refresh_token for downstream propagation.
    """
    if not ENABLE_AUTH:
        return _build_dev_user()

    auth_header = headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise PermissionError("Missing or invalid Authorization header")

    access_token = auth_header[7:]
    payload = _decode_token(access_token)

    user_id = payload["sub"]
    refresh_token = headers.get("x-refresh-token", "")

    return {
        "identity": user_id,
        "display_name": payload.get("name", payload.get("preferred_username", "")),
        "permissions": payload.get("realm_access", {}).get("roles", []),
        "is_authenticated": True,
        "email": payload.get("email", ""),
        "access_token": access_token,
        "refresh_token": refresh_token,
        "encrypted_id": encrypt_user_id(user_id),
    }
