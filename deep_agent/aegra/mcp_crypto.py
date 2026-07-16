"""Symmetric encryption for MCP OAuth tokens and client secrets at rest."""

from __future__ import annotations

import os
from typing import cast

from cryptography.fernet import Fernet, InvalidToken

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_fernet_primary: Fernet | None = None
_fernet_previous: Fernet | None | bool = False


def _fernet_key_help() -> str:
    return (
        'Generate one with: python -c "from cryptography.fernet import Fernet; '
        'print(Fernet.generate_key().decode())"'
    )


def _parse_fernet_key(env_var: str) -> str:
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise RuntimeError(
            f"{env_var} is required for MCP OAuth token storage. {_fernet_key_help()}"
        )
    return key


def _get_fernet_primary() -> Fernet:
    """Return the current encryption key (``MCP_TOKEN_ENCRYPTION_KEY``)."""
    global _fernet_primary  # noqa: PLW0603
    if _fernet_primary is None:
        _fernet_primary = Fernet(_parse_fernet_key("MCP_TOKEN_ENCRYPTION_KEY").encode())
    return _fernet_primary


def _get_fernet_previous() -> Fernet | None:
    """Return the optional previous key used during rotation."""
    global _fernet_previous  # noqa: PLW0603
    if _fernet_previous is False:
        previous_key = os.environ.get("MCP_TOKEN_ENCRYPTION_KEY_PREVIOUS", "").strip()
        _fernet_previous = Fernet(previous_key.encode()) if previous_key else None
    return cast(Fernet | None, _fernet_previous)


def reset_mcp_crypto_cache() -> None:
    """Clear cached Fernet instances (used by tests and config reload)."""
    global _fernet_primary, _fernet_previous  # noqa: PLW0603
    _fernet_primary = None
    _fernet_previous = False


def encrypt_secret(plaintext: str | None) -> str | None:
    """Encrypt a secret value for Redis/Postgres storage."""
    if plaintext is None:
        return None
    return cast(str, _get_fernet_primary().encrypt(plaintext.encode()).decode())


def decrypt_secret(ciphertext: str | None) -> str | None:
    """Decrypt a value previously stored by :func:`encrypt_secret`.

    Tries ``MCP_TOKEN_ENCRYPTION_KEY`` first, then ``MCP_TOKEN_ENCRYPTION_KEY_PREVIOUS``
    when set, so ciphertext can be read during key rotation.
    """
    if ciphertext is None:
        return None

    keys: list[Fernet] = [_get_fernet_primary()]
    previous = _get_fernet_previous()
    if previous is not None:
        keys.append(previous)

    last_exc: InvalidToken | None = None
    for fernet in keys:
        try:
            return cast(str, fernet.decrypt(ciphertext.encode()).decode())
        except InvalidToken as exc:
            last_exc = exc

    logger.error("Failed to decrypt MCP secret — key mismatch or corrupt data")
    raise RuntimeError("MCP token decryption failed") from last_exc
