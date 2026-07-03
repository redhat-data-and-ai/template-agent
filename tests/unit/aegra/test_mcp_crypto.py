"""Unit tests for MCP OAuth token encryption."""

from __future__ import annotations

import os

import pytest
from cryptography.fernet import Fernet, InvalidToken

from deep_agent.aegra.mcp_crypto import (
    decrypt_secret,
    encrypt_secret,
    reset_mcp_crypto_cache,
)


@pytest.fixture(autouse=True)
def _clear_crypto_cache():
    reset_mcp_crypto_cache()
    yield
    reset_mcp_crypto_cache()


@pytest.fixture
def fernet_keys():
    primary = Fernet.generate_key().decode()
    previous = Fernet.generate_key().decode()
    return primary, previous


class TestMcpCrypto:
    def test_encrypt_decrypt_round_trip(self, fernet_keys, monkeypatch):
        primary, _ = fernet_keys
        monkeypatch.setenv("MCP_TOKEN_ENCRYPTION_KEY", primary)
        ciphertext = encrypt_secret("secret-token")
        assert ciphertext is not None
        assert decrypt_secret(ciphertext) == "secret-token"

    def test_none_passthrough(self, fernet_keys, monkeypatch):
        primary, _ = fernet_keys
        monkeypatch.setenv("MCP_TOKEN_ENCRYPTION_KEY", primary)
        assert encrypt_secret(None) is None
        assert decrypt_secret(None) is None

    def test_decrypt_with_previous_key(self, fernet_keys, monkeypatch):
        primary, previous = fernet_keys
        monkeypatch.setenv("MCP_TOKEN_ENCRYPTION_KEY", previous)
        ciphertext = encrypt_secret("rotated-secret")

        monkeypatch.setenv("MCP_TOKEN_ENCRYPTION_KEY", primary)
        monkeypatch.setenv("MCP_TOKEN_ENCRYPTION_KEY_PREVIOUS", previous)
        reset_mcp_crypto_cache()

        assert decrypt_secret(ciphertext) == "rotated-secret"

    def test_encrypt_uses_primary_only(self, fernet_keys, monkeypatch):
        primary, previous = fernet_keys
        monkeypatch.setenv("MCP_TOKEN_ENCRYPTION_KEY", primary)
        monkeypatch.setenv("MCP_TOKEN_ENCRYPTION_KEY_PREVIOUS", previous)
        ciphertext = encrypt_secret("new-secret")

        with pytest.raises(InvalidToken):
            Fernet(previous.encode()).decrypt(ciphertext.encode())
        assert (
            Fernet(primary.encode()).decrypt(ciphertext.encode()).decode()
            == "new-secret"
        )

    def test_missing_primary_key_raises(self, monkeypatch):
        monkeypatch.delenv("MCP_TOKEN_ENCRYPTION_KEY", raising=False)
        with pytest.raises(RuntimeError, match="MCP_TOKEN_ENCRYPTION_KEY"):
            encrypt_secret("x")

    def test_wrong_keys_raise(self, fernet_keys, monkeypatch):
        primary, previous = fernet_keys
        other = Fernet.generate_key().decode()
        monkeypatch.setenv("MCP_TOKEN_ENCRYPTION_KEY", other)
        ciphertext = encrypt_secret("lost-secret")

        monkeypatch.setenv("MCP_TOKEN_ENCRYPTION_KEY", primary)
        monkeypatch.setenv("MCP_TOKEN_ENCRYPTION_KEY_PREVIOUS", previous)
        reset_mcp_crypto_cache()

        with pytest.raises(RuntimeError, match="decryption failed"):
            decrypt_secret(ciphertext)
