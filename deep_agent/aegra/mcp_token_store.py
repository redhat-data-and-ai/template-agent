"""Repository for MCP OAuth tokens (Redis) and DCR client records (Postgres)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from deep_agent.aegra.mcp_crypto import decrypt_secret, encrypt_secret
from deep_agent.aegra.redis import cache_delete, cache_get, cache_set_persistent
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_TABLES_ENSURED = False
_TOKEN_KEY_PREFIX = "mcp_oauth_token:"

CREATE_OAUTH_CLIENTS_TABLE = """
CREATE TABLE IF NOT EXISTS mcp_oauth_clients (
    mcp_name           TEXT PRIMARY KEY,
    client_id          TEXT NOT NULL,
    client_secret      TEXT,
    registration_data  JSONB,
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

MIGRATE_OAUTH_TABLES = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'mcp_oauth_clients'
    ) AND (
        NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'mcp_oauth_clients'
              AND column_name = 'client_id'
        )
        OR NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'mcp_oauth_clients'
              AND column_name = 'registration_data'
        )
        OR NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'mcp_oauth_clients'
              AND column_name = 'updated_at'
        )
    ) THEN
        DROP TABLE mcp_oauth_clients;
    END IF;
END $$;
"""


@dataclass
class McpOAuthClient:
    """Registered OAuth client for a DCR-backed MCP server."""

    mcp_name: str
    client_id: str
    client_secret: str | None = None
    registration_data: dict[str, Any] | None = None
    updated_at: datetime | None = None


@dataclass
class McpOAuthToken:
    """Stored OAuth tokens for a (user, MCP) pair."""

    user_id: str
    mcp_name: str
    access_token: str
    refresh_token: str | None = None
    expires_at: datetime | None = None
    scopes: list[str] | None = None
    updated_at: datetime | None = None


class McpTokenStore:
    """Async store for MCP OAuth user tokens (Redis) and DCR clients (Postgres)."""

    def __init__(self, database_uri: str) -> None:
        """Initialize with a Postgres connection URI for DCR client records."""
        self._uri = database_uri

    @staticmethod
    def _token_key(user_id: str, mcp_name: str) -> str:
        return f"{_TOKEN_KEY_PREFIX}{user_id}:{mcp_name}"

    @staticmethod
    def _serialize_datetime(value: datetime | None) -> str | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat()

    @staticmethod
    def _deserialize_datetime(value: str | None) -> datetime | None:
        if not value:
            return None
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    def _token_to_payload(
        self,
        access_token: str,
        refresh_token: str | None,
        expires_at: datetime | None,
        scopes: list[str] | None,
    ) -> dict[str, Any]:
        now = datetime.now(UTC)
        return {
            "access_token": encrypt_secret(access_token),
            "refresh_token": encrypt_secret(refresh_token),
            "expires_at": self._serialize_datetime(expires_at),
            "scopes": scopes,
            "updated_at": self._serialize_datetime(now),
        }

    def _payload_to_token(
        self, user_id: str, mcp_name: str, payload: dict[str, Any]
    ) -> McpOAuthToken:
        return McpOAuthToken(
            user_id=user_id,
            mcp_name=mcp_name,
            access_token=decrypt_secret(payload.get("access_token")) or "",
            refresh_token=decrypt_secret(payload.get("refresh_token")),
            expires_at=self._deserialize_datetime(payload.get("expires_at")),
            scopes=list(payload["scopes"]) if payload.get("scopes") else None,
            updated_at=self._deserialize_datetime(payload.get("updated_at")),
        )

    async def ensure_tables(self) -> None:
        """Create MCP OAuth client table in Postgres if it does not already exist."""
        global _TABLES_ENSURED  # noqa: PLW0603
        async with await psycopg.AsyncConnection.connect(self._uri) as conn:
            await conn.execute(MIGRATE_OAUTH_TABLES)
            await conn.execute(CREATE_OAUTH_CLIENTS_TABLE)
            await conn.commit()
        if not _TABLES_ENSURED:
            _TABLES_ENSURED = True
            logger.info("MCP OAuth client table ensured")

    async def get_client(self, mcp_name: str) -> McpOAuthClient | None:
        """Return the registered OAuth client for *mcp_name*, if any."""
        await self.ensure_tables()
        async with await psycopg.AsyncConnection.connect(
            self._uri, row_factory=dict_row
        ) as conn:
            cur = await conn.execute(
                "SELECT * FROM mcp_oauth_clients WHERE mcp_name = %s",
                (mcp_name,),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            return McpOAuthClient(
                mcp_name=row["mcp_name"],
                client_id=row["client_id"],
                client_secret=decrypt_secret(row["client_secret"]),
                registration_data=row["registration_data"],
                updated_at=row["updated_at"],
            )

    async def upsert_client(
        self,
        mcp_name: str,
        client_id: str,
        client_secret: str | None = None,
        registration_data: dict[str, Any] | None = None,
    ) -> McpOAuthClient:
        """Insert or update the OAuth client record for *mcp_name*."""
        await self.ensure_tables()
        enc_secret = encrypt_secret(client_secret)
        async with await psycopg.AsyncConnection.connect(self._uri) as conn:
            await conn.execute(
                """
                INSERT INTO mcp_oauth_clients (
                    mcp_name, client_id, client_secret, registration_data, updated_at
                )
                VALUES (%s, %s, %s, %s, now())
                ON CONFLICT (mcp_name) DO UPDATE SET
                    client_id = EXCLUDED.client_id,
                    client_secret = EXCLUDED.client_secret,
                    registration_data = EXCLUDED.registration_data,
                    updated_at = now()
                """,
                (
                    mcp_name,
                    client_id,
                    enc_secret,
                    Jsonb(registration_data) if registration_data is not None else None,
                ),
            )
            await conn.commit()
        return McpOAuthClient(
            mcp_name=mcp_name,
            client_id=client_id,
            client_secret=client_secret,
            registration_data=registration_data,
        )

    async def get_token(self, user_id: str, mcp_name: str) -> McpOAuthToken | None:
        """Return stored OAuth tokens for *(user_id, mcp_name)* from Redis."""
        raw = await asyncio.to_thread(cache_get, self._token_key(user_id, mcp_name))
        if raw is None:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.error(
                "Corrupt MCP OAuth token payload for user '%s' MCP '%s'",
                user_id,
                mcp_name,
            )
            return None
        if not isinstance(payload, dict):
            logger.error(
                "Invalid MCP OAuth token payload type for user '%s' MCP '%s'",
                user_id,
                mcp_name,
            )
            return None
        return self._payload_to_token(user_id, mcp_name, payload)

    async def upsert_token(
        self,
        user_id: str,
        mcp_name: str,
        access_token: str,
        refresh_token: str | None = None,
        expires_at: datetime | None = None,
        scopes: list[str] | None = None,
    ) -> McpOAuthToken:
        """Insert or update OAuth tokens for *(user_id, mcp_name)* in Redis."""
        payload = self._token_to_payload(
            access_token, refresh_token, expires_at, scopes
        )
        key = self._token_key(user_id, mcp_name)
        stored = await asyncio.to_thread(cache_set_persistent, key, json.dumps(payload))
        if not stored:
            raise RuntimeError(
                f"Failed to persist MCP OAuth token for user '{user_id}' MCP '{mcp_name}'"
            )
        return McpOAuthToken(
            user_id=user_id,
            mcp_name=mcp_name,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            scopes=scopes,
            updated_at=self._deserialize_datetime(payload["updated_at"]),
        )

    async def delete_token(self, user_id: str, mcp_name: str) -> bool:
        """Delete stored OAuth tokens for *(user_id, mcp_name)* from Redis."""
        return await asyncio.to_thread(cache_delete, self._token_key(user_id, mcp_name))

    @staticmethod
    def expires_at_from_token_response(data: dict[str, Any]) -> datetime | None:
        """Compute expiry from an OAuth token endpoint JSON body."""
        expires_in = data.get("expires_in")
        if expires_in is None:
            return None
        try:
            return datetime.now(UTC) + timedelta(seconds=int(expires_in))
        except (TypeError, ValueError):
            return None
