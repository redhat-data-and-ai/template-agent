"""Repository for user policy settings in PostgreSQL."""

from __future__ import annotations

import json
from typing import Any

import psycopg
from psycopg.rows import dict_row

from deep_agent.src.policy.models import PolicySettings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_TABLES_ENSURED = False

CREATE_POLICY_SETTINGS_TABLE = """
CREATE TABLE IF NOT EXISTS user_policy_settings (
    user_id TEXT PRIMARY KEY,
    settings JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

CREATE_POLICY_SETTINGS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_user_policy_updated
ON user_policy_settings(updated_at);
"""


class PolicySettingsRepository:
    """Repository for managing user policy settings in PostgreSQL."""

    def __init__(self, database_uri: str) -> None:
        """Initialize repository with database connection URI.

        Args:
            database_uri: PostgreSQL connection string
        """
        self._uri = database_uri

    async def ensure_table(self) -> None:
        """Create policy settings table if it doesn't exist."""
        global _TABLES_ENSURED  # noqa: PLW0603
        if _TABLES_ENSURED:
            return

        async with await psycopg.AsyncConnection.connect(self._uri) as conn:
            await conn.execute(CREATE_POLICY_SETTINGS_TABLE)
            await conn.execute(CREATE_POLICY_SETTINGS_INDEX)
            await conn.commit()

        _TABLES_ENSURED = True
        logger.info("Policy settings table ensured")

    async def get_user_settings(self, user_id: str) -> PolicySettings | None:
        """Get policy settings for a specific user.

        Args:
            user_id: User identifier

        Returns:
            PolicySettings if user has custom settings, None otherwise
        """
        await self.ensure_table()

        async with await psycopg.AsyncConnection.connect(
            self._uri, row_factory=dict_row
        ) as conn:
            cur = await conn.execute(
                "SELECT user_id, settings as values, updated_at "
                "FROM user_policy_settings WHERE user_id = %s",
                (user_id,),
            )
            row = await cur.fetchone()
            if row:
                return PolicySettings(**row)
            return None

    async def save_user_settings(
        self, user_id: str, settings: dict[str, Any]
    ) -> PolicySettings:
        """Save or update user policy settings.

        Args:
            user_id: User identifier
            settings: Policy settings dictionary

        Returns:
            Updated PolicySettings object
        """
        await self.ensure_table()

        async with await psycopg.AsyncConnection.connect(self._uri) as conn:
            cur = await conn.execute(
                """
                INSERT INTO user_policy_settings (user_id, settings, updated_at)
                VALUES (%s, %s, now())
                ON CONFLICT (user_id)
                DO UPDATE SET settings = EXCLUDED.settings, updated_at = now()
                RETURNING user_id, settings as values, updated_at
                """,
                (user_id, json.dumps(settings)),
            )
            await conn.commit()
            row = await cur.fetchone()

        logger.info(f"Saved policy settings for user {user_id}")
        return PolicySettings(
            user_id=row[0], values=row[1], updated_at=row[2]
        )

    async def delete_user_settings(self, user_id: str) -> bool:
        """Delete user policy settings (revert to defaults).

        Args:
            user_id: User identifier

        Returns:
            True if settings were deleted, False if none existed
        """
        await self.ensure_table()

        async with await psycopg.AsyncConnection.connect(self._uri) as conn:
            cur = await conn.execute(
                "DELETE FROM user_policy_settings WHERE user_id = %s",
                (user_id,),
            )
            await conn.commit()
            deleted = cur.rowcount > 0

        if deleted:
            logger.info(f"Deleted policy settings for user {user_id}")
        return deleted

    async def list_all_settings(self) -> list[PolicySettings]:
        """List all user policy settings.

        Returns:
            List of PolicySettings for all users with custom settings
        """
        await self.ensure_table()

        async with await psycopg.AsyncConnection.connect(
            self._uri, row_factory=dict_row
        ) as conn:
            cur = await conn.execute(
                "SELECT user_id, settings as values, updated_at "
                "FROM user_policy_settings "
                "ORDER BY updated_at DESC"
            )
            rows = await cur.fetchall()
            return [PolicySettings(**row) for row in rows]
