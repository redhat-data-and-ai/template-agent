"""Async Postgres repository for user memories and rules.

Uses ``psycopg`` (async) against the same database that stores
LangGraph checkpoints. Tables are created lazily on first use via
:meth:`PersonalizationRepository.ensure_tables`.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import psycopg
from psycopg.rows import dict_row

from deep_agent.src.personalization.models import Memory, Rule
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_TABLES_ENSURED = False

CREATE_MEMORIES_TABLE = """
CREATE TABLE IF NOT EXISTS user_memories (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     TEXT NOT NULL,
    content     TEXT NOT NULL,
    score       FLOAT NOT NULL DEFAULT 1.0,
    cluster_id  UUID,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_user_memories_user_id
    ON user_memories (user_id);
"""

MIGRATE_MEMORIES_TABLE = """
ALTER TABLE user_memories ADD COLUMN IF NOT EXISTS score FLOAT NOT NULL DEFAULT 1.0;
ALTER TABLE user_memories ADD COLUMN IF NOT EXISTS cluster_id UUID;
"""

CREATE_RULES_TABLE = """
CREATE TABLE IF NOT EXISTS user_rules (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     TEXT NOT NULL,
    content     TEXT NOT NULL,
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_user_rules_user_id
    ON user_rules (user_id);
"""


class PersonalizationRepository:
    """Thin async wrapper around the personalization tables."""

    def __init__(self, database_uri: str) -> None:
        """Initialise with a Postgres connection URI."""
        self._uri = database_uri

    async def ensure_tables(self) -> None:
        """Create personalization tables if they do not already exist."""
        global _TABLES_ENSURED  # noqa: PLW0603
        if _TABLES_ENSURED:
            return
        async with await psycopg.AsyncConnection.connect(self._uri) as conn:
            await conn.execute(CREATE_MEMORIES_TABLE)
            await conn.execute(CREATE_RULES_TABLE)
            await conn.execute(MIGRATE_MEMORIES_TABLE)
            await conn.commit()
        _TABLES_ENSURED = True
        logger.info("Personalization tables ensured")

    # ── Memories ──────────────────────────────────────────────

    async def list_memories(self, user_id: str) -> list[Memory]:
        """Return all memories for *user_id*, newest first."""
        await self.ensure_tables()
        async with await psycopg.AsyncConnection.connect(
            self._uri, row_factory=dict_row
        ) as conn:
            cur = await conn.execute(
                "SELECT * FROM user_memories WHERE user_id = %s ORDER BY created_at DESC",
                (user_id,),
            )
            return [Memory(**row) for row in await cur.fetchall()]

    async def list_top_memories(self, user_id: str, limit: int = 20) -> list[Memory]:
        """Return top-N memories for *user_id*, ranked by score descending."""
        await self.ensure_tables()
        async with await psycopg.AsyncConnection.connect(
            self._uri, row_factory=dict_row
        ) as conn:
            cur = await conn.execute(
                "SELECT * FROM user_memories WHERE user_id = %s "
                "ORDER BY score DESC, updated_at DESC LIMIT %s",
                (user_id, limit),
            )
            return [Memory(**row) for row in await cur.fetchall()]

    async def create_memory(self, user_id: str, content: str) -> Memory:
        """Insert a new memory and return the created model."""
        await self.ensure_tables()
        mem = Memory(user_id=user_id, content=content)
        async with await psycopg.AsyncConnection.connect(self._uri) as conn:
            await conn.execute(
                "INSERT INTO user_memories (id, user_id, content, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (str(mem.id), mem.user_id, mem.content, mem.created_at, mem.updated_at),
            )
            await conn.commit()
        return mem

    async def delete_memory(self, user_id: str, memory_id: uuid.UUID) -> bool:
        """Delete a memory by id; return True if a row was removed."""
        await self.ensure_tables()
        async with await psycopg.AsyncConnection.connect(self._uri) as conn:
            cur = await conn.execute(
                "DELETE FROM user_memories WHERE id = %s AND user_id = %s",
                (str(memory_id), user_id),
            )
            await conn.commit()
            return bool(cur.rowcount > 0)

    # ── Rules ─────────────────────────────────────────────────

    async def list_rules(self, user_id: str, *, active_only: bool = True) -> list[Rule]:
        """Return rules for *user_id*, optionally filtering to active only."""
        await self.ensure_tables()
        clause = " AND is_active = TRUE" if active_only else ""
        async with await psycopg.AsyncConnection.connect(
            self._uri, row_factory=dict_row
        ) as conn:
            cur = await conn.execute(
                f"SELECT * FROM user_rules WHERE user_id = %s{clause} ORDER BY created_at DESC",
                (user_id,),
            )
            return [Rule(**row) for row in await cur.fetchall()]

    async def upsert_rule(
        self,
        user_id: str,
        content: str,
        rule_id: uuid.UUID | None = None,
        is_active: bool = True,
    ) -> Rule:
        """Create or update a rule and return the model."""
        await self.ensure_tables()
        now = datetime.utcnow()
        rid = rule_id or uuid.uuid4()
        rule = Rule(
            id=rid,
            user_id=user_id,
            content=content,
            is_active=is_active,
            created_at=now,
            updated_at=now,
        )
        async with await psycopg.AsyncConnection.connect(self._uri) as conn:
            await conn.execute(
                """
                INSERT INTO user_rules (id, user_id, content, is_active, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id)
                DO UPDATE SET content = EXCLUDED.content,
                              is_active = EXCLUDED.is_active,
                              updated_at = EXCLUDED.updated_at
                """,
                (
                    str(rule.id),
                    rule.user_id,
                    rule.content,
                    rule.is_active,
                    rule.created_at,
                    rule.updated_at,
                ),
            )
            await conn.commit()
        return rule

    async def delete_rule(self, user_id: str, rule_id: uuid.UUID) -> bool:
        """Delete a rule by id; return True if a row was removed."""
        await self.ensure_tables()
        async with await psycopg.AsyncConnection.connect(self._uri) as conn:
            cur = await conn.execute(
                "DELETE FROM user_rules WHERE id = %s AND user_id = %s",
                (str(rule_id), user_id),
            )
            await conn.commit()
            return bool(cur.rowcount > 0)
