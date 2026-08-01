"""Exponential decay scoring for user memories.

Memories lose relevance over time unless accessed. The score formula:

    score = base_score * e^(-λ * age_days) + access_boost

Where:
    - base_score: initial score (1.0 for new memories)
    - λ (lambda): decay rate from MEMORY_DECAY_LAMBDA
    - age_days: days since last update
    - access_boost: small bump each time the memory is referenced

This runs as a **background job** — never in the request path.
"""

import math
from datetime import datetime, timezone

from deep_agent.src.memory.config import memory_settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

ACCESS_BOOST = 0.1
MIN_SCORE = 0.01


def compute_decay_score(
    base_score: float,
    updated_at: datetime,
    now: datetime | None = None,
) -> float:
    """Compute the decayed score for a memory.

    Args:
        base_score: The memory's current stored score.
        updated_at: When the memory was last updated or accessed.
        now: Current time (defaults to utcnow for testability).

    Returns:
        Decayed score, floored at MIN_SCORE.
    """
    now = now or datetime.now(timezone.utc)
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)

    age_days = max((now - updated_at).total_seconds() / 86400, 0)
    lam = memory_settings.MEMORY_DECAY_LAMBDA
    score = base_score * math.exp(-lam * age_days)
    return max(score, MIN_SCORE)


def apply_access_boost(current_score: float) -> float:
    """Bump a memory's score when it's referenced in a conversation.

    Capped at 1.0 to prevent runaway scores.
    """
    return min(current_score + ACCESS_BOOST, 1.0)


async def decay_all_memories(database_uri: str) -> int:
    """Recalculate scores for all memories in the database.

    Returns the number of memories updated.
    """
    import psycopg
    from psycopg.rows import dict_row

    if not memory_settings.is_enabled("decay"):
        logger.debug("Memory decay disabled — skipping")
        return 0

    now = datetime.now(timezone.utc)
    updated = 0

    async with await psycopg.AsyncConnection.connect(
        database_uri, row_factory=dict_row
    ) as conn:
        cur = await conn.execute("SELECT id, score, updated_at FROM user_memories")
        rows = await cur.fetchall()

        for row in rows:
            old_score = float(row.get("score", 1.0) or 1.0)
            new_score = compute_decay_score(old_score, row["updated_at"], now)

            if abs(new_score - old_score) > 0.001:
                await conn.execute(
                    "UPDATE user_memories SET score = %s WHERE id = %s",
                    (new_score, str(row["id"])),
                )
                updated += 1

        if updated:
            await conn.commit()

    logger.info("Decay scoring: updated %d / %d memories", updated, len(rows))
    return updated
