"""Relationship inference between user memories.

Detects memories that share significant keywords or entities,
and stores links in the ``memory_relationships`` table so the
agent can surface related context.

Runs as a **background job** — never in the request path.
No LLM calls — pure keyword overlap.
"""

import re
from collections import Counter

from deep_agent.src.memory.config import memory_settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could i me my we our you your "
    "he she it they them their this that these those of in to for with "
    "on at by from as into through during before after above below "
    "and or but not no nor so yet also very too".split()
)

MIN_SHARED_KEYWORDS = 2


def extract_keywords(text: str, top_n: int = 10) -> list[str]:
    """Extract significant keywords from text.

    Strips stopwords, short tokens, and returns the most frequent
    meaningful words.
    """
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    meaningful = [t for t in tokens if t not in STOPWORDS]
    counts = Counter(meaningful)
    return [word for word, _ in counts.most_common(top_n)]


def find_related_pairs(
    memories: list[dict[str, str]],
    min_shared: int = MIN_SHARED_KEYWORDS,
) -> list[tuple[int, int, list[str]]]:
    """Find pairs of memories that share significant keywords.

    Args:
        memories: List of dicts with ``content`` key.
        min_shared: Minimum shared keywords to consider related.

    Returns:
        List of (idx_a, idx_b, shared_keywords) tuples.
    """
    keyword_sets = [set(extract_keywords(m["content"])) for m in memories]
    pairs: list[tuple[int, int, list[str]]] = []

    for i in range(len(memories)):
        for j in range(i + 1, len(memories)):
            shared = keyword_sets[i] & keyword_sets[j]
            if len(shared) >= min_shared:
                pairs.append((i, j, sorted(shared)))

    return pairs


async def infer_user_relationships(
    database_uri: str,
    user_id: str,
) -> int:
    """Detect and store relationships between a user's memories.

    Returns the number of new relationships created.
    """
    import psycopg
    from psycopg.rows import dict_row

    async with await psycopg.AsyncConnection.connect(
        database_uri, row_factory=dict_row
    ) as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_relationships (
                memory_a    UUID NOT NULL,
                memory_b    UUID NOT NULL,
                keywords    TEXT NOT NULL,
                user_id     TEXT NOT NULL,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (memory_a, memory_b)
            )
            """
        )

        cur = await conn.execute(
            "SELECT id, content FROM user_memories "
            "WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,),
        )
        memories = [dict(row) for row in await cur.fetchall()]

        if len(memories) < 2:
            return 0

        pairs = find_related_pairs(memories)
        if not pairs:
            return 0

        created = 0
        for i, j, keywords in pairs:
            id_a = str(memories[i]["id"])
            id_b = str(memories[j]["id"])
            a, b = min(id_a, id_b), max(id_a, id_b)
            try:
                await conn.execute(
                    """
                    INSERT INTO memory_relationships (memory_a, memory_b, keywords, user_id)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (memory_a, memory_b) DO NOTHING
                    """,
                    (a, b, ",".join(keywords), user_id),
                )
                created += 1
            except Exception:
                logger.debug("Relationship insert failed", exc_info=True)

        if created:
            await conn.commit()
            logger.info(
                "Relationships for user %s: %d pair(s) from %d memories",
                user_id[:8],
                created,
                len(memories),
            )
        return created


async def infer_all_relationships(database_uri: str) -> int:
    """Run relationship inference across all users."""
    if not memory_settings.is_enabled("relationships"):
        logger.debug("Relationship inference disabled — skipping")
        return 0

    import psycopg

    async with await psycopg.AsyncConnection.connect(database_uri) as conn:
        cur = await conn.execute("SELECT DISTINCT user_id FROM user_memories")
        user_ids = [row[0] for row in await cur.fetchall()]

    total = 0
    for uid in user_ids:
        total += await infer_user_relationships(database_uri, uid)

    logger.info(
        "Relationships complete: %d pairs across %d users", total, len(user_ids)
    )
    return total
