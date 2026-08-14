"""Memory consolidation — merge duplicate/similar memories.

When a user accumulates many memories, this module:
1. Detects near-duplicates (exact or fuzzy match)
2. Merges them into a single consolidated memory
3. Deletes the originals

Runs as a **background job** — never in the request path.
"""

import re
from collections import defaultdict

from deep_agent.src.memory.config import memory_settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


def _token_set(text: str) -> set[str]:
    """Return a set of normalised tokens."""
    return set(_normalise(text).split())


def token_similarity(a: str, b: str) -> float:
    """Jaccard similarity between token sets of two strings."""
    sa, sb = _token_set(a), _token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def find_duplicates(
    memories: list[dict[str, str]],
    threshold: float | None = None,
) -> list[list[int]]:
    """Group memory indices that are near-duplicates.

    Args:
        memories: List of dicts with at least a ``content`` key.
        threshold: Similarity threshold (default from config).

    Returns:
        List of groups, where each group is a list of indices
        into *memories* that should be consolidated.
    """
    threshold = threshold or memory_settings.MEMORY_CLUSTER_THRESHOLD

    n = len(memories)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            sim = token_similarity(memories[i]["content"], memories[j]["content"])
            if sim >= threshold:
                union(i, j)

    groups: defaultdict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    return [g for g in groups.values() if len(g) >= 2]


def pick_representative(
    memories: list[dict[str, str]],
    indices: list[int],
) -> int:
    """Choose the best memory from a duplicate group.

    Picks the longest content (most informative), breaking ties
    by highest score.
    """
    best = indices[0]
    for idx in indices[1:]:
        cur_len = len(memories[idx]["content"])
        best_len = len(memories[best]["content"])
        if cur_len > best_len:
            best = idx
        elif cur_len == best_len:
            cur_score = float(memories[idx].get("score", 0))
            best_score = float(memories[best].get("score", 0))
            if cur_score > best_score:
                best = idx
    return best


async def consolidate_user_memories(
    database_uri: str,
    user_id: str,
) -> int:
    """Consolidate duplicate memories for a single user.

    Returns the number of memories deleted.
    """
    import psycopg
    from psycopg.rows import dict_row

    async with await psycopg.AsyncConnection.connect(
        database_uri, row_factory=dict_row
    ) as conn:
        cur = await conn.execute(
            "SELECT id, content, score FROM user_memories "
            "WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,),
        )
        memories = [dict(row) for row in await cur.fetchall()]

        if len(memories) < 2:
            return 0

        groups = find_duplicates(memories)
        if not groups:
            return 0

        deleted = 0
        for group in groups:
            keep = pick_representative(memories, group)
            to_delete = [i for i in group if i != keep]
            for idx in to_delete:
                await conn.execute(
                    "DELETE FROM user_memories WHERE id = %s",
                    (str(memories[idx]["id"]),),
                )
                deleted += 1

        if deleted:
            await conn.commit()
            logger.info(
                "Consolidated user %s: deleted %d duplicate(s) from %d group(s)",
                user_id[:8],
                deleted,
                len(groups),
            )

        return deleted


async def consolidate_all_users(database_uri: str) -> int:
    """Run consolidation across all users. Returns total deletions."""
    if not memory_settings.MEMORY_CONSOLIDATION_ENABLED:
        logger.debug("Memory consolidation disabled — skipping")
        return 0

    import psycopg

    async with await psycopg.AsyncConnection.connect(database_uri) as conn:
        cur = await conn.execute("SELECT DISTINCT user_id FROM user_memories")
        user_ids = [row[0] for row in await cur.fetchall()]

    total = 0
    for uid in user_ids:
        total += await consolidate_user_memories(database_uri, uid)

    logger.info(
        "Consolidation complete: %d total deletions across %d users",
        total,
        len(user_ids),
    )
    return total
