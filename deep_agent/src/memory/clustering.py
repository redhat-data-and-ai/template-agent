"""Semantic clustering of user memories.

Groups similar memories by content similarity using token-based
cosine similarity (TF-IDF style, zero API calls). Assigns a
``cluster_id`` to each memory in the database.

Runs as a **background job** — never in the request path.
No embedding API calls — pure local computation.
"""

import math
import uuid
from collections import Counter, defaultdict

from deep_agent.src.memory.config import memory_settings
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokeniser with lowercasing."""
    return text.lower().split()


def _build_tfidf(documents: list[str]) -> list[dict[str, float]]:
    """Build TF-IDF vectors for a list of documents.

    Returns a list of {token: tfidf_weight} dicts, one per document.
    """
    n = len(documents)
    if n == 0:
        return []

    doc_tokens = [_tokenize(d) for d in documents]

    df: Counter[str] = Counter()
    for tokens in doc_tokens:
        df.update(set(tokens))

    vectors: list[dict[str, float]] = []
    for tokens in doc_tokens:
        tf: Counter[str] = Counter(tokens)
        total = len(tokens) or 1
        vec: dict[str, float] = {}
        for term, count in tf.items():
            idf = math.log((n + 1) / (df[term] + 1)) + 1
            vec[term] = (count / total) * idf
        vectors.append(vec)

    return vectors


def _cosine_sim(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    common = set(a.keys()) & set(b.keys())
    if not common:
        return 0.0
    dot = sum(a[k] * b[k] for k in common)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def cluster_memories(
    contents: list[str],
    threshold: float | None = None,
) -> list[list[int]]:
    """Cluster memory indices by TF-IDF cosine similarity.

    Uses single-linkage agglomerative clustering (union-find).

    Args:
        contents: List of memory content strings.
        threshold: Minimum similarity to merge (default from config).

    Returns:
        List of clusters (each a list of indices). Singletons are excluded.
    """
    threshold = threshold or memory_settings.MEMORY_CLUSTER_THRESHOLD
    vectors = _build_tfidf(contents)
    n = len(vectors)

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
            if _cosine_sim(vectors[i], vectors[j]) >= threshold:
                union(i, j)

    groups: defaultdict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    return [g for g in groups.values() if len(g) >= 2]


async def cluster_user_memories(
    database_uri: str,
    user_id: str,
) -> int:
    """Assign cluster_id to similar memories for a user.

    Returns the number of clusters created.
    """
    import psycopg
    from psycopg.rows import dict_row

    async with await psycopg.AsyncConnection.connect(
        database_uri, row_factory=dict_row
    ) as conn:
        cur = await conn.execute(
            "SELECT id, content FROM user_memories "
            "WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,),
        )
        memories = [dict(row) for row in await cur.fetchall()]

        if len(memories) < 2:
            return 0

        contents = [m["content"] for m in memories]
        clusters = cluster_memories(contents)

        if not clusters:
            return 0

        for group in clusters:
            cid = str(uuid.uuid4())
            for idx in group:
                await conn.execute(
                    "UPDATE user_memories SET cluster_id = %s WHERE id = %s",
                    (cid, str(memories[idx]["id"])),
                )

        await conn.commit()
        logger.info(
            "Clustered user %s: %d cluster(s) from %d memories",
            user_id[:8],
            len(clusters),
            len(memories),
        )
        return len(clusters)


async def cluster_all_users(database_uri: str) -> int:
    """Run clustering across all users. Returns total clusters created."""
    if not memory_settings.is_enabled("clustering"):
        logger.debug("Memory clustering disabled — skipping")
        return 0

    import psycopg

    async with await psycopg.AsyncConnection.connect(database_uri) as conn:
        cur = await conn.execute("SELECT DISTINCT user_id FROM user_memories")
        user_ids = [row[0] for row in await cur.fetchall()]

    total = 0
    for uid in user_ids:
        total += await cluster_user_memories(database_uri, uid)

    logger.info(
        "Clustering complete: %d clusters across %d users", total, len(user_ids)
    )
    return total
