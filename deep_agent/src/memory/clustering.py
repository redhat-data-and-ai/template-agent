"""Semantic clustering of user memories stored in LangGraph Store.

Groups similar memories by content similarity using token-based
TF-IDF cosine similarity (zero API calls — pure local computation).

Can be used to:
- Identify related memory facts for context grouping
- Detect near-duplicates before injection into prompts
- Present grouped memories in the personalization API
"""

import math
import re
from collections import Counter, defaultdict

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

DEFAULT_CLUSTER_THRESHOLD = 0.4

_UNIT_PATTERN = re.compile(r"(\d+)\s*(kg|cm|lbs|ft|in|bmi|mph|km|m|g|lb)\b", re.I)
_NUMBER_PATTERN = re.compile(r"(\d+\.?\d*)")


def _normalize_text(text: str) -> str:
    """Normalize text for better similarity matching.

    - Lowercase
    - Separate numbers from units (70kg → 70 kg)
    - Normalize common word variants
    """
    text = text.lower()
    text = _UNIT_PATTERN.sub(r"\1 \2", text)
    text = re.sub(r"[^\w\s.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_STEM_MAP = {
    "weighs": "weight",
    "weigh": "weight",
    "weighted": "weight",
    "heights": "height",
    "tall": "height",
    "prefers": "prefer",
    "preferred": "prefer",
    "preference": "prefer",
    "likes": "like",
    "liked": "like",
    "uses": "use",
    "using": "use",
    "used": "use",
}


def _tokenize(text: str) -> list[str]:
    """Tokenize with normalization and lightweight stemming."""
    normalized = _normalize_text(text)
    tokens = normalized.split()
    return [_STEM_MAP.get(t, t) for t in tokens]


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
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
) -> list[list[int]]:
    """Cluster memory indices by TF-IDF cosine similarity.

    Uses single-linkage agglomerative clustering (union-find).

    Args:
        contents: List of memory content strings.
        threshold: Minimum similarity to merge (default 0.4).

    Returns:
        List of clusters (each a list of indices). Singletons are excluded.
    """
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


async def cluster_store_memories(
    database_uri: str,
    namespace: tuple[str, ...],
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
) -> list[list[str]]:
    """Cluster memories in a LangGraph Store namespace.

    Reads all memory items from the store, clusters them by content
    similarity, and returns groups of related memory facts.

    Args:
        database_uri: Postgres connection URI.
        namespace: LangGraph Store namespace tuple for the user.
        threshold: Minimum cosine similarity to group together.

    Returns:
        List of clusters, where each cluster is a list of memory
        content strings that are semantically related.
    """
    import re

    from langgraph.store.postgres.aio import AsyncPostgresStore

    async with AsyncPostgresStore.from_conn_string(database_uri) as store:
        await store.setup()
        items = await store.asearch(namespace, limit=200)

    all_facts: list[str] = []
    for item in items:
        value: dict = item.value
        content_lines: list[str] = value.get("content", [])
        raw_text = "\n".join(content_lines)
        for fact in re.split(r"\n+", raw_text):
            cleaned = re.sub(r"^[-*•]\s*", "", fact).strip()
            if cleaned:
                all_facts.append(cleaned)

    if len(all_facts) < 2:
        return []

    clusters = cluster_memories(all_facts, threshold=threshold)

    result: list[list[str]] = []
    for group in clusters:
        result.append([all_facts[i] for i in group])

    if result:
        logger.info(
            "Clustered %d facts into %d groups (namespace=%s)",
            len(all_facts),
            len(result),
            namespace,
        )

    return result
