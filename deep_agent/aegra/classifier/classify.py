"""Core classification logic - one MaaS LLM call does everything.

classify_query() is the main entry point. It:
  1. Sends the query to DiffusionGemma with the unified system prompt
  2. Parses the structured JSON response (with repair for quirks)
  3. Attaches table metadata from product_tables.json
  4. Falls back to keyword-based classification if LLM fails
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import httpx

from deep_agent.aegra.classifier.config import (
    MAAS_API_KEY,
    MAAS_BASE_URL,
    MAAS_MAX_TOKENS,
    MAAS_MODEL_ID,
    MAAS_TIMEOUT,
    MAAS_VERIFY_SSL,
    load_classify_config,
    load_product_tables,
    load_system_prompt,
)
from deep_agent.aegra.classifier.json_repair import safe_parse_json

logger = logging.getLogger(__name__)

VALID_INTENTS = frozenset(["data-query", "unstructured", "catalog", "chart", "direct-response"])
VALID_ENTITY_TYPES = frozenset([
    "person", "rh_product", "dv_product", "metric", "time_period",
    "geography", "organization", "account", "document", "sku",
])


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IntentResult:
    intent_class: str
    confidence: float = 0.0


@dataclass
class TableInfo:
    name: str
    schema: str = "MARTS"
    description: str = ""


@dataclass
class Entity:
    text: str
    type: str


@dataclass
class SubQuery:
    query: str
    intents: list[IntentResult] = field(default_factory=list)
    product: str | None = None
    confidence: float = 0.0
    tables_list: list[TableInfo] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class RouteInfo:
    success: bool = True
    decision: str = "clear_winner"
    selected_product: str | None = None
    reasoning: str = ""


@dataclass
class ClassifyResult:
    entities: list[Entity] = field(default_factory=list)
    sub_queries: list[SubQuery] = field(default_factory=list)
    is_compound: bool = False
    intents: list[IntentResult] = field(default_factory=list)
    routing: RouteInfo | None = None
    tables_list: list[TableInfo] = field(default_factory=list)
    reasoning: str = ""

    def to_classification_block(self) -> str:
        """Build a <classification> text block for injecting into the human message."""
        if not self.sub_queries:
            return ""
        lines = ["<classification>"]
        if self.is_compound:
            lines.append(f"compound_query: true ({len(self.sub_queries)} parts)")
        for i, sq in enumerate(self.sub_queries):
            intent_str = ", ".join(it.intent_class for it in sq.intents) or "unknown"
            lines.append(f'[Part {i + 1}] "{sq.query}"')
            lines.append(f"  intent: {intent_str}")
            if sq.product:
                lines.append(f"  product: {sq.product}")
            if sq.tables_list:
                table_str = ", ".join(f"{t.name} ({t.schema})" for t in sq.tables_list)
                lines.append(f"  tables: {table_str}")
            if sq.entities:
                lines.append(f"  entities: {', '.join(sq.entities)}")
        lines.append("</classification>")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table lookup
# ---------------------------------------------------------------------------

def _lookup_tables(product_name: str) -> list[TableInfo]:
    """Look up tables for a product by normalized name."""
    tables = load_product_tables()
    normalized = product_name.lower().replace(" ", "").replace("_", "").replace("-", "")
    return [
        TableInfo(name=t["name"], schema=t.get("schema", "MARTS"), description=t.get("description", ""))
        for t in tables.get(normalized, [])
    ]


# ---------------------------------------------------------------------------
# Fallback classification (keyword-based)
# ---------------------------------------------------------------------------

def _fallback_classify(query: str) -> ClassifyResult:
    """Keyword-based fallback when the LLM call fails."""
    config = load_classify_config()
    lower = query.lower()
    intents: list[IntentResult] = []

    fallback = config.get("fallback", {})
    direct_kw = fallback.get("direct_response_signals", [])
    if any(kw in lower for kw in direct_kw):
        intents.append(IntentResult(intent_class="direct-response", confidence=0.8))

    unstruct = config.get("unstructured_documents", {})
    for cat_cfg in unstruct.get("categories", {}).values():
        if any(kw in lower for kw in cat_cfg.get("keywords", [])):
            intents.append(IntentResult(intent_class="unstructured", confidence=0.7))
            break

    if not intents:
        default = fallback.get("default_intent", {"intent": "data-query", "confidence": 0.5})
        intents.append(IntentResult(intent_class=default["intent"], confidence=default["confidence"]))

    intents.sort(key=lambda i: i.confidence, reverse=True)

    return ClassifyResult(
        sub_queries=[SubQuery(query=query, intents=intents, reasoning="Fallback: LLM unavailable")],
        is_compound=False,
        intents=intents,
        reasoning="Fallback classification (LLM call failed)",
    )


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------

def _parse_unified_response(raw: str, original_query: str) -> tuple[list[Entity], list[SubQuery]]:
    """Parse the unified LLM response into entities and sub-queries with tables."""
    parsed = safe_parse_json(raw)
    if parsed is None or not isinstance(parsed, dict):
        return [], []

    raw_entities = parsed.get("entities", [])
    entities = [
        Entity(text=e["text"].strip(), type=e["type"].strip().lower())
        for e in raw_entities
        if isinstance(e, dict) and e.get("text", "").strip() and e.get("type", "").strip().lower() in VALID_ENTITY_TYPES
    ]

    raw_subs = parsed.get("sub_queries", [])
    if not raw_subs:
        return entities, []

    sub_queries: list[SubQuery] = []
    for raw_sq in raw_subs:
        query_text = raw_sq.get("query", original_query)
        raw_intents = raw_sq.get("intents", ["data-query"])
        valid = [i for i in raw_intents if i.lower().strip() in VALID_INTENTS]
        if not valid:
            valid = ["data-query"]

        confidence = min(1.0, max(0.0, float(raw_sq.get("confidence", 0)) / 100))
        intent_results = [IntentResult(intent_class=ic, confidence=confidence) for ic in valid]

        product = raw_sq.get("product")
        tables: list[TableInfo] = []
        if product and any(i.intent_class == "data-query" for i in intent_results):
            tables = _lookup_tables(product)

        sub_queries.append(SubQuery(
            query=query_text,
            intents=intent_results,
            product=product,
            confidence=confidence,
            tables_list=tables,
            entities=raw_sq.get("entities", []),
            reasoning=raw_sq.get("reasoning", ""),
        ))

    return entities, sub_queries


# ---------------------------------------------------------------------------
# MaaS client (httpx, with retry)
# ---------------------------------------------------------------------------

async def _call_maas(query: str) -> str | None:
    """Call MaaS Enterprise and return the assistant content, or None on failure."""
    system_prompt = load_system_prompt()
    if not system_prompt:
        logger.warning("No system prompt loaded, skipping LLM classification")
        return None

    body = {
        "model": MAAS_MODEL_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        "max_tokens": MAAS_MAX_TOKENS,
    }

    async with httpx.AsyncClient(verify=MAAS_VERIFY_SSL, timeout=MAAS_TIMEOUT) as client:
        response = await client.post(
            f"{MAAS_BASE_URL}/chat/completions",
            json=body,
            headers={
                "Authorization": f"Bearer {MAAS_API_KEY}",
                "Content-Type": "application/json",
            },
        )

    if response.status_code not in (200, 202):
        logger.warning("MaaS HTTP %d: %s", response.status_code, response.text[:300])
        return None

    payload = response.json()
    choices = payload.get("choices", [])
    if not choices:
        return None

    content = choices[0].get("message", {}).get("content", "")
    if not content.strip():
        content = choices[0].get("message", {}).get("reasoning", "")

    return content if content.strip() else None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def classify_query(query: str) -> ClassifyResult:
    """Classify a user query via DiffusionGemma on MaaS.

    1. Sends query to DiffusionGemma with the unified system prompt
    2. Parses structured JSON response (NER + decompose + classify + identify)
    3. Attaches table metadata from product_tables.json
    4. Falls back to keyword classification if LLM fails

    Returns a ClassifyResult with sub_queries, intents, routing, and tables.
    Use result.to_classification_block() to get the <classification> text.
    """
    if not MAAS_API_KEY:
        logger.warning("MAAS_API_KEY not set, using fallback classification")
        return _fallback_classify(query)

    start = time.perf_counter()

    try:
        content = await _call_maas(query)
    except Exception as exc:
        latency = (time.perf_counter() - start) * 1000
        logger.warning("MaaS call failed (%.0fms): %s", latency, exc)
        return _fallback_classify(query)

    if not content:
        return _fallback_classify(query)

    entities, sub_queries = _parse_unified_response(content, query)
    if not sub_queries:
        logger.warning("Parse returned no sub_queries, falling back")
        return _fallback_classify(query)

    is_compound = len(sub_queries) > 1
    primary = sub_queries[0]

    all_intents: list[IntentResult] = []
    seen: set[str] = set()
    for sq in sub_queries:
        for intent in sq.intents:
            if intent.intent_class not in seen:
                all_intents.append(intent)
                seen.add(intent.intent_class)
    all_intents.sort(key=lambda i: i.confidence, reverse=True)

    routing: RouteInfo | None = None
    if primary.product:
        routing = RouteInfo(
            decision="clear_winner" if primary.confidence >= 0.8 else "ambiguous",
            selected_product=primary.product,
            reasoning=primary.reasoning,
        )

    latency = (time.perf_counter() - start) * 1000
    logger.info(
        "classify_query: %d entities, %d sub_queries, intents=%s, products=%s, %.0fms",
        len(entities),
        len(sub_queries),
        [i.intent_class for i in all_intents],
        [sq.product for sq in sub_queries if sq.product],
        latency,
    )

    return ClassifyResult(
        entities=entities,
        sub_queries=sub_queries,
        is_compound=is_compound,
        intents=all_intents,
        routing=routing,
        tables_list=primary.tables_list,
        reasoning=primary.reasoning,
    )
