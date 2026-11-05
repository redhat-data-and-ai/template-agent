"""Classify a user query. classify_query() is the only entry point.

1. Stream one DiffusionGemma completion from MaaS (key from the environment)
2. Parse the structured JSON (with repair for model quirks)
3. Attach tables from product_tables.yaml
4. Fall back to keyword rules in logic.yaml when the model is unavailable

The agent answer is streamed separately by ClassifierRunnable. This function
returns the classification, it does not buffer the answer.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from deep_agent.aegra.classifier.config import (
    entity_types,
    get_maas_settings,
    intent_classes,
    load_logic,
    load_product_tables,
    load_system_prompt,
)
from deep_agent.aegra.classifier.json_repair import safe_parse_json
from deep_agent.aegra.classifier.maas_client import stream_completion

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class IntentResult:
    """One intent label and its confidence."""

    intent_class: str
    confidence: float = 0.0


@dataclass
class TableInfo:
    """A table attached to a classified data product."""

    name: str
    schema: str = "MARTS"
    description: str = ""


@dataclass
class Entity:
    """A named entity extracted from the user query."""

    text: str
    type: str


@dataclass
class SubQuery:
    """One part of a user query after decomposition."""

    query: str
    intents: list[IntentResult] = field(default_factory=list)
    product: str | None = None
    confidence: float = 0.0
    tables_list: list[TableInfo] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class RouteInfo:
    """Product selected for the primary sub-query."""

    success: bool = True
    decision: str = "clear_winner"
    selected_product: str | None = None
    reasoning: str = ""


@dataclass
class ClassifyResult:
    """Full classification for one user query."""

    entities: list[Entity] = field(default_factory=list)
    sub_queries: list[SubQuery] = field(default_factory=list)
    is_compound: bool = False
    intents: list[IntentResult] = field(default_factory=list)
    routing: RouteInfo | None = None
    tables_list: list[TableInfo] = field(default_factory=list)
    reasoning: str = ""
    source: str = "model"
    fallback_reason: str = ""
    latency_ms: float = 0.0

    def to_thought(self) -> str:
        """Return the classification record shown in the thought process."""
        intent_text = (
            ", ".join(
                f"{item.intent_class} ({item.confidence:.2f})" for item in self.intents
            )
            or "none"
        )
        entity_text = (
            ", ".join(f"{entity.text} ({entity.type})" for entity in self.entities)
            or "none"
        )
        table_text = (
            ", ".join(f"{table.name} ({table.schema})" for table in self.tables_list)
            or "none"
        )
        decision = self.routing.decision if self.routing else "no product selected"
        product = "none"
        if self.routing and self.routing.selected_product:
            product = self.routing.selected_product
        lines = [
            "**Intent classification**",
            f"Source: {self.source}",
            f"Why: {self.fallback_reason or self.reasoning or 'model result'}",
            f"Latency: {self.latency_ms:.0f} ms",
            f"Decision: {decision}",
            f"Product: {product}",
            f"Intents: {intent_text}",
            f"Entities: {entity_text}",
            f"Tables: {table_text}",
            f"Sub-queries: {len(self.sub_queries)}",
            f"Compound: {'yes' if self.is_compound else 'no'}",
        ]
        for index, part in enumerate(self.sub_queries, start=1):
            part_intents = (
                ", ".join(item.intent_class for item in part.intents) or "unknown"
            )
            part_tables = ", ".join(table.name for table in part.tables_list) or "none"
            lines.append(
                f"{index}. {part.query} | intent {part_intents} | "
                f"product {part.product or 'none'} | "
                f"confidence {part.confidence:.2f} | tables {part_tables}"
            )
            if part.reasoning:
                lines.append(f"   note: {part.reasoning}")
        return "\n".join(lines)

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
        TableInfo(
            name=t["name"],
            schema=t.get("schema", "MARTS"),
            description=t.get("description", ""),
        )
        for t in tables.get(normalized, [])
    ]


# ---------------------------------------------------------------------------
# Fallback classification (keyword-based)
# ---------------------------------------------------------------------------


def _log_classification(result: ClassifyResult) -> None:
    """Write the same record the UI shows as a thought."""
    logger.info("%s", result.to_thought())


def _fallback_classify(
    query: str, reason: str, latency_ms: float = 0.0
) -> ClassifyResult:
    """Keyword-based fallback when the LLM call fails."""
    config = load_logic()
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
        default = fallback.get(
            "default_intent", {"intent": "data-query", "confidence": 0.5}
        )
        intents.append(
            IntentResult(
                intent_class=default["intent"], confidence=default["confidence"]
            )
        )

    intents.sort(key=lambda i: i.confidence, reverse=True)

    result = ClassifyResult(
        sub_queries=[
            SubQuery(query=query, intents=intents, reasoning=f"Fallback: {reason}")
        ],
        is_compound=False,
        intents=intents,
        reasoning=f"Fallback: {reason}",
        source="fallback",
        fallback_reason=reason,
        latency_ms=latency_ms,
    )
    _log_classification(result)
    return result


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------


def _parse_unified_response(
    raw: str, original_query: str
) -> tuple[list[Entity], list[SubQuery]]:
    """Parse the unified LLM response into entities and sub-queries with tables."""
    parsed = safe_parse_json(raw)
    if parsed is None or not isinstance(parsed, dict):
        return [], []

    raw_entities = parsed.get("entities", [])
    entities = [
        Entity(text=e["text"].strip(), type=e["type"].strip().lower())
        for e in raw_entities
        if isinstance(e, dict)
        and e.get("text", "").strip()
        and e.get("type", "").strip().lower() in entity_types()
    ]

    raw_subs = parsed.get("sub_queries", [])
    if not raw_subs:
        return entities, []

    sub_queries: list[SubQuery] = []
    for raw_sq in raw_subs:
        query_text = raw_sq.get("query", original_query)
        raw_intents = raw_sq.get("intents", ["data-query"])
        valid = [i for i in raw_intents if i.lower().strip() in intent_classes()]
        if not valid:
            valid = ["data-query"]

        confidence = min(1.0, max(0.0, float(raw_sq.get("confidence", 0)) / 100))
        intent_results = [
            IntentResult(intent_class=ic, confidence=confidence) for ic in valid
        ]

        product = raw_sq.get("product")
        tables: list[TableInfo] = []
        if product and any(i.intent_class == "data-query" for i in intent_results):
            tables = _lookup_tables(product)

        sub_queries.append(
            SubQuery(
                query=query_text,
                intents=intent_results,
                product=product,
                confidence=confidence,
                tables_list=tables,
                entities=raw_sq.get("entities", []),
                reasoning=raw_sq.get("reasoning", ""),
            )
        )

    return entities, sub_queries


async def _call_maas(query: str) -> str | None:
    """Stream the classifier completion. Returns the joined model text."""
    system_prompt = load_system_prompt()
    if not system_prompt:
        logger.warning("No system prompt loaded, skipping LLM classification")
        return None
    return await stream_completion(
        query=query,
        system_prompt=system_prompt,
        settings=get_maas_settings(),
    )


async def classify_query(query: str) -> ClassifyResult:
    """Classify a user query via DiffusionGemma on MaaS.

    The API key is read from MAAS_API_KEY at call time. Rules and the product
    catalog come from config/agent/data/classification/*.yaml.

    Returns a ClassifyResult. The agent answer is streamed by the graph after
    this returns; this function does not hold those tokens.
    """
    text = (query or "").strip()
    if not text:
        return _fallback_classify(text, "query was empty")

    if not get_maas_settings().api_key:
        return _fallback_classify(text, "MAAS_API_KEY is not set")

    start = time.perf_counter()

    try:
        content = await _call_maas(text)
    except Exception as exc:
        latency = (time.perf_counter() - start) * 1000
        return _fallback_classify(text, f"MaaS call failed: {exc}", latency)

    latency = (time.perf_counter() - start) * 1000
    if not content:
        return _fallback_classify(text, "MaaS returned no text", latency)

    entities, sub_queries = _parse_unified_response(content, text)
    if not sub_queries:
        return _fallback_classify(text, "model JSON had no sub-queries", latency)

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

    result = ClassifyResult(
        entities=entities,
        sub_queries=sub_queries,
        is_compound=is_compound,
        intents=all_intents,
        routing=routing,
        tables_list=primary.tables_list,
        reasoning=primary.reasoning,
        source="model",
        latency_ms=latency,
    )
    _log_classification(result)
    return result
