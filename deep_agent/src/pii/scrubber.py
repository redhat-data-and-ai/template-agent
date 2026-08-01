"""PII scrubbing engine with per-request token map.

The token map is stored in a ContextVar so it is:
  - isolated per async task (multi-user safe)
  - propagated to child tasks (parallel tool calls share the same map)
  - ephemeral (rebuilt from Postgres message history on each request)
"""

import hashlib
import hmac
import os
import re
from contextvars import ContextVar
from typing import Any

from deep_agent.src.pii.config import PIIConfig
from deep_agent.src.pii.detector import PIIDetector, PIIMatch
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

# Per-request state stored in ContextVars.
_token_map: ContextVar[dict[str, str] | None] = ContextVar(
    "pii_token_map", default=None
)
_value_map: ContextVar[dict[str, str] | None] = ContextVar(
    "pii_value_map", default=None
)  # reverse: value→token
_label_counters: ContextVar[dict[str, int] | None] = ContextVar(
    "pii_label_counters", default=None
)

# Shared mutable container for cross-context token map sharing.
#
# LangGraph runs each node via copy_context().run(node_fn), which copies the
# ContextVar *reference* (not the dict value). A mutable dict registered here
# by PIIAwareRunnable before the graph runs can be mutated from inside the
# node (where _token_map is populated) and read back in the outer scope.
_shared_token_map_ref: ContextVar[dict[str, str] | None] = ContextVar(
    "pii_shared_token_map_ref", default=None
)

# Per-thread stable token map store — capped with TTL to bound memory.
# Holds at most 10,000 threads; evicts least-recently-used after 30 days.
# Each entry is ~500 bytes, so worst-case memory footprint is ~5 MB.
try:
    from cachetools import TTLCache

    _thread_token_maps: TTLCache = TTLCache(maxsize=10_000, ttl=7 * 86_400)
except ImportError:
    _thread_token_maps: dict = {}  # type: ignore[no-redef]

# Regex to find tokens already present in text (for restoration).
_TOKEN_RE = re.compile(r"\[([A-Z_]+)_(\d+)\]")

# Keys that hold identifiers, not free-text content. Scanning these for PII
# regularly produces false positives (e.g. a UUID substring matching a phone
# number pattern), corrupting values that must remain byte-for-byte stable
# for correlation (message ids, run ids, tool_call_id, thread/checkpoint ids).
_ID_LIKE_KEYS = frozenset(
    {
        "id",
        "run_id",
        "parent_run_id",
        "tool_call_id",
        "thread_id",
        "checkpoint_id",
        "checkpoint_ns",
        "trace_id",
        "span_id",
        "session_id",
        "request_id",
        "correlation_id",
        "call_id",
    }
)


class PerRuleDetector:
    """Routes each rule to its configured detector backend (regex / presidio / custom).

    Replaces the global detector setting — each rule declares its own backend:
      detector: regex    → compiled regex (BUILTIN_PATTERNS or rule.regex)
      detector: presidio → Presidio NLP AnalyzerEngine
      detector: custom   → rule.regex field (same path as regex, just explicit)
    """

    def __init__(self, rules: list) -> None:
        """Build regex and/or Presidio sub-detectors from the rule list."""
        regex_rules = [r for r in rules if r.detector in ("regex", "custom")]
        presidio_rules = [r for r in rules if r.detector == "presidio"]

        self._regex = PIIDetector(regex_rules) if regex_rules else None
        if presidio_rules:
            from deep_agent.src.pii.presidio_detector import PresidioDetector

            self._presidio: Any = PresidioDetector(presidio_rules)
        else:
            self._presidio = None

    def find_all(self, text: str) -> list[PIIMatch]:
        """Return all non-overlapping PII matches ordered by position."""
        combined: list[PIIMatch] = []
        if self._regex:
            combined.extend(self._regex.find_all(text))
        if self._presidio:
            combined.extend(self._presidio.find_all(text))
        combined.sort(key=lambda m: (m.start, m.end))
        deduped: list[PIIMatch] = []
        last_end = -1
        for match in combined:
            if match.start >= last_end:
                deduped.append(match)
                last_end = match.end
        return deduped


def _build_detector(config: PIIConfig) -> Any:
    """Build a PerRuleDetector from the config rules."""
    if not config.rules:
        if config.enabled:
            logger.warning(
                "pii_enabled_no_rules: PII is enabled but no rules are defined — scrubbing is inactive"
            )
        return None
    return PerRuleDetector(config.rules)


class PIIScrubber:
    """Scrubs and restores PII values using a per-request token map.

    Args:
        config: PII rule configuration.
        hash_key: HMAC key bytes for deterministic hashing. Defaults to a
            random process-scoped key if not provided.
    """

    def __init__(self, config: PIIConfig, hash_key: bytes = b"") -> None:
        """Build the scrubber and its per-strategy detectors from config."""
        self._config = config
        self._detector = _build_detector(config)
        self._hash_key = hash_key or os.urandom(32)
        # Pre-compute block-rule names and a dedicated detector for fast input blocking.
        # _check_input_blocked uses this instead of running all rules then filtering.
        block_rules = [r for r in config.rules if r.action.value == "block"]
        self._block_rule_names: frozenset[str] = frozenset(r.name for r in block_rules)
        self._block_detector = (
            _build_detector(PIIConfig(enabled=True, rules=block_rules))
            if block_rules
            else None
        )

    # ------------------------------------------------------------------
    # ContextVar helpers
    # ------------------------------------------------------------------

    def _get_map(self) -> dict[str, str]:
        m = _token_map.get()
        if m is None:
            m = {}
            _token_map.set(m)
        return m

    def _get_value_map(self) -> dict[str, str]:
        """Reverse map: value → token. O(1) lookup to avoid duplicate token assignment.

        Rebuilds from _token_map if empty but _token_map already has entries — guards
        against code paths that set _token_map without populating _value_map, which
        would cause duplicate token creation for the same PII value.
        """
        vm = _value_map.get()
        if vm is None:
            existing = _token_map.get()
            vm = {v: k for k, v in existing.items()} if existing else {}
            _value_map.set(vm)
        return vm

    def _get_counters(self) -> dict[str, int]:
        c = _label_counters.get()
        if c is None:
            c = {}
            _label_counters.set(c)
        return c

    # ------------------------------------------------------------------
    # Token assignment
    # ------------------------------------------------------------------

    def _assign_token(self, value: str, label: str) -> str:
        """Return an existing token for *value* or create a new one — O(1) via reverse map."""
        vm = self._get_value_map()
        existing = vm.get(value)
        if existing:
            return existing
        counters = self._get_counters()
        n = counters.get(label, 0) + 1
        counters[label] = n
        token = f"[{label}_{n}]"
        self._get_map()[token] = value
        vm[value] = token
        return token

    def _mask_value(self, value: str) -> str:
        """Mask a value, preserving the last 4 chars."""
        if len(value) <= 4:
            return "*" * len(value)
        return "*" * (len(value) - 4) + value[-4:]

    def _hash_value(self, value: str) -> str:
        digest = hmac.new(self._hash_key, value.encode(), hashlib.sha256).hexdigest()[
            :12
        ]
        return f"[HASH:{digest}]"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _apply_matches(self, text: str, matches: list, *, one_way: bool) -> str:
        parts: list[str] = []
        cursor = 0
        for m in matches:
            parts.append(text[cursor : m.start])
            if m.action == "hash":
                parts.append(self._hash_value(m.value))
            elif m.action in ("tokenize", "scrub") and not one_way:
                parts.append(self._assign_token(m.value, m.label))
            elif m.action == "mask":
                parts.append(self._mask_value(m.value))
            else:
                # redact / block / one-way scrub — no token assigned
                parts.append("***REDACTED***")
            cursor = m.end
        parts.append(text[cursor:])
        return "".join(parts)

    def scrub(self, text: str) -> str:
        """Replace PII in *text* using the token map (tokenize rules are reversible)."""
        if not text or not self._detector:
            return text
        matches = self._detector.find_all(text)
        return self._apply_matches(text, matches, one_way=False) if matches else text

    def restore(self, text: str) -> str:
        """Replace tokens in *text* with their original values."""
        if not text:
            return text
        token_map = _token_map.get()
        if not token_map:
            return text
        for token, value in token_map.items():
            if token in text:
                text = text.replace(token, value)
        return text

    def scrub_one_way(self, text: str) -> str:
        """Stateless one-way sanitization for logs/observability — token map not modified."""
        if not text or not self._detector:
            return text
        matches = self._detector.find_all(text)
        return self._apply_matches(text, matches, one_way=True) if matches else text

    def scrub_for_trace_hash(self, text: str) -> str:
        """Like scrub_one_way but replaces all PII with deterministic HMAC hashes.

        Used by mask_otel_spans when trace_strategy: hash — allows log correlation
        across requests without exposing real values.
        """
        if not text or not self._detector:
            return text
        matches = self._detector.find_all(text)
        if not matches:
            return text
        parts: list[str] = []
        cursor = 0
        for m in matches:
            parts.append(text[cursor : m.start])
            parts.append(self._hash_value(m.value))
            cursor = m.end
        parts.append(text[cursor:])
        return "".join(parts)

    # ------------------------------------------------------------------
    # Shared-container helpers (cross-context SSE restoration)
    # ------------------------------------------------------------------

    def set_shared_container(self, container: dict[str, str]) -> None:
        """Register *container* as the shared token map target for this request.

        Stored both as a ContextVar (for contexts where propagation works) and
        as an instance attribute (_instance_container) so that snapshot_to_container()
        can always find it even when LangGraph runs the middleware in an async
        context that doesn't inherit the ContextVar from the outer PIIAwareRunnable.
        """
        _shared_token_map_ref.set(container)
        self._instance_container: "dict[str, str] | None" = container

    def _get_shared_container(self) -> "dict[str, str] | None":
        # Prefer ContextVar (per-request isolation); fall back to instance attr.
        return _shared_token_map_ref.get() or getattr(self, "_instance_container", None)

    def snapshot_to_container(self) -> None:
        """Copy the current token map into the shared container (if registered).

        Uses both the ContextVar and the instance attribute fallback so the
        middleware can populate the container regardless of which async context
        it runs in relative to the outer PIIAwareRunnable.
        """
        container = self._get_shared_container()
        if container is not None:
            container.update(self.snapshot_token_map())

    # ------------------------------------------------------------------
    # Token map persistence helpers
    # ------------------------------------------------------------------

    def load_token_map(self, token_map: dict[str, str]) -> None:
        """Populate the ContextVar token map from a cached snapshot."""
        _token_map.set(dict(token_map))
        # Rebuild reverse map so _assign_token stays O(1) after loading.
        _value_map.set({v: k for k, v in token_map.items()})
        counters: dict[str, int] = {}
        for token in token_map:
            m = _TOKEN_RE.fullmatch(token)
            if m:
                label, n = m.group(1), int(m.group(2))
                counters[label] = max(counters.get(label, 0), n)
        _label_counters.set(counters)

    def snapshot_token_map(self) -> dict[str, str]:
        """Return a copy of the current token map for Redis persistence."""
        return dict(_token_map.get() or {})

    # ------------------------------------------------------------------
    # Per-thread stable token map (keeps the same token for the same PII
    # value throughout a full conversation thread).
    # ------------------------------------------------------------------

    def load_thread_map(self, thread_id: str) -> None:
        """Seed the per-request ContextVar map from the stable thread store.

        Must be called before _sanitize_input so that previously assigned
        tokens are reused and counters continue from where they left off.
        """
        existing = _thread_token_maps.get(thread_id)
        if existing:
            self.load_token_map(existing)

    def save_thread_map(self, thread_id: str) -> None:
        """Merge the current ContextVar map back into the stable thread store.

        Call this after _sanitize_input so any newly detected PII values are
        persisted for future calls in the same thread.
        """
        current = self.snapshot_token_map()
        if current:
            stored = _thread_token_maps.setdefault(thread_id, {})
            stored.update(current)
