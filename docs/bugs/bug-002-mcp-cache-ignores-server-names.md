# Bug — MCP Tool Cache Ignores `server_names`

| Field | Detail |
|---|---|
| **File** | `deep_agent/aegra/mcp.py` |
| **Function** | `get_mcp_tools()` |
| **Severity** | High |
| **Type** | Silent data corruption |
| **Discovered** | Jun 6, 2026 |
| **Status** | Fixed |

---

## Summary

The module-level MCP tool cache used a flat list with no key. The first caller's tool list was
returned to **all subsequent callers** for the entire TTL window (default 300s), regardless of
which MCP servers they requested. The orchestrator and analyst subagent request different server
subsets — so one always received the other's tools silently.

---

## Background

Every request, `graph.py` builds the agent by calling `get_mcp_tools()` twice:

1. **Orchestrator** — `get_mcp_tools(server_names=["main-mcp"])` → expects `[validate_email]`
2. **Analyst subagent** (inside `load_subagents()`) — `get_mcp_tools(server_names=["analytics-mcp"])` → expects `[calculate_bmi, search_web]`

---

## Root Cause

```python
# mcp.py — before fix
_cached_tools: list[Any] = []     # one flat bucket for all callers
_cached_tools_ts: float = 0.0

async def get_mcp_tools(sso_token=None, server_names=None):
    if _cached_tools and (time.time() - _cached_tools_ts) < _MCP_TOOL_CACHE_TTL:
        return _cached_tools   # ← server_names is never checked
```

The cache check asked **"is there anything cached?"** — not **"is what's cached relevant to what I asked for?"**

---

## What Happened at Runtime

```
Request arrives
│
├── Call 1: get_mcp_tools(server_names=["main-mcp"])
│         → cache MISS → connects to main-mcp
│         → gets [validate_email]
│         → stores in _cached_tools  ← flat list, no key
│         → returns [validate_email]  ✓ correct
│
└── Call 2: get_mcp_tools(server_names=["analytics-mcp"])
          → cache HIT (_cached_tools is not empty, TTL not expired)
          → returns [validate_email]  ✗ WRONG
             analytics-mcp was never contacted
             analyst never received [calculate_bmi, search_web]
```

---

## Why It Was Dangerous

This failure was **completely silent**:

- No exception raised
- No log warning at ERROR level
- Client received a normal-looking response

When the analyst LLM ran with `validate_email` instead of `calculate_bmi`, it had two paths:

1. Call `validate_email` with height/weight data → MCP server error → analyst fails mid-stream
2. Answer from internal knowledge → **hallucinated BMI value with no tool call** → client receives fabricated data presented as fact

Option 2 is the dangerous case. The system appeared to work correctly but was producing wrong results.

---

## Reproduction

```python
# tests/unit/cache/test_mcp_cache.py

async def test_bug2_analyst_gets_own_tools_after_orchestrator_cached():
    # Step 1 — orchestrator populates the cache
    orchestrator_tools = await get_mcp_tools(server_names=["main-mcp"])
    # → [validate_email]  ✓

    # Step 2 — analyst requests a different server
    analyst_tools = await get_mcp_tools(server_names=["analytics-mcp"])
    # → [validate_email]  ✗  (should be [calculate_bmi, search_web])

    # Fails: analytics-mcp was never contacted (call_count == 1, not 2)
    assert connect_mock.call_count == 2
```

Run:

```bash
uv run pytest tests/unit/cache/test_mcp_cache.py -v
```

Before the fix, `test_bug2_*` tests **fail** (proving the bug).  
After the fix, all 6 tests **pass**.

---

## Fix

Key the cache by `frozenset(server_names)` so each unique combination of requested servers gets
its own cache entry.

### 1 — Variables: flat list → keyed dict

```diff
- _cached_tools: list[Any] = []
- _cached_tools_ts: float = 0.0
+ # Cache keyed by frozenset(server_names) so each unique combination of
+ # requested servers gets its own entry.
+ _tool_cache: dict[frozenset, list[Any]] = {}
+ _tool_cache_ts: dict[frozenset, float] = {}
```

### 2 — Lookup: build a key before checking

```diff
- global _cached_tools, _cached_tools_ts
- if _cached_tools and (time.time() - _cached_tools_ts) < _MCP_TOOL_CACHE_TTL:
-     return _cached_tools
+ global _tool_cache, _tool_cache_ts
+ cache_key = frozenset(server_names) if server_names else frozenset()
+ cached = _tool_cache.get(cache_key)
+ if cached is not None and (now - _tool_cache_ts.get(cache_key, 0)) < _MCP_TOOL_CACHE_TTL:
+     return cached
```

### 3 — Store: write under the key

```diff
- _cached_tools = tools
- _cached_tools_ts = time.time()
+ _tool_cache[cache_key] = tools
+ _tool_cache_ts[cache_key] = time.time()
```

### Why `frozenset`?

`["main-mcp", "analytics-mcp"]` and `["analytics-mcp", "main-mcp"]` are the same request —
order should not matter. `frozenset` is order-independent and hashable (usable as a dict key).
`frozenset()` (empty set) cleanly represents "connect to all enabled servers".

---

## Cache Behavior After Fix

```
Before (one flat bucket):            After (one bucket per server combination):

  _cached_tools                        _tool_cache
  ┌───────────────────┐               ┌──────────────────┬────────────────────────┐
  │ [validate_email]  │               │ {"main-mcp"}     │ [validate_email]       │
  └───────────────────┘               ├──────────────────┼────────────────────────┤
         ↑                            │ {"analytics-mcp"}│ [calculate_bmi,        │
  every caller gets this              │                  │  search_web]           │
                                      └──────────────────┴────────────────────────┘
                                              ↑                    ↑
                                        orchestrator           analyst
                                        gets its own          gets its own
```

Cache hits within TTL still work — repeated calls with the same `server_names` return the cached
result without reconnecting.

---

## Test Suite Impact

| Test | Before | After |
|---|---|---|
| `test_cache_miss_fetches_correct_tools_for_main_mcp` | PASS | PASS |
| `test_cache_miss_fetches_correct_tools_for_analytics_mcp` | PASS | PASS |
| `test_bug2_analyst_gets_own_tools_after_orchestrator_cached` | **FAIL** | PASS |
| `test_bug2_orchestrator_gets_own_tools_after_analyst_cached` | **FAIL** | PASS |
| `test_same_server_names_cache_hit_is_correct` | PASS | PASS |
| `test_expired_cache_refetches_from_network` | PASS | PASS |

**Overall suite:** 538 passed → 543 passed (net +5, includes previously blocked tests in `test_mcp.py` that were unblocked as a side effect of the `sys.modules` patch in the new test file).

---

## Files Changed

| File | Change |
|---|---|
| `deep_agent/aegra/mcp.py` | Replace flat cache with `frozenset`-keyed dict; update lookup and store logic |
| `tests/unit/cache/test_mcp_cache.py` | New — 6 tests covering cache miss, bug regression (×2), correct hit, TTL expiry |
| `tests/unit/infrastructure/test_mcp.py` | Update `_reset_mcp_cache()` to reset new dict names |
