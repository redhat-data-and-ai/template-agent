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

---

## Addendum — Rebase onto upstream/deep-agent + CodeRabbit follow-ups

Before this PR merged, `upstream/deep-agent` moved 57 commits ahead and landed a large OAuth/DCR
rewrite of `mcp.py` (token-injector interceptor, per-server credential resolution, retry logic,
auth placeholder tools, `user_id` param). That rewrite **reintroduced the exact bug described
above** — it kept the old flat `_cached_tools` / `_cached_tools_ts` globals, since it branched
off before this fix merged. This PR was rebased onto the latest `upstream/deep-agent` and the
`frozenset`-keyed cache was reapplied on top of the new function bodies, with all of upstream's
new OAuth functionality left untouched. `invalidate_mcp_tool_cache()` (new upstream, called after
an OAuth connect completes) now does `_tool_cache.clear(); _tool_cache_ts.clear()`.

CodeRabbit flagged three follow-up issues on the original diff, all addressed in the rebase:

### 1 — Empty results were never cached (Major)

`get_mcp_tools()` returned `[]` on both the "no enabled servers" and "all servers failed" paths
*before* the cache-write lines, so a repeated request that resolves to zero tools reconnects on
every single call instead of respecting the TTL like a successful result does.

Fix: write to `_tool_cache[cache_key]` / `_tool_cache_ts[cache_key]` on both early-return paths
too — **with one deliberate exception**. The "no auth token at startup" deferral is *not* cached,
because the cache key is keyed only by `server_names`, not by auth state. Caching that negative
result would mask the first legitimately authenticated request for the same servers for the rest
of the TTL window. Caching is applied only when `has_auth` (`bool(sso_token or user_id)`) is true:

```python
if not tools:
    ...
    if has_auth:
        _tool_cache[cache_key] = []
        _tool_cache_ts[cache_key] = time.time()
    return []
```

The "no enabled servers" path has no such risk — it's a static, deterministic outcome of config
+ `server_names` only, independent of any token — so it is cached unconditionally.

### 2 — Missing order-independence regression test (Minor)

Added `test_reversed_server_names_order_is_still_a_cache_hit`: requests
`["main-mcp", "analytics-mcp"]` then `["analytics-mcp", "main-mcp"]` and asserts the second call
is a cache hit (`connect_mock.call_count` stays at 2, one per server, not 4).

### 3 — Missing empty-result caching regression tests (Minor)

Added two tests mirroring the "Bug 2" test structure — they fail without the Section 1 fix and
pass with it:

- `test_no_enabled_servers_result_is_cached` — asserts `_get_server_configs` is called once
  across two identical requests that resolve to zero enabled servers.
- `test_all_servers_failed_result_is_cached_when_authenticated` — asserts
  `connect_mock.call_count == 1` across two identical authenticated requests where every server
  fails to connect.

A third test, `test_unauthenticated_deferral_is_not_cached`, guards the deliberate exception
above — an unauthenticated call followed by an authenticated call for the same `server_names`
must both hit the network (`connect_mock.call_count == 2`).
