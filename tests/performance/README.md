# Performance Benchmarks

Standalone scripts for measuring template-agent latency before and after dependency upgrades. Uses only Python stdlib — no extra dependencies required.

## Scripts

### `benchmark_agent.py` — LLM Path Benchmark

Measures end-to-end latency through the full LLM pipeline (middleware chain → model → streaming → response).

**What it measures:**
- **TTFT (Time to First Token)** — parsed from SSE stream
- **Total response time** — request to final SSE event
- **Health endpoint latency**
- Splits results by prompt type: `simple` (no tool calls) vs `subagent` (triggers orchestration)

**Usage:**
```bash
# Run 20 measurement iterations (+ 3 warmup)
python tests/performance/benchmark_agent.py \
  --base-url http://localhost:5002 \
  --runs 20 \
  --output results-llm.json

# Compare two result files
python tests/performance/benchmark_agent.py \
  --compare before.json after.json
```

### `benchmark_framework.py` — Framework Path Benchmark (No LLM)

Measures framework overhead without LLM calls — isolates Aegra/LangGraph infrastructure from model inference.

**What it measures:**
- `/health` and `/readyz` endpoint latency
- Thread lifecycle: create → get → delete
- Graph build: cold (first compile) vs cached (fingerprint hit)

**Usage:**
```bash
# Run 100 iterations
python tests/performance/benchmark_framework.py \
  --base-url http://localhost:5002 \
  --iterations 100 \
  --output results-framework.json

# Compare two result files
python tests/performance/benchmark_framework.py \
  --compare before.json after.json
```

## Upgrade Validation Workflow

1. **Before upgrade** — run both benchmarks, save output JSONs
2. **Apply version bumps** — edit `pyproject.toml`
3. **Rebuild + restart** — rebuild Kind image, restart pod
4. **After upgrade** — run both benchmarks again
5. **Compare** — use `--compare` to see deltas

### Key metrics and thresholds

| Metric | Source | Pass | Regression signal |
|--------|--------|------|-------------------|
| TTFT p50/p95 | `benchmark_agent.py` | ±10% | >15% slower |
| Total response p50/p95 | `benchmark_agent.py` | ±10% | >15% slower |
| Health latency | Both scripts | ≤ baseline | Any increase = Aegra overhead |
| Graph cold build | `benchmark_framework.py` | ±20% | >30% slower |
| Graph cached | `benchmark_framework.py` | ≤ baseline | Cache invalidation bug |
| Thread CRUD | `benchmark_framework.py` | ±10% | Checkpoint layer changed |

## Notes

- Both scripts use **Python stdlib only** (`urllib`, `json`, `ssl`) — no pip install needed.
- Output JSON files are environment-specific and should **not** be committed. Add `*.json` results to `.gitignore`.
- TTFT is parsed from SSE `data:` events, not HTTP headers — it reflects actual user-perceived first token time.
- Warmup runs (default 3) are excluded from statistics to avoid cold-start skew.
