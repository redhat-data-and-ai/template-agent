# Eval Runner — Flow Reference

## Overview

The eval runner is a standalone FastAPI service (`eval_api.py`) that runs
pre-written eval cases against a live agent and stores results in Postgres.
It is deployed as a separate pod (KEDA HTTP-scaled, min=0 max=2) in the same
namespace as the agent.

```
UI / agentpod
    │
    └─► POST /evals/run          ← trigger (from agentpod eval_routes.py)
             │
         eval_api.py             ← orchestrator
             │
         run_eval.py (subprocess, one per tag)
             │
         Agent (LangGraph Platform)   Lightspeed-eval scorer
             │                              │
             └──────── results ─────────────┘
                              │
                         Postgres (evaluation_results table)
                              │
                    eval_api.py aggregates and writes
                              │
                    Postgres (evals table) ← agentpod reads this
```

---

## 1. Configuration

All paths are resolved at startup:

| Env var | Default | Description |
|---|---|---|
| `AGENT_CONFIG_DIR` | auto-detected | Root of the agent config PVC |
| `EVAL_CASES_PATH` | `$AGENT_CONFIG_DIR/evals/lightspeed/eval_cases.yaml` | Eval dataset |
| `EVAL_SYSTEM_CONFIG` | `$AGENT_CONFIG_DIR/evals/lightspeed/system.yaml` | Lightspeed-eval system config |
| `EVAL_OUTPUT_DIR` | `$TMPDIR/eval_output` | Temp output dir for scorer reports |
| `AGENT_HOST` | `http://localhost:5002` | Agent base URL |
| `EVAL_MAX_CONCURRENCY` | `3` | Max tag batches running in parallel |
| `AGENT_CONFIG_HASH` | computed | SHA-256 of config files (16 chars) — links eval results to a config version |
| `AGENT_AUTH_TOKEN` | `""` | Bearer token forwarded to agent for MCP tool calls |

Auto-detection walks up from `eval_api.py` looking for `config/agent/`. Falls
back to `/agent-config` if not found.

---

## 2. Eval Dataset — `eval_cases.yaml`

The dataset is pre-written to the config PVC by agent-engine at deploy time.
The eval runner never writes to it — it is read-only.

### Structure

```yaml
- conversation_group_id: tool_use_abc123
  description: "..."
  tag: tool_use          # groups cases into parallel batches
  turns:
    - turn_id: turn_1
      query: "What is my BMI for 175cm and 70kg?"
      expected_response: "BMI is 22.9, Normal weight."
      expected_keywords:
        - ["22.9", "22.8"]   # OR group — any of these counts
        - ["Normal"]
      expected_tool_calls:
        - calculate_bmi
      turn_metrics:
        - custom:answer_correctness
        - custom:keywords_eval
        - custom:tool_eval
      turn_metrics_metadata:
        custom:tool_eval:
          ordered: false
          full_match: false
```

### Supported tags

| Tag | Default metrics |
|---|---|
| `tool_use` | `custom:answer_correctness` |
| `hitl` | `custom:intent_eval` — scores the pre-approval response |
| `structured_output` | `geval:tone_safety` |
| `multi_agent` | `custom:answer_correctness`, `geval:delegation_compliance` |
| anything else | `custom:answer_correctness` (fallback) |

`custom:keywords_eval` and `custom:tool_eval` are auto-added when
`expected_keywords` / `expected_tool_calls` are present.

### Keyword normalization

Keywords are normalized to `list[list[str]]` at read time. Each inner list is
an OR group — the eval passes if any string in the group appears in the
response.

```
"22.9, Normal"           → [["22.9"], ["Normal"]]
["22.9", "Normal"]       → [["22.9"], ["Normal"]]
[["22.9", "22.8"], ["Normal"]]  → used as-is
```

---

## 3. Trigger Flow

### `POST /evals/run` — run all tags

1. Validates `eval_cases.yaml` and `system.yaml` exist.
2. Generates a `run_id` (`YYYYMMDD-HHMMSS`).
3. Sets in-memory state → `running`.
4. Fires `_run_eval(pattern=None)` as a background task.
5. Returns `{run_id, status: "started", pattern: "all"}` immediately (202).

### `POST /evals/run/{pattern}` — run one tag

Same as above but `pattern` is one of `tool_use`, `hitl`, `structured_output`,
`multi_agent`. Returns 400 for unknown patterns.

---

## 4. `_run_eval` — Background Orchestrator

```
_run_eval(pattern)
    │
    ├─ _system_yaml_path()        inject Postgres creds into system.yaml → temp file
    │
    ├─ _find_eval_files(pattern)  split eval_cases.yaml by tag → temp YAML per tag
    │
    ├─ asyncio.gather(            run all tag files concurrently (semaphore = EVAL_MAX_CONCURRENCY)
    │     _run_eval_pattern(tag1_file),
    │     _run_eval_pattern(tag2_file),
    │     ...
    │  )
    │
    ├─ cleanup temp files
    │
    ├─ load_results_since(run_started_at)   read aggregated results from Postgres
    │
    └─ write_eval_result(...)               write final summary to evals table
```

### Tag splitting

`eval_cases.yaml` is loaded once. Cases are grouped into one temp YAML file
per tag in a single pass (`defaultdict`). Each temp file is passed to a
separate `run_eval.py` subprocess. This keeps turns sequential within a tag
while tags run in parallel.

### System YAML credential injection

`system.yaml` from the PVC has placeholder Postgres credentials. Before
spawning subprocesses, `_get_system_yaml_content()` overlays actual
credentials from env vars (`POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`,
`POSTGRES_USER`, `POSTGRES_PASSWORD`) and writes the result to a temp file.
The original PVC file is never modified.

---

## 5. `run_eval.py` — Subprocess (one per tag)

Each subprocess independently:

```
run_eval.py --agent-url ... --eval-data tag.yaml --system system.yaml --output-dir ...
    │
    ├─ load eval_cases.yaml (tag slice)
    │
    ├─ _populate_dataset()          call agent for each conversation group (parallel, max 10 workers)
    │     │
    │     └─ per group: _populate_group()
    │           │
    │           ├─ POST /threads                      create LangGraph thread
    │           │
    │           └─ per turn: _call_agent()
    │                 │
    │                 ├─ POST /threads/{id}/runs/stream   stream initial run
    │                 │
    │                 ├─ if __interrupt__:
    │                 │     capture pre_approval_response
    │                 │     loop: POST resume until no interrupt (max 50 approvals)
    │                 │
    │                 └─ return {response, pre_approval_response, tool_calls, contexts}
    │
    ├─ write populated YAML to temp file
    │
    ├─ _run_lightspeed()            invoke lightspeed-eval scorer subprocess
    │     └─ lightspeed-eval --system-config ... --eval-data ... --output-dir ...
    │           └─ writes results to Postgres (evaluation_results table)
    │           └─ writes *_summary.json to output-dir
    │
    └─ _log_summary()               log overall pass rate (INFO) + threshold failures (WARNING)
                                    per-metric detail logged at DEBUG only
```

### Agent interaction — SSE stream parsing

Each turn streams via LangGraph's `POST /threads/{id}/runs/stream` with
`stream_mode: ["updates", "events"]` and `stream_subgraphs: True`.

Two event types are consumed:

| Event | What's extracted |
|---|---|
| `events` (on_tool_start) | Every MCP tool call at any subagraph depth, deduplicated by `run_id` |
| `updates` | Final AI response text, orchestrator-level tool calls, tool result contexts |

Internal housekeeping tools (`write_todos`, `task`, `read_file`, etc.) are
excluded from `tool_calls`.

### HITL handling

When `__interrupt__` is detected in the stream:

1. The pre-approval response (what the agent said before pausing) is captured.
2. The turn's `hitl: true` flag causes `intent_eval` to score the
   pre-approval response rather than the final response.
3. Auto-approval resumes the run: `POST /threads/{id}/runs/stream` with
   `command.resume.decisions: [{type: approve}]`.
4. Capped at `MAX_HITL_APPROVALS = 50` per turn.

### For `tool_use` tag — subagent tool call fetch

After the main stream completes, `_fetch_subagent_tool_calls()` calls
`GET /v1/eval/thread-tool-calls/{thread_id}` on the agent. This endpoint reads
directly from Postgres `checkpoint_blobs` across all subagent namespaces,
since LangGraph's HTTP API only exposes subgraph state during interrupts.
Retried up to 4 times with back-off because subagent checkpoints may not be
flushed immediately after the stream closes.

---

## 6. Results Storage

### `evaluation_results` table (written by lightspeed-eval)

One row per turn × metric combination. Key columns:

| Column | Description |
|---|---|
| `run_id` | Timestamp-based ID from the subprocess run |
| `conversation_group_id` | Links back to the eval case |
| `tag` | Eval tag |
| `turn_id` | Turn identifier |
| `metric_identifier` | e.g. `custom:answer_correctness` |
| `result` | `PASS` / `FAIL` / `ERROR` |
| `score` | Float 0–1 |
| `reason` | Judge explanation |
| `query`, `response`, `expected_response` | Not logged — stored in DB only |

### `evals` table (written by `write_eval_result`)

One row per eval run. Stores the aggregated summary:

| Column | Description |
|---|---|
| `org`, `name`, `config_hash` | Identifies which agent version was evaluated |
| `eval_status` | `in_progress` → `completed` / `error` |
| `eval_score` | `pass / total` across all metrics |
| `pass`, `fail`, `error` | Counts |
| `ls_run_ids` | Array of run_ids from `evaluation_results` |
| `results_detail` | Full JSONB summary (turns + by_metric + by_conversation) |

`write_eval_result` does an `UPDATE ... WHERE eval_status IN ('in_progress', 'error')` — it updates the row that was inserted by the agentpod trigger, not a new one.

---

## 7. API Endpoints

### `GET /evals/status`

Returns the current in-memory run state:

```json
{"state": "idle" | "running" | "completed" | "error", "run_id": "..."}
```

### `GET /evals/results`

Returns the latest in-memory result dict (populated after `_run_eval` completes).

### `GET /evals/results/{run_id}`

Queries `evaluation_results` for a specific run and returns per-turn rows
(`conversation_group_id`, `turn_id`, `metric_identifier`, `result`, `score`, `reason`).

### `GET /health`

Always returns `{"status": "ok"}`. Used as liveness probe.

### `POST /mcp`

Stub — returns an error. Suppresses 404 noise from lightspeed-eval's MCP probe
at startup.

---

## 8. Concurrency Model

```
FastAPI event loop (single process)
    │
    ├─ incoming HTTP requests handled here
    │
    └─ _run_eval() runs as asyncio background task
          │
          ├─ asyncio.Semaphore(EVAL_MAX_CONCURRENCY=3) — limits parallel tag subprocesses
          │
          └─ each _run_eval_pattern() offloads blocking subprocess.run() to thread pool
                (loop.run_in_executor) so the event loop stays free
```

Only one eval run at a time is enforced by checking `_status["state"] == "running"`
before accepting a new trigger. Concurrent trigger requests receive 409.

---

## 9. Temp File Lifecycle

| File | Created by | Cleaned up by |
|---|---|---|
| `eval_{tag}_*.yaml` | `_find_eval_files` | `_run_eval` finally block |
| `eval_system_*.yaml` | `_system_yaml_path` | `_run_eval` finally block |
| `lightspeed_live_*.yaml` | `run_eval.py main()` | `run_eval.py` finally block |
| `gcp_sa_*.json` | `_subprocess_env` | `_run_lightspeed` finally block |

All temp files are cleaned up unconditionally in `finally` blocks, including on
subprocess failure.

---

## 10. What is NOT Logged

The following are written to Postgres only — never appear in logs:

- Query text / expected response / actual agent response content
- Tool call arguments or tool result contexts
- `results_detail` JSONB payload
- Auth tokens or credentials

Logs contain only: counts, IDs, file paths, exit codes, pass rates (aggregate
stats), and threshold failure warnings.
