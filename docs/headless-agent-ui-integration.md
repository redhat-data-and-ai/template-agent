# Headless Agent — UI Integration Specification

## Overview

This document specifies the UI changes needed to support headless agent creation and management in the AI Factory interface. The backend (template-agent) already supports headless mode — this doc covers what the UI needs to expose and what APIs/configs it generates.

## Current Agent Creation Flow (Server Only)

Today the UI creates a regular server agent with:

```
User selects → Model, Skills, MCP Servers, Subagents
UI generates → PROMPT.md, agent.yaml, mcp.json, subagent .md files
Platform     → Deploys single Deployment (server mode via Aegra)
```

## New Flow: Agent with Headless Worker

```
User selects → Model, Skills, MCP Servers, Subagents
               + Enables headless worker
               + Configures triggers and sinks
UI generates → PROMPT.md (orchestrator, includes queue_task tool)
               HEADLESS_PROMPT.md (worker, auto-generated)
               agent.yaml (includes triggers + sinks + health_check)
               mcp.json (shared)
Platform     → Deploys TWO Deployments (server + headless)
```

---

## UI Changes Required

### 1. Agent Creation Form — New Section

Add a collapsible "Background Worker" section to the agent creation form:

```
┌─────────────────────────────────────────────────────┐
│  Create Agent                                        │
│                                                      │
│  Name:        [Health Assistant          ]           │
│  Model:       [gemini-2.5-pro         ▼ ]           │
│  Skills:      [bmi-report] [client-intake] [+]      │
│  MCP Servers: [template-mcp-server] [+]              │
│  Subagents:   [analyst] [publisher] [+]              │
│                                                      │
│  ── Background Worker (optional) ──────────────────  │
│  [ ] Enable headless worker                          │
│                                                      │
│  ▶ Triggers (collapsed when disabled)                │
│  ▶ Output Sinks                                      │
│  ▶ Worker Prompt                                     │
│  ▶ Health Check                                      │
│                                                      │
│  [Create Agent]                                      │
└─────────────────────────────────────────────────────┘
```

### 2. Triggers Configuration

When "Enable headless worker" is checked, expand the triggers section:

```
┌─────────────────────────────────────────────────────┐
│  Triggers                                            │
│                                                      │
│  ☑ Queue Consumer                                    │
│    Backend:        [Redis Streams ▼]                 │
│                     ├── Redis Streams                │
│                     └── Kafka                        │
│    Stream/Topic:   [agent-tasks              ]       │
│    Consumer Group: [agent-workers            ]       │
│                                                      │
│    (if Kafka selected)                               │
│    Bootstrap Servers: [localhost:9092         ]       │
│                                                      │
│  ☐ Webhook Listener                                  │
│    Port:           [8888    ]                        │
│    Path:           [/trigger]                        │
│                                                      │
│  ☐ Cron Jobs                                         │
│    [+ Add Job]                                       │
│    ┌─ Job 1 ────────────────────────────────┐       │
│    │ Name:     [daily-report             ]  │       │
│    │ Schedule: [0 9 * * *                ]  │       │
│    │ Payload:  [{"task": "gen report"}   ]  │       │
│    └────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────┘
```

### 3. Output Sinks Configuration

```
┌─────────────────────────────────────────────────────┐
│  Output Sinks (results fan out to all enabled sinks) │
│                                                      │
│  [+ Add Sink]                                        │
│                                                      │
│  ┌─ Sink 1 ─────────────────────────────────┐       │
│  │ Type: [Stdout ▼]                         │       │
│  │       ├── Stdout (console/logs)          │       │
│  │       ├── File (JSONL)                   │       │
│  │       ├── Webhook (HTTP POST)            │       │
│  │       └── Redis Stream                   │       │
│  └──────────────────────────────────────────┘       │
│                                                      │
│  ┌─ Sink 2 ─────────────────────────────────┐       │
│  │ Type: [File ▼]                           │       │
│  │ Path: [/var/log/agent/results.jsonl]     │       │
│  └──────────────────────────────────────────┘       │
│                                                      │
│  ┌─ Sink 3 ─────────────────────────────────┐       │
│  │ Type: [Webhook ▼]                        │       │
│  │ URL:  [https://example.com/callback]     │       │
│  │ Headers:                                  │       │
│  │   Authorization: [Bearer ${TOKEN}]       │       │
│  └──────────────────────────────────────────┘       │
│                                                      │
│  ┌─ Sink 4 ─────────────────────────────────┐       │
│  │ Type: [Redis Stream ▼]                   │       │
│  │ Stream: [agent-results]                  │       │
│  └──────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────┘
```

### 4. Worker Prompt Configuration

```
┌─────────────────────────────────────────────────────┐
│  Worker Prompt                                       │
│                                                      │
│  ○ Auto-generate from agent skills (recommended)     │
│    The worker prompt is generated automatically       │
│    using the agent's skills and tools. It strips      │
│    conversational behavior (TODO lists, greetings)    │
│    and focuses on silent task processing.             │
│                                                      │
│  ○ Custom prompt                                     │
│    ┌──────────────────────────────────────────┐      │
│    │ You are a background task processor...   │      │
│    │                                          │      │
│    │                                          │      │
│    └──────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────┘
```

### 5. Health Check Configuration

```
┌─────────────────────────────────────────────────────┐
│  Health Check                                        │
│                                                      │
│  ☑ Enable health endpoint                            │
│  Port: [8080]                                        │
│                                                      │
│  Endpoints:                                          │
│    /healthz  — liveness probe (always 200 if up)     │
│    /readyz   — readiness probe (checks sources/loop) │
└─────────────────────────────────────────────────────┘
```

---

## Files Generated by UI

When the user clicks "Create Agent" with headless enabled, the UI must generate these files:

### 1. PROMPT.md (Orchestrator — modified)

The existing orchestrator prompt, but with `queue_task`, `check_task_status`, and `get_pending_results` tools auto-added:

```yaml
---
name: orchestrator
model: gemini-2.5-pro            # from user selection
tools:
  - validate_email               # from user selection
  - queue_task                   # AUTO-ADDED when headless enabled
  - check_task_status            # AUTO-ADDED when headless enabled
  - get_pending_results          # AUTO-ADDED when headless enabled
skills:
  - client-intake                # from user selection
mcps:
  - template-mcp-server          # from user selection
---

# {Agent Name}

{User-provided or auto-generated orchestrator prompt}

## Background Tasks (Headless Worker)

{AUTO-INJECTED section — see template below}
```

**Auto-injected background tasks section:**

```markdown
## Background Tasks (Headless Worker)

A headless worker runs alongside you as a background processor.
Use `queue_task` to delegate work that is long-running, bulk,
or doesn't need an immediate response.

**When to use queue_task:**
- Bulk operations (e.g., "generate reports for all clients")
- Long-running processing (e.g., "export all data")
- Fire-and-forget notifications

**When NOT to use queue_task:**
- Anything the user expects an immediate answer to

**Status tracking — CRITICAL RULES:**
1. `queue_task` returns a task ID — give this to the user
2. When the user asks about task status, ALWAYS call `check_task_status(task_id)`
3. Show complete results from `check_task_status` — never say "check a file"
4. At the start of every conversation, call `get_pending_results(user_id)`
   to deliver completed background task results proactively
```

### 2. HEADLESS_PROMPT.md (Worker — auto-generated)

```yaml
---
name: headless-worker
model: gemini-2.5-pro            # SAME model as orchestrator
tools:                           # SAME tools as orchestrator (minus queue_task etc.)
  - calculate_bmi
  - search_web
skills:                          # SAME skills as orchestrator
  - bmi-report
mcps:                            # SAME MCP servers
  - template-mcp-server
---

# Background Task Processor

You are a background task processor. You receive tasks from a queue
and process them silently.

## Rules

1. No greetings, no TODO lists, no conversational responses.
2. Process the payload directly using your tools.
3. Return structured JSON results.
4. Handle errors gracefully with clear error messages.

## Task Processing

When you receive a task payload:
1. Parse the task name and data
2. Execute the work using your tools
3. Return a JSON result:
   - status: "success" or "error"
   - summary: Brief description
   - data: The actual results
   - error: Error message if failed
```

**Auto-generation logic:**
- `model` → copy from orchestrator
- `tools` → copy from orchestrator, REMOVE: `queue_task`, `check_task_status`, `get_pending_results`, `validate_email`
- `skills` → copy from orchestrator
- `mcps` → copy from orchestrator
- System prompt → use the standard worker template above

### 3. agent.yaml (Runtime config — extended)

Append these sections to the existing agent.yaml:

```yaml
# ── Triggers (headless mode only) ──
triggers:
  webhook:
    enabled: true                # from UI checkbox
    host: "0.0.0.0"
    port: 8888                   # from UI input
    path: "/trigger"             # from UI input
  cron:
    enabled: false               # from UI checkbox
    jobs: []                     # from UI job list
  queue:
    enabled: true                # from UI checkbox
    backend: "redis_streams"     # from UI dropdown
    stream: "agent-tasks"        # from UI input
    consumer_group: "agent-workers"  # from UI input
    consumer_name: ""            # auto: defaults to $HOSTNAME
    # Kafka-specific (only when backend=kafka):
    bootstrap_servers: "localhost:9092"  # from UI input
    topic: "agent-tasks"         # from UI input

# ── Output Sinks (headless mode only) ──
output_sinks:                    # from UI sink list
  - type: stdout
  - type: file
    path: "/var/log/agent/results.jsonl"
  - type: redis
    stream: "agent-results"

# ── Health Check (headless mode only) ──
health_check:
  enabled: true                  # from UI checkbox
  host: "0.0.0.0"
  port: 8080                     # from UI input
```

---

## Backend API Requirements

The UI needs these backend APIs to support headless agent management:

### 1. Agent Creation (existing, extended)

```
POST /api/agents
```

Existing payload, with new optional `headless` field:

```json
{
  "name": "Health Assistant",
  "model": "gemini-2.5-pro",
  "skills": ["bmi-report"],
  "mcps": ["template-mcp-server"],
  "subagents": ["analyst", "publisher"],
  "headless": {
    "enabled": true,
    "triggers": {
      "webhook": {"enabled": true, "port": 8888, "path": "/trigger"},
      "cron": {"enabled": false, "jobs": []},
      "queue": {
        "enabled": true,
        "backend": "redis_streams",
        "stream": "agent-tasks",
        "consumer_group": "agent-workers"
      }
    },
    "output_sinks": [
      {"type": "stdout"},
      {"type": "file", "path": "/var/log/agent/results.jsonl"}
    ],
    "health_check": {"enabled": true, "port": 8080},
    "worker_prompt": "auto"
  }
}
```

The backend should:
1. Generate `PROMPT.md` with `queue_task` tools auto-added
2. Generate `HEADLESS_PROMPT.md` (auto or custom)
3. Write trigger/sink config to `agent.yaml`
4. Create two Deployments (server + headless) in the kustomize overlay

### 2. Task Status API (new)

```
GET /api/agents/{agent_id}/tasks
```

Returns task history from PostgreSQL audit table:

```json
{
  "tasks": [
    {
      "task_id": "abc123",
      "task_name": "bulk-bmi-report",
      "status": "completed",
      "user_id": "naveen",
      "created_at": "2026-06-24T13:34:50Z",
      "completed_at": "2026-06-24T13:35:12Z",
      "duration_seconds": 22,
      "delivered": true
    }
  ],
  "total": 1,
  "page": 1
}
```

Query parameters:
- `user_id` — filter by user
- `status` — filter by status (queued, processing, completed, failed)
- `limit`, `offset` — pagination

SQL behind this API:
```sql
SELECT * FROM tasks
WHERE user_id = $1 AND status = $2
ORDER BY created_at DESC
LIMIT $3 OFFSET $4;
```

### 3. Task Detail API (new)

```
GET /api/agents/{agent_id}/tasks/{task_id}
```

Returns full task detail including result:

```json
{
  "task_id": "abc123",
  "task_name": "bulk-bmi-report",
  "status": "completed",
  "payload": {"employee_count": 500},
  "result": "{\"status\": \"success\", \"data\": [...]}",
  "error": null,
  "user_id": "naveen",
  "thread_id": "thread-xyz",
  "delivered": true,
  "created_at": "2026-06-24T13:34:50Z",
  "updated_at": "2026-06-24T13:35:12Z",
  "completed_at": "2026-06-24T13:35:12Z"
}
```

### 4. Headless Worker Status API (new)

```
GET /api/agents/{agent_id}/headless/status
```

Proxies to the headless worker's health endpoint:

```json
{
  "status": "ready",
  "sources": 2,
  "sinks": 3,
  "loop_running": true,
  "uptime_seconds": 3600
}
```

---

## UI Pages / Views

### 1. Agent Detail Page — New "Background Tasks" Tab

```
┌──────────────────────────────────────────────────────┐
│  Health Assistant                                     │
│                                                       │
│  [Chat] [Settings] [Background Tasks] [Monitoring]   │
│                                                       │
│  ── Background Tasks ────────────────────────────────│
│                                                       │
│  Worker Status: ● Ready (2 sources, 3 sinks)         │
│                                                       │
│  ┌────────────────────────────────────────────────┐  │
│  │ ID          Name              Status   Duration│  │
│  │ abc123      bulk-bmi-report   ✅ Done   22s    │  │
│  │ def456      weekly-digest     🔄 Running 5s    │  │
│  │ ghi789      data-export       ❌ Failed  3s    │  │
│  │ jkl012      email-blast       ⏳ Queued  —     │  │
│  └────────────────────────────────────────────────┘  │
│                                                       │
│  Showing 4 of 127 tasks  [< 1 2 3 ... 13 >]         │
└──────────────────────────────────────────────────────┘
```

### 2. Task Detail Modal

Click on a task row to see details:

```
┌──────────────────────────────────────────────────────┐
│  Task: bulk-bmi-report (abc123)                      │
│                                                       │
│  Status:     ✅ Completed                             │
│  Created:    2026-06-24 13:34:50                     │
│  Completed:  2026-06-24 13:35:12                     │
│  Duration:   22 seconds                               │
│  User:       naveen                                   │
│  Thread:     thread-xyz                               │
│  Delivered:  Yes                                      │
│                                                       │
│  ── Payload ──────────────────────────────────────── │
│  {                                                    │
│    "employee_count": 500                              │
│  }                                                    │
│                                                       │
│  ── Result ───────────────────────────────────────── │
│  {                                                    │
│    "status": "success",                               │
│    "summary": "Processed 500 BMI calculations",       │
│    "data": [...]                                      │
│  }                                                    │
│                                                       │
│  [Close]                                              │
└──────────────────────────────────────────────────────┘
```

### 3. Agent Dashboard — Headless Worker Card

On the main dashboard, show headless worker status alongside the server agent:

```
┌─────────────────────┐  ┌─────────────────────────┐
│  Server Agent       │  │  Headless Worker         │
│  ● Running          │  │  ● Ready                 │
│  Port: 5002         │  │  Sources: webhook, queue │
│  Threads: 42        │  │  Sinks: stdout, file     │
│  Uptime: 3h 15m     │  │  Tasks today: 23         │
│                     │  │  Success rate: 96%        │
└─────────────────────┘  └─────────────────────────┘
```

---

## Data Flow Summary

```
┌────────────────────────────────────────────────────────────┐
│                        AI Factory UI                        │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Create Agent Form                                         │
│  ├── Model, Skills, MCP, Subagents                        │
│  └── Headless: Triggers, Sinks, Health Check              │
│       │                                                    │
│       ▼                                                    │
│  POST /api/agents { headless: { ... } }                   │
│       │                                                    │
│       ▼                                                    │
│  Backend generates:                                        │
│  ├── PROMPT.md (with queue_task auto-added)                │
│  ├── HEADLESS_PROMPT.md (auto-generated)                  │
│  ├── agent.yaml (triggers + sinks + health_check)         │
│  └── Kustomize overlays (server + headless Deployments)   │
│       │                                                    │
│       ▼                                                    │
│  Deployed to OpenShift:                                    │
│  ├── Deployment: agent (server mode, port 5002)           │
│  └── Deployment: agent-headless (headless mode)           │
│                                                            │
│  Task Monitoring:                                          │
│  ├── GET /api/agents/{id}/tasks → PostgreSQL audit table  │
│  ├── GET /api/agents/{id}/headless/status → health check  │
│  └── Background Tasks tab in agent detail page            │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## Configuration Reference

### Trigger Types

| Type | Config Fields | Description |
|------|--------------|-------------|
| Queue (Redis Streams) | `backend`, `stream`, `consumer_group` | Consumes from Redis Stream. Default for server→headless delegation. |
| Queue (Kafka) | `backend`, `topic`, `bootstrap_servers`, `consumer_group` | Consumes from Kafka topic. For external system events. |
| Webhook | `port`, `path` | Minimal HTTP listener. For external system webhooks. |
| Cron | `jobs[].name`, `jobs[].schedule`, `jobs[].payload` | Scheduled triggers. Standard 5-field crontab syntax. |

### Sink Types

| Type | Config Fields | Description |
|------|--------------|-------------|
| Stdout | (none) | Prints to process stdout / container logs |
| File | `path` | Appends JSONL to file |
| Webhook | `url`, `headers` | POSTs result to URL with retry (3 attempts) |
| Redis Stream | `stream` | XADD to Redis Stream |

### Queue Backend Comparison

| Feature | Redis Streams | Kafka |
|---------|--------------|-------|
| Setup complexity | None (already in stack) | Needs Kafka cluster |
| Message ordering | Per-stream | Per-partition |
| Consumer groups | Yes (XREADGROUP) | Yes (native) |
| Message persistence | Configurable | Default persistent |
| Multi-replica | Yes (consumer groups) | Yes (consumer groups) |
| Use case | Internal delegation | External events, high throughput |

---

## How Skills Drive Headless Processing

The headless agent is **not one generic worker**. It's specific to the agent it was created with. The **skills** define what the headless worker knows how to do.

### The Three Layers

```
HEADLESS_PROMPT.md (generic scaffolding)
  "You are a background processor. Process tasks silently. Return JSON."

         +

Skill Documents (domain knowledge)
  config/agent/skills/order-fulfillment/README.md
  "To process an order:
   1. Call check_inventory(sku)
   2. If in stock, call process_payment(amount)
   3. Call update_order(status='confirmed')
   4. Return order confirmation JSON"

         +

Tools / MCP Servers (capabilities)
  check_inventory, process_payment, update_order

         =

Headless worker that knows how to process orders
```

### Each Agent Gets Its Own Headless Worker

```
AI Factory
├── Health Assistant Agent
│   ├── Server: PROMPT.md (orchestrator for health)
│   └── Headless: HEADLESS_PROMPT.md
│       Skills: [bmi-report]
│       Tools: [calculate_bmi, search_web]
│       → Knows how to calculate BMI and generate health reports
│
├── Order Processing Agent
│   ├── Server: PROMPT.md (orchestrator for orders)
│   └── Headless: HEADLESS_PROMPT.md
│       Skills: [order-fulfillment, inventory-check]
│       Tools: [check_inventory, process_payment, update_order]
│       → Knows how to validate, process, and confirm orders
│
├── Data Pipeline Agent
│   ├── Server: PROMPT.md (orchestrator for data)
│   └── Headless: HEADLESS_PROMPT.md
│       Skills: [data-transform, quality-check]
│       Tools: [query_warehouse, write_report]
│       → Knows how to run ETL and quality checks
```

### What the Skill Document Contains

A skill README is where all domain-specific processing logic lives:

```markdown
# Order Fulfillment Skill

## Input Format
- order_id: string (required)
- items: list of {sku, quantity}

## Processing Steps
1. Validate all items exist in catalog (use check_inventory tool)
2. Verify stock availability for each item
3. Calculate total price including tax
4. Process payment (use process_payment tool)
5. Update order status to confirmed (use update_order tool)
6. Generate confirmation with estimated delivery date

## Output Format
{
  "status": "success",
  "order_id": "ORD-1234",
  "total": 149.99,
  "items_fulfilled": 3,
  "estimated_delivery": "2026-06-28"
}

## Error Handling
- Out of stock → return status "partial", list unavailable items
- Payment failed → return status "error", include payment error code
- Invalid order_id → return status "error", message "Order not found"
```

### How It Flows at Runtime

```
External system sends:
  {"name": "process-order", "task": "Process order ORD-1234", "order_id": "ORD-1234"}
         │
         ▼
Headless worker receives via Kafka/Redis/Webhook
         │
         ▼
HEADLESS_PROMPT.md: "You are a background processor. Use your skills."
         │
         ▼
LLM loads order-fulfillment skill: "Step 1: Call check_inventory..."
         │
         ▼
LLM calls tools: check_inventory → process_payment → update_order
         │
         ▼
Result: {"status": "success", "order_id": "ORD-1234", "total": 149.99}
         │
         ▼
Output sinks (Redis, file, webhook callback)
```

### UI Implication

When the user creates a headless agent in the UI:
- The **skills they attach** define what the worker can do
- The **tools/MCP servers** give it the capabilities
- The **HEADLESS_PROMPT.md** is generic scaffolding (auto-generated)
- **No custom code needed per use case** — just a skill document and the right tools

The UI should show a preview: "This headless worker will be able to handle: BMI calculations (bmi-report skill), health data search (search_web tool)."

---

## Implementation Notes for UI Team

1. **The `headless` field in the agent creation API is optional.** If omitted, only the server agent is created (current behavior).

2. **Auto-generation of HEADLESS_PROMPT.md** should copy model/skills/MCP from the orchestrator and use the standard worker template. The UI should show a preview before creation.

3. **The `queue_task`, `check_task_status`, `get_pending_results` tools are built-in** — they don't need to be in any MCP server. The backend auto-registers them when headless is enabled.

4. **Task status polling:** The UI should poll `GET /api/agents/{id}/tasks?status=processing` every 5-10 seconds when the Background Tasks tab is open, to show real-time status updates.

5. **The headless worker uses the same container image** as the server agent. Only the startup command differs (`python -m deep_agent.headless` vs `aegra dev`). No separate build needed.

6. **Health check:** The `/readyz` endpoint returns `503` when the worker is not ready (sources not started, loop not running). Use this for the status indicator in the dashboard.
