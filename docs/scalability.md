# Scaling to 1000 concurrent users

What breaks first, in the order it breaks, with the numbers behind
each claim. Nothing here is a projection from a load test — these are
the configured limits read off the code and the manifests, and the
arithmetic of running them at ten pods.

The target is 1000 people using the agent at the same time. An agent
turn is not a web request: it is tens of seconds of mostly waiting on
a model, holding a checkpointer transaction and a thread of state the
whole way. That shape is why the usual web-tier intuitions give the
wrong answer here, and it is the thread running through every finding
below.

## Summary

| # | Finding | Fails as | Severity |
|---|---|---|---|
| 1 | Postgres connection budget exceeds `max_connections` at 2 pods, and is uncapped besides | `FATAL: sorry, too many clients already` on a random request | Outage |
| 2 | HPA scales on CPU, load is I/O-bound | Never scales out; saturates in place | Outage |
| 3 | Redis client is synchronous, 10 connections/pod | Latency cliff, lock contention | Degradation |
| 4 | No prompt caching; ~120k input tokens on a one-sentence turn | Cost, linear in traffic | Cost |
| 5 | No agent-side concurrency guard | Accepts load it cannot serve | Amplifier |
| 6 | Pools have no liveness check on checkout | A server-side close fails the *next* caller's query | Degradation |

Findings 1 and 2 compound: the HPA will not add the pods that would
relieve the pressure, and if it did, finding 1 means the new pods
would make things worse rather than better.

## 1. The Postgres connection budget

Four independent pools open per pod. None of their sizes is set in
`config/agent/deployment/values.yaml` or any overlay, so every one of
them is an upstream default that nobody chose for this workload.

| Pool | Declared at | Max conns |
|---|---|---|
| SQLAlchemy engine | `aegra_api/core/database.py:37` | `pool_size=10` + `max_overflow=20` = 30 |
| LangGraph shared (checkpointer + store) | `aegra_api/core/database.py:55` | `LANGGRAPH_MAX_POOL_SIZE=20` |
| Personalization | `deep_agent/src/personalization/repository.py:98` | `max_size=10` per URI |
| Analytics usage | `deep_agent/src/analytics/usage.py:107` | `max_size=4` |
| | **per pod** | **64** |

The HPA is `minReplicas: 2, maxReplicas: 10`
(`config/agent/deployment/values.yaml:70`). So the fleet asks for
between 128 and 640 connections. The pgvector instance backing the
local and dev environments reports `max_connections = 100`.

Two pods under load already exceed it. This is not a 1000-user
problem; it is a two-pod problem, and the only reason it has not been
hit is that nothing has yet driven both pods to saturate their pools
at the same moment.

The failure mode is the unpleasant kind. Pools do not fail at the
pool — they fail at whichever request happens to be the one that
needed a new connection, which surfaces as an unrelated endpoint
returning 500 while the rest of the pod looks healthy.

### 64 per pod is a floor, not a cap

Two modules open connections outside every pool, so they do not
appear in the budget above and nothing bounds them.

`deep_agent/aegra/eval_routes.py` acquires a raw, unpooled
`psycopg.AsyncConnection.connect()` through its `_pg_conn()` helper
(`:646`) at **sixteen** call sites. Each one is a full TCP, TLS and
authentication handshake for the duration of one request. Under
concurrent `/evals` traffic these stack on top of the 64 with no
ceiling at all.

`deep_agent/aegra/personalization_routes.py:102` goes further and
builds an entire `asyncpg.create_pool(min_size=1, max_size=2)` per
call, to read a single row, and tears it down again. A TTL cache
(`_ASSISTANT_TTL`) keeps this off the hot path most of the time, but
the cache check at `:91` holds no lock — so when it expires, every
concurrent request in that instant creates its own pool. It is also
the only use of `asyncpg` in a codebase that is otherwise psycopg,
which means a second driver, a second set of pool semantics and a
second place to configure.

Both should borrow from the shared pools. Until they do, any
connection budget computed from the four pools understates the real
demand by an amount nobody can predict.

### On PgBouncer and RDS Proxy

A transaction-pooling proxy is the standard answer, and the question
of whether to run PgBouncer when RDS Proxy is available has a clean
answer: **they are the same tool and you want exactly one of them.**
RDS Proxy is a managed transaction pooler; putting PgBouncer behind
it buys nothing and gives you two places where a connection can pin.

Three things are worth knowing before treating either as the fix:

- **A proxy shares the ceiling, it does not raise it.** RDS Proxy's
  `MaxConnectionsPercent` is a share of the database's own
  `max_connections`. Multiplexing works because most pooled
  connections are idle between statements. Connections *actively in a
  transaction* cannot be multiplexed by anyone. If the application
  genuinely holds 640 concurrent transactions, the proxy queues or
  errors exactly as the database would.
- **Pinning is already handled.** Session pinning is what usually
  defeats a transaction pooler — prepared statements, `SET`,
  advisory locks, temp tables. Upstream aegra already sets
  `prepared_statement_cache_size: 0` on the SQLAlchemy engine
  (`database.py:43`) and `prepare_threshold: None` on the LangGraph
  pool (`database.py:48`), both commented as PgBouncer compatibility.
  `repository.py:101` does the same. This is a solved problem and we
  should not re-solve it.
- **Right-sizing is required either way.** 64 connections per pod is
  wrong on its own terms. The analytics pool holds 4 connections to
  write best-effort telemetry; the personalization pool holds 10 per
  URI for a feature most turns never touch. A proxy would hide that
  waste rather than remove it, and hidden waste is what makes the
  next capacity question unanswerable.

**Recommendation.** Right-size the four pools to a per-pod budget
that fits `maxReplicas` against the real `max_connections`, and make
each size a config key in `values.yaml` rather than an inherited
default. Use RDS Proxy as the multiplexer where it exists; do not
add PgBouncer. The pool sizes are the fix, the proxy is the headroom.

`POSTGRES_HOST` resolves from Vault
(`values.yaml:143`), and no RDS or Aurora resource exists anywhere in
`sre/infra-aws` — the only datastore module there is `elasticache`.
So whether a proxy is in the path for preprod and prod is not
knowable from this repo and must be confirmed with SRE before the
budget is set.

## 2. The HPA measures the wrong thing

`targetCPUUtilizationPercentage: 70` against `limits.cpu: 1000m`.

An agent turn spends almost all of its wall-clock time awaiting a
model response. A pod can hold a hundred concurrent streams — every
one of them occupying an event-loop task, a checkpointer connection
and a slot in the thread pool — while reporting CPU in the teens.
The resource that is exhausted is never the resource being measured,
so the autoscaler sits at two replicas through precisely the load it
exists to absorb.

`scaleDown.stabilizationWindowSeconds: 300` makes it worse in the
other direction: the fleet scales *in* during a quiet five minutes
and meets the next burst from a standing start, paying cold-start on
an import graph heavy enough to have forced the memory limit to 2Gi
(`values.yaml:62`).

The manifest says so itself. `deployment/overlays/openshift/hpa.yaml`
annotates `maxReplicas: 10` with "Peak capacity - handles ~100-150
concurrent users". Against a target of 1000 that is a stated 7-10x
shortfall, written down and not yet acted on.

The signal that tracks the real constraint is in-flight runs per pod,
or connection-pool utilisation. The first of those **already exists**:
`deep_agent/aegra/inflight_metrics.py` publishes an
`aegra_runs_in_flight` gauge, refreshed on a loop, counting runs in
pending or running status. Nothing consumes it — `hpa.yaml` declares
only the two `type: Resource` CPU and memory metrics. Wiring it as a
`type: Pods` metric is most of the work already done.

## 3. Redis is the synchronous client

`deep_agent/aegra/redis.py:61` calls `redis.from_url(...)` — the
blocking client, not `redis.asyncio`. Every call site compensates by
wrapping it in `asyncio.to_thread` (`auth.py:273`, on the
authenticated-request path).

That keeps the event loop free, which is the important part, and it
is why this is a degradation rather than an outage. But it routes all
caching, distributed locking and circuit-breaker state through the
default `ThreadPoolExecutor` and onto `REDIS_MAX_CONNECTIONS = 10`
per pod (`redis.py:27`), with `REDIS_SOCKET_TIMEOUT = 5`.

Three consequences worth naming:

- The thread pool is shared with every other `to_thread` caller in
  the process, so Redis latency and unrelated blocking work contend
  for the same scarce slots.
- `acquire_distributed_lock` and the `distributed_lock` context
  manager run through this path. A lock whose acquisition is queued
  behind a saturated thread pool holds its critical section longer,
  which lengthens the queue — the classic shape that turns slow into
  stuck.
- Connection-pool exhaustion inside the sync client raises rather
  than waits, and `retry_on_timeout` does not cover it.

Moving to `redis.asyncio` removes the thread pool from the picture
entirely and lets the connection count be sized against concurrency
instead of against threads. The call sites already `await`, so the
change is mechanical: the `asyncio.to_thread` wrappers come out at
the same time.

Note also that `sre/infra-aws/modules/elasticache` exists, so Redis
in the deployed environments is likely ElastiCache rather than the
in-cluster `redis` Deployment used locally. Its own connection and
TLS behaviour should be checked against the client configuration
before the pool size is chosen.

## 4. Prompt caching, and the 193 tools

Measured, not estimated. `PromptSizeMiddleware` reports what each
model call actually sends, and on a turn whose entire answer was the
word "ok":

| Component | Characters | Share |
|---|---|---|
| Tool schemas (331 tools) | 615,705 | **98.2%** |
| System prompt | 11,040 | 1.8% |
| Conversation | 23 | ~0% |

That call recorded **119,769 input tokens** — about 362 tokens per
bound tool, paid on every turn, by every user, whether a tool is used
or not. The question needed none of them.

Two things are worth separating. The ratio is absurd on its own
terms: roughly four thousand input tokens per output token for a
one-word reply. And the tool count is not stable — it was 193 when
this document was first written and is 331 now, having grown without
anyone noticing, because nothing reported what it cost. The budget
logging below exists so that cannot happen again.

Note the startup log is misleading here and cost a round of analysis.
`Agent ready: 11 tool(s)` is printed before MCP discovery completes;
the figure that matters is the one at model-call time.

### Where the tool cost sits

From the `Tool budget:` line, ordered by cost. Tokens are scaled to
the recorded total rather than the character estimate.

| Server | Tools | ≈ tokens | Share | Tokens per tool |
|---|---|---|---|---|
| gworkspace | 101 | 35,800 | 30% | 354 |
| datadog | 34 | 23,300 | 19% | **685** |
| gitlab | 116 | 19,100 | 16% | 165 |
| atlan | 15 | 13,500 | 11% | 900 |
| atlassian | 21 | 12,500 | 10% | 595 |
| miro | 23 | 10,500 | 9% | 457 |
| dataverse | 6 | 4,500 | 4% | 750 |
| unstructured-data | 3 | 540 | 0.5% | 180 |

The useful result is that **count and cost do not correlate**.
gitlab has the most tools and is a sixth of the bill; atlan has
fifteen tools that cost 900 tokens each. Anyone reasoning about this
from tool counts — as this document did at first — reaches for the
wrong server.

No `cache_control`, context caching or equivalent appears anywhere in
`deep_agent/`. The tool schemas and system prefix are stable across
every turn of every user — they are the ideal cache content, and
caching them is the single largest cost lever available.

The second lever is the token count itself. 193 tools is more than
the model can usefully discriminate between in one call, so this is a
quality argument as much as a cost one: tool-selection accuracy
degrades well before 193 options. Per-subagent scoping is the
existing mechanism, and the `/subagents` endpoint already reports
`tools` and `tool_scope`, so the data model does not need inventing.

These two interact. Shrinking the orchestrator's tool set shrinks the
cacheable prefix but also shrinks what has to be cached; do the
scoping first, then cache what remains, or the cache gets invalidated
by the scoping work.

## 5. No concurrency guard

There is no semaphore and no in-flight cap anywhere in `deep_agent/`.
`RATE_LIMIT_ERROR` exists in `src/exceptions.py:84` as an error code,
but nothing in the agent raises it from a concurrency decision.

Rate limiting lives at `platform/gateway`, which is the right place
for per-user fairness and abuse control, and the wrong place for this
problem: the gateway protects itself and knows nothing about how many
checkpointer connections a pod has left.

Without a cap, the pod accepts every request it is given and
discovers it cannot serve them only once the pools are gone — at
which point the failure lands on requests that were already
half-served rather than on the ones that should have been shed. A
bounded queue with a fast 503 is worse for the marginal user and
much better for everyone already in flight.

## 6. Connection lifecycle: release, and liveness on checkout

The instinct to close connections after each operation and on failure
is the right instinct aimed at the wrong layer, so it is worth
separating the two halves.

### Release is already correct, and should not be made stricter

Every pooled acquisition in `deep_agent/` uses `async with` — all
fifteen in `src/personalization/repository.py`, the one in
`src/analytics/usage.py:148`, the one in
`aegra/personalization_routes.py:105`. The context manager returns
the connection to the pool on success *and* on exception, so a raised
error cannot leak a connection.

The sixteen raw connections in `eval_routes.py` are also safe, for a
reason worth writing down because it differs from psycopg2:
psycopg3's `AsyncConnection.__aexit__` rolls back on exception,
commits otherwise, and then **closes the connection if it does not
belong to a pool**. Fourteen sites use `async with`; the remaining
two (`:657`, `:710`) use `try/finally` with an explicit `close()`.
Both forms are correct.

So there is nothing to fix here, and adding explicit closes would be
actively harmful: closing a *pooled* connection after each operation
destroys the thing the pool exists for, trading a microsecond
checkout for a TCP, TLS and authentication round trip on every query.
The leak this guards against does not exist.

### Liveness on checkout is the real gap

What is missing is the other direction — a connection that the pool
believes is good but the server has already closed.

All three psycopg pools are constructed with `check` left at its
default of `None` (`aegra_api/core/database.py:55`,
`repository.py:95`, `usage.py:107`), which means no liveness probe
when a connection is handed out. The pool's other lifecycle defaults
are sane — `max_idle=600`, `max_lifetime=3600`, `timeout=30`,
`reconnect_timeout=300` — but none of them help in the window that
matters.

A connection can die server-side at any moment without the client
learning: an RDS Proxy or PgBouncer idle timeout, a database
failover, a network partition, an administrative termination. The
pool hands that corpse to the next borrower, and the failure lands on
*that* request — a caller with no relationship to whatever caused the
close. After a failover this is not one bad request but every pooled
connection failing once, as a burst, which looks like an application
fault and is not one.

The SQLAlchemy engine already handles this: `pool_pre_ping=True` at
`database.py:41`. The three psycopg pools have no equivalent enabled,
and `psycopg_pool` ships exactly the right callable —
`AsyncConnectionPool.check_connection`. Passing `check=` is a
one-line change per pool and costs a round trip on checkout, which is
the correct trade against failing a user's request.

This matters more, not less, behind a proxy. Transaction poolers
recycle backend connections aggressively by design, so the window
where the client holds a stale handle is wider than it would be
against the database directly.

### Statement timeouts

No `statement_timeout` is set on any pooled connection. The only two
occurrences in the codebase (`eval_routes.py:660`, `:713`) are `SET`
statements on raw unpooled connections, where the session-scoped
setting dies with the connection and so cannot leak to another
borrower — correct as written, but not coverage for anything else.

Without one, a query that hangs holds its connection until
`max_lifetime` expires an hour later. At a 64-connection budget that
is a meaningful fraction of the pod's capacity removed by a single
stuck statement. A `statement_timeout` applied at the pool's
`configure` hook, or on the database role, bounds the damage.

Note that `SET` is a session-level pinning trigger for a transaction
pooler. If a timeout is applied per connection rather than on the
role, it belongs in the pool's `configure` callback — run once when
the connection is created — not inside request handlers.

## What was changed

All six findings are addressed in code. The table maps each to where
it now lives; the sections above remain as the reasoning behind them.

| # | Fix | Where |
|---|---|---|
| 1 | One shared pool replaces two pools and three unpooled call sites | `deep_agent/src/db/pool.py` |
| 1 | Pool bounds are config, not inherited defaults | `deployment/base/configmap-tuning.yaml` |
| 2 | HPA consumes `aegra_runs_in_flight`; slower scale-down | `scripts/render_deployment.py`, `values.yaml` |
| 3 | Native async distributed lock; dedicated Redis executor | `deep_agent/aegra/redis.py` |
| 4 | Prompt prefix made cacheable | `deep_agent/src/infrastructure/middleware.py` |
| 4 | Per-call prompt size logged, split by component | `middleware.py` (`PromptSizeMiddleware`) |
| 4 | Tool budget logged per MCP server, warns past a threshold | `deep_agent/aegra/mcp.py` |
| 4 | Cached-token count recorded and surfaced | `usage.py`, `queries.py`, `AnalyticsPage.tsx` |
| 5 | Run admission control with 503 shedding | `deep_agent/aegra/admission.py` |
| 6 | Liveness check on checkout, both pools | `pool.py`, `deep_agent/aegra/startup.py` |

### The connection budget, now

Per pod, with every pool bounded and nothing opening connections
outside one:

| Pool | Max |
|---|---|
| SQLAlchemy engine (`SQLALCHEMY_POOL_SIZE` + overflow) | 15 |
| LangGraph shared (`LANGGRAPH_MAX_POOL_SIZE`) | 12 |
| `deep_agent` shared (`DB_POOL_MAX_SIZE`) | 8 |
| **total** | **35** |

Against `maxReplicas: 10` that is 350, down from an unbounded figure
whose floor was 640. It still exceeds a `max_connections` of 100, so
the remaining work is not in this repo: either raise
`max_connections`, lower `maxReplicas`, or put a transaction pooler
in the path. The difference is that the number is now computable, and
changing it is a config edit rather than a code change.

Three call sites stopped opening their own connections:
`eval_routes._pg_conn` (sixteen), `analytics.queries._connect`
(five), and `personalization_routes._get_assistant_id_for_store`,
which built and destroyed a whole asyncpg pool per cache miss and was
the last use of a second driver.

### The prompt cache fix

`get_current_datetime()` is injected into the system message on every
model call, and it carried seconds. The system message heads the
prefix a provider caches and the tool schemas sit behind it, so a
value changing every second meant the prefix was unique on every
call — nothing downstream of it could ever be cached, including the
hundred-odd tool definitions that are most of the input tokens.

It is now rounded to `DATETIME_PROMPT_GRANULARITY_SECONDS`, default
one hour, which makes the prefix identical across every user and
every turn within that hour. An agent needing the time to the second
can call a tool for it.

This is the enabling change, not the saving itself. It makes caching
possible where it was impossible; it does not reduce the token count
we record, because cached input tokens are still input tokens — they
are billed at a discount, and the discount shows up in a field
(`cached_content_token_count`) that `record_llm_usage` does not
currently capture.

So the next step is measurement, and it needs a change to do it:
record the cached-token count alongside the input count, then compare
turns. Until that exists, the claim here is "the prefix is now
stable", not "caching is working".

## Long-running agents

A turn can run for minutes. That makes almost every piece of
lifecycle machinery in Kubernetes hostile by default, because the
defaults assume a request finishes in milliseconds. Four things had
to change, and one was already right.

**Grace period.** Was `terminationGracePeriodSeconds: 60`, which is
shorter than a single long turn — a rollout or an eviction killed
work in progress. Now 600. Shutdown persists in-flight runs so
another pod resumes them, but finishing in place is always better
than resuming elsewhere, and the grace period is what buys that.

**Endpoint removal races the TERM signal.** Kubernetes sends both at
once, so a pod can begin shutting down while the router is still
sending it traffic. A `preStop` sleep of 15 seconds holds the
container open until the removal has propagated; shutdown starts
after.

**Liveness was too eager.** `periodSeconds: 10`, `timeoutSeconds: 5`,
`failureThreshold: 3` — three slow answers and the pod dies with
every run on it. A loop busy with long turns can be slow to answer
without being unhealthy, and a GC pause could fire it. Now
`20s × 6 × 10s`: two minutes of genuine silence before a kill.
Readiness is unchanged and still quick, because the trade runs the
other way — shedding traffic from a struggling pod is cheap and
reversible.

**Scale-down retires pods slowly.** `stabilizationWindowSeconds` went
from 300 to 900, with a 10%-per-minute policy and `selectPolicy:
Min`. Each retired pod has to drain or hand off whatever it is
working on, so a burst of scale-down during a lull is how long work
gets interrupted for nothing.

**The route was already right.** `haproxy.router.openshift.io/timeout:
18000s` in `deployment/overlays/openshift/route.yaml` — five hours.
Worth knowing it is deliberate, because the HAProxy default of 30
seconds would sever every long turn at the ingress regardless of
anything the pod does.

One interaction worth stating: `DB_STATEMENT_TIMEOUT_MS` bounds a
*statement*, not a run. A long turn is many short queries against the
checkpointer, so a 30-second statement timeout never truncates one —
it only catches a query that has genuinely hung. The two limits look
like they conflict and do not.

## Latency: where a turn's wall clock goes

A CPU flame graph is the wrong instrument for this and would have
misled. The agent spends nearly all of a turn awaiting a model over
the network, so a sampled Python stack shows an idle event loop and
attributes the time to `epoll`. (It is also not obtainable in the
normal way: py-spy needs `CAP_SYS_PTRACE`, which the pod should not
have.) `misc/planning/templates/local/latency-probe.sh` measures
wall clock per span instead.

Six consecutive turns, prompt "Reply with one word: ok", answer "ok":

| | thread create | run | total |
|---|---|---|---|
| warm, steady | 0.07s | **4.13s** | 4.20s |
| spread over six turns | — | 4.11–4.14s | — |

Then the same probe immediately after a pod roll:

| turn | run |
|---|---|
| 1 | 32.2s |
| 2 | 42.2s |
| 3 | 20.2s |
| …later, cache expired mid-probe | 16.1s |

Two separate problems, with different fixes.

### The 4.1s floor is the token count

One model call, 119,769 input tokens, one output token. Nothing else
in the turn is measurable: thread creation is 0.07s and the
conversation is 23 characters. 4.1s is what it costs a model to read
120,000 tokens of tool schema before it can say "ok".

That makes the tool budget a latency problem as well as a cost one,
and it is the more visible of the two: nobody sees the bill, everyone
sees four seconds.

### The 16–42s cliff is MCP discovery

`get_mcp_tools` caches on `f"{user_id}:{server_key}"`
(`deep_agent/aegra/mcp.py:837`) with a 300s TTL
(`config/agent/runtime/agent.yaml:364`), in a process-local dict.
Every cache miss reconnects to all eight MCP servers.

Keyed per user, that means:

- Every user pays 16–42s on their first turn.
- They pay it again after five minutes idle.
- The cache is in-process, so the same user pays it again on each
  pod they are routed to — up to `maxReplicas` times over.
- At 1000 users that is thousands of discovery sweeps against eight
  third-party services, which is a thundering herd pointed at
  somebody else's infrastructure.

What makes this worth fixing rather than tuning: **tool schemas are
not user-specific**. Which servers a user may reach is; the shape of
`gitlab`'s 116 tools is not. The cache is keyed on the user because
authorisation is per user, but the expensive part of what it stores
is identical for everyone. A shared schema cache with per-user
*filtering* would turn 1000 discoveries into one.

### `_cached_tools` has no eviction

`_cached_tools: dict[str | None, list[Any]] = {}`
(`deep_agent/aegra/mcp.py:58`) is a plain dict with no size bound.
TTL is checked when an entry is read, so an entry for a user who
stopped using the agent is never freed — it is simply never read
again.

Measured, each user's entry holds 615,705 characters of tool
description and schema, before Python object overhead for 319 tool
instances. Growth is proportional to unique users seen since the pod
started, against a 2Gi limit that already carries ~1223 MiB of
resident interpreter. This is the shape of an OOM at a user count
nobody will connect to the cause.

## Results after the latency work

Measured with `latency-probe.sh`, same prompt ("Reply with one word:
ok"), same answer, before and after scoping the orchestrator's MCP
list to the four servers no subagent owns.

| | before | after |
|---|---|---|
| input tokens | 119,769 | **57,883** |
| of which cached | 0 | **53,103 (91.7%)** |
| warm turn | 4.11–4.14s | 2.09–4.11s |
| tools bound | 319 | ~120 |

Three things worth separating out.

**The token count halved** because four servers moved behind the
subagents that already owned them — atlan and dataverse to curator,
datadog and gitlab to investigator. No capability was removed; a
turn that needs gitlab now delegates instead of calling directly.

**Prompt caching started working.** It had been reporting zero. The
rounding fix made the prefix stable, but the prefix also has to
*repeat*, and with 319 tools discovered fresh per pod the list was
not identical turn to turn. A smaller, stabler tool set is what made
the cache reachable. Cached input tokens are billed at a discount
rather than free, so this is not a 92% saving — but it is the
difference between paying full price for 58k tokens every turn and
paying it once.

Note this was only visible because the cached-token figure is now
recorded. Before that, a working cache and a broken one produced
identical numbers, which is how a per-second timestamp sat in the
cacheable prefix unnoticed for the life of the feature.

**Latency is no longer flat.** It was 4.13s ± 0.02 every turn; it is
now 2.09s on a cache hit and ~4.1s otherwise. The floor is prefill,
and prefill is what caching avoids.

### What is left on the table

The four remaining servers cost about 59,340 tokens:

| Server | ≈ tokens | Owned by |
|---|---|---|
| gworkspace | 35,800 | nobody |
| atlassian | 12,500 | nobody |
| miro | 10,500 | nobody |
| unstructured-data | 540 | nobody |

These cannot simply be deleted from `config/agent/PROMPT.md`: an
agent only sees the servers it names, so removing one makes it
unreachable rather than indirect. Shortening the list further means
giving them a subagent to own — a workspace-facing one would cover
gworkspace, atlassian and miro, and take roughly another 59,000
tokens off every turn. That is a product decision about how the
agent is organised, so it is proposed rather than done.

## Second latency pass: where it actually went

The first pass concluded "4.1s is prefill of 120,000 tokens". That
was wrong, and the way it was caught is worth recording: cutting the
prompt to 7,965 tokens — a fifteenfold reduction — did not move the
number at all. A tenfold cut in work with no change in wall clock is
not a model being slow.

What the clustering showed instead: turn latencies sat at 2.09s,
4.10s and 6.11s, spaced 2.01s apart. Quantisation, not computation.

`aegra_api/services/worker_executor.py:119` polls a Redis done-key
and then `await asyncio.sleep(2.0)`. It is a poll, not a
subscription, so a run finishing in 300ms is reported to the caller
at 2.0s. `deep_agent/aegra/run_wait.py` replaces it with an
exponential backoff from 50ms, installed at startup — the same
arrangement as the pool liveness check, and for the same reason: the
publisher side is in a vendored package.

The first version of that patch made things *slower*, which is
instructive. Upstream checks Postgres on every second poll; at a 2s
interval that is every 4s. Keeping the ratio against a 50ms interval
meant a database query twenty times a second per waiting request.
The fallback is now rate-limited by time, not by poll count.

With that fixed, the breakdown of a 3.18s turn:

| Phase | Time |
|---|---|
| POST → run row created | 0.05s |
| run executing | 2.65–2.87s |
| done → response delivered | 0.26–0.49s |

And `model_call seconds=3.317` against a client-observed 3.19s: the
provider round trip *is* the turn. Graph construction is cached and
costs 0.05s; completion detection is now 0.3–0.5s where it was up to
2s. There is no longer meaningful latency in this codebase to remove.

Which means further latency work is provider-shaped, not code-shaped:

- **Stream.** `/runs/stream` puts a first token in front of the user
  in a fraction of the time. The 3.2s measured here is a property of
  `/runs/wait`, which by definition cannot return until the whole
  turn is done. This is the single largest remaining win for
  *perceived* latency and it needs no change to the agent.
- A smaller or faster model, or a lower thinking budget, for turns
  that do not need reasoning.

`misc/planning/templates/local/latency-probe.sh` reproduces the
table. `PromptSizeMiddleware` logs `prompt_size` and `model_call` on
every call, so the "is it us or the model" question does not need
re-deriving.

## Deliberately not done

**Cutting the tool set.** The orchestrator names all eight MCP
servers in `config/agent/PROMPT.md`, and the scoping mechanism
already exists — `_filter_tools_by_mcp_names`, which the subagents
already use. At 98.2% of every request this is far and away the
largest cost lever, ahead of caching, and it improves tool selection
too: accuracy degrades well before 331 options.

It is not done here because it changes what the orchestrator can
answer directly rather than by delegating to a subagent. That is a
product decision about how the agent behaves, not a capacity fix,
and it should be made deliberately rather than as a side effect of
a scalability pass. What this pass does provide is the measurement
to make it with: `Tool budget:` logs the per-server breakdown on
every discovery, so the cost of keeping each server is now a number
rather than a guess.

**The seventy synchronous Redis call sites.** `cache_get` and its
siblings are still the blocking client, called from thirteen modules,
many of them synchronous. They no longer share the default executor —
Redis has its own, sized to its connection pool — so the starvation
path is closed. Converting them to `redis.asyncio` cascades async
through every caller and is a separate change.

## Suggested order for what remains

1. **Confirm `max_connections` and proxy topology with SRE.** The
   budget is computable now but 350 against 100 is still over. This
   is the only open item that can still cause an outage.
2. **Deploy the custom-metrics adapter** so the HPA's
   `aegra_runs_in_flight` metric resolves. Until it does, the CPU and
   memory targets remain in place as a fallback and finding 2 is only
   half fixed.
3. **Load test.** Everything above is reasoned from configured limits,
   not observed behaviour. `test/load-tests` is where the ordering
   gets confirmed or corrected.
4. **Decide on tool scoping** — a product call, with a measurable
   cost attached.

## What this document does not cover

No load test has been run. Every number here is a configured limit or
a measured per-turn cost, not an observed breaking point, and the
interaction effects between the findings are reasoned rather than
demonstrated.

The prompt-cache change makes caching possible; whether the provider
then caches, and what that saves, is unmeasured.

One observation from earlier investigation remains unexplained and
may bear on reliability: 10 of roughly 27 recorded runs ended in
`interrupted` status, and the cause is not established. The longer
grace period and slower scale-down below plausibly address some of
it — a pod retired mid-run is exactly how a run ends up interrupted —
but that is a hypothesis, not a diagnosis.
