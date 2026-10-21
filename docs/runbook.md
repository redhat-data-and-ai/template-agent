# Runbook

Operational procedures for an agent built from this template. The
symptom sections are real — they describe this codebase's actual failure
modes. The contact and escalation details are `TODO`: only you know them.

Keep this page short enough to be read under pressure. If a section grows
past a screen, it has become a design document and should move.

## Before you need it

| | |
|---|---|
| Escalation | `TODO` — who to wake, and after how long |
| Owning team | `TODO` |
| Dashboards | `TODO` — link them here, not in a bookmark folder |
| SLOs and burn-rate alerts | [slo.md](slo.md) |
| Where each signal lands | [observability.md](observability.md) |

## Orientation, in order

1. **Is it the agent or a dependency?** `GET /health` reports per-check
   status for Postgres, Redis, config, MCP servers, the LLM provider and
   OPA. `degraded` means a non-critical dependency is down and the pod is
   deliberately still serving.
2. **Is it one pod or all of them?** `oc get pods -l app=agent`. Restart
   counts and age usually answer it.
3. **Did something ship?** Compare the alert's start with the rollout:
   `oc rollout history deployment/agent`. The pods carry
   `app.kubernetes.io/version`, so the metrics backend can split any
   series by release.

---

## Symptom: 429s from the agent

**Expected under load, not a fault.** Admission control
(`deep_agent/aegra/admission.py`) bounds concurrent run starts per pod and
sheds the excess at the door. Reads, health and in-flight runs are never
gated.

- Confirm: `aegra_runs_in_flight` at or near `MAX_CONCURRENT_RUNS` on
  every pod.
- If the HPA is not adding pods, check `targetRunsInFlightPerPod` in
  `values.yaml` — CPU and memory targets do not track this load, because
  a turn spends its wall-clock time waiting on a model at low CPU.
- Raising `MAX_CONCURRENT_RUNS` without raising the Postgres pool bounds
  moves the failure from the door to the connection pool, which is the
  worse place for it. The arithmetic is in [scalability.md](scalability.md).

## Symptom: pods restarting, database unreachable

Liveness is on `/livez`, which does not touch Postgres, so a database
outage should **not** restart pods — it should make them unready and stop
traffic. If pods are restarting during a database incident, check that
nothing has repointed `livenessProbe` at `/health`; that turns a database
blip into a fleet-wide restart and a reconnection stampede against the
database that was already struggling.

## Symptom: runs stuck, never completing

- `REDIS_BROKER_ENABLED` must be `true`. A run dispatched without the
  Redis queue holds no lease, and a run with no lease cannot be reclaimed
  by the LeaseReaper when the pod holding it dies — crash recovery is
  silently a no-op.
- Check Redis reachability in `/health`.
- Orphaned runs from a dead pod should be reclaimed automatically; if
  they are not, the lease machinery is the place to look.

## Symptom: streams cut off mid-answer

The signal is `stream_errors_total` rising while HTTP 5xx stays flat —
these runs returned 200 and died later, so SLO 1 will not show it. Usual
causes:

- **Pods terminated during long turns.** Check the drain settings:
  `terminationGracePeriodSeconds` must exceed the longest turn you intend
  to let finish, and `preStopSleepSeconds` must exceed the time the router
  takes to stop sending new traffic.
- **HPA scale-down during a lull.** Scale-down is deliberately slow for
  this reason; if it has been made aggressive, every removed pod is a set
  of interrupted runs.
- **Model provider timeouts.** Langfuse shows which calls failed and
  against which provider.

## Symptom: MCP tools failing or asking for authorization repeatedly

- `/health` reports per-server MCP status; a failing server degrades the
  pod rather than taking it down.
- Token refresh is locked across pods via Redis. Repeated authorization
  prompts usually mean the lock is unavailable, i.e. Redis is.
- `CUSTOM_CA_URL` — an internal MCP server over HTTPS fails the handshake
  without the corporate trust bundle appended at container start.

## Symptom: config changes not taking effect

Config is mounted from a PVC at `CONFIG_PATH`, not baked into the image.
Check that the volume holds what you think it does, and that the reload
picked it up. A rollout is not required for a config change and should not
be used as the fix before you know the volume contents are right.

## Symptom: deploy stuck at "progressing"

`progressDeadlineSeconds` is 600 and `minReadySeconds` is 30, so a release
whose pods pass their probes and then fall over stalls after the first pod
instead of replacing the fleet. That is the intended behaviour — the
stalled rollout is the safety mechanism, not the bug.

- `oc rollout status deployment/agent` and `oc describe` the new pod.
- The `db-migrate` init container runs Alembic once per rollout. If it
  fails, no app pod starts. Check its logs first.
- To abandon: `oc rollout undo deployment/agent`.

---

## Procedure: releasing

A plain Deployment cannot do a real canary — that needs two workloads and
a traffic split, which is a platform decision. What this template gives
you is a rollout that fails safe:

1. `maxUnavailable: 0` — capacity never dips below the current replica
   count, so no in-flight turn is dropped to make room.
2. `maxSurge: 1` — one new pod at a time.
3. `minReadySeconds: 30` — it must stay ready for 30s before the next one
   starts. A release that is broken on its second minute stops after one
   pod.

While it rolls, split your SLI panels by `app.kubernetes.io/version` and
watch the new one. The label exists for exactly this.

If you need a true canary, run a second Deployment with a distinct
`app.kubernetes.io/version` and split traffic at the gateway, then compare
the same SLIs across the two labels.

## Procedure: rolling back

```
oc rollout undo deployment/agent
oc rollout status deployment/agent
```

Two things do not roll back with it, and both have bitten people:

- **Database migrations.** The init container has already applied them.
  Check whether the previous image can run against the new schema before
  you assume the rollback is clean.
- **Config on the PVC.** It is not versioned with the image. If the
  release included a config change, revert that separately.

## Procedure: after an incident

Write it up. The template is [postmortem.md](postmortem.md); it is
blameless by construction. The bar for writing one is "it surprised us",
not "it breached an SLO" — the near-misses are where the cheap lessons
are.
