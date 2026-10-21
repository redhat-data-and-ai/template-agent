# scale-to-zero

Scales the agent to **0 replicas** when idle, waking it on demand.

Opt in from an overlay:

```yaml
components:
  - ../../components/scale-to-zero
```

## Cluster prerequisites

- Custom Metrics Autoscaler operator (KEDA)
- KEDA **http-add-on** — mandatory, see below
- Prometheus scraping the agent, if using the in-flight guard

## Why the HTTP add-on is mandatory

A Redis queue-depth trigger deadlocks: at zero replicas no pod receives
the HTTP request, so nothing enqueues a run, so the queue stays empty
and the workload never wakes. The interceptor instead accepts the
connection, holds it, triggers the scale-up, and forwards once Ready.

## Scaling down safely

`scaledownPeriod` alone is a guess — it has to be longer than the
longest run you can imagine, which means idling pods for that long
after every burst.

`in-flight-scaledobject.yaml` replaces the guess with a measurement:
the agent exposes `aegra_runs_in_flight`, and KEDA keeps at least one
replica while it is above zero. HTTP activation and the in-flight guard
live in **one** `ScaledObject` via the add-on's external scaler, because
two autoscalers on one Deployment oscillate.

Use `httpscaledobject.yaml` (simpler, timer-based) **or**
`in-flight-scaledobject.yaml` (metric-based), not both — the
kustomization ships the metric-based one.

## Cron

Aegra's scheduler claims crons whose `next_run_date` is in the past, so
nothing is lost while the pod sleeps — it is deferred. `agent-wakeup`
pokes `/health` every five minutes so overdue jobs fire; a cron is late
by at most one poke interval plus the cold start.

## Costs

| | |
|---|---|
| Cold start | ~60s on the first request |
| Cron latency | Up to one poke interval + cold start |
| In-flight runs | Protected by the gauge; without it, severed on scale-in |
