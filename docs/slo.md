# Service level objectives

**This is a skeleton. The SLIs are specified; the targets are `TODO` and
only you can fill them in.** A target copied from a template is a number
nobody agreed to, which is worse than no number at all — it will be
missed, and nobody will care, and then no SLO in the repo will mean
anything.

Fill in the targets with whoever is accountable for this agent being up,
then delete this paragraph.

## Why these four

An agent is not a CRUD service and the usual latency SLI misleads badly
here. A turn legitimately takes two minutes; the user has not been failed
by that. What fails them is a turn that errors, a turn that sits in
silence before the first token, or an answer that is wrong. So:
availability, time to first token, stream completion, and quality.

Everything below is measured from metrics in `deep_agent/aegra/otel.py`
and therefore needs `ENABLE_OTEL=true` — see
[observability.md](observability.md). The metric prefix is derived from
the service name, so substitute yours.

---

## SLO 1 — Availability

**SLI.** The proportion of HTTP requests that do not return 5xx, measured
at the agent, over a 28-day rolling window.

```
sum(rate(http_server_duration_count{status_code!~"5.."}[5m]))
  / sum(rate(http_server_duration_count[5m]))
```

From FastAPI auto-instrumentation (`tracing.fastapi_auto_instrument`).

**Exclusions, and why.** 429 from the admission controller
(`deep_agent/aegra/admission.py`) counts as *served*. Shedding at the door
is the system working: the alternative is accepting the request and
failing it three minutes in, having also degraded everyone already in
flight. Do not let load shedding burn the error budget, or the budget will
push you toward removing the shedding.

4xx generally counts as served. A rejected malformed request is not an
outage.

**Target.** `TODO` — e.g. 99.5% over 28 days. Consider what the gateway
promises upstream; this agent cannot be more available than it is.

**Error budget.** `TODO` — at 99.5%, 3h 22m per 28 days.

---

## SLO 2 — Time to first token

**SLI.** The proportion of runs whose first token arrives within the
threshold.

```
histogram_quantile(0.95,
  sum(rate(<prefix>_time_to_first_token_seconds_bucket[5m])) by (le))
```

**Why this and not total duration.** This is the number the user
experiences as "is it broken". Total turn duration is governed by how hard
the question was, and alerting on it alerts you to users asking harder
questions. TTFT is governed by you: model routing, cold caches, MCP
handshakes, queue depth.

**Target.** `TODO` — e.g. p95 under 3s. Measure your actual distribution
for a fortnight before committing; the right target is roughly today's p95
if today's p95 is acceptable, and a project if it is not.

---

## SLO 3 — Stream completion

**SLI.** The proportion of started runs that finish without a stream
error.

```
1 - (sum(rate(<prefix>_stream_errors_total[5m]))
       / sum(rate(<prefix>_conversations_total[5m])))
```

**Why it is separate from availability.** A stream that dies at token 400
returned HTTP 200 and is invisible to SLO 1. From the user's side it is a
total failure — worse than an error, because the partial answer looks like
a complete one. This is the SLI that catches pod evictions during long
turns, model provider timeouts, and the drain window being too short.

**Target.** `TODO` — e.g. 99.0%.

---

## SLO 4 — Answer quality

**SLI.** The proportion of eval cases passing on the committed suites
(`config/agent/skills/*/evals/`, run by `eval-runner/`), plus the
proportion of user feedback scores that are positive
(`deep_agent/aegra/feedback.py` → Langfuse scores).

**Why an SLO at all.** An agent that is up, fast, and confidently wrong
has met every objective above. This is the only one the other three cannot
substitute for, and it is the reason the eval suite is in the repo rather
than in someone's notebook.

**Target.** `TODO` — e.g. no regression against the previous release, and
≥ 90% on the committed suites.

**Caveat.** Not a paging signal. It is slow, noisy at low volume, and a
dip is a conversation to have, not an incident to declare.

---

## Alerting

Do not alert on every SLI breach — alert on burn rate, so that a brief
blip stays quiet and a sustained one pages before the budget is gone. The
standard two-window pair, per SLO:

| Window | Burn rate | Action | Catches |
|---|---|---|---|
| 1h and 5m | 14.4x | page | fast burn — 2% of a 28d budget in an hour |
| 6h and 30m | 6x | page | medium burn |
| 24h and 2h | 3x | ticket | slow burn |

Both windows in a row must be breaching, which is what stops a 90-second
blip from paging. Define these in your monitoring backend, not here; the
template has no credentials and should not acquire any.

Each alert must link to the matching section of
[runbook.md](runbook.md). An alert with no runbook link is a page that
starts with ten minutes of orientation.

## Review

Revisit these quarterly, and after any incident that the SLOs did not
catch. An SLO that has never been close to breaching is not evidence of a
reliable service; it is usually evidence of a target set too low to be
informative.
