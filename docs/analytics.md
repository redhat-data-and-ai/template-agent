# Usage analytics

`GET /analytics/*`, gated on the same privileged LDAP roles as
`/admin` and `/evals`. The UI renders it at `/analytics`.

| Endpoint | Answers |
|---|---|
| `/analytics/summary?days=7` | Is it used, does it work, how slow, what does it cost, is anyone happy |
| `/analytics/timeseries?days=30&bucket=day` | Runs, errors and p95 over time |
| `/analytics/models?days=7` | Per-model volume, tokens, p95, errors |
| `/analytics/errors?days=7` | Failures by kind |
| `/analytics/users?days=7` | Per-person usage — **404 unless enabled** |

## No write path

Everything is a query over `runs`, `thread` and `message_feedback` —
rows the platform already produces as a side effect of working. There
is no events table, no collector, no background job.

open-webui maintains a separate `chat_messages` table to back its
analytics. A second write path is a second thing to keep correct, and
when the two disagree the dashboard is the one nobody checks against
reality. A query cannot drift from its source.

The cost of that choice is that analytics is bounded by what Aegra
already records. Tokens come out of
`runs.output -> messages[] -> usage_metadata`, so a run that failed
before any model call contributes no tokens and reports its model as
`unknown`. That is accurate, and the alternative was instrumenting
every call site.

## No message content

No endpoint returns anyone's text. open-webui's analytics API includes
one that returns users' actual messages to an admin; on an internal
platform that is a surveillance surface wearing a dashboard's clothes,
and a template should not hand that to a fork by default.

**Error strings are classified, not echoed.** An exception routinely
carries the input that produced it — a filename, a path, a prompt
fragment. `/analytics/errors` returns a kind and a count:

| Class | Means |
|---|---|
| `model_config` | Credentials or model name wrong — fixed in config |
| `model_rate_limit` | Provider pushed back |
| `model_timeout` | Model did not answer in time |
| `model_refused` | Safety, recitation, or an unusable finish reason |
| `context_too_long` | Conversation outgrew the window |
| `retries_exhausted` | Gave up after repeated worker failures |
| `storage` | Postgres or Redis unreachable |
| `agent_defect` | A Python exception reached the user — our bug |
| `tool_failure` | A tool or MCP call failed |
| `cancelled` | The user stopped it |

The classifier is grounded in failures this deployment actually
produced. Its first version sorted all 17 errors in the live database
into `other`, which is the useful thing a classifier can tell you about
itself — the three real classes it was missing were `model_config`,
`retries_exhausted` and `agent_defect`.

`agent_defect` is separate on purpose: a stack trace reaching a user is
not the same incident as a rate limit, and a dashboard that buckets
them together will have you tuning quota while the bug stays.

## Per-user figures are off by default

```yaml
analytics:
  expose_users: false   # default
```

Cost attribution is a real need, which is why `/analytics/users`
exists. "Who talked to the assistant most this month" is also a
management report nobody consented to, and on an internal tool the
difference between those two is a config flag somebody should have to
set deliberately. Aggregate figures are unaffected.

When disabled the endpoint returns **404 with the reason**, not an
empty list — an empty list reads as "nobody used it".

## Statistics that are not invented

- **`success_rate` is `null`, not 0 or 100, when nothing has finished.**
  A fresh deployment has no success rate, and putting a green number on
  a system nobody has used is worse than a dash.
- **In-flight runs never count as failures.** Otherwise every dashboard
  looks worse the busier the system gets.
- **Percentiles cover finished runs only.** An in-flight run's elapsed
  time is not its duration; including it drags p95 toward "however long
  ago the request arrived".
- **Empty time buckets are emitted as zero**, not skipped. A gap in a
  chart reads as missing data, not as a quiet weekend.
- **A failed query returns 503, not zeroes.** A flat chart reads as
  "nobody used it", and somebody will act on that.

## Windows

`days` is capped at 365. Unbounded, a dashboard refresh is a sequential
scan of every run ever recorded; `runs` is indexed on `created_at`, so
a bounded window is an index scan.
