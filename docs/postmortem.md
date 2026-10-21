# Postmortem template

Copy this into a new document per incident. Do not fill it in here.

**Blameless means blameless.** Name systems, not people. Where a person
acted, describe what information they had at the time and why the action
was reasonable given it — if it was not reasonable, the gap is in the
tooling, the docs, or the alert, and that is the finding. A postmortem
that identifies someone as the cause has stopped early: it has found who
was holding the system when it broke, not why the system let it.

Write one whenever an incident surprised you. Not only when an SLO burned
— the near-misses are the cheap lessons, and they are the ones a
budget-triggered rule never captures.

---

# Incident: `<short description>`

| | |
|---|---|
| Date | `YYYY-MM-DD` |
| Authors | |
| Status | draft / in review / final |
| Severity | |
| Error budget consumed | |

## Summary

Two or three sentences. What broke, who it affected, how long, how it was
resolved. Written so that someone who was not there, reading it in a
year, gets the shape of it before the detail.

## Impact

- Users affected, and how they experienced it. For an agent this is
  rarely a clean "down": turns that errored, turns that silently
  truncated mid-stream, answers that were wrong or stale.
- Duration from first user impact — not from first alert. The gap between
  the two is itself a finding.
- Which SLO, and how much budget.

## Timeline

All times UTC. Include when people *learned* things, not only when things
happened; the lag between an event and its discovery is usually where the
improvement is.

| Time | Event |
|---|---|
| | first impact (often established retrospectively) |
| | detection — alert, or a human noticing, and say which |
| | escalation |
| | mitigation |
| | resolution |

## Root cause

What actually happened, mechanically. Keep going past the first
plausible stopping point: "Postgres was slow" is a symptom. Why was it
slow, why did the agent respond to slowness that way, and why did nothing
catch it earlier?

There is frequently more than one cause, and the interesting ones are
usually a latent condition plus a trigger.

## Detection

How did you find out? If a user told you before monitoring did, that is
the most important finding in the document and it belongs here rather
than buried in the actions.

## Response

What was tried, in order, including what did not work. The failed
attempts are the most useful part of a postmortem for the next person —
they are the only record of the hypotheses that looked right and were
not.

Was the runbook used? Was it right? Did it exist?

## What went well

Not a formality. Mechanisms that worked — the load shedding held, the
rollout stalled instead of completing, the lease reaper recovered the
orphaned runs — are the things to protect when someone later proposes
removing them for being noisy.

## What went wrong

Conditions, not people.

## Where we got lucky

The things that would have made this much worse and happened not to
apply. These are the next incident, described in advance.

## Action items

Each one owned, dated, and tracked somewhere that is not this document.
Prefer actions that remove a class of failure over actions that remove
this instance of it. Be honest about which are which — "add an alert" is
usually a detection fix, not a prevention one, and labelling it correctly
stops the list from looking more complete than it is.

| Action | Type | Owner | Issue | Due |
|---|---|---|---|---|
| | prevent / detect / mitigate / process | | | |

## Lessons

What someone who did not live through this should take away.
