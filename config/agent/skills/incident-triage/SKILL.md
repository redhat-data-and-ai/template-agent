---
name: incident-triage
description: >
  Investigate a production alert or outage by correlating observability
  signals with recent changes. Use when the user reports something is
  slow, erroring, or down, or pastes an alert, trace id or error string.
---

# Incident Triage

**Purpose:** Turn "something is broken" into a specific, evidenced
statement of what changed and what to do next.

## When to Use

- An alert, monitor name, trace id, or stack trace is pasted in
- "Why is X slow / erroring / down?"
- "Did anything deploy before this started?"
- A post-incident write-up is requested

Not for: routine metric questions with no suspicion of fault ("what was
p99 last week"). Answer those directly.

## The order matters

Resolve **time window** before anything else. Every query below is
scoped by it, and a window guessed wrong makes every subsequent
observation worthless while looking perfectly convincing.

1. **Establish the window.** When did it start? If the user does not
   say, find the first bad signal and use that; state the window you
   chose in the answer so it can be challenged.
2. **Characterise the symptom** from Datadog — error rate, latency,
   saturation. Get the *shape*: step change, ramp, or spike. A step
   change points at a deploy or config flip; a ramp points at a leak
   or growth; a spike points at load or a dependency.
3. **Find what changed** in the same window via GitLab. `list_commits`
   and `list_merge_requests` on the suspect project, then
   `list_merge_request_pipelines` to see what actually shipped.
   `get_commit_diff` on the candidate.
4. **Correlate, then say how strongly.** Co-occurrence in a window is
   not causation, and an incident write-up that states a cause it
   cannot support sends people to the wrong place for an hour.
5. **Record it** — see `references/writeup.md` for the shape.

## Guardrails

- **Never restart, redeploy, revert, merge or close anything.** This
  skill is read-only. Recommend the action and let a human take it.
  GitLab's write tools (`merge_merge_request`, `create_branch`,
  `push_files`) are not part of this workflow.
- **Quote real numbers.** "Error rate rose" is not a finding; "error
  rate went from 0.2% to 11% at 14:05 UTC" is. If you could not get
  the number, say the number is missing rather than characterising it.
- **Name the window in the answer.** Every figure is meaningless
  without it.
- **One suspect is a hypothesis, not a conclusion.** If two changes
  landed in the window, say both.
- **If an integration is not connected**, say which one and what you
  could not check. A triage missing its deploy history is a guess, and
  the user needs to know it was a guess.

## Output

Lead with the finding. Then evidence, then the recommendation.

```
**Finding.** <what is wrong, one sentence>
**Window.** <start–end, timezone>
**Evidence.** <numbers, with where each came from>
**Likely cause.** <change, with confidence and why>
**Recommended next step.** <the action a human should take>
**Not checked.** <integrations unavailable, if any>
```
