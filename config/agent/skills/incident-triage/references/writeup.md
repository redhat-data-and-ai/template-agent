# Incident write-up

Read this when the user asks for a post-incident note, a summary to
paste into a ticket, or a handover.

## Shape

```markdown
## <service> — <symptom>, <date>

**Impact.** Who or what was affected, and for how long. Users, not
components: "checkout failed for ~4% of sessions" beats "pod CPU
throttled".

**Timeline.** UTC, newest last.
| Time | Event |
|---|---|
| 14:02 | MR !482 merged to main |
| 14:05 | Error rate 0.2% → 11% |
| 14:31 | Rollback to v2.9.1 |
| 14:34 | Error rate back to baseline |

**Cause.** One paragraph. Say "likely" unless the evidence is
conclusive — a confident wrong cause costs more than an honest
uncertain one, because nobody re-checks a stated fact.

**Evidence.** Links and figures. Every claim in Cause traces to a row
here.

**What is not known.** The questions this investigation could not
answer, and why. A write-up without this section reads as complete
when it is not.
```

## Rules that are easy to get wrong

- **Timestamps in UTC**, always, with the offset of any local time
  quoted alongside. An incident spanning a timezone boundary where
  half the entries are local is unreadable afterwards.
- **Impact before cause.** The first question anyone asks is "who was
  hurt", not "which commit".
- **Do not assign blame to a person.** Name the change, not the
  author. `!482` is the subject; the author is incidental and naming
  them changes how the write-up gets read.
- **Distinguish trigger from cause.** The deploy that surfaced a
  latent bug is the trigger; the bug is the cause. Conflating them
  produces "cause: deployed code", which prevents nothing.
- **Leave the remediation to humans.** Suggest; do not schedule, merge
  or revert.
