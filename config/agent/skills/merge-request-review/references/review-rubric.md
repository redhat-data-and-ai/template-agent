# What counts as a finding

Rank by what happens if it ships unchanged.

| Rank | Means | Examples |
|---|---|---|
| **Blocking** | Breaks correctness, security or data | Unvalidated input reaching a query; a dropped `await`; an auth check behind a feature flag defaulting open; a migration with no rollback |
| **Should fix** | Works, but will cost someone later | Unbounded query in a request path; an error swallowed into a generic message; a retry without backoff |
| **Worth noting** | Fine to merge, worth saying | Dead code left behind; a test that asserts the mock rather than the behaviour |
| **Preference** | Say it at most once, collectively | Naming, formatting, file layout |

Do not pad. Three real findings beat three real findings plus nine
observations, because the padding is what gets the list skimmed.

## Things reliably missed in diffs

- **The absent case.** A new branch with no matching test; a new error
  path that nothing exercises. Diffs show what was added, so what was
  *not* added is invisible unless deliberately looked for.
- **Call sites.** A changed signature or semantics with callers left
  alone. The diff shows the definition, not who depends on it.
- **Config and migrations.** A code change that needs a config key or
  schema change that is not in the MR, so it passes CI and fails on
  deploy.
- **Partial renames.** The old name still live somewhere, often in a
  string or a config file the rename did not touch.
- **Widened permissions.** A role, scope or CORS origin quietly
  broadened as a side effect of something else.

## Writing a finding

```
**<file>:<line> — <one-line claim>**
<What breaks, concretely: the input or state, and the wrong result.>
<Suggested change.>
```

A finding without a concrete failure is a feeling. If you cannot say
what input makes it go wrong, either dig until you can or drop it to
"worth noting" and say it is a smell rather than a defect.

## Tone

Address the change, not the author. "This drops the error" rather than
"you dropped the error" — the same information, and it survives being
read by someone having a bad day.
