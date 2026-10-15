---
name: merge-request-review
description: >
  Review a GitLab merge request — read the diff, check it against its
  issue and pipeline, and leave findings as draft notes. Use when given
  an MR URL or number, or asked to review, summarise or check a change.
---

# Merge Request Review

**Purpose:** Read a change properly and report what is wrong with it,
with enough precision that someone can act without re-reading the diff.

## When to Use

- An MR URL or `!number` is given
- "Review this", "what does this change do", "is this safe to merge"
- "Why is the pipeline failing on !482"

## Workflow

1. **Load the MR.** `get_merge_request` for title, description, author,
   target branch and state.
2. **Read the actual diff** — `list_merge_request_changed_files` first
   to see the shape, then `get_merge_request_diffs`, or
   `get_merge_request_file_diff` per file when the change is large.
   **Never review from the description.** A description says what the
   author believes they did; the diff says what they did, and the gap
   between those two is where most review findings live.
3. **Check the pipeline.** `list_merge_request_pipelines`. A red
   pipeline usually makes everything else secondary — lead with it.
4. **Check approvals and conflicts** when merge-readiness is the
   question: `get_merge_request_approval_state`,
   `get_merge_request_conflicts`.
5. **Trace the intent.** If the description or branch names an issue,
   `get_issue` it and ask whether the diff actually does that — and
   nothing else. Scope creep is a finding.
6. **Report.** See `references/review-rubric.md` for what counts as a
   finding and how to rank them.

## Leaving comments

Only when the user asks you to comment. Then:

- `create_draft_note` per finding, **not** `create_note`. Drafts let
  the user read everything before any of it is published; a direct
  note is visible to the author the instant it is wrong.
- `bulk_publish_draft_notes` only on explicit instruction. "Review
  this MR" is not permission to post on it.

## Guardrails

- **Never merge, approve, or close.** `merge_merge_request`,
  `approve_merge_request` and `delete_*` are outside this workflow
  regardless of how clearly the change passes. Approval is a human
  accountability signal; an agent cannot hold it.
- **Never push commits or edit files** to "just fix" something.
  Propose the change in a comment.
- **Say when the diff was too large to read in full**, and which files
  you skipped. A review that silently covered 3 of 40 files is worse
  than no review, because it is read as covering 40.
- **Distinguish defects from preferences.** A bug and a naming quibble
  do not belong in the same list at the same weight.
