---
name: issue-tracking
description: >
  Work with Jira issues and Confluence pages — find them, read them,
  report status. Use for a Jira key, a JQL-shaped question, "what is
  the status of", or finding the page that documents something.
---

# Issue Tracking

**Purpose:** Answer a question from Jira or Confluence precisely
enough that the user does not have to open either.

Distinct from **workspace-briefing**, which assembles a short brief
across calendar, mail and tickets. This skill is for going deep on
Jira and Confluence themselves.

## When to Use

- A Jira key (`RHITAIF-123`), or "what is assigned to me"
- "What is the status of / who owns / when did we close …"
- "Where is the page about X?"
- Summarising a Confluence page

## Workflow

1. **Resolve the site first.** `getAccessibleAtlassianResources` —
   a cloud id is required by most other calls, and guessing it fails
   in a way that reads as "no access" rather than "wrong site".
2. **Fetch exactly what was named.** `getJiraIssue` for a key,
   `getConfluenceContent` for a page. Searching for something you
   were given the identifier of wastes a call and can return a
   different object with a similar title.
3. **Search by JQL when the question is a set** —
   `searchJiraIssuesUsingJql`. Scope it: project, assignee, status
   and an updated-within bound. An unbounded JQL returns volume
   instead of an answer.
4. **`searchConfluence` for prose**, then fetch the page. Search
   results carry excerpts, which read as answers and are not.
5. **Report status with its holder and its date.** "In Progress" on
   its own does not tell anyone whether to wait or chase.

## Guardrails

- **Read-only.** `createJiraIssue`, `editJiraIssue`,
  `transitionJiraIssue`, `addOrEditJiraIssueComment`,
  `createConfluenceContent` and `updateConfluenceContent` are not
  part of this workflow. Transitioning someone's issue rewrites a
  board other people are standing in front of.
- **`executeWrite` and `executeDestructive` are never used here**,
  whatever the task looks like. They exist to run arbitrary
  operations and bypass every specific guardrail above by design.
  `executeRead` is acceptable when no specific tool fits.
- **Quote the issue key and the page id** in the answer. A status
  without the record it came from cannot be checked.
- **A stale page is not a wrong page.** Give the last-updated date
  and let the reader judge; Confluence is full of accurate documents
  describing a system that has since changed.
- **An empty JQL result may be a permissions boundary**, not an
  absence. Say which you cannot distinguish.
