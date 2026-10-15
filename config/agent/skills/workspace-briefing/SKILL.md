---
name: workspace-briefing
description: >
  Assemble a brief from Google Workspace and Atlassian — calendar, mail,
  Jira issues and Confluence pages. Use for meeting prep, catch-up after
  time away, status on a project or person, or "what do I need to know".
---

# Workspace Briefing

**Purpose:** Gather scattered context into one short brief, and be
explicit about what it does not cover.

## When to Use

- "Prep me for my 3pm" / "what's this meeting about"
- "What did I miss?" after leave
- "Where is <project> up to?"
- "What's assigned to me / to <person>?"

## Tool discovery

Unlike the GitLab, Atlan and Miro workflows, this skill does **not**
name specific tools. The Google Workspace and Atlassian servers expose
different tool sets depending on the scopes the user granted at
connect time, so a hardcoded tool name here would be right for one
user and wrong for the next. Look at the tools you actually have under
the `gworkspace` and `atlassian` prefixes and use those.

If a capability is missing, that is a scope the user did not grant —
say so rather than working around it.

## Workflow

1. **Fix the window and the subject** before fetching anything. "What
   did I miss" over four days is a different query from over four
   weeks, and an unscoped sweep returns volume instead of a brief.
2. **Calendar first** when the trigger is a meeting — attendees,
   agenda and any attached document define what is relevant. Without
   it you are summarising the inbox.
3. **Mail and issues, filtered to the subject.** Threads the user is
   on, issues they own or are mentioned in.
4. **Confluence last**, for background that has not changed recently.
   It is reference, not news, and leading with it buries the new
   information.
5. **Compress hard.** A brief that takes as long to read as the
   sources has failed. Aim for something readable in under a minute.

## Guardrails

- **Never send, reply, accept, decline or create** anything. This
  skill reads. Drafting text for the user to send is fine; putting it
  in someone else's inbox or calendar is not.
- **This is the user's private correspondence.** Summarise it to them.
  Do not carry its content into other tools — do not paste mail into
  a Jira comment or a Miro board, even when asked to "put this
  somewhere", without the user confirming that specific destination.
- **Say what you could not see.** A private calendar, an unshared
  page, a mailbox scope not granted. A brief that silently omits the
  one meeting that mattered is actively misleading.
- **Do not infer sentiment or intent** from terse mail. "Fine, do it"
  is not necessarily agreement, and a brief that reports it as such
  misrepresents a colleague.
- **Quote dates and names exactly.** A brief is used to walk into a
  room; a wrong name or time costs the user credibility directly.

## Output

```
**Subject.** <meeting / project / period>
**The short version.** <two or three sentences>
**Recent activity.** <what changed, newest first, with dates>
**Needs you.** <decisions or replies waiting on the user>
**Background.** <standing context, briefly>
**Not covered.** <sources unavailable or out of scope>
```
