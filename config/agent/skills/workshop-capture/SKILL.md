---
name: workshop-capture
description: >
  Read a Miro board and turn it into written output — decisions, themes,
  owners and open questions. Use when given a board URL or asked to
  summarise, write up or extract actions from a workshop or diagram.
---

# Workshop Capture

**Purpose:** Convert a board — which is spatial, unordered and full of
shorthand — into prose someone who was not there can act on.

## When to Use

- A Miro board URL, or "summarise the board / workshop / retro"
- "What did we decide?" "What were the actions?"
- Turning a diagram into a written description

## Workflow

1. **Find the board.** `board_search_boards`, or go straight there if
   given a URL. `user_who_am_i` when you need to know whose boards are
   visible.
2. **Read the content.** `canvas_read_as_svg` renders the board
   including its *layout*, which is most of the meaning: clusters are
   themes, arrows are sequence or causation, and a sticky's position
   often matters more than its text. `canvas_search` to locate
   specific content on a large board.
3. **Read the discussion.** `comment_list_comments` — objections and
   decisions frequently live in comments rather than on stickies, and
   a summary built from stickies alone reports the proposal as the
   outcome.
4. **Structure it.** Themes, then decisions, then actions, then open
   questions.

## Guardrails

- **Preserve the uncertainty.** Workshop output is mostly unresolved.
  A board of competing proposals summarised as a plan invents a
  consensus that the room did not reach — the single most damaging
  thing this skill can do.
- **Mark what is illegible.** Handwriting, cropped images and
  unlabelled arrows exist. List them rather than guessing; a guessed
  sticky is indistinguishable from a read one in the output.
- **Attribute only what is attributed.** Authorship on a board is
  often missing or wrong. Do not infer who said what from proximity
  or sticky colour.
- **Do not write to the board** unless asked. `canvas_update_from_svg`,
  `comment_create` and `board_create` change a shared artefact other
  people are looking at.

## Output

```
**Board.** <name, date if known>
**Themes.** <clusters, with what sits in each>
**Decided.** <only what is explicitly marked or stated as decided>
**Proposed, not decided.** <the rest — this section is usually longer>
**Actions.** <action — owner, if genuinely named>
**Open questions.** <including anything unreadable>
```
