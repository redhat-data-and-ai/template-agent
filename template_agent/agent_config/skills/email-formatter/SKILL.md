---
name: email-formatter
description: >
  Provides Gmail-compatible HTML template and inline CSS rules for formatting
  fitness report emails. Use when formatting and sending reports via email
  with send_email.
---

# Email Formatter — Gmail-Compatible HTML

Format fitness reports as HTML emails with inline CSS only.

## When to Use

When sending a fitness report via `send_email`. Gmail strips `<style>` blocks and CSS classes.

## Resources

- **HTML Template:** `assets/template.html`
- **Email Structure:** `references/email_structure.md`
- **CSS Styling Rules:** `references/inline_css_rules.md`

## Critical Requirements

- **Inline CSS only** — no `<style>` tags or CSS classes (Gmail strips them)
- **Include disclaimer footer** — mandatory in every email
- **Skip empty sections** — only render sections with data, no placeholders
- **Max width 600px** — mobile email client compatibility

## Gotchas

- **Gmail strips `<style>` blocks and CSS classes** — all styles must be inline on every element
- **Always include the disclaimer** — it's mandatory in every email, never skip it
- **Skip sections without data** — don't render empty placeholders or "N/A" messages for missing content
