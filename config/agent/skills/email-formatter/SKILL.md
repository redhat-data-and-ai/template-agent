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

## Core Workflow

1. **Parse Input:** Extract BMI data, health tips, and any other content
2. **Format as HTML:** Create HTML email structure with inline CSS
3. **Include All Required Sections:**
   - Email body opening
   - BMI result
   - Health tips (if provided)
   - **Disclaimer footer (MANDATORY - never skip)**
4. **Return Complete HTML:** Full HTML email ready for `send_email`

## Critical Requirements

**MUST DO:**
- ✅ **Use HTML tags** — `<p>`, `<strong>`, `<br>`, etc. (NOT plain text or markdown)
- ✅ **Inline CSS only** — `style="..."` on every element (Gmail strips `<style>` blocks)
- ✅ **Include disclaimer footer** — MANDATORY in every email, no exceptions
- ✅ **Max width 600px** — for mobile compatibility

**OPTIONAL:**
- Skip empty sections (e.g., if no health tips provided, don't show empty tips section)

## Output Format

**CRITICAL: Always return HTML, never plain text.**

Minimum structure (even for simple emails):
```html
<div style="max-width: 600px; font-family: Arial, sans-serif;">
  <p>Hello,</p>

  <p><strong>Your BMI:</strong> <value> (<category>)</p>

  <!-- Health tips section only if tips are provided -->

  <p style="font-size: 12px; color: #666; margin-top: 20px;">
    <strong>Disclaimer:</strong> This information is for general purposes only
    and is not a substitute for professional medical advice.
  </p>
</div>
```

**Example with tips:**
```html
<div style="max-width: 600px; font-family: Arial, sans-serif;">
  <p>Hello,</p>
  <p><strong>Your BMI:</strong> 22.5 (Normal)</p>
  <p><strong>Health Tips:</strong></p>
  <ul style="line-height: 1.6;">
    <li>Continue balanced nutrition</li>
    <li>Maintain regular exercise</li>
    <li>Stay hydrated</li>
  </ul>
  <p style="font-size: 12px; color: #666; margin-top: 20px;">
    <strong>Disclaimer:</strong> This information is for general purposes only
    and is not a substitute for professional medical advice.
  </p>
</div>
```

**Example without tips (minimal):**
```html
<div style="max-width: 600px; font-family: Arial, sans-serif;">
  <p>Hello,</p>
  <p><strong>Your BMI:</strong> 28.3 (Overweight)</p>
  <p style="font-size: 12px; color: #666; margin-top: 20px;">
    <strong>Disclaimer:</strong> This information is for general purposes only
    and is not a substitute for professional medical advice.
  </p>
</div>
```

## What NOT to Do

❌ **DO NOT return plain text** — Always use HTML tags
❌ **DO NOT skip the disclaimer** — It's legally required in every email
❌ **DO NOT use markdown** — Use HTML (`<p>`, `<strong>`) not markdown (`**`, `#`)
❌ **DO NOT use `<style>` blocks** — Use inline styles only

## Gotchas

- **Gmail strips `<style>` blocks and CSS classes** — all styles must be inline on every element
- **Always include the disclaimer** — it's mandatory in every email, never skip it
- **Skip sections without data** — don't render empty placeholders or "N/A" messages for missing content
- **Even simple emails need HTML** — Don't fall back to plain text for brevity
