# Email Structure

The email has three layers: **header**, **body sections**, and **footer**.

## Fixed Sections (Always Present)

1. **Header** — Red Hat branded banner
2. **Disclaimer** — footer with medical advice disclaimer

## Dynamic Sections (Include Only What is Provided)

Each piece of content provided as input gets its own section in the body.

**Skip any section whose data was not provided — never leave an empty section or placeholder.**

### Common Sections

| Section | When to Include | Content Format |
|---------|----------------|----------------|
| BMI Result | BMI value + category provided | Value, category, one-line interpretation in a highlight box |
| Health Tips | Tips list provided | Actionable tips as a bullet list |

### Rendering Additional Sections

This list is not exhaustive. If the input contains additional named sections not listed above, render them using the same styling conventions:
- Section heading (h3 with Red Hat color)
- Content using appropriate HTML tags (p, ul/li, table, strong)
- Section divider (hr) after content

## Assembly Order

1. Header (fixed)
2. BMI Result (if provided)
3. Health Tips (if provided)
4. Additional sections (if provided)
5. Disclaimer footer (fixed)
