---
name: bmi-report
description: >
  Defines BMI categories, report structure, and tone guidelines for BMI
  analysis reports. Use when generating a BMI report for a user.
---

# BMI Report Content

Generate consistent, category-specific BMI reports based on calculated values.

## When to Use

When generating a BMI analysis report for a user based on calculated BMI value and category.

## BMI Categories

| BMI Range | Category |
|-----------|----------|
| < 18.5 | Underweight |
| 18.5–24.9 | Normal |
| 25–29.9 | Overweight |
| ≥ 30 | Obese |

## Resources

- **Categories & Ranges:** `references/bmi_categories.md`
- **Health Tips:** `references/health_tips_by_category.md`
- **Report Structure & Tone:** `assets/report_template.txt`

## Critical Requirements

- **Always include disclaimer** — mandatory in every report
- **Tips must be category-specific** — see health_tips_by_category.md
- **Tone: friendly, encouraging, non-judgmental** — never use "bad" or "failing"
