# Edge Cases and Special Handling

## Age and Pregnancy Restrictions

| Situation | Validation Result |
|-----------|-------------------|
| User is under 18 | **INVALID** - Advise that standard BMI may not apply. Recommend consulting a healthcare professional. |
| User is pregnant | **INVALID** - Advise that standard BMI does not apply during pregnancy. Recommend consulting a healthcare professional. |

**Rationale:** BMI calculations are designed for adult, non-pregnant populations. Different standards apply to children and pregnant individuals.

## Unrealistic Goals

| Situation | Response |
|-----------|----------|
| Unrealistic timeline (e.g., lose 20 kg in 1 week) | Explain that safe weight loss is 0.5–1 kg/week. Offer a realistic alternative timeline. |
| Extreme dietary restrictions | Acknowledge concern, emphasize balanced nutrition, suggest consulting a nutritionist. |

**Safe weight loss rate:** 0.5–1 kg per week (1–2 lbs per week)

## Duplicate Measurements

| Situation | Action |
|-----------|--------|
| Identical height/weight within same conversation | **Flag as duplicate** - Acknowledge that values haven't changed. |
| Similar but not identical measurements | **Accept as new** - Treat as fresh measurement. |

**Check window:** Current conversation only (not across sessions)

## Missing or Incomplete Data

| Missing Field | Action |
|---------------|--------|
| Height only | **Incomplete** - Prompt for weight |
| Weight only | **Incomplete** - Prompt for height |
| Units unclear | **Ambiguous** - Ask user to clarify (metric or imperial) |
| Partial imperial (e.g., "6 feet" without inches) | **Auto-complete** - Treat missing inches as 0 |

**Never guess or estimate missing values.**

## Invalid Measurements

| Issue | Action |
|-------|--------|
| Negative values | **Invalid** - Ask user to re-enter (likely typo) |
| Extreme outliers (e.g., height > 300cm, weight > 500kg) | **Suspicious** - Confirm with user |
| Zero values | **Invalid** - Ask user to provide valid measurement |

**Validation thresholds:**
- Height: 50–300 cm (reasonable human range)
- Weight: 10–500 kg (reasonable human range)
