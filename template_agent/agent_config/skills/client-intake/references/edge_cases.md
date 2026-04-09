# Edge Cases and Special Handling

## Age and Pregnancy Restrictions

| Situation | Action |
|-----------|--------|
| User is under 18 | Advise that standard BMI may not apply. Recommend consulting a healthcare professional. **Do not route to analyst.** |
| User is pregnant | Advise that standard BMI does not apply during pregnancy. Recommend consulting a healthcare professional. **Do not route to analyst.** |

**Rationale:** BMI calculations are designed for adult, non-pregnant populations. Different standards apply to children and pregnant individuals.

## Unrealistic Goals

| Situation | Response |
|-----------|----------|
| Unrealistic timeline (e.g., lose 20 kg in 1 week) | Explain that safe weight loss is 0.5–1 kg/week. Offer a realistic alternative timeline before routing to analyst. |
| Extreme dietary restrictions | Acknowledge concern, emphasize balanced nutrition, suggest consulting a nutritionist. |

**Safe weight loss rate:** 0.5–1 kg per week (1–2 lbs per week)

## Duplicate Measurements

| Situation | Action |
|-----------|--------|
| Identical height/weight within same conversation | Acknowledge that values haven't changed. Skip re-analysis unless user explicitly requests it. |
| Similar but not identical measurements | Treat as new measurement and route to analyst. |

**Check window:** Current conversation only (not across sessions)

## Missing or Incomplete Data

| Missing Field | Action |
|---------------|--------|
| Height only | Ask for weight before routing |
| Weight only | Ask for height before routing |
| Units unclear | Ask user to clarify (metric or imperial) |
| Partial imperial (e.g., "6 feet" without inches) | Treat missing inches as 0 |

**Never guess or estimate missing values.**

## Invalid Measurements

| Issue | Action |
|-------|--------|
| Negative values | Ask user to re-enter (likely typo) |
| Extreme outliers (e.g., height > 300cm, weight > 500kg) | Confirm with user before routing |
| Zero values | Ask user to provide valid measurement |

**Validation thresholds:**
- Height: 50–300 cm (reasonable human range)
- Weight: 10–500 kg (reasonable human range)
