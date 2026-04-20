---
name: client-intake
description: >
  Utility skill for gathering and normalizing health metrics (height, weight).
  Handles input parsing, validation, and unit conversion from imperial to metric.
  Use when you need to collect or convert user measurements.
---

# Client Intake

**Purpose:** Parse user input and convert health measurements to standardized metric units.

## When to Use

- Gathering height and weight from user input
- Converting imperial units (ft, in, lbs) to metric (cm, kg)
- Validating measurement inputs
- Standardizing measurements before analysis

## Resources

- **Input Gathering Guidelines:** `references/input_gathering.md`
- **Unit Conversion Formulas:** `references/unit_conversion_formulas.md`
- **Conversion Script:** `scripts/convert_units.py`
- **Edge Cases & Validation:** `references/edge_cases.md`

## Core Workflow

1. **Parse Input:** Extract height and weight from user message
   - Height may be in: cm, ft+in, or inches
   - Weight may be in: kg or lbs

2. **Validate:** Ensure both measurements are present and reasonable
   - Height: 50-272 cm (or equivalent)
   - Weight: 20-300 kg (or equivalent)

3. **Convert to Metric:** If measurements are in imperial units, convert to metric
   - See `references/unit_conversion_formulas.md` for exact formulas
   - Or use `scripts/convert_units.py` for programmatic conversion

4. **Return Standardized Values:** Always return height in cm and weight in kg

## Unit Conversion

**Use the conversion script, never hardcode conversions:**

```bash
# Convert height (5 feet 10 inches to cm):
python3 scripts/convert_units.py feet_inches 5 10

# Convert weight (180 pounds to kg):
python3 scripts/convert_units.py pounds 180

# Other supported conversions:
python3 scripts/convert_units.py feet 6        # feet to cm
python3 scripts/convert_units.py inches 70     # inches to cm
```

## Input Validation

**Required fields:**
- Height (must be positive, within 50-272 cm range)
- Weight (must be positive, within 20-300 kg range)

**Edge cases to handle:**
- Mixed units (e.g., "5 feet 10 inches and 80 kg")
- Implicit units (e.g., "5'10" assumes imperial)
- Missing measurements (prompt user for both)
- Out-of-range values (ask user to verify)

## Output Format

Always return measurements in this format:
- **Height:** `<value>` cm
- **Weight:** `<value>` kg

## Gotchas

- **Use `python3` not `python`** — python command is not available on all systems
- **Always convert before returning** — downstream processes expect metric units (cm, kg) only
- **Don't assume units** — if ambiguous, ask user to clarify (e.g., "Is that 180 lbs or kg?")
- **Validate ranges** — catch unrealistic measurements early (e.g., 1000 cm height)
