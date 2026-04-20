# Input Gathering Guidelines

## Data Collection

### Required Measurements

Both measurements must be collected:
- **Height** (accept in cm, ft+in, or inches)
- **Weight** (accept in kg or lbs)

If either is missing: prompt user. Never guess or estimate.

### Optional Information

- **Email address** — only collect if explicitly mentioned
- Do NOT ask proactively — wait for user to mention it

## Input Parsing

### Height Formats

Accept any of these formats:
- `175 cm` or `175cm`
- `5 feet 10 inches`, `5'10"`, `5ft 10in`
- `70 inches` or `70"`

### Weight Formats

Accept any of these formats:
- `70 kg` or `70kg`
- `180 lbs`, `180 pounds`, `180lb`

### Edge Cases

- **Missing inches:** treat as 0 (e.g., "6ft" → 6 feet 0 inches)
- **Decimal values:** accept and convert precisely (e.g., "5.5 feet" → 167.64 cm)
- **Mixed units:** handle gracefully (e.g., "5'10 and 80kg")
- **Ambiguous input:** if unclear, ask user to clarify

## Unit Conversion

Always convert to metric before returning:
- **Target:** Height in cm, Weight in kg
- **Methods:** See `unit_conversion_formulas.md` or use `scripts/convert_units.py`

### Conversion Priority

1. Use `scripts/convert_units.py` for programmatic conversion
2. Use formulas from `unit_conversion_formulas.md` for manual calculation
3. Never hardcode conversion factors inline

## Validation

### Sanity Checks

Apply reasonable ranges:
- **Height:** 50-272 cm (approx. 1.6-9 feet)
- **Weight:** 20-300 kg (approx. 44-660 lbs)

If values fall outside these ranges, ask user to verify.

### Common Mistakes

- User says "180" without units → ask: "Is that 180 lbs or 180 kg?"
- User gives height in meters → convert to cm (e.g., 1.75m → 175cm)
- User provides only one measurement → prompt for the missing one
