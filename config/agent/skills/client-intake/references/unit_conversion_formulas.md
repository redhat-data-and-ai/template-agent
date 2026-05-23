# Unit Conversion Formulas

## Shell Commands (Using sympy for precision)

These commands use exact rational arithmetic to avoid floating point errors.

| Input Format | Examples | Command |
|--------------|----------|---------|
| Inches | "72 inches", "72in" | `python3 -c "from sympy import Rational, N; print(N(Rational(x) * Rational(254, 100), 5))"` |
| Feet | "6ft", "6 feet" | `python3 -c "from sympy import Rational, N; print(N(Rational(x) * 12 * Rational(254, 100), 5))"` |
| Feet + Inches | "6ft 2in", "6'2" | `python3 -c "from sympy import Rational, N; print(N((Rational(x) * 12 + Rational(y)) * Rational(254, 100), 5))"` |
| Pounds | "180lbs", "180 pounds" | `python3 -c "from sympy import Rational, N; print(N(Rational(x) / Rational(2205, 1000), 5))"` |

**Important:** Always use `python3`, never `python` — `python` is not available on all systems.

## Python Script Alternative

For more complex conversions or batch processing, use the conversion script:

```bash
python3 scripts/convert_units.py inches 72
python3 scripts/convert_units.py feet 6
python3 scripts/convert_units.py feet_inches 6 2
python3 scripts/convert_units.py pounds 180
```

## Conversion Constants

- **Inch to CM:** 1 inch = 2.54 cm (exact: 254/100)
- **Foot to CM:** 1 foot = 12 inches = 30.48 cm (exact: 12 × 254/100)
- **Pound to KG:** 1 lb = 0.45359237 kg (exact: 1/2.205)

## Edge Cases

- Missing inches in feet format: treat as 0 (e.g., "6ft" → 6 feet 0 inches)
- Decimal values: accepted and converted precisely using rational arithmetic
