#!/usr/bin/env python3
"""Unit conversion utilities for imperial to metric conversions.

Uses sympy for exact rational arithmetic to avoid floating point errors.
"""

from sympy import N, Rational


def inches_to_cm(inches: float) -> float:
    """Convert inches to centimeters."""
    return float(N(Rational(inches) * Rational(254, 100), 5))


def feet_to_cm(feet: float) -> float:
    """Convert feet to centimeters."""
    return float(N(Rational(feet) * 12 * Rational(254, 100), 5))


def feet_inches_to_cm(feet: float, inches: float = 0) -> float:
    """Convert feet and inches to centimeters."""
    total_inches = feet * 12 + inches
    return float(N(Rational(total_inches) * Rational(254, 100), 5))


def pounds_to_kg(pounds: float) -> float:
    """Convert pounds to kilograms."""
    return float(N(Rational(pounds) / Rational(2205, 1000), 5))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: convert_units.py <type> <value1> [value2]")
        print("Types: inches, feet, feet_inches, pounds")
        sys.exit(1)

    conversion_type = sys.argv[1]

    if conversion_type == "inches":
        print(inches_to_cm(float(sys.argv[2])))
    elif conversion_type == "feet":
        print(feet_to_cm(float(sys.argv[2])))
    elif conversion_type == "feet_inches":
        feet = float(sys.argv[2])
        inches = float(sys.argv[3]) if len(sys.argv) > 3 else 0
        print(feet_inches_to_cm(feet, inches))
    elif conversion_type == "pounds":
        print(pounds_to_kg(float(sys.argv[2])))
    else:
        print(f"Unknown conversion type: {conversion_type}")
        sys.exit(1)
