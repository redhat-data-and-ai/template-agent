#!/usr/bin/env python3
"""Test script for JSON Schema extraction from Rego templates."""

import asyncio
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from deep_agent.src.policy.schema_parser import (
    extract_schema_from_template,
    validate_schema,
    get_field_defaults,
    validate_user_settings,
    merge_user_settings_with_defaults,
)


def test_schema_extraction():
    """Test extracting schema from template."""
    print("=" * 70)
    print("Test 1: Extract Schema from Template")
    print("=" * 70)

    template_path = project_root / "config" / "compliance" / "policy_templates" / "agent_authz.rego.tmpl"

    try:
        schema = extract_schema_from_template(template_path)
        print(f"✓ Successfully extracted schema")
        print(f"  Title: {schema.get('title')}")
        print(f"  Version: {schema.get('version')}")
        print(f"  Properties: {len(schema.get('properties', {}))}")
        print(f"  Sections: {len(schema.get('sections', {}))}")
        return schema
    except Exception as exc:
        print(f"✗ Failed to extract schema: {exc}")
        return None


def test_schema_validation(schema):
    """Test schema validation."""
    print("\n" + "=" * 70)
    print("Test 2: Validate Schema")
    print("=" * 70)

    try:
        validate_schema(schema)
        print("✓ Schema is valid")
    except Exception as exc:
        print(f"✗ Schema validation failed: {exc}")


def test_defaults_extraction(schema):
    """Test extracting defaults from schema."""
    print("\n" + "=" * 70)
    print("Test 3: Extract Defaults")
    print("=" * 70)

    defaults = get_field_defaults(schema)
    print(f"✓ Extracted {len(defaults)} default values:")
    for key, value in defaults.items():
        print(f"  {key}: {value}")
    return defaults


def test_user_settings_validation(schema):
    """Test validating user settings."""
    print("\n" + "=" * 70)
    print("Test 4: Validate User Settings")
    print("=" * 70)

    # Test valid settings
    valid_settings = {
        "max_trajectory_length": 50,
        "enable_trajectory_limits": True
    }

    is_valid, errors = validate_user_settings(valid_settings, schema)
    if is_valid:
        print(f"✓ Valid settings accepted")
    else:
        print(f"✗ Valid settings rejected: {errors}")

    # Test invalid settings (out of range)
    invalid_settings = {
        "max_trajectory_length": 2000,  # Max is 1000
        "enable_trajectory_limits": True
    }

    is_valid, errors = validate_user_settings(invalid_settings, schema)
    if not is_valid:
        print(f"✓ Invalid settings correctly rejected:")
        for error in errors:
            print(f"  - {error}")
    else:
        print(f"✗ Invalid settings were accepted")

    # Test invalid type
    invalid_type = {
        "max_trajectory_length": "fifty",  # Should be int
    }

    is_valid, errors = validate_user_settings(invalid_type, schema)
    if not is_valid:
        print(f"✓ Type mismatch correctly rejected:")
        for error in errors:
            print(f"  - {error}")
    else:
        print(f"✗ Type mismatch was accepted")


def test_merge_settings(schema):
    """Test merging user settings with defaults."""
    print("\n" + "=" * 70)
    print("Test 5: Merge Settings with Defaults")
    print("=" * 70)

    # Partial user settings
    user_settings = {
        "max_trajectory_length": 200
    }

    merged = merge_user_settings_with_defaults(user_settings, schema)
    print(f"✓ Merged settings:")
    print(f"  User provided: {len(user_settings)} fields")
    print(f"  Total merged: {len(merged)} fields")
    print(f"  Custom max_trajectory_length: {merged.get('max_trajectory_length')}")
    print(f"  Default enable_trajectory_limits: {merged.get('enable_trajectory_limits')}")


def test_schema_structure(schema):
    """Test schema structure details."""
    print("\n" + "=" * 70)
    print("Test 6: Schema Structure Analysis")
    print("=" * 70)

    properties = schema.get("properties", {})
    sections = schema.get("sections", {})

    print("Properties with UI metadata:")
    for prop_name, prop_schema in properties.items():
        ui = prop_schema.get("ui", {})
        print(f"  {prop_name}:")
        print(f"    - Widget: {ui.get('widget')}")
        print(f"    - Section: {ui.get('section')}")
        print(f"    - Order: {ui.get('order')}")
        if ui.get("controls"):
            print(f"    - Controls: {ui.get('controls')}")

    print(f"\nSections (sorted by order):")
    sorted_sections = sorted(
        sections.items(),
        key=lambda x: x[1].get("order", 999)
    )
    for section_id, section_data in sorted_sections:
        print(f"  {section_id}:")
        print(f"    - Title: {section_data.get('title')}")
        print(f"    - Order: {section_data.get('order')}")
        print(f"    - Icon: {section_data.get('icon')}")


def test_api_schema_output(schema):
    """Test API-ready schema output."""
    print("\n" + "=" * 70)
    print("Test 7: API Schema Output (JSON)")
    print("=" * 70)

    # This is what the API endpoint will return
    api_output = json.dumps(schema, indent=2)
    print("Schema ready for /api/v1/policy/schema/agent_authz:")
    print(api_output[:500] + "..." if len(api_output) > 500 else api_output)
    print(f"\nTotal JSON size: {len(api_output)} bytes")


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "Schema Extraction Test Suite" + " " * 23 + "║")
    print("╚" + "═" * 68 + "╝")

    schema = test_schema_extraction()
    if not schema:
        print("\n✗ Schema extraction failed, cannot continue")
        sys.exit(1)

    test_schema_validation(schema)
    defaults = test_defaults_extraction(schema)
    test_user_settings_validation(schema)
    test_merge_settings(schema)
    test_schema_structure(schema)
    test_api_schema_output(schema)

    print("\n" + "=" * 70)
    print("✓ All tests complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Start agent: make dev")
    print("  2. Test API: curl http://localhost:5002/api/v1/policy/schema/agent_authz")
    print("  3. Build dynamic UI component")
    print()


if __name__ == "__main__":
    main()
