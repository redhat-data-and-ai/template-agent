"""Parse JSON Schema from Rego template metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


def extract_schema_from_template(template_path: Path) -> dict[str, Any]:
    """Extract JSON Schema from Jinja template comment block.

    Looks for {# SCHEMA ... #} block at the top of the template.

    Args:
        template_path: Path to .rego.tmpl file

    Returns:
        Parsed JSON Schema dict

    Raises:
        ValueError: If schema block not found or invalid JSON
    """
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    content = template_path.read_text(encoding="utf-8")

    # Match {# SCHEMA ... #} (non-greedy, multiline, dotall)
    pattern = r'\{#\s*SCHEMA\s*(.*?)\s*#\}'
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        raise ValueError(f"No SCHEMA block found in {template_path}")

    schema_json = match.group(1).strip()

    try:
        schema = json.loads(schema_json)
        logger.info(f"Extracted schema from {template_path.name}: {schema.get('title', 'Untitled')}")
        return schema
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in SCHEMA block: {exc}") from exc


def validate_schema(schema: dict[str, Any]) -> bool:
    """Validate that schema has required fields.

    Args:
        schema: Parsed JSON Schema

    Returns:
        True if valid

    Raises:
        ValueError: If schema is invalid
    """
    required_fields = ["title", "type", "properties"]
    for field in required_fields:
        if field not in schema:
            raise ValueError(f"Schema missing required field: {field}")

    # Validate each property has UI metadata
    missing_ui = []
    for prop_name, prop_schema in schema.get("properties", {}).items():
        if "ui" not in prop_schema:
            missing_ui.append(prop_name)

    if missing_ui:
        logger.warning(f"Properties missing 'ui' metadata: {', '.join(missing_ui)}")

    return True


def get_field_defaults(schema: dict[str, Any]) -> dict[str, Any]:
    """Extract default values from schema.

    Args:
        schema: JSON Schema

    Returns:
        Dict of field_name -> default_value
    """
    defaults = {}
    for prop_name, prop_schema in schema.get("properties", {}).items():
        if "default" in prop_schema:
            defaults[prop_name] = prop_schema["default"]
    return defaults


def merge_user_settings_with_defaults(
    user_settings: dict[str, Any] | None,
    schema: dict[str, Any]
) -> dict[str, Any]:
    """Merge user settings with schema defaults.

    Args:
        user_settings: User's custom settings (can be None or partial)
        schema: JSON Schema with defaults

    Returns:
        Complete settings dict (defaults + user overrides)
    """
    defaults = get_field_defaults(schema)

    if not user_settings:
        return defaults

    # Merge: defaults < user overrides
    return {**defaults, **user_settings}


def validate_user_settings(
    user_settings: dict[str, Any],
    schema: dict[str, Any]
) -> tuple[bool, list[str]]:
    """Validate user settings against schema.

    Simple validation for min/max/type. For full JSON Schema validation,
    use jsonschema library.

    Args:
        user_settings: User's settings
        schema: JSON Schema

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    properties = schema.get("properties", {})

    for field_name, value in user_settings.items():
        if field_name not in properties:
            errors.append(f"Unknown field: {field_name}")
            continue

        prop_schema = properties[field_name]
        expected_type = prop_schema.get("type")

        # Type check
        if expected_type == "integer" and not isinstance(value, int):
            errors.append(f"{field_name}: expected integer, got {type(value).__name__}")
        elif expected_type == "boolean" and not isinstance(value, bool):
            errors.append(f"{field_name}: expected boolean, got {type(value).__name__}")
        elif expected_type == "array" and not isinstance(value, list):
            errors.append(f"{field_name}: expected array, got {type(value).__name__}")

        # Range check for integers
        if expected_type == "integer" and isinstance(value, int):
            if "minimum" in prop_schema and value < prop_schema["minimum"]:
                errors.append(f"{field_name}: {value} < minimum {prop_schema['minimum']}")
            if "maximum" in prop_schema and value > prop_schema["maximum"]:
                errors.append(f"{field_name}: {value} > maximum {prop_schema['maximum']}")

        # Array validations
        if expected_type == "array" and isinstance(value, list):
            if "maxItems" in prop_schema and len(value) > prop_schema["maxItems"]:
                errors.append(f"{field_name}: {len(value)} items > max {prop_schema['maxItems']}")
            if "uniqueItems" in prop_schema and prop_schema["uniqueItems"]:
                if len(value) != len(set(value)):
                    errors.append(f"{field_name}: contains duplicate items")

    return (len(errors) == 0, errors)
