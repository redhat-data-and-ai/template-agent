#!/usr/bin/env python3
"""Render Rego policy files from Jinja2 templates.

This script reads .rego.tmpl files from config/compliance/policy_templates/
and renders them with default values to config/compliance/policies/.

The rendered .rego files are loaded by OPA at runtime.
"""

import sys
from pathlib import Path

from jinja2 import Template

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from deep_agent.src.policy.schema_parser import extract_schema_from_template, get_field_defaults


def render_policy_template(template_path: Path, output_path: Path) -> None:
    """Render a Rego template with default values.

    Args:
        template_path: Path to .rego.tmpl file
        output_path: Path to write rendered .rego file
    """
    print(f"Rendering {template_path.name}...")

    # Read template
    template_content = template_path.read_text()

    # Extract schema and defaults
    try:
        schema = extract_schema_from_template(template_path)
        defaults = get_field_defaults(schema)
        print(f"  Found {len(defaults)} default values")
    except Exception as exc:
        print(f"  Warning: Could not extract schema: {exc}")
        print(f"  Rendering without defaults")
        defaults = {}

    # Extract Rego part (after SCHEMA block)
    schema_end = template_content.find('#}')
    if schema_end != -1:
        rego_template = template_content[schema_end + 2:].strip()
    else:
        rego_template = template_content

    # Render with defaults
    template = Template(rego_template)

    # Convert Python booleans to Rego booleans (lowercase)
    render_context = {}
    for key, value in defaults.items():
        if isinstance(value, bool):
            render_context[key] = str(value).lower()
        else:
            render_context[key] = value

    rendered = template.render(**render_context)

    # Write output
    output_path.write_text(rendered)
    print(f"  ✓ Wrote {output_path} ({len(rendered)} bytes)")


def main():
    """Render all policy templates."""
    template_dir = project_root / "config" / "compliance" / "policy_templates"
    output_dir = project_root / "config" / "compliance" / "policies"

    if not template_dir.exists():
        print(f"Error: Template directory not found: {template_dir}")
        sys.exit(1)

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all .rego.tmpl files
    templates = list(template_dir.glob("*.rego.tmpl"))

    if not templates:
        print(f"No .rego.tmpl files found in {template_dir}")
        sys.exit(1)

    print(f"Found {len(templates)} template(s)\n")

    # Render each template
    for template_path in templates:
        output_filename = template_path.stem  # Removes .tmpl extension
        output_path = output_dir / output_filename

        try:
            render_policy_template(template_path, output_path)
        except Exception as exc:
            print(f"  ✗ Failed to render {template_path.name}: {exc}")
            sys.exit(1)

    print(f"\n✓ Successfully rendered {len(templates)} policy file(s)")
    print(f"\nOPA will automatically reload policies from {output_dir}")


if __name__ == "__main__":
    main()
