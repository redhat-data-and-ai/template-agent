"""Compare an OpenAPI schema against a committed baseline.

The UI builds against the agent's HTTP surface but ships separately, so
a removed route or a renamed field only fails in integration — or in
production. `test_ui_contract` pins the handful of paths the UI calls by
name; this catches the rest.

The asymmetry is deliberate: **additive changes pass, removals fail.**
Adding a route or an optional field cannot break an existing client;
removing one can.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SchemaDiff:
    """Breaking and additive changes between two OpenAPI schemas."""

    removed_operations: list[str] = field(default_factory=list)
    removed_schemas: list[str] = field(default_factory=list)
    newly_required: list[str] = field(default_factory=list)
    added_operations: list[str] = field(default_factory=list)

    @property
    def breaking(self) -> list[str]:
        return (
            [f"operation removed: {o}" for o in self.removed_operations]
            + [f"schema removed: {s}" for s in self.removed_schemas]
            + [f"request field newly required: {f}" for f in self.newly_required]
        )

    @property
    def is_breaking(self) -> bool:
        return bool(self.breaking)


def operations(schema: dict[str, Any]) -> set[str]:
    """Every `METHOD /path` pair the schema declares."""
    found: set[str] = set()
    for path, methods in (schema.get("paths") or {}).items():
        for method in methods:
            if method.lower() in {
                "get",
                "post",
                "put",
                "patch",
                "delete",
                "head",
                "options",
            }:
                found.add(f"{method.upper()} {path}")
    return found


def _required_fields(schema: dict[str, Any]) -> dict[str, set[str]]:
    """Required property names per component schema."""
    components = (schema.get("components") or {}).get("schemas") or {}
    return {
        name: set(body.get("required") or [])
        for name, body in components.items()
        if isinstance(body, dict)
    }


def diff_schemas(baseline: dict[str, Any], current: dict[str, Any]) -> SchemaDiff:
    """What changed, split into breaking and additive."""
    old_ops, new_ops = operations(baseline), operations(current)

    old_required = _required_fields(baseline)
    new_required = _required_fields(current)

    # A field that becomes required breaks callers that already omit it.
    newly_required = [
        f"{name}.{prop}"
        for name, props in new_required.items()
        if name in old_required
        for prop in sorted(props - old_required[name])
    ]

    return SchemaDiff(
        removed_operations=sorted(old_ops - new_ops),
        removed_schemas=sorted(set(old_required) - set(new_required)),
        newly_required=sorted(newly_required),
        added_operations=sorted(new_ops - old_ops),
    )
