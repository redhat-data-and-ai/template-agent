"""OpenAPI drift against a committed baseline.

`test_ui_contract` pins the handful of routes the UI calls by name. This
covers the rest of the surface: 107 operations the UI, the SDK, or any
other client may depend on.

Additive changes pass. Removals and newly-required request fields fail,
because only those can break a client that already works.

When a change is intentional:

    make openapi-baseline

and commit the result — the diff in review then shows exactly what the
API contract gained or lost.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import pytest

from tests.support.openapi import diff_schemas, operations

BASELINE = Path(__file__).parents[2] / "contracts" / "openapi-baseline.json"


@pytest.fixture(scope="module")
def baseline() -> dict:
    assert BASELINE.is_file(), (
        f"No OpenAPI baseline at {BASELINE} — generate with: make openapi-baseline"
    )
    return json.loads(BASELINE.read_text())


@pytest.fixture(scope="module")
def current() -> dict:
    from aegra_api.main import create_app

    # create_app logs a page of startup noise.
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        return create_app().openapi()


class TestOpenApiDrift:
    def test_no_breaking_changes(self, baseline: dict, current: dict):
        diff = diff_schemas(baseline, current)

        assert not diff.is_breaking, (
            "Breaking OpenAPI change:\n  "
            + "\n  ".join(diff.breaking)
            + "\n\nIf intended, regenerate with: make openapi-baseline"
        )

    def test_additive_changes_are_reported_not_failed(
        self, baseline: dict, current: dict, capsys
    ):
        """Adding a route cannot break a client, so it must not fail CI."""
        diff = diff_schemas(baseline, current)

        if diff.added_operations:
            print("New operations since baseline:", diff.added_operations)

    def test_baseline_covers_a_realistic_surface(self, baseline: dict):
        """Guards against a baseline regenerated from a half-built app."""
        assert len(operations(baseline)) > 50


class TestDifferSemantics:
    """The differ must not pass everything — that is the failure mode
    that makes a drift test worthless."""

    def test_removed_operation_is_breaking(self):
        before = {"paths": {"/a": {"get": {}}, "/b": {"post": {}}}}
        after = {"paths": {"/a": {"get": {}}}}

        assert diff_schemas(before, after).is_breaking

    def test_removed_method_on_a_kept_path_is_breaking(self):
        before = {"paths": {"/a": {"get": {}, "delete": {}}}}
        after = {"paths": {"/a": {"get": {}}}}

        assert diff_schemas(before, after).is_breaking

    def test_added_operation_is_not_breaking(self):
        before = {"paths": {"/a": {"get": {}}}}
        after = {"paths": {"/a": {"get": {}}, "/b": {"get": {}}}}

        diff = diff_schemas(before, after)
        assert not diff.is_breaking
        assert diff.added_operations == ["GET /b"]

    def test_newly_required_field_is_breaking(self):
        """A caller that already omits the field starts failing."""
        before = {"components": {"schemas": {"S": {"required": ["a"]}}}}
        after = {"components": {"schemas": {"S": {"required": ["a", "b"]}}}}

        assert diff_schemas(before, after).is_breaking

    def test_relaxing_a_requirement_is_not_breaking(self):
        before = {"components": {"schemas": {"S": {"required": ["a", "b"]}}}}
        after = {"components": {"schemas": {"S": {"required": ["a"]}}}}

        assert not diff_schemas(before, after).is_breaking

    def test_removed_schema_is_breaking(self):
        before = {"components": {"schemas": {"S": {"required": []}}}}
        after = {"components": {"schemas": {}}}

        assert diff_schemas(before, after).is_breaking

    def test_identical_schemas_are_clean(self):
        schema = {
            "paths": {"/a": {"get": {}}},
            "components": {"schemas": {"S": {"required": ["x"]}}},
        }

        diff = diff_schemas(schema, schema)
        assert not diff.is_breaking
        assert diff.added_operations == []
