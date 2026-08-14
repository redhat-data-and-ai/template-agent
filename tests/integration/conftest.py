"""Shared fixtures for integration tests.

Auto-marks all tests in this directory as integration tests so they
can be skipped in fast CI runs via ``pytest -m "not integration"``.
"""

import pytest


def pytest_collection_modifyitems(items):
    """Auto-add 'integration' marker to all tests in this directory."""
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
