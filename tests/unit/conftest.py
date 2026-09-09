"""Auto-apply ``@pytest.mark.unit`` to every test in tests/unit/."""

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        item.add_marker(pytest.mark.unit)
