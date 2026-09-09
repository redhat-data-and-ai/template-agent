"""Auto-apply ``@pytest.mark.unit`` to every test in tests/unit/."""

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        item.add_marker(pytest.mark.unit)


@pytest.fixture(autouse=True)
def isolate_rover_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep local .env DEVELOPER_GROUP / USER_GROUP out of unit tests.

    Tests that need groups mock them explicitly. The Settings singleton and
    auth.py module-level copies are captured at import from the environment.
    """
    monkeypatch.setenv("DEVELOPER_GROUP", "")
    monkeypatch.setenv("USER_GROUP", "")
    monkeypatch.setattr("deep_agent.src.settings.settings.DEVELOPER_GROUP", [])
    monkeypatch.setattr("deep_agent.src.settings.settings.USER_GROUP", [])
    monkeypatch.setattr("deep_agent.aegra.auth.DEVELOPER_GROUP", [])
    monkeypatch.setattr("deep_agent.aegra.auth.USER_GROUP", [])
