"""Where a `filesystem_readonly` route reads from.

`/skills/` is image content and resolves under CONFIG_PATH. Datasets
are a mounted PVC: large, changing without a release, and resolving
them under CONFIG_PATH would put a data tree inside the directory
`_disk_signature` rglobs on every reload check.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import deep_agent.src.infrastructure.backend as backend


@pytest.fixture
def base(tmp_path: Path):
    with patch.object(backend, "agent_config") as cfg:
        cfg.base_dir = tmp_path
        yield tmp_path


class TestReadonlyRoot:
    def test_bare_name_resolves_under_the_config_dir(self, base):
        # The existing form must keep working untouched.
        assert (
            backend._readonly_root("/skills/", "filesystem_readonly") == base / "skills"
        )

    def test_an_absolute_path_is_used_as_given(self, base):
        assert backend._readonly_root(
            "/datasets/", "filesystem_readonly:/app/data/datasets"
        ) == Path("/app/data/datasets")

    def test_a_relative_path_stays_under_the_config_dir(self, base):
        # Not an escape hatch out of the config tree by accident.
        assert (
            backend._readonly_root("/x/", "filesystem_readonly:extra") == base / "extra"
        )

    def test_the_route_prefix_is_not_the_path(self, base):
        # With an explicit root the prefix is only a mount point —
        # `/datasets/` must not also append "datasets".
        assert backend._readonly_root(
            "/datasets/", "filesystem_readonly:/mnt/corpus"
        ) == Path("/mnt/corpus")
