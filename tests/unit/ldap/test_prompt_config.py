"""Unit tests for deep_agent.src.ldap.prompt_config."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from deep_agent.src.ldap.prompt_config import (
    PRIVILEGED_ROLES,
    ROLE_HIERARCHY,
    GroupRoleMapping,
    PromptAccessConfig,
    _parse_groups_from_frontmatter,
    get_prompt_access_config,
    reset_prompt_config,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_prompt_config()
    yield
    reset_prompt_config()


def _write_prompt_md(tmpdir: str, content: str) -> Path:
    path = Path(tmpdir) / "PROMPT.md"
    path.write_text(content)
    return path


class TestRoleHierarchy:
    def test_hierarchy_values(self):
        assert ROLE_HIERARCHY["owners"] == 4
        assert ROLE_HIERARCHY["admins"] == 3
        assert ROLE_HIERARCHY["builders"] == 2
        assert ROLE_HIERARCHY["users"] == 1

    def test_privileged_roles(self):
        assert PRIVILEGED_ROLES == {"owners", "admins", "builders"}
        assert "users" not in PRIVILEGED_ROLES


class TestParseGroupsFromFrontmatter:
    def test_full_groups(self, tmp_path):
        content = """\
---
accessibility: public
groups:
  - role: owners
    group: team-owners
  - role: admins
    group: team-admins
  - role: builders
    group: team-builders
  - role: users
    group: team-users
---
System prompt here.
"""
        path = tmp_path / "PROMPT.md"
        path.write_text(content)
        config = _parse_groups_from_frontmatter(path)

        assert config.accessibility == "public"
        assert config.groups is not None
        assert len(config.groups) == 4
        assert config.groups[0].role == "owners"
        assert config.groups[0].group == "team-owners"

    def test_private_by_default(self, tmp_path):
        content = """\
---
groups:
  - role: admins
    group: team-admins
---
"""
        path = tmp_path / "PROMPT.md"
        path.write_text(content)
        config = _parse_groups_from_frontmatter(path)

        assert config.accessibility == "private"
        assert config.groups is not None
        assert len(config.groups) == 1

    def test_no_frontmatter(self, tmp_path):
        path = tmp_path / "PROMPT.md"
        path.write_text("Just a system prompt, no YAML.")
        config = _parse_groups_from_frontmatter(path)

        assert config.groups is None
        assert config.accessibility == "private"

    def test_empty_groups_list(self, tmp_path):
        content = """\
---
groups: []
---
"""
        path = tmp_path / "PROMPT.md"
        path.write_text(content)
        config = _parse_groups_from_frontmatter(path)
        assert config.groups is None

    def test_invalid_group_entries_skipped(self, tmp_path):
        content = """\
---
groups:
  - role: owners
    group: team-owners
  - role: 123
    group: invalid-role-type
  - bad: entry
  - role: builders
    group: team-builders
---
"""
        path = tmp_path / "PROMPT.md"
        path.write_text(content)
        config = _parse_groups_from_frontmatter(path)

        assert config.groups is not None
        assert len(config.groups) == 2
        assert config.groups[0].role == "owners"
        assert config.groups[1].role == "builders"

    def test_roles_lowercased(self, tmp_path):
        content = """\
---
groups:
  - role: OWNERS
    group: team-owners
---
"""
        path = tmp_path / "PROMPT.md"
        path.write_text(content)
        config = _parse_groups_from_frontmatter(path)
        assert config.groups[0].role == "owners"

    def test_file_not_found(self, tmp_path):
        config = _parse_groups_from_frontmatter(tmp_path / "missing.md")
        assert config.groups == []
        assert config.accessibility == "private"

    def test_accessibility_non_public_defaults_private(self, tmp_path):
        content = """\
---
accessibility: restricted
groups:
  - role: owners
    group: team-owners
---
"""
        path = tmp_path / "PROMPT.md"
        path.write_text(content)
        config = _parse_groups_from_frontmatter(path)
        assert config.accessibility == "private"

    def test_no_groups_key(self, tmp_path):
        content = """\
---
accessibility: public
---
"""
        path = tmp_path / "PROMPT.md"
        path.write_text(content)
        config = _parse_groups_from_frontmatter(path)
        assert config.groups is None
        assert config.accessibility == "public"


class TestGetPromptAccessConfig:
    def test_caches_by_mtime(self, tmp_path):
        content = """\
---
groups:
  - role: owners
    group: team-owners
---
"""
        prompt_path = tmp_path / "PROMPT.md"
        prompt_path.write_text(content)

        with patch(
            "deep_agent.src.ldap.prompt_config._prompt_md_path",
            return_value=prompt_path,
        ):
            c1 = get_prompt_access_config()
            c2 = get_prompt_access_config()
            assert c1 is c2

    def test_reloads_on_mtime_change(self, tmp_path):
        content1 = """\
---
groups:
  - role: owners
    group: team-owners
---
"""
        content2 = """\
---
groups:
  - role: admins
    group: team-admins
  - role: builders
    group: team-builders
---
"""
        prompt_path = tmp_path / "PROMPT.md"
        prompt_path.write_text(content1)

        with patch(
            "deep_agent.src.ldap.prompt_config._prompt_md_path",
            return_value=prompt_path,
        ):
            c1 = get_prompt_access_config()
            assert len(c1.groups) == 1

            import time

            time.sleep(0.05)
            prompt_path.write_text(content2)

            c2 = get_prompt_access_config()
            assert len(c2.groups) == 2

    def test_missing_file_returns_denied(self, tmp_path):
        with patch(
            "deep_agent.src.ldap.prompt_config._prompt_md_path",
            return_value=tmp_path / "nonexistent.md",
        ):
            config = get_prompt_access_config()
            assert config.groups == []
            assert config.accessibility == "private"


class TestResetPromptConfig:
    def test_clears_cache(self, tmp_path):
        content = """\
---
groups:
  - role: owners
    group: team-owners
---
"""
        prompt_path = tmp_path / "PROMPT.md"
        prompt_path.write_text(content)

        with patch(
            "deep_agent.src.ldap.prompt_config._prompt_md_path",
            return_value=prompt_path,
        ):
            c1 = get_prompt_access_config()
            reset_prompt_config()
            c2 = get_prompt_access_config()
            assert c1 is not c2


class TestDataclasses:
    def test_group_role_mapping(self):
        m = GroupRoleMapping(role="owners", group="team-owners")
        assert m.role == "owners"
        assert m.group == "team-owners"

    def test_prompt_access_config_defaults(self):
        c = PromptAccessConfig()
        assert c.groups is None
        assert c.accessibility == "private"
