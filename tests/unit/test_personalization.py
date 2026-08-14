"""Unit tests for personalization models and injector."""

import uuid
from datetime import datetime

import pytest

from deep_agent.src.personalization.injector import inject_personalization
from deep_agent.src.personalization.models import Memory, Rule


class TestMemoryModel:
    def test_create_with_defaults(self):
        m = Memory(user_id="u1", content="Likes Python")
        assert m.user_id == "u1"
        assert m.content == "Likes Python"
        assert isinstance(m.id, uuid.UUID)
        assert isinstance(m.created_at, datetime)

    def test_create_with_explicit_id(self):
        uid = uuid.uuid4()
        m = Memory(id=uid, user_id="u1", content="test")
        assert m.id == uid


class TestRuleModel:
    def test_create_with_defaults(self):
        r = Rule(user_id="u1", content="Be concise")
        assert r.is_active is True

    def test_inactive_rule(self):
        r = Rule(user_id="u1", content="Old rule", is_active=False)
        assert r.is_active is False


class TestInjectPersonalization:
    def test_no_personalization(self):
        result = inject_personalization("Base prompt", [], [])
        assert result == "Base prompt"

    def test_memories_only(self):
        result = inject_personalization("Base", ["Likes Python", "Uses Linux"], [])
        assert "User Memories" in result
        assert "Likes Python" in result
        assert "Uses Linux" in result
        assert "Custom Instructions" not in result

    def test_rules_only(self):
        result = inject_personalization("Base", [], ["Be concise", "Use code blocks"])
        assert "Custom Instructions" in result
        assert "Be concise" in result
        assert "User Memories" not in result

    def test_both_memories_and_rules(self):
        result = inject_personalization(
            "Base prompt",
            ["Prefers dark mode"],
            ["Always use TypeScript"],
        )
        assert "User Memories" in result
        assert "Custom Instructions" in result
        assert "Prefers dark mode" in result
        assert "Always use TypeScript" in result
        assert result.startswith("Base prompt")

    def test_separator_between_sections(self):
        result = inject_personalization("Base", ["m1"], ["r1"])
        assert "---" in result
