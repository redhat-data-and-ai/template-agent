"""Unit tests for graph personalization helpers."""

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_agent.aegra.graph import (
    _active_rule_contents,
    _personalization_uid,
    _resolve_personalization_uid,
    _rule_lookup_ids,
)


def _jwt(claims: dict) -> str:
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"hdr.{payload}.sig"


class TestResolvePersonalizationUid:
    def test_prefers_preferred_username(self):
        user = MagicMock()
        user.identity = "uuid-1"
        token = _jwt({"preferred_username": "dpundir", "sub": "uuid-1"})
        assert _resolve_personalization_uid(user, token) == "dpundir"

    def test_falls_back_to_sub(self):
        token = _jwt({"sub": "uuid-1"})
        assert _resolve_personalization_uid(MagicMock(), token) == "uuid-1"

    def test_returns_none_without_token_or_on_bad_payload(self):
        assert _resolve_personalization_uid(MagicMock(), None) is None
        assert _resolve_personalization_uid(MagicMock(), "not-a-jwt") is None


class TestRuleLookupIds:
    def test_prefers_personalization_uid(self):
        assert _rule_lookup_ids("uuid-1", "dpundir") == ["dpundir"]
        assert _rule_lookup_ids("uuid-1", None) == ["uuid-1"]
        assert _rule_lookup_ids(None, None) == []


class TestPersonalizationUid:
    def test_falls_back_to_identity(self):
        runtime = MagicMock()
        user = MagicMock()
        user.identity = "dpundir"
        with (
            patch(
                "deep_agent.src.infrastructure.backend._get_user_id_from_runtime",
                return_value=None,
            ),
            patch("deep_agent.utils.pylogger._user_id_var") as user_id_var,
        ):
            user_id_var.get.return_value = None
            assert _personalization_uid(runtime, user, None) == "dpundir"


class TestActiveRuleContents:
    @pytest.mark.asyncio
    async def test_empty_ids(self):
        assert await _active_rule_contents([]) == []

    @pytest.mark.asyncio
    async def test_uses_cache(self):
        with patch(
            "deep_agent.src.cache.personalization_cache.get_rules",
            new_callable=AsyncMock,
            return_value=[{"content": "Be concise"}, {"content": ""}],
        ):
            assert await _active_rule_contents(["u1"]) == ["Be concise"]

    @pytest.mark.asyncio
    async def test_loads_from_repo_and_dedupes(self):
        rule_a = MagicMock()
        rule_a.content = "Be concise"
        rule_b = MagicMock()
        rule_b.content = "Be concise"
        repo = MagicMock()
        repo.list_rules = AsyncMock(return_value=[rule_a, rule_b])

        with (
            patch(
                "deep_agent.src.cache.personalization_cache.get_rules",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "deep_agent.src.cache.personalization_cache.set_rules",
                new_callable=AsyncMock,
            ) as mock_set,
            patch(
                "deep_agent.src.personalization.repository.PersonalizationRepository",
                return_value=repo,
            ),
        ):
            result = await _active_rule_contents(["u1", "u1"])
        assert result == ["Be concise"]
        mock_set.assert_awaited()
