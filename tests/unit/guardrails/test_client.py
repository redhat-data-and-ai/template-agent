"""Unit tests for deep_agent.src.guardrails.client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import deep_agent.src.guardrails.client as client_mod
from deep_agent.src.guardrails.client import (
    _build_guardian_block,
    _call_guardian,
    _get_guardian_client,
    _guardian_model,
    check_injection,
    check_safety,
)


@pytest.fixture(autouse=True)
def reset_guardian_client():
    """Reset cached client between tests."""
    original = client_mod._guardian_client
    client_mod._guardian_client = None
    yield
    client_mod._guardian_client = original


class TestGetGuardianClient:
    def test_creates_client_on_first_call(self):
        with (
            patch("deep_agent.src.guardrails.client.AsyncOpenAI") as mock_cls,
            patch("deep_agent.src.guardrails.client.httpx.AsyncClient"),
            patch("deep_agent.src.guardrails.client.settings") as mock_settings,
        ):
            mock_settings.GUARDIAN_API_KEY = "key"
            mock_settings.GUARDIAN_API_BASE = "http://example.com"
            mock_settings.GUARDIAN_SSL_VERIFY = True
            mock_cls.return_value = MagicMock()

            result = _get_guardian_client()
            assert result is mock_cls.return_value

    def test_returns_cached_client_on_second_call(self):
        with (
            patch("deep_agent.src.guardrails.client.AsyncOpenAI") as mock_cls,
            patch("deep_agent.src.guardrails.client.httpx.AsyncClient"),
            patch("deep_agent.src.guardrails.client.settings") as mock_settings,
        ):
            mock_settings.GUARDIAN_API_KEY = "key"
            mock_settings.GUARDIAN_API_BASE = "http://example.com"
            mock_settings.GUARDIAN_SSL_VERIFY = False
            mock_cls.return_value = MagicMock()

            first = _get_guardian_client()
            second = _get_guardian_client()
            assert first is second
            mock_cls.assert_called_once()


class TestBuildGuardianBlock:
    def test_contains_criteria(self):
        block = _build_guardian_block("some criteria")
        assert "some criteria" in block
        assert "Yes" in block
        assert "No" in block


class TestGuardianModel:
    def test_returns_config_model_when_set(self):
        cfg = MagicMock()
        cfg.model = "my-model"
        with patch("deep_agent.src.guardrails.get_guardrails_config", return_value=cfg):
            assert _guardian_model() == "my-model"

    def test_returns_default_when_no_config(self):
        with patch(
            "deep_agent.src.guardrails.get_guardrails_config", return_value=None
        ):
            assert "granite-guardian" in _guardian_model()


class TestCallGuardian:
    @pytest.mark.asyncio
    async def test_returns_safe_when_verdict_is_no(self):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "No"

        with (
            patch(
                "deep_agent.src.guardrails.client._guardian_model", return_value="model"
            ),
            patch("deep_agent.src.guardrails.client._get_guardian_client"),
            patch(
                "deep_agent.src.guardrails.client.litellm.acompletion",
                new=AsyncMock(return_value=mock_response),
            ),
        ):
            is_safe, verdict = await _call_guardian(
                [{"role": "user", "content": "hi"}], "input"
            )
            assert is_safe is True
            assert verdict == "No"

    @pytest.mark.asyncio
    async def test_returns_unsafe_when_verdict_starts_with_yes(self):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Yes, this is unsafe."

        with (
            patch(
                "deep_agent.src.guardrails.client._guardian_model", return_value="model"
            ),
            patch("deep_agent.src.guardrails.client._get_guardian_client"),
            patch(
                "deep_agent.src.guardrails.client.litellm.acompletion",
                new=AsyncMock(return_value=mock_response),
            ),
        ):
            is_safe, verdict = await _call_guardian(
                [{"role": "user", "content": "bad"}], "input"
            )
            assert is_safe is False
            assert verdict == "Yes,"

    @pytest.mark.asyncio
    async def test_returns_safe_on_exception(self):
        with (
            patch(
                "deep_agent.src.guardrails.client._guardian_model", return_value="model"
            ),
            patch("deep_agent.src.guardrails.client._get_guardian_client"),
            patch(
                "deep_agent.src.guardrails.client.litellm.acompletion",
                new=AsyncMock(side_effect=RuntimeError("network error")),
            ),
        ):
            is_safe, verdict = await _call_guardian(
                [{"role": "user", "content": "hi"}], "input"
            )
            assert is_safe is True
            assert verdict == "error"


class TestCheckSafety:
    @pytest.mark.asyncio
    async def test_delegates_to_call_guardian(self):
        with patch(
            "deep_agent.src.guardrails.client._call_guardian",
            new=AsyncMock(return_value=(True, "No")),
        ) as mock_call:
            result = await check_safety("some content", context="input")
            assert result == (True, "No")
            mock_call.assert_called_once()
            args = mock_call.call_args
            assert args[1]["context"] == "input"


class TestCheckInjection:
    @pytest.mark.asyncio
    async def test_sends_two_messages(self):
        with patch(
            "deep_agent.src.guardrails.client._call_guardian",
            new=AsyncMock(return_value=(False, "Yes")),
        ) as mock_call:
            result = await check_injection("inject me", context="input")
            assert result == (False, "Yes")
            messages = mock_call.call_args.kwargs["messages"]
            assert len(messages) == 2
