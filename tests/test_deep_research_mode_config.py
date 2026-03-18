"""Tests for deep research mode configuration module."""

from __future__ import annotations

import dataclasses

import pytest

from template_agent.src.core.deep_research.mode_config import (
    MODES,
    ModelSpec,
    QualityGate,
    resolve_mode,
    resolve_model_spec,
)


class TestQualityGate:
    """Test cases for QualityGate dataclass."""

    def test_quality_gate_default_values(self) -> None:
        """QualityGate stores all dimension thresholds correctly."""
        gate = QualityGate(
            coverage_min=0.5,
            factual_grounding_min=0.6,
            data_utilization_min=0.45,
            synthesis_quality_min=0.4,
            structural_compliance_min=0.5,
            communication_quality_min=0.5,
            actionability_min=0.35,
            target_word_count_range=(400, 1000),
            min_tables=0,
            min_sections=2,
            max_sections=4,
            cross_refs_required=False,
            confidence_table_required=False,
            methodology_required=False,
        )
        assert gate.coverage_min == pytest.approx(0.5)
        assert gate.target_word_count_range == (400, 1000)
        assert gate.min_tables == 0
        assert gate.methodology_required is False

    def test_quality_gate_is_frozen(self) -> None:
        """QualityGate is immutable (frozen dataclass)."""
        gate = QualityGate(
            coverage_min=0.5,
            factual_grounding_min=0.6,
            data_utilization_min=0.45,
            synthesis_quality_min=0.4,
            structural_compliance_min=0.5,
            communication_quality_min=0.5,
            actionability_min=0.35,
            target_word_count_range=(400, 1000),
            min_tables=0,
            min_sections=2,
            max_sections=4,
            cross_refs_required=False,
            confidence_table_required=False,
            methodology_required=False,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            gate.coverage_min = 0.6


class TestResearchModeConfig:
    """Test cases for ResearchModeConfig dataclass."""

    def test_mode_config_max_output_tokens_property(self) -> None:
        """max_output_tokens returns synthesis_max_output_tokens."""
        config = MODES["fast"]
        assert config.max_output_tokens == 16384

    def test_mode_config_session_timeout_seconds_property(self) -> None:
        """session_timeout_seconds returns session_timeout_floor."""
        config = MODES["fast"]
        assert config.session_timeout_seconds == 180

    def test_mode_config_extended_has_higher_limits(self) -> None:
        """Extended mode has higher token and timeout limits than fast."""
        fast = MODES["fast"]
        extended = MODES["extended"]
        assert extended.max_output_tokens > fast.max_output_tokens
        assert extended.session_timeout_seconds > fast.session_timeout_seconds

    def test_mode_config_fast_max_has_confidence_table_required(self) -> None:
        """fast_max mode requires confidence table in quality gate."""
        config = MODES["fast_max"]
        assert config.quality_gate.confidence_table_required is True

    def test_mode_config_extended_max_has_methodology_required(self) -> None:
        """extended_max is the only mode requiring methodology section."""
        config = MODES["extended_max"]
        assert config.quality_gate.methodology_required is True

    def test_mode_config_legacy_max_alias(self) -> None:
        """MODES['max'] aliases to extended_max for backward compatibility."""
        assert MODES["max"] is MODES["extended_max"]

    def test_mode_config_all_modes_have_required_fields(self) -> None:
        """All predefined modes have non-empty planning and synthesis instructions."""
        for name, config in MODES.items():
            if name == "max":
                continue
            assert config.planning_instruction
            assert config.synthesis_instruction
            assert config.worker_instruction
            assert config.review_instruction
            assert config.name


class TestModelSpec:
    """Test cases for ModelSpec dataclass."""

    def test_model_spec_stores_context_and_output_limits(self) -> None:
        """ModelSpec stores context window and max output tokens."""
        spec = ModelSpec(context_window=200_000, max_output_tokens=64_000)
        assert spec.context_window == 200_000
        assert spec.max_output_tokens == 64_000

    def test_model_spec_is_frozen(self) -> None:
        """ModelSpec is immutable."""
        spec = ModelSpec(context_window=200_000, max_output_tokens=64_000)
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.context_window = 300_000


class TestResolveModelSpec:
    """Test cases for resolve_model_spec function."""

    def test_resolve_model_spec_none_returns_default(self) -> None:
        """None model name returns default spec."""
        spec = resolve_model_spec(None)
        assert spec.context_window == 200_000
        assert spec.max_output_tokens == 64_000

    def test_resolve_model_spec_empty_string_returns_default(self) -> None:
        """Empty string model name returns default spec."""
        spec = resolve_model_spec("")
        assert spec.context_window == 200_000

    def test_resolve_model_spec_claude_sonnet_matches(self) -> None:
        """Claude Sonnet family is matched by substring."""
        spec = resolve_model_spec("claude-sonnet-4@20250514")
        assert spec.context_window == 200_000
        assert spec.max_output_tokens == 64_000

    def test_resolve_model_spec_gemini_matches(self) -> None:
        """Gemini family is matched by substring."""
        spec = resolve_model_spec("gemini-2.5-pro")
        assert spec.context_window == 1_000_000
        assert spec.max_output_tokens == 65_536

    def test_resolve_model_spec_unknown_returns_default(self) -> None:
        """Unknown model family returns default spec."""
        spec = resolve_model_spec("unknown-model-v1")
        assert spec.context_window == 200_000

    def test_resolve_model_spec_case_insensitive(self) -> None:
        """Model name matching is case insensitive."""
        spec = resolve_model_spec("GEMINI-2.5-FLASH")
        assert spec.context_window == 1_000_000


class TestResolveMode:
    """Test cases for resolve_mode function."""

    def test_resolve_mode_fast_depth_returns_fast(self) -> None:
        """depth='fast' returns fast mode when max_mode is False."""
        config = resolve_mode("claude-sonnet-4", max_mode=False, depth="fast")
        assert config.name == "fast"

    def test_resolve_mode_extended_depth_returns_extended(self) -> None:
        """depth='extended' returns extended mode when max_mode is False."""
        config = resolve_mode("claude-sonnet-4", max_mode=False, depth="extended")
        assert config.name == "extended"

    def test_resolve_mode_fast_with_max_returns_fast_max(self) -> None:
        """depth='fast' with max_mode returns fast_max."""
        config = resolve_mode("claude-sonnet-4", max_mode=True, depth="fast")
        assert config.name == "fast_max"

    def test_resolve_mode_extended_with_max_returns_extended_max(self) -> None:
        """depth='extended' with max_mode returns extended_max."""
        config = resolve_mode("claude-sonnet-4", max_mode=True, depth="extended")
        assert config.name == "extended_max"

    def test_resolve_mode_auto_gemini_returns_extended(self) -> None:
        """depth='auto' with gemini model returns extended mode."""
        config = resolve_mode("gemini-2.5-pro", max_mode=False, depth="auto")
        assert config.name == "extended"

    def test_resolve_mode_auto_non_gemini_returns_fast(self) -> None:
        """depth='auto' with non-gemini model returns fast mode."""
        config = resolve_mode("claude-sonnet-4", max_mode=False, depth="auto")
        assert config.name == "fast"

    def test_resolve_mode_max_clamps_synthesis_tokens_to_model_limit(self) -> None:
        """When max_mode and model limit is lower, synthesis tokens are clamped."""
        config = resolve_mode("claude-sonnet-4", max_mode=True, depth="extended")
        assert config.synthesis_max_output_tokens <= 64_000

    def test_resolve_mode_auto_none_model_returns_fast(self) -> None:
        """depth='auto' with None model returns fast mode."""
        config = resolve_mode(None, max_mode=False, depth="auto")
        assert config.name == "fast"
