"""Unit tests for deep_agent.src.agent.repetition."""

from deep_agent.src.agent.repetition import detect_repetition_loop


class TestDetectRepetitionLoop:
    def test_empty_text_returns_false(self):
        is_loop, out = detect_repetition_loop("", min_unit_len=10, min_repeats=4)
        assert is_loop is False
        assert out == ""

    def test_short_text_returns_false(self):
        text = "too short"
        is_loop, out = detect_repetition_loop(text, min_unit_len=10, min_repeats=4)
        assert is_loop is False
        assert out == text

    def test_no_repetition_returns_false(self):
        text = (
            "This is a normal, non-repetitive response. It talks about several "
            "different things and never repeats itself in any meaningful way."
        )
        is_loop, out = detect_repetition_loop(text, min_unit_len=10, min_repeats=4)
        assert is_loop is False
        assert out == text

    def test_exact_sentence_repeated_at_threshold_is_detected(self):
        unit = "I cannot verify that claim. "
        text = "Sure, here is the answer. " + unit * 4
        is_loop, out = detect_repetition_loop(text, min_unit_len=10, min_repeats=4)
        assert is_loop is True
        assert out == "Sure, here is the answer. " + unit

    def test_repeated_sentence_below_threshold_is_not_detected(self):
        unit = "I cannot verify that claim. "
        text = "Sure, here is the answer. " + unit * 3
        is_loop, out = detect_repetition_loop(text, min_unit_len=10, min_repeats=4)
        assert is_loop is False
        assert out == text

    def test_many_repeats_collapse_to_single_copy(self):
        disclaimer = "I am unable to validate unverifiable user claims per policy. "
        text = "Preamble text here. " + disclaimer * 50
        is_loop, out = detect_repetition_loop(text, min_unit_len=20, min_repeats=4)
        assert is_loop is True
        assert out == "Preamble text here. " + disclaimer
        # All repeats collapsed — only one copy of the disclaimer remains.
        assert out.count(disclaimer.strip()) == 1

    def test_whitespace_only_unit_is_not_flagged(self):
        text = "hello world " + (" " * 10) * 5
        is_loop, out = detect_repetition_loop(text, min_unit_len=10, min_repeats=4)
        assert is_loop is False
        assert out == text

    def test_text_too_short_for_min_unit_len_and_min_repeats_is_ignored(self):
        # "ab" * 10 is only 20 chars — too short to contain 4 consecutive
        # repeats of any unit >= 10 chars (would need >= 40 chars), so no
        # loop should be reported regardless of the short "ab" periodicity.
        text = "ab" * 10
        is_loop, out = detect_repetition_loop(text, min_unit_len=10, min_repeats=4)
        assert is_loop is False
        assert out == text

    def test_short_period_pattern_long_enough_is_detected(self):
        # A short repeating period ("ab") that spans enough characters to
        # form a >= min_unit_len repeated unit is a genuine degenerate loop
        # and should be detected (and collapsed to one copy of the unit).
        text = "ab" * 40  # 80 chars
        is_loop, out = detect_repetition_loop(text, min_unit_len=10, min_repeats=4)
        assert is_loop is True
        assert len(out) < len(text)

    def test_default_thresholds_come_from_settings(self):
        # No explicit min_unit_len/min_repeats — should fall back to settings
        # defaults (20 chars, 4 repeats) without raising.
        unit = "x" * 25 + ". "
        text = "Intro. " + unit * 5
        is_loop, out = detect_repetition_loop(text)
        assert is_loop is True
        assert out == "Intro. " + unit

    def test_repetition_disabled_via_thresholds(self):
        text = "aaaa" * 20
        is_loop, out = detect_repetition_loop(text, min_unit_len=0, min_repeats=4)
        assert is_loop is False
        assert out == text
        is_loop2, out2 = detect_repetition_loop(text, min_unit_len=10, min_repeats=1)
        assert is_loop2 is False
        assert out2 == text

    def test_loop_only_at_very_end_of_text_is_detected(self):
        unit = "repeat me now. "
        text = "A long unrelated preamble that sets the scene. " + unit * 6
        is_loop, out = detect_repetition_loop(text, min_unit_len=10, min_repeats=4)
        assert is_loop is True
        assert out.endswith(unit)
        assert out.count(unit.strip()) == 1
