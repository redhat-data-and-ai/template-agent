"""PII substitution is recorded on the reply.

The restored text reads exactly like what the user typed, so without
a note the fact that the model saw something else cannot be
recovered — and that is the thing someone weighing the answer would
want to know.
"""

from __future__ import annotations

from types import SimpleNamespace

from deep_agent.src.pii.middleware import _note_scrubbed


class TestNoteScrubbed:
    def test_records_a_count_on_the_first_annotatable_message(self):
        message = SimpleNamespace(additional_kwargs={})
        _note_scrubbed([message], 3)
        assert message.additional_kwargs["pii_scrubbed"] == 3

    def test_records_a_count_never_the_values(self):
        message = SimpleNamespace(additional_kwargs={})
        _note_scrubbed([message], 2)
        assert message.additional_kwargs == {"pii_scrubbed": 2}

    def test_skips_a_message_that_cannot_carry_it(self):
        # Best effort: a missing note costs a line of provenance.
        unannotatable = SimpleNamespace(additional_kwargs=None)
        annotatable = SimpleNamespace(additional_kwargs={})
        _note_scrubbed([unannotatable, annotatable], 1)
        assert annotatable.additional_kwargs["pii_scrubbed"] == 1

    def test_does_not_raise_when_nothing_can_carry_it(self):
        _note_scrubbed([SimpleNamespace(additional_kwargs=None)], 1)
