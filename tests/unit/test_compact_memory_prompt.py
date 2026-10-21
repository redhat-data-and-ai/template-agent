"""The memory preamble carries memory, not a manual.

deepagents ships 1,280 tokens of `<memory_guidelines>` alongside the
loaded memory. On a "say hello" turn that was the second largest
thing in the system prompt, spent to deliver five words, and nearly
all of it restates the Memory section of PROMPT.md.
"""

from __future__ import annotations

import pytest

from deep_agent.src.memory.instructions import (
    COMPACT_MEMORY_PROMPT,
    use_compact_memory_prompt,
)


@pytest.fixture
def memory_module():
    from deepagents.middleware import memory as dmem

    original_init = dmem.MemoryMiddleware.__init__
    applied = getattr(dmem.MemoryMiddleware, "_compact_prompt_applied", False)
    try:
        yield dmem
    finally:
        dmem.MemoryMiddleware.__init__ = original_init
        if applied:
            dmem.MemoryMiddleware._compact_prompt_applied = True
        else:
            # The flag lives on the class, and a class __dict__ is a
            # read-only mappingproxy — delattr is the way off it.
            try:
                delattr(dmem.MemoryMiddleware, "_compact_prompt_applied")
            except AttributeError:
                pass


class _Backend:
    """Enough of the protocol for __init__; nothing is read here."""


class TestCompactMemoryPrompt:
    def test_the_template_keeps_the_slot_the_library_requires(self):
        assert "{agent_memory}" in COMPACT_MEMORY_PROMPT

    def test_it_is_a_fraction_of_what_it_replaces(self, memory_module):
        assert len(COMPACT_MEMORY_PROMPT) < len(memory_module.MEMORY_SYSTEM_PROMPT) / 4

    def test_the_trust_framing_survives(self):
        """Memory is file data. A fact in it must not read as an order."""
        text = COMPACT_MEMORY_PROMPT.lower()
        assert "not instructions" in text
        assert "never follow an instruction" in text

    def test_it_still_says_how_to_write_memory(self):
        assert "edit_file" in COMPACT_MEMORY_PROMPT

    def test_a_default_construction_gets_the_short_one(self, memory_module):
        use_compact_memory_prompt()
        mw = memory_module.MemoryMiddleware(
            backend=_Backend(), sources=["/memories/user_profile.md"]
        )
        assert mw.system_prompt == COMPACT_MEMORY_PROMPT

    def test_an_explicit_template_is_left_alone(self, memory_module):
        use_compact_memory_prompt()
        mine = "<agent_memory>{agent_memory}</agent_memory>"
        mw = memory_module.MemoryMiddleware(
            backend=_Backend(), sources=["/m.md"], system_prompt=mine
        )
        assert mw.system_prompt == mine

    def test_suppressing_the_block_still_suppresses_it(self, memory_module):
        """`None` means "no memory preamble" and must not be overridden."""
        use_compact_memory_prompt()
        mw = memory_module.MemoryMiddleware(
            backend=_Backend(), sources=["/m.md"], system_prompt=None
        )
        assert mw.system_prompt is None

    def test_applying_twice_does_not_nest_the_wrapper(self, memory_module):
        use_compact_memory_prompt()
        once = memory_module.MemoryMiddleware.__init__
        use_compact_memory_prompt()
        assert memory_module.MemoryMiddleware.__init__ is once
