"""Distilling a corpus into an agent skill.

The failure mode that matters: a skill whose frontmatter is missing or
whose name does not match its directory is silently ignored by the
loader. No error, no skill — so the shape is worth pinning.
"""

from __future__ import annotations

import pytest

from deep_agent.src.knowledge.skill_builder import (
    MAX_SKILL_CHARS,
    SourceDoc,
    build_prompt_for_reference,
    build_prompt_for_skill,
    ensure_frontmatter,
    reference_filename,
    slugify,
    strip_code_fence,
)


class TestNaming:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("HR Policies", "hr-policies"),
            ("  spaced  out  ", "spaced-out"),
            ("Already-Fine", "already-fine"),
            ("sym!!bols##here", "sym-bols-here"),
        ],
    )
    def test_names_become_safe_slugs(self, raw, expected):
        assert slugify(raw) == expected

    def test_an_empty_name_still_yields_something(self):
        assert slugify("") == "corpus"
        assert slugify("!!!") == "corpus"

    def test_slugs_are_bounded(self):
        assert len(slugify("x" * 200)) <= 48

    def test_reference_filenames_drop_the_source_extension(self):
        assert reference_filename("Travel Policy.docx") == "travel-policy.md"
        assert reference_filename("notes.md") == "notes.md"


class TestFrontmatter:
    def test_valid_frontmatter_is_left_alone(self):
        text = "---\nname: hr\ndescription: x\n---\n\n# Body"
        assert ensure_frontmatter(text, "hr", "d") == text

    def test_a_mismatched_name_is_corrected(self):
        """A name that disagrees with the directory means the loader
        silently skips the skill."""
        text = "---\nname: something-else\ndescription: x\n---\n\n# Body"
        out = ensure_frontmatter(text, "hr", "d")
        assert "name: hr" in out
        assert "something-else" not in out

    def test_missing_frontmatter_is_added(self):
        out = ensure_frontmatter("# Just a body", "hr", "my description")
        assert out.startswith("---\nname: hr")
        assert "my description" in out
        assert "# Just a body" in out

    def test_a_wrapping_code_fence_is_removed(self):
        """Models like to wrap markdown in a fence; the fence would
        become part of the frontmatter."""
        text = "```markdown\n---\nname: hr\ndescription: x\n---\n\n# Body\n```"
        out = ensure_frontmatter(text, "hr", "d")
        assert out.startswith("---")
        assert "```" not in out

    def test_frontmatter_without_a_name_gains_one(self):
        out = ensure_frontmatter("---\ndescription: x\n---\n\nbody", "hr", "d")
        assert "name: hr" in out


class TestCodeFences:
    def test_plain_text_is_untouched(self):
        assert strip_code_fence("no fence here") == "no fence here"

    def test_a_fence_is_stripped(self):
        assert strip_code_fence("```\ninner\n```") == "inner"

    def test_a_labelled_fence_is_stripped(self):
        assert strip_code_fence("```md\ninner\n```") == "inner"


class TestPrompts:
    def test_the_reference_prompt_forbids_invention(self):
        prompt = build_prompt_for_reference(SourceDoc("a.md", "content"))
        assert "Do not add anything the document does not say" in prompt
        assert "content" in prompt

    def test_the_reference_prompt_asks_for_lookup_not_narrative(self):
        """An agent reads a reference to look something up."""
        prompt = build_prompt_for_reference(SourceDoc("a.md", "x"))
        assert "look something up" in prompt

    def test_the_skill_prompt_demands_a_routing_description(self):
        """The description is how an agent decides to load the skill —
        it must describe when to use it, not the subject."""
        prompt = build_prompt_for_skill("hr", "HR", {"a.md": "body"})
        assert "Do not use for" in prompt
        assert "about *routing*" in prompt

    def test_the_skill_prompt_bounds_the_output(self):
        """SKILL.md is resident context whenever the skill is active."""
        prompt = build_prompt_for_skill("hr", "HR", {"a.md": "body"})
        assert str(MAX_SKILL_CHARS) in prompt

    def test_every_reference_is_listed(self):
        prompt = build_prompt_for_skill("hr", "HR", {"a.md": "x", "b.md": "y"})
        assert "references/a.md" in prompt
        assert "references/b.md" in prompt

    def test_it_asks_for_boundaries_so_the_agent_declines(self):
        prompt = build_prompt_for_skill("hr", "HR", {"a.md": "x"})
        assert "declines instead of inventing" in prompt or "does NOT cover" in prompt


class TestOutputHygiene:
    def test_latex_is_forbidden(self):
        """The model emitted `$\\le$` in a generated skill — it renders
        as literal LaTeX in a terminal and a chat bubble."""
        prompt = build_prompt_for_skill("hr", "HR", {"a.md": "x"})
        assert "LaTeX" in prompt and "Plain markdown only" in prompt

    def test_the_reference_bodies_are_only_truncated_for_the_prompt(self):
        """The files on disk keep their full text; only the prompt is
        bounded, or a long corpus would stop fitting."""
        long_body = "y" * 10_000
        prompt = build_prompt_for_skill("hr", "HR", {"a.md": long_body})
        assert len(prompt) < len(long_body) + 5_000


class TestDatasetPrompt:
    def test_it_forbids_inventing_figures(self):
        """A data dictionary that misstates a column is wrong in a way
        nobody checks."""
        from deep_agent.src.knowledge.skill_builder import (
            build_prompt_for_dataset_skill,
        )

        prompt = build_prompt_for_dataset_skill("s", "S", {"a.md": "| x |"})
        assert "measured facts" in prompt
        assert "Do not estimate" in prompt

    def test_join_candidates_are_marked_unverified(self):
        """Matched by name is not a relationship; stating one would put
        a join in the skill that does not work."""
        from deep_agent.src.knowledge.skill_builder import (
            build_prompt_for_dataset_skill,
        )

        prompt = build_prompt_for_dataset_skill(
            "s", "S", {"a.md": "x"}, [("id", ["a.csv", "b.csv"])]
        )
        assert "matched by name only" in prompt
        assert "Do not repeat this instruction" in prompt

    def test_no_join_section_without_shared_columns(self):
        from deep_agent.src.knowledge.skill_builder import (
            build_prompt_for_dataset_skill,
        )

        assert "## Joining" not in build_prompt_for_dataset_skill(
            "s", "S", {"a.md": "x"}
        )
