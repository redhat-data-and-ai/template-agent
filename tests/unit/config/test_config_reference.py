"""The config reference must not drift from the models.

Hand-written config docs answer "what can I change?" once and then rot.
Generating from the Pydantic models means the reference cannot disagree
with the code that enforces it — but only if something checks that the
committed file was regenerated.
"""

from pathlib import Path

import pytest

REPO = Path(__file__).parents[3]
REFERENCE = REPO / "docs" / "config-reference.md"


@pytest.fixture(scope="module")
def generated() -> str:
    import sys

    sys.path.insert(0, str(REPO))
    from scripts.config_reference import build

    return build()


class TestCommittedFileIsCurrent:
    def test_reference_exists(self):
        assert REFERENCE.is_file(), "run: make config-reference"

    def test_reference_matches_the_models(self, generated: str):
        """Fails when a config field is added without regenerating."""
        assert REFERENCE.read_text() == generated, (
            "docs/config-reference.md is stale — run: make config-reference"
        )


class TestCoverage:
    def test_documents_a_realistic_number_of_keys(self, generated: str):
        """Guards against a generator that silently emits empty tables."""
        assert generated.count("\n| `") > 50

    @pytest.mark.parametrize(
        "key",
        [
            "human_approval.policies",
            "human_approval.default_decision",
            "effects",
            "enabled",
        ],
    )
    def test_documents_keys_people_actually_edit(self, generated: str, key: str):
        assert f"`{key}`" in generated

    def test_covers_every_validated_file(self, generated: str):
        """Anything the validator checks should be documented."""
        for where in (
            "runtime/agent.yaml",
            "runtime/observability.yaml",
            "runtime/router.yaml",
        ):
            assert where in generated, f"{where} missing from the reference"

    def test_names_the_files_it_does_not_cover(self, generated: str):
        """Silence about PROMPT.md or skills/ reads as 'not configurable'."""
        for unvalidated in ("PROMPT.md", "mcp.json", "skills/"):
            assert unvalidated in generated


class TestRendering:
    def test_union_types_do_not_break_table_cells(self, generated: str):
        """A bare pipe inside a cell ends the column early."""
        for line in generated.splitlines():
            if line.startswith("| `") and "'all'" in line:
                assert r"\|" in line, f"unescaped union pipe: {line}"

    def test_every_table_row_has_four_columns(self, generated: str):
        for line in generated.splitlines():
            if not line.startswith("| `"):
                continue
            # Escaped pipes are content, not separators.
            columns = line.replace(r"\|", "").split("|")
            assert len(columns) == 6, f"malformed row: {line}"
