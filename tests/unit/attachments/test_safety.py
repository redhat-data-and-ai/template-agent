"""Uploaded filenames are attacker-controlled and end up in a path."""

from __future__ import annotations

import pytest

from deep_agent.src.attachments.safety import (
    MAX_NAME_LENGTH,
    sanitise_filename,
    unique_name,
)


class TestPathTraversal:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("../../etc/passwd", "passwd"),
            ("..\\..\\windows\\win.ini", "win.ini"),
            ("/absolute/path/file.txt", "file.txt"),
            ("./relative.txt", "relative.txt"),
            ("dir/sub/deep.md", "deep.md"),
        ],
    )
    def test_directory_components_are_stripped_not_escaped(self, raw, expected):
        """A name is a name; anything path-shaped is someone trying to
        write outside the mount."""
        assert sanitise_filename(raw) == expected

    def test_no_separator_survives(self):
        for raw in ["a/b", "a\\b", "..", "../..", "...."]:
            out = sanitise_filename(raw)
            assert "/" not in out and "\\" not in out

    def test_a_null_byte_cannot_truncate_the_path(self):
        """Null bytes terminate strings in the C APIs underneath."""
        assert "\x00" not in sanitise_filename("evil\x00.sh")

    def test_unicode_lookalikes_are_normalised_first(self):
        # Fullwidth solidus normalises to '/' under NFKC, so it must be
        # treated as a separator rather than kept as a character.
        assert "/" not in sanitise_filename("a／..／b.txt")


class TestNameHygiene:
    def test_an_ordinary_name_is_untouched(self):
        assert sanitise_filename("quarterly-report_v2.pdf") == "quarterly-report_v2.pdf"

    def test_an_empty_name_gets_a_placeholder(self):
        assert sanitise_filename("") == "upload"
        assert sanitise_filename("   ") == "upload"

    def test_windows_reserved_names_are_defused(self):
        """A fork may well mount the same volume from Windows."""
        assert sanitise_filename("con.txt") == "con_file.txt"
        assert sanitise_filename("LPT1.log").lower().startswith("lpt1_file")

    def test_long_names_are_truncated_but_keep_their_extension(self):
        out = sanitise_filename("a" * 400 + ".txt")
        assert len(out) <= MAX_NAME_LENGTH
        assert out.endswith(".txt")

    def test_exotic_characters_are_replaced(self):
        out = sanitise_filename("rm -rf *; $(whoami).txt")
        assert all(c.isalnum() or c in "._-" for c in out)


class TestCollisions:
    def test_a_free_name_is_kept(self):
        assert unique_name("a.txt", set()) == "a.txt"

    def test_a_clash_is_suffixed_rather_than_overwriting(self):
        """Two files called report.pdf is normal; losing one is not."""
        assert unique_name("report.pdf", {"report.pdf"}) == "report-2.pdf"

    def test_suffixing_continues_past_the_second(self):
        taken = {"r.pdf", "r-2.pdf", "r-3.pdf"}
        assert unique_name("r.pdf", taken) == "r-4.pdf"

    def test_extensionless_names_still_get_suffixed(self):
        assert unique_name("README", {"README"}) == "README-2"
