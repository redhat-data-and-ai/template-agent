"""Unit tests for upload checksum manifest parsing and comparison."""

import hashlib
import json

import pytest

from deep_agent.aegra.upload_checksum import (
    ChecksumInvalid,
    compare_manifest,
    normalize_path,
    parse_digest,
    parse_manifest,
)

MAX = 500


def _sha(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


class TestParseDigest:
    def test_prefixed_digest(self):
        digest = _sha("A")
        assert parse_digest(f"sha256:{digest}") == digest

    def test_bare_digest(self):
        """A caller who pipes sha256sum straight through sends no prefix."""
        digest = _sha("A")
        assert parse_digest(digest) == digest

    def test_uppercase_hex_is_accepted(self):
        digest = _sha("A")
        assert parse_digest(digest.upper()) == digest

    def test_surrounding_whitespace_is_tolerated(self):
        digest = _sha("A")
        assert parse_digest(f"  sha256:{digest}\n") == digest

    def test_unsupported_algorithm_is_rejected(self):
        with pytest.raises(ChecksumInvalid, match="unsupported checksum algorithm"):
            parse_digest(f"md5:{'a' * 32}")

    def test_wrong_length_is_rejected(self):
        with pytest.raises(ChecksumInvalid, match="64 hex characters"):
            parse_digest("abc123")

    def test_non_hex_characters_are_rejected(self):
        with pytest.raises(ChecksumInvalid, match="64 hex characters"):
            parse_digest("z" * 64)

    def test_empty_digest_is_rejected(self):
        with pytest.raises(ChecksumInvalid):
            parse_digest("")


class TestNormalizePath:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("./docs/a.md", "docs/a.md"),
            ("docs//a.md", "docs/a.md"),
            ("docs/a.md", "docs/a.md"),
        ],
    )
    def test_equivalent_spellings_collapse(self, raw, expected):
        """find(1) prints './x', zipfile reports 'x' -- the caller shouldn't care."""
        assert normalize_path(raw) == expected

    @pytest.mark.parametrize("raw", [" a.md", "a.md ", " a.md "])
    def test_whitespace_is_preserved(self, raw):
        """A space is a legal part of a filename, so it names a different file.

        Stripping it folded " a.md" and "a.md" into one key, which let one
        entry's digest vouch for the other and published an undeclared file.
        """
        assert normalize_path(raw) == raw
        assert normalize_path(raw) != "a.md"


class TestParseManifest:
    def test_parses_prefixed_and_bare_values(self):
        manifest = parse_manifest(
            json.dumps({"a.md": f"sha256:{_sha('A')}", "b.md": _sha("B")}), MAX
        )
        assert manifest == {"a.md": _sha("A"), "b.md": _sha("B")}

    def test_keys_are_normalized(self):
        manifest = parse_manifest(json.dumps({"./docs/a.md": _sha("A")}), MAX)
        assert manifest == {"docs/a.md": _sha("A")}

    def test_invalid_json_is_rejected(self):
        with pytest.raises(ChecksumInvalid, match="not valid JSON"):
            parse_manifest("{not json", MAX)

    def test_deeply_nested_input_is_rejected(self):
        """json.loads raises RecursionError, which is not a ValueError.

        Unhandled it escapes the route's (ValueError, AssertionError)
        handler too, so the caller got a 500 for what is a bad field.
        """
        with pytest.raises(ChecksumInvalid, match="not valid JSON"):
            parse_manifest("[" * 100_000, MAX)

    def test_oversized_integer_literal_is_rejected(self):
        """CPython's int/str conversion guard raises a bare ValueError.

        That one did reach the client, but as a "malformed multipart body"
        400 from the route -- blaming the wrong part of the request, and
        skipping the checksum rejection audit.
        """
        with pytest.raises(ChecksumInvalid, match="not valid JSON"):
            parse_manifest('{"a.md": ' + "1" * 5000 + "}", MAX)

    def test_non_object_is_rejected(self):
        with pytest.raises(ChecksumInvalid, match="must be a JSON object"):
            parse_manifest(json.dumps(["a.md"]), MAX)

    def test_empty_manifest_is_rejected(self):
        """An empty manifest asserts nothing; a 200 would mean the opposite."""
        with pytest.raises(ChecksumInvalid, match="empty"):
            parse_manifest("{}", MAX)

    def test_non_string_value_is_rejected(self):
        with pytest.raises(ChecksumInvalid, match="not a string"):
            parse_manifest(json.dumps({"a.md": 123}), MAX)

    def test_bad_digest_names_the_path(self):
        with pytest.raises(ChecksumInvalid, match="a.md"):
            parse_manifest(json.dumps({"a.md": "nope"}), MAX)

    def test_duplicate_paths_after_normalization_are_rejected(self):
        """JSON keeps both spellings; silently dropping one loses an assertion."""
        raw = json.dumps({"./a.md": _sha("A"), "a.md": _sha("B")})
        with pytest.raises(ChecksumInvalid, match="duplicate path"):
            parse_manifest(raw, MAX)

    def test_empty_path_is_rejected(self):
        with pytest.raises(ChecksumInvalid, match="empty path"):
            parse_manifest(json.dumps({"": _sha("A")}), MAX)

    def test_entry_cap_is_enforced(self):
        raw = json.dumps({f"f{i}.md": _sha(str(i)) for i in range(5)})
        with pytest.raises(ChecksumInvalid, match="exceeds 3 entries"):
            parse_manifest(raw, 3)

    def test_manifest_at_the_cap_is_accepted(self):
        raw = json.dumps({f"f{i}.md": _sha(str(i)) for i in range(3)})
        assert len(parse_manifest(raw, 3)) == 3


class TestCompareManifest:
    def test_exact_match_returns_none(self):
        digests = {"a.md": _sha("A"), "b.md": _sha("B")}
        assert compare_manifest(digests, dict(digests)) is None

    def test_declared_file_that_did_not_land(self):
        """The truncated-archive case this feature exists for."""
        reason = compare_manifest(
            {"a.md": _sha("A"), "b.md": _sha("B")}, {"a.md": _sha("A")}
        )
        assert reason is not None
        assert "b.md" in reason and "not published" in reason

    def test_published_file_that_was_not_declared(self):
        reason = compare_manifest(
            {"a.md": _sha("A")}, {"a.md": _sha("A"), "extra.md": _sha("X")}
        )
        assert reason is not None
        assert "extra.md" in reason and "not listed" in reason

    def test_digest_mismatch_reports_both_sides(self):
        reason = compare_manifest({"a.md": _sha("A")}, {"a.md": _sha("B")})
        assert reason is not None
        assert _sha("A") in reason and _sha("B") in reason

    def test_reason_is_deterministic_across_runs(self):
        """Same payload, same message -- not whichever key iterated first.

        Insertion order is DESCENDING on purpose. With ascending keys an
        unsorted implementation also reaches f3.md first, so the assertion
        below would hold either way and the test could never fail. Built
        f19..f0, unsorted iteration hits f7.md first and sorted hits
        f3.md, so this now actually pins the sorted() in compare_manifest.
        """
        expected = {f"f{i}.md": _sha(str(i)) for i in reversed(range(20))}
        actual = {k: v for k, v in expected.items() if k not in ("f3.md", "f7.md")}
        reasons = {compare_manifest(expected, dict(actual)) for _ in range(5)}
        assert len(reasons) == 1
        assert "f3.md" in reasons.pop()

    def test_long_paths_are_truncated_in_the_message(self):
        long_path = "d/" * 200 + "a.md"
        reason = compare_manifest({long_path: _sha("A")}, {})
        assert reason is not None
        assert len(reason) < len(long_path)
        assert "…" in reason
