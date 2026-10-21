"""Dataset profiling.

Every figure a dataset skill states is computed here. A model asked to
read rows invents a plausible distribution, and a data dictionary that
misstates a column is wrong in a way nobody checks — so these are the
tests that keep the numbers honest.
"""

from __future__ import annotations

import io

import pytest

from deep_agent.src.knowledge.dataset_profile import (
    CATEGORICAL_MAX_DISTINCT,
    find_join_keys,
    profile_bytes,
    profile_to_markdown,
)

CSV = (
    b"region,revenue,status,notes\n"
    b"EMEA,50000,active,first\n"
    b"APAC,31000,active,second\n"
    b"EMEA,42000,churned,\n"
)


class TestColumnFacts:
    def test_rows_and_columns_are_counted(self):
        p = profile_bytes("sales.csv", CSV)
        assert p.row_count == 3
        assert p.column_names == ["region", "revenue", "status", "notes"]

    def test_numeric_ranges_are_measured(self):
        p = profile_bytes("sales.csv", CSV)
        revenue = next(c for c in p.columns if c.name == "revenue")
        assert revenue.minimum == 31000
        assert revenue.maximum == 50000

    def test_nulls_are_counted(self):
        p = profile_bytes("sales.csv", CSV)
        notes = next(c for c in p.columns if c.name == "notes")
        assert notes.null_count == 1
        assert notes.null_pct == pytest.approx(33.3, abs=0.5)

    def test_distinct_counts_are_measured(self):
        p = profile_bytes("sales.csv", CSV)
        region = next(c for c in p.columns if c.name == "region")
        assert region.distinct == 2

    def test_low_cardinality_columns_yield_a_vocabulary(self):
        """The allowed values are what let an agent write a valid filter."""
        p = profile_bytes("sales.csv", CSV)
        status = next(c for c in p.columns if c.name == "status")
        assert status.values == ["active", "churned"]
        assert status.is_categorical

    def test_high_cardinality_columns_do_not(self):
        """Listing every value of a free-text column is noise."""
        rows = b"id\n" + b"\n".join(
            str(i).encode() for i in range(CATEGORICAL_MAX_DISTINCT + 10)
        )
        p = profile_bytes("ids.csv", rows)
        assert not p.columns[0].is_categorical

    def test_text_columns_get_no_misleading_range(self):
        """min/max on free text is alphabetical noise, not a range."""
        p = profile_bytes("sales.csv", CSV)
        notes = next(c for c in p.columns if c.name == "notes")
        assert notes.minimum is None and notes.maximum is None


class TestFormats:
    def test_tsv_is_read_with_the_right_delimiter(self):
        p = profile_bytes("a.tsv", b"a\tb\n1\t2\n")
        assert p.column_names == ["a", "b"]

    def test_jsonl_is_read(self):
        data = b'{"a": 1, "b": "x"}\n{"a": 2, "b": "y"}\n'
        p = profile_bytes("a.jsonl", data)
        assert p.row_count == 2

    def test_parquet_is_read(self):
        pa = pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq

        buf = io.BytesIO()
        pq.write_table(pa.table({"a": [1, 2, 3]}), buf)
        p = profile_bytes("a.parquet", buf.getvalue())
        assert p.row_count == 3

    def test_xlsx_is_read(self):
        openpyxl = pytest.importorskip("openpyxl")
        book = openpyxl.Workbook()
        book.active.append(["region", "revenue"])
        book.active.append(["EMEA", 50000])
        buf = io.BytesIO()
        book.save(buf)
        p = profile_bytes("a.xlsx", buf.getvalue())
        assert "region" in p.column_names

    def test_a_non_tabular_file_is_declined_not_guessed(self):
        assert profile_bytes("notes.md", b"# heading") is None

    def test_a_corrupt_file_does_not_raise(self):
        assert profile_bytes("a.parquet", b"not parquet") is None


class TestRendering:
    def test_the_sheet_states_measured_numbers(self):
        md = profile_to_markdown(profile_bytes("sales.csv", CSV))
        assert "31000" in md and "50000" in md
        assert "`revenue`" in md

    def test_vocabularies_are_rendered(self):
        md = profile_to_markdown(profile_bytes("sales.csv", CSV))
        assert "Allowed values" in md
        assert "active, churned" in md

    def test_sampling_is_disclosed(self):
        """A range computed from a sample must not read as the whole."""
        from deep_agent.src.knowledge.dataset_profile import DatasetProfile

        md = profile_to_markdown(DatasetProfile("big.csv", 10, [], truncated=True))
        assert "sampled" in md


class TestJoinKeys:
    def test_shared_column_names_are_found(self):
        a = profile_bytes("a.csv", b"id,x\n1,2\n")
        b = profile_bytes("b.csv", b"id,y\n1,3\n")
        assert find_join_keys([a, b]) == [("id", ["a.csv", "b.csv"])]

    def test_unshared_columns_are_not_reported(self):
        a = profile_bytes("a.csv", b"id,x\n1,2\n")
        b = profile_bytes("b.csv", b"id,y\n1,3\n")
        names = [n for n, _ in find_join_keys([a, b])]
        assert "x" not in names and "y" not in names

    def test_a_single_dataset_has_no_joins(self):
        assert find_join_keys([profile_bytes("a.csv", b"id\n1\n")]) == []


class TestVocabularyHonesty:
    """A vocabulary states a constraint. Getting it wrong writes a rule
    into the skill that the data does not actually have."""

    def test_numeric_columns_never_get_a_vocabulary(self):
        """Three distinct revenues in a sample is not an allowed-values
        list; it is three rows."""
        p = profile_bytes("sales.csv", CSV)
        revenue = next(c for c in p.columns if c.name == "revenue")
        assert revenue.values == []
        assert revenue.minimum is not None  # a range is the honest summary

    def test_nulls_are_a_rate_not_a_value(self):
        """An empty string among the allowed values reads as a value
        the column can take."""
        p = profile_bytes("sales.csv", CSV)
        notes = next(c for c in p.columns if c.name == "notes")
        assert "" not in notes.values
        assert notes.null_count == 1

    def test_a_blank_cell_counts_as_missing(self):
        """Without strings_can_be_null a blank reads as "" and the
        column reports 0% null while being half empty.

        Two columns, so the blank field is unambiguous — a blank line
        in a single-column CSV is just a trailing newline.
        """
        p = profile_bytes("a.csv", b"x,y\nvalue,1\n,2\n")
        x = next(c for c in p.columns if c.name == "x")
        assert x.null_count == 1
