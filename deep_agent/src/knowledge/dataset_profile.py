"""Profile a dataset into facts a skill can be written from.

Every number here is **computed**, never produced by a model. A model
asked to read rows will invent a plausible distribution, and a data
dictionary that misstates a column's range is worse than no dictionary
at all — it is wrong in a way nobody checks.

So the division is strict: this module measures, and the model is
later allowed only to narrate what it measured.

pyarrow and openpyxl are already in the image, so this adds no
dependency and no second service to keep running.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

# A column with few distinct values is a vocabulary the agent should
# know; past this it is free text and listing it is noise.
CATEGORICAL_MAX_DISTINCT = 25

# Rows read for profiling. Enough to characterise, bounded so a large
# file does not have to fit in memory.
SAMPLE_ROWS = 50_000


@dataclass
class ColumnProfile:
    """What one column of a dataset contains."""

    name: str
    dtype: str
    null_count: int
    row_count: int
    distinct: int | None = None
    minimum: Any = None
    maximum: Any = None
    # Populated only for low-cardinality columns — the vocabulary.
    values: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)

    @property
    def null_pct(self) -> float:
        """Share of rows that are null, as a percentage."""
        return (self.null_count / self.row_count * 100) if self.row_count else 0.0

    @property
    def is_categorical(self) -> bool:
        """Does this column have a small enough value set to enumerate?"""
        return bool(self.values)


@dataclass
class DatasetProfile:
    """Shape and column summary for one dataset."""

    name: str
    row_count: int
    columns: list[ColumnProfile]
    truncated: bool = False

    @property
    def column_names(self) -> list[str]:
        """Column names, in file order."""
        return [c.name for c in self.columns]


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text[:60]


def profile_table(table: Any, name: str, truncated: bool = False) -> DatasetProfile:
    """Profile a pyarrow Table."""
    import pyarrow.compute as pc

    rows = table.num_rows
    columns: list[ColumnProfile] = []

    for field_ in table.schema:
        col = table.column(field_.name)
        profile = ColumnProfile(
            name=field_.name,
            dtype=str(field_.type),
            null_count=col.null_count,
            row_count=rows,
        )

        try:
            distinct = pc.count_distinct(col).as_py()
            profile.distinct = distinct
        except Exception:
            distinct = None

        # Ranges only where ordering means something. min/max on a
        # free-text column is alphabetical noise.
        if _is_ordered(field_.type):
            try:
                profile.minimum = pc.min(col).as_py()
                profile.maximum = pc.max(col).as_py()
            except Exception:
                pass

        # Strings only. A numeric column with three distinct values in
        # a sample is not a vocabulary — listing 31000, 42000, 50000 as
        # "allowed values" states a constraint the data does not have.
        if (
            _is_textual(field_.type)
            and distinct is not None
            and 0 < distinct <= CATEGORICAL_MAX_DISTINCT
        ):
            try:
                counts = pc.value_counts(col)
                # Nulls are reported as a null rate, not as a member of
                # the vocabulary — an empty string listed among the
                # allowed values reads as a value the column can take.
                profile.values = sorted(
                    _fmt(entry["values"])
                    for entry in counts.to_pylist()
                    if entry["values"] is not None
                )
            except Exception:
                pass
        else:
            profile.examples = [
                _fmt(v) for v in col.slice(0, 3).to_pylist() if v is not None
            ]

        columns.append(profile)

    return DatasetProfile(
        name=name, row_count=rows, columns=columns, truncated=truncated
    )


def _is_textual(arrow_type: Any) -> bool:
    import pyarrow as pa

    # bool() because pyarrow ships no stubs, so these predicates are
    # Any as far as the type checker is concerned.
    return bool(
        pa.types.is_string(arrow_type)
        or pa.types.is_large_string(arrow_type)
        or pa.types.is_dictionary(arrow_type)
    )


def _is_ordered(arrow_type: Any) -> bool:
    import pyarrow as pa

    return bool(
        pa.types.is_integer(arrow_type)
        or pa.types.is_floating(arrow_type)
        or pa.types.is_decimal(arrow_type)
        or pa.types.is_temporal(arrow_type)
    )


def profile_bytes(filename: str, data: bytes) -> DatasetProfile | None:
    """Profile a dataset file. Returns None if it is not tabular."""
    import io

    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    try:
        if suffix in ("csv", "tsv"):
            from pyarrow import csv as pacsv

            delimiter = "\t" if suffix == "tsv" else ","
            # strings_can_be_null: otherwise a blank cell reads as ""
            # and appears in the vocabulary as a real value, while the
            # null rate reports 0% for a column that is a third empty.
            table = pacsv.read_csv(
                io.BytesIO(data),
                parse_options=pacsv.ParseOptions(delimiter=delimiter),
                convert_options=pacsv.ConvertOptions(strings_can_be_null=True),
            )
        elif suffix == "parquet":
            import pyarrow.parquet as pq

            table = pq.read_table(io.BytesIO(data))
        elif suffix in ("json", "jsonl", "ndjson"):
            from pyarrow import json as pajson

            table = pajson.read_json(io.BytesIO(data))
        elif suffix == "xlsx":
            return _profile_xlsx(filename, data)
        else:
            return None
    except Exception as exc:
        logger.warning("Could not profile %s: %s", filename, exc)
        return None

    truncated = table.num_rows > SAMPLE_ROWS
    if truncated:
        table = table.slice(0, SAMPLE_ROWS)
    return profile_table(table, filename, truncated)


def _profile_xlsx(filename: str, data: bytes) -> DatasetProfile | None:
    """Profile the first sheet of a workbook.

    First sheet only: a workbook with several unrelated sheets is
    several datasets, and merging their columns would describe a table
    that does not exist.
    """
    import io

    import pyarrow as pa
    from openpyxl import load_workbook

    book = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    sheet = book.worksheets[0]
    rows = sheet.iter_rows(values_only=True)

    try:
        header = [
            str(h) if h is not None else f"col{i}" for i, h in enumerate(next(rows))
        ]
    except StopIteration:
        book.close()
        return None

    collected: list[list[Any]] = [[] for _ in header]
    count = 0
    for row in rows:
        for index in range(len(header)):
            collected[index].append(row[index] if index < len(row) else None)
        count += 1
        if count >= SAMPLE_ROWS:
            break
    book.close()

    if not count:
        return None

    arrays = []
    for values in collected:
        try:
            arrays.append(pa.array(values))
        except Exception:
            # Mixed types in one column: fall back to strings rather
            # than dropping the column entirely.
            arrays.append(pa.array([None if v is None else str(v) for v in values]))

    table = pa.table(dict(zip(header, arrays, strict=True)))
    return profile_table(table, f"{filename}:{sheet.title}", count >= SAMPLE_ROWS)


def profile_to_markdown(profile: DatasetProfile) -> str:
    """A reference sheet, rendered deterministically.

    Not model-written: this is the ground truth the model is later
    allowed to summarise, so it must not pass through one.
    """
    lines = [
        f"# {profile.name}",
        "",
        f"- Rows profiled: {profile.row_count:,}"
        + (" (sampled from a larger file)" if profile.truncated else ""),
        f"- Columns: {len(profile.columns)}",
        "",
        "## Columns",
        "",
        "| Column | Type | Nulls | Distinct | Range |",
        "|---|---|---|---|---|",
    ]

    for col in profile.columns:
        span = ""
        if col.minimum is not None or col.maximum is not None:
            span = f"{_fmt(col.minimum)} … {_fmt(col.maximum)}"
        lines.append(
            f"| `{col.name}` | {col.dtype} | {col.null_pct:.0f}% "
            f"| {col.distinct if col.distinct is not None else ''} | {span} |"
        )

    categorical = [c for c in profile.columns if c.is_categorical]
    if categorical:
        lines += ["", "## Allowed values", ""]
        for col in categorical:
            lines.append(f"- `{col.name}`: {', '.join(col.values)}")

    examples = [c for c in profile.columns if c.examples]
    if examples:
        lines += ["", "## Example values", ""]
        for col in examples:
            lines.append(f"- `{col.name}`: {', '.join(col.examples)}")

    return "\n".join(lines)


def find_join_keys(profiles: list[DatasetProfile]) -> list[tuple[str, list[str]]]:
    """Columns shared by name across datasets — candidate join keys.

    By name only. Confirming a real relationship needs the values, and
    guessing one would put a join in the skill that does not work.
    """
    seen: dict[str, list[str]] = {}
    for profile in profiles:
        for name in profile.column_names:
            seen.setdefault(name, []).append(profile.name)
    return sorted(
        (name, sorted(where)) for name, where in seen.items() if len(where) > 1
    )
